"""Lock a positive OSA coefficient using model-validation labels only.

Called after a direct-transfer diagnostic failed. This is explicitly an
exploratory validation-only follow-up, not a new untouched test benchmark.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import faith_ogb as O

ROOT, torch = O.ROOT, O.torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Host GPU access required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    O.C.support.released_guide.F = O.F
    output = ROOT / 'results/faith_ogb_tuning'
    output.mkdir(parents=True, exist_ok=True)
    grid = [1., .1, .01, .001]
    protocol = dict(criterion='model-validation accuracy only', seed=42,
                    updates_per_candidate=150, positive_reference_gamma_multipliers=grid,
                    tie_break='larger positive gamma', beta=.02,
                    calibration_labels_used=False, test_labels_used=False,
                    preceding_trial='Failed direct transfer; arxiv test results had been observed; follow-up is exploratory',
                    implementation_sha256=O.C.digest(Path(__file__)),
                    training_implementation_sha256=O.C.digest(ROOT / 'faith_ogb.py'))
    O.write(output / 'protocol.json', protocol)
    args = SimpleNamespace(data=ROOT / 'ogb-data', output=output, max_epochs=150, selection=None)
    selection = {}
    for dataset in ['ogbn-arxiv', 'ogbn-products']:
        data, base = O.prepare(dataset, args)
        source = ROOT / 'results/faith_ogb_transfer_diagnostic' / dataset / 'seed42/teacher.pt'
        checkpoint = torch.load(source, map_location='cuda')
        teacher = checkpoint['logits']
        teacher_energy = O.node_energy(teacher, data['adjacency'], data['degree']).detach()
        data['adj_teacher_energy'] = torch.sparse.mm(data['adjacency'], teacher_energy[:, None]).squeeze(1)
        data['teacher_edge_norm'] = (data['degree'] * teacher_energy.square()).sum().sqrt().clamp_min(1e-8)
        O.C.support.released_utils.set_seed(1042)
        initial = O.C.support.released_models.Model(base, model_type='student').cuda()
        candidates = []
        for factor in grid:
            config = dict(base, gamma=base['gamma'] * factor, seed=42, patience=150)
            O.C.support.released_utils.set_seed(2042)
            model = copy.deepcopy(initial)
            folder = output / dataset / f'multiplier{factor}'
            folder.mkdir(parents=True, exist_ok=True)
            O.write(folder / 'config.json', config)
            _, trace = O.fit(model, data, config, folder / 'FAITH.pt', teacher, teacher_energy)
            assert trace[-1]['epoch'] == 150
            score = max(t['validation_accuracy'] for t in trace)
            candidates.append(dict(multiplier=factor, gamma=config['gamma'], validation_accuracy=score,
                                   best_epoch=max(t['epoch'] for t in trace if t['validation_accuracy'] == score),
                                   updates=150, checkpoint_sha256=O.C.digest(folder / 'FAITH.pt')))
            print(json.dumps(dict(dataset=dataset, candidate_complete=True, **candidates[-1])), flush=True)
            del model
        chosen = max(candidates, key=lambda r: (r['validation_accuracy'], r['gamma']))
        selection[dataset] = dict(criterion='model-validation accuracy only', selected_gamma=chosen['gamma'],
                                 selected_multiplier=chosen['multiplier'], candidates=candidates,
                                 teacher_sha256=O.C.digest(source), protocol_sha256=O.C.digest(output / 'protocol.json'))
        O.write(ROOT / 'results/faith_ogb/validation_selection.json', selection)
        print(json.dumps(dict(dataset=dataset, selected_gamma=chosen['gamma'], validation_accuracy=chosen['validation_accuracy'])), flush=True)
        del data, teacher, teacher_energy, initial, checkpoint
        torch.cuda.empty_cache()
    print('Validation-only positive coefficients locked for the final five-seed study.', flush=True)


if __name__ == '__main__':
    main()
