"""Original GLNN student on the FAITH reproduction's teacher and public split.

Uses the GLNN authors' student architecture/config and training routine directly.
This is an explicitly specified reference; FAITH's GLNN rerun config is absent.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DGLBACKEND'] = 'pytorch'

import argparse
import hashlib
import json
import logging
import sys
import time
import numpy as np
import torch
import dgl

sys.path.insert(0, str(ROOT / 'glnn-upstream'))
from models import Model
from utils import get_training_config, get_evaluator, set_seed
from train_and_eval import distill_run_transductive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / 'results/glnn_source_reference'
    output.mkdir(exist_ok=True)
    logger = logging.getLogger('glnn_reference')
    logger.addHandler(logging.NullHandler())
    results = []
    for name in args.datasets:
        cls = {'cora': dgl.data.CoraGraphDataset, 'citeseer': dgl.data.CiteseerGraphDataset, 'pubmed': dgl.data.PubmedGraphDataset}[name]
        graph = cls(raw_dir=str(ROOT / 'dgl-reproduction-data'), verbose=False)[0]
        x, y = graph.ndata['feat'], graph.ndata['label']
        train, valid, test = [graph.ndata[k + '_mask'].nonzero().flatten() for k in ['train', 'val', 'test']]
        assert len(valid) == 500
        for seed in [42, 43, 45, 48, 52]:
            destination = output / f'{name}_seed{seed}'
            destination.mkdir(exist_ok=True)
            metric_file = destination / 'metrics.json'
            if metric_file.exists():
                results.append(json.loads(metric_file.read_text()))
                continue
            teacher_path = ROOT / 'results/released_reproduction' / name / f'seed{seed}' / 'teacher.pt'
            if not teacher_path.exists():
                raise RuntimeError(f'FAITH teacher not yet available: {teacher_path}')
            set_seed(seed)
            conf = dict(seed=seed, device='cpu', batch_size=512, lamb=0., max_epoch=200, patience=50, eval_interval=1, norm_type='none', dataset=name, feat_dim=x.shape[1], label_dim=int(y.max()) + 1)
            conf.update(get_training_config(str(ROOT / 'glnn-upstream/train.conf.yaml'), 'MLP', name))
            (destination / 'config.json').write_text(json.dumps(conf, indent=2))
            model = Model(conf)
            optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'], weight_decay=conf['weight_decay'])
            teacher_logits = torch.load(teacher_path, map_location='cpu')['logits']
            # GLNN expects teacher log probabilities, not raw logits.
            targets = teacher_logits.log_softmax(1)
            # Preserve the GLNN source's selection of soft-label training nodes.
            soft_indices = torch.cat([train, valid, test])
            curve = []
            start = time.perf_counter()
            logits, val_acc, test_acc = distill_run_transductive(conf, model, x, y, targets, (train, soft_indices, valid, test), torch.nn.NLLLoss(), torch.nn.KLDivLoss(reduction='batchmean', log_target=True), get_evaluator(name), optimizer, logger, curve)
            assert abs(get_evaluator(name)(logits[test], y[test]) - test_acc) < 1e-7
            result = dict(dataset=name, seed=seed, accuracy=test_acc, validation_accuracy=val_acc, epochs=len(curve), seconds=time.perf_counter()-start, source='original GLNN student routine and released student YAML', teacher_source='FAITH released reproduction checkpoint', data_protocol='FAITH public DGL citation split', soft_training_nodes=len(soft_indices), original_glnn_paper_protocol=False, faithful_to_unreleased_faith_glnn_rerun_config='unknown', teacher_checkpoint_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest())
            torch.save(dict(state=model.state_dict(), log_probabilities=logits), destination / 'student.pt')
            (destination / 'curve.json').write_text(json.dumps(curve, indent=2))
            metric_file.write_text(json.dumps(result, indent=2))
            results.append(result)
            print(json.dumps(result), flush=True)
    (output / 'metrics.json').write_text(json.dumps(results, indent=2))
    for name in args.datasets:
        values = [r['accuracy']*100 for r in results if r['dataset'] == name]
        print(name, float(np.mean(values)), float(np.std(values)), flush=True)


if __name__ == '__main__':
    main()
