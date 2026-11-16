"""Run released FAITH models and training loops, with recorded runtime repairs.

CPU DGL is used because the official CUDA wheel server is unavailable.
The previous pilot and its models are not imported here.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib-reproduction')

import argparse
import contextlib
import copy
import functools
import hashlib
import inspect
import json
import re
import shutil
import statistics
import sys
import time

import dgl
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch.nn import functional as F

sys.path.insert(0, str(ROOT / 'upstream/src_trans'))
import dataloader as released_data
import models as released_models
import train_and_eval as released_training
import utils as released_utils
import utils_guide as released_guide

PAPER = {
    'cora': {'SAGE': (82.18, .95), 'GLNN': (81.50, 1.21), 'FAITH': (83.96, .47)},
    'citeseer': {'SAGE': (70.86, .35), 'GLNN': (71.46, 1.79), 'FAITH': (74.10, .60)},
    'pubmed': {'SAGE': (77.38, .81), 'GLNN': (77.97, 3.35), 'FAITH': (81.53, .33)},
}
GAMMA = {'cora': 2e-5, 'citeseer': 1e-5, 'pubmed': 2e-6}
SOURCE_SHA = '94be4423aee12cdf4d4d098d369fe472aeedcbe7'


def write_json(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare_data():
    """Supply original Planetoid pickles to DGL's own processor, unchanged.

    Only the Cora filename prefix is renamed to DGL's cora_v2 convention.
    This avoids a second dependency on the unavailable data.dgl.ai server.
    """
    raw_dir = ROOT / 'dgl-reproduction-data'
    hashes = {}
    for public_name in ['Cora', 'CiteSeer', 'PubMed']:
        name = public_name.lower()
        dgl_name = 'cora_v2' if name == 'cora' else name
        url = f'https://data.dgl.ai/dataset/{dgl_name}.zip'
        directory = raw_dir / (dgl_name + '_' + hashlib.sha1(url.encode()).hexdigest()[:8])
        directory.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / 'data' / public_name / 'raw').glob(f'ind.{name}.*'):
            target = directory / source.name.replace(f'ind.{name}.', f'ind.{dgl_name}.')
            if not target.exists():
                shutil.copyfile(source, target)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            assert hashlib.sha256(target.read_bytes()).hexdigest() == source_hash
            hashes[str(target.relative_to(ROOT))] = source_hash
    # Keep the released load_dgl_data call path, changing storage location only.
    for class_name in ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset']:
        original = getattr(dgl.data, class_name)
        setattr(dgl.data, class_name, functools.partial(original, raw_dir=str(raw_dir), verbose=False))
    return hashes


class StaticTensors:
    def __init__(self):
        self.clear()

    def clear(self):
        self.delta = None
        self.adjacency = None
        self.fairness_laplacian = None
        self.uncached_fairness = released_training.get_IF if not hasattr(self, 'uncached_fairness') else self.uncached_fairness

    def matrices(self, delta, adjacency):
        if self.delta is None:
            values = torch.tensor(delta.data, dtype=torch.float32)
            indices = torch.tensor(np.stack([delta.row, delta.col]), dtype=torch.int64)
            self.delta = torch.sparse_coo_tensor(indices, values, size=delta.shape, dtype=torch.float32)
            self.adjacency = torch.tensor(adjacency, dtype=torch.float32).to_sparse()
        return self.delta, self.adjacency

    def fairness(self, logits, features, adjacency):
        if self.fairness_laplacian is None:
            similarity = released_guide.calculate_similarity_matrix(sp.csr_matrix(adjacency), features, metric='cosine')
            lap = released_guide.laplacian(similarity)
            self.fairness_laplacian = released_guide.convert_sparse_matrix_to_sparse_tensor(lap).to(logits.device)
        # Same operation order as the source; only graph-static construction is cached.
        return torch.trace(torch.mm(logits.t(), torch.sparse.mm(self.fairness_laplacian, logits))).item() / 1000


STATIC = StaticTensors()


def install_runtime_repairs(output):
    # The source function refers to F without importing it in utils_guide.py.
    released_guide.F = F
    original_source = inspect.getsource(released_training.train_mini_batch)
    cpu_source = original_source.replace("device='cuda'", 'device=logits.device')
    old_delta = """    values_delta = torch.tensor(Delta.data, dtype=torch.float32, device=logits.device)
    indices_delta = torch.tensor([Delta.row, Delta.col], dtype=torch.int64, device=logits.device)
    Delta_sparse = torch.sparse_coo_tensor(indices_delta, values_delta, size=Delta.shape, dtype=torch.float32)"""
    assert old_delta in cpu_source
    cached_source = cpu_source.replace(old_delta, '    Delta_sparse, adj_sparse = _static_matrices(Delta, adj)')
    old_adj = '    adj_sparse = torch.tensor(adj, dtype=torch.float32).to_sparse()'
    assert old_adj in cached_source
    cached_source = cached_source.replace(old_adj, '')
    namespace = released_training.__dict__
    namespace['_static_matrices'] = STATIC.matrices
    exec(compile(cached_source, '<released-train-step-with-runtime-repairs>', 'exec'), namespace)
    released_training.get_IF = STATIC.fairness
    write_json(output / 'runtime_repairs.json', {
        'source_sha': SOURCE_SHA,
        'original_train_step': original_source,
        'cpu_reference_train_step': cpu_source,
        'executed_train_step': cached_source,
        'repairs': ['Resolve beta and gamma from released README', 'Supply missing utils_guide.F import', 'Replace hardcoded cuda tensor placement with logits.device', 'Cache unchanged adjacency and incidence tensors', 'Cache unchanged similarity Laplacian for diagnostic IF'],
        'unchanged': ['DGL Model and Com_KD_Prob implementations', 'train_teacher and train_student control flow', 'loss formulas and reduction order', 'optimizer settings', 'original full validation split and tie handling'],
    })
    return cpu_source


def verify_repairs(cpu_source):
    """Compare an actual original-vs-cached optimizer step and fairness metric."""
    g = dgl.add_self_loop(dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]), num_nodes=4))
    adjacency = g.adjacency_matrix().to_dense().numpy()
    features = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    teacher = torch.randn(4, 2)
    src, dst = g.adjacency_matrix().indices()
    energy = torch.zeros(4).index_add(0, src, (teacher[src] - teacher[dst]).square().sum(1))
    delta = released_guide.GDO_optimized(released_guide.calculate_similarity_matrix(sp.csr_matrix(adjacency), features, metric='cosine'))
    param = dict(teacher='SAGE', student='MLP', num_layers=2, feat_dim=3, label_dim=2, hidden_dim=4, dropout_s=0., norm_type='none', lamb=.3, tau=1.1)
    model = released_models.Model(param, model_type='student')
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    optimizer_ref = torch.optim.Adam(reference.parameters(), lr=.01)
    loss_l = torch.nn.NLLLoss()
    loss_t = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    released_training.beta, released_training.gamma = .02, 2e-5
    namespace = dict(released_training.__dict__)
    exec(compile(cpu_source, '<released-cpu-reference>', 'exec'), namespace)
    arguments = (g.adjacency_matrix().indices(), features, labels, teacher, loss_l, loss_t)
    result_ref = namespace['train_mini_batch'](reference, *arguments, optimizer_ref, torch.arange(2), adjacency, delta, None, energy, param)
    STATIC.clear()
    result = released_training.train_mini_batch(model, *arguments, optimizer, torch.arange(2), adjacency, delta, None, energy, param)
    np.testing.assert_allclose(result, result_ref, rtol=1e-7, atol=1e-7)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual = STATIC.fairness(teacher, features, adjacency)
    expected = STATIC.uncached_fairness(teacher, features, adjacency)
    assert actual == expected
    STATIC.clear()
    return dict(optimizer_step_equivalence=True, gradient_equivalence=True, diagnostic_fairness_equivalence=True)


def seed_schedule(config_seed, count):
    # Literal behavior of the released main: param['seed'] += seed in range(5).
    value = config_seed
    result = []
    for increment in range(count):
        value += increment
        result.append(value)
    return result


def run_seed(name, seed, output, max_epochs=None):
    destination = output / name / f'seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / 'metrics.json'
    if complete.exists():
        previous = json.loads(complete.read_text())
        assert previous['requested_epochs'] == max_epochs
        return previous
    param = json.loads((ROOT / 'upstream/param/best_parameters.json').read_text())[name]['tran']['SAGE']
    param['seed'] = seed
    if max_epochs is not None:
        param['max_epoch'] = max_epochs
    start = time.perf_counter()
    with (destination / 'training.log').open('w') as stream, contextlib.redirect_stdout(stream):
        released_utils.set_seed(seed)
        graph, labels, train, valid, test = released_data.load_data(name, seed=seed)
        assert len(valid) == 500
        assert not (set(train.tolist()) & set(valid.tolist()) or set(train.tolist()) & set(test.tolist()) or set(valid.tolist()) & set(test.tolist()))
        features = graph.ndata['feat']
        param['feat_dim'] = features.shape[1]
        param['label_dim'] = labels.int().max().item() + 1
        write_json(destination / 'config.json', param)
        write_json(destination / 'split.json', dict(train=train.tolist(), validation=valid.tolist(), test=test.tolist(), nodes=graph.num_nodes(), directed_edges_with_loops=graph.num_edges()))
        loss_l = torch.nn.NLLLoss()
        loss_t = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        evaluator = released_utils.get_evaluator(name)
        model_t = released_models.Model(param, model_type='teacher')
        optimizer_t = torch.optim.Adam(model_t.parameters(), lr=.01, weight_decay=param['weight_decay'])
        STATIC.clear()
        teacher_result = released_training.train_teacher(param, model_t, graph, features, labels, (train, valid, test), loss_l, evaluator, optimizer_t)
        logits_t, _, teacher_accuracy, _, _, energy, teacher_if = teacher_result
        torch.save(dict(state=model_t.state_dict(), logits=logits_t, energy=energy), destination / 'teacher.pt')
        # Source initialization consumes random feature noise before student creation.
        kd_model = released_models.Com_KD_Prob(param)
        kd_model.initialization(model_t, graph, features)
        student = released_models.Model(param, model_type='student')
        init = copy.deepcopy(student.state_dict())
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        import random
        python_state = random.getstate()
        values = {}
        for method, beta, gamma in [('FAITH', .02, GAMMA[name]), ('KL_ablation', 0., 0.)]:
            student.load_state_dict(init)
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            random.setstate(python_state)
            released_training.beta, released_training.gamma = beta, gamma
            optimizer = torch.optim.Adam(student.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
            STATIC.clear()
            print(f'REPRODUCTION METHOD {method}', flush=True)
            outcome = released_training.train_student(param, student, graph, features, labels, logits_t, (train, valid, test), loss_l, loss_t, evaluator, optimizer, energy, copy.deepcopy(kd_model))
            logits, _, validation_selected_test, _, state, metric_if = outcome
            checked = evaluator(logits[test].log_softmax(1), labels[test])
            assert checked == validation_selected_test
            values[method] = dict(accuracy=checked, individual_unfairness=metric_if, beta=beta, gamma=gamma)
            torch.save(dict(state=state, logits=logits), destination / f'{method}.pt')
        result = dict(dataset=name, seed=seed, requested_epochs=max_epochs, teacher_accuracy=teacher_accuracy, teacher_individual_unfairness=teacher_if, methods=values, seconds=time.perf_counter()-start, validation_nodes=len(valid), device='cpu')
        write_json(complete, result)
    print(json.dumps(result), flush=True)
    return result


def report(results, output):
    write_json(output / 'metrics.json', results)
    lines = ['# Released FAITH reproduction attempt', '', 'Uniform node classification accuracy (%), with the full original validation mask and no serving cache. Five seeds use the released cumulative schedule: 42, 43, 45, 48, 52. Population standard deviation matches the released aggregation code. These CPU runs retain the released DGL models and training loops, with recorded compatibility and static-tensor caching repairs.', '', '| Dataset | Method | Published main table | Reproduction mean ± std | Difference pp | Runs |', '|---|---|---:|---:|---:|---:|']
    for name in sorted({r['dataset'] for r in results}):
        selected = [r for r in results if r['dataset'] == name]
        for method in ['SAGE', 'FAITH', 'KL_ablation']:
            values = [100*(r['teacher_accuracy'] if method == 'SAGE' else r['methods'][method]['accuracy']) for r in selected]
            mean, std = float(np.mean(values)), float(np.std(values))
            paper = PAPER[name].get(method)
            published = f'{paper[0]:.2f} ± {paper[1]:.2f}' if paper else 'Not separately reported'
            delta = f'{mean-paper[0]:+.2f}' if paper else '—'
            lines.append(f'| {name} | {method} | {published} | {mean:.2f} ± {std:.2f} | {delta} | {len(values)} |')
    lines += ['', '## Scope and unresolved comparability', '', 'FAITH means the released Dirichlet energy and Oracle Similarity Alignment losses together, using beta/gamma from the README. FAITH-star counterfactual group fairness is a separate experiment and is excluded from this citation-graph comparison.', '', 'KL_ablation disables both extra FAITH losses but retains FAITH-tuned architectures and optimizer settings. It must not be labeled a reproduction of the GLNN row. The FAITH repository does not release its GLNN baseline training configuration or checkpoints. The original GLNN repository uses different citation-graph preprocessing and splits; its original-paper results are not the GLNN rerun in the FAITH table.', '', 'The main-table published GLNN targets are Cora 81.50 ± 1.21, CiteSeer 71.46 ± 1.79, PubMed 77.97 ± 3.35. The included FAITH appendix instead lists Cora GLNN std 2.23, CiteSeer std 3.16, PubMed mean 77.90; it also lists PubMed FAITH 81.72 rather than the main-table 81.53. These discrepancies are preserved rather than resolved by choosing a closer number.', '', 'The unavailable DGL CUDA wheel endpoint required CPU DGL 2.1.0 and PyTorch 2.1.2 rather than the README versions 0.6.1 and 1.6.0. NumPy 1.21.6, SciPy 1.7.3, and scikit-learn 1.0.2 match the release. Hardware/runtime differences prevent bitwise reproduction. The original Planetoid raw bytes were passed through DGL’s own citation loader; their hashes and filename-only Cora prefix change are recorded in environment.json.', '', 'Runtime repairs and their source text are in runtime_repairs.json. A complete original-vs-cached training step checks loss, gradients, and updated model parameters; the cached fairness diagnostic is checked against its released implementation. Checkpoint selection uses validation accuracy only; printed test-best fields from the source are not used as the reported result.', '', 'Sources: [FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705), [released FAITH code](https://github.com/shashivipul/FAITH), [original GLNN code](https://github.com/snap-research/graphless-neural-networks).']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/released_reproduction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    raw_hashes = prepare_data()
    source = install_runtime_repairs(args.output)
    checks = verify_repairs(source)
    write_json(args.output / 'checks.json', checks)
    write_json(args.output / 'environment.json', dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__, device='cpu', threads=args.threads, source_sha=SOURCE_SHA, raw_hashes=raw_hashes, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    config = json.loads((ROOT / 'upstream/param/best_parameters.json').read_text())
    results = []
    for name in args.datasets:
        seeds = seed_schedule(config[name]['tran']['SAGE']['seed'], args.runs)
        for seed in seeds:
            results.append(run_seed(name, seed, args.output, args.max_epochs))
            report(results, args.output)
    write_json(args.output / 'verification.json', dict(completed_dataset_seed_runs=len(results), expected=len(args.datasets)*args.runs, full_validation_masks=True, source_training_loops=True, reported_metric='uniform accuracy at validation-selected checkpoint', **checks))
    print(f'Completed reproduction attempt: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
