"""Released FAITH on CUDA using a tested message-sum compatibility adapter.

The DGL SAGEConv class and released training loops remain unchanged. Only the
homogeneous copy_u/sum graph primitive is supplied by torch.sparse.mm on CUDA.
"""
import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

import reproduce_hybrid as H

ROOT, support = H.ROOT, H.support
torch, np, F, dgl = H.torch, H.np, H.F, H.dgl


class SparseCUDAProxy:
    """Minimal graph API used by the released homogeneous SAGE gcn teacher."""
    def __init__(self, graph, device='cuda'):
        assert not graph.is_block
        self.graph = graph
        self.device = torch.device(device)
        self.is_block = False
        src, dst = graph.edges(order='eid')
        # Destination rows sum incoming source features. Duplicate edges retain
        # their multiplicity after coalescing, matching DGL update_all(sum).
        indices = torch.stack([dst, src]).to(self.device)
        self.adj = torch.sparse_coo_tensor(indices, torch.ones(len(src), device=self.device), (graph.num_nodes(), graph.num_nodes())).coalesce()
        self.srcdata = {}
        self.dstdata = self.srcdata

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def to(self, device):
        if torch.device(device) != self.device:
            raise ValueError('This adapter supports one fixed CUDA device')
        return self

    @contextlib.contextmanager
    def local_scope(self):
        old_src, old_dst = self.srcdata, self.dstdata
        self.srcdata = {}
        self.dstdata = self.srcdata
        try:
            yield self
        finally:
            self.srcdata, self.dstdata = old_src, old_dst

    def update_all(self, message, reducer):
        if message.name != 'copy_u' or reducer.name != 'sum':
            raise NotImplementedError('Only released SAGE gcn copy_u/sum is supported')
        self.dstdata['neigh'] = torch.sparse.mm(self.adj, self.srcdata['h'])


def verify_graph_primitive():
    # A directed graph with parallel edges and an isolated node tests direction,
    # edge multiplicity, degrees, and the SAGE self contribution separately.
    raw = dgl.graph(([0, 0, 1, 2, 2], [1, 1, 2, 0, 2]), num_nodes=4)
    proxy = SparseCUDAProxy(raw)
    checks = []
    for inputs, hidden, outputs, layers in [(3, 8, 2, 2), (8, 3, 2, 3)]:
        param = dict(teacher='SAGE', student='MLP', num_layers=layers, feat_dim=inputs, label_dim=outputs, hidden_dim=hidden, dropout_t=0., norm_type='none')
        reference = support.released_models.Model(param, model_type='teacher').eval()
        actual = copy.deepcopy(reference).cuda()
        x = torch.randn(4, inputs, requires_grad=True)
        x_cuda = x.detach().clone().cuda().requires_grad_()
        expected = reference(raw, x)
        result = actual(proxy, x_cuda)
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-5, atol=2e-6)
        expected.square().sum().backward()
        result.square().sum().backward()
        torch.testing.assert_close(x_cuda.grad.cpu(), x.grad, rtol=3e-5, atol=3e-6)
        for a, b in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=3e-5, atol=3e-6)
        checks.append(dict(inputs=inputs, hidden=hidden, outputs=outputs, layers=layers))
    return dict(released_dgl_forward_equivalence=True, released_dgl_parameter_gradient_equivalence=True, input_gradient_equivalence=True, directed_edges=True, duplicate_edges=True, isolated_nodes=True, cases=checks)


def install(output):
    training = support.released_training
    support.released_guide.F = F
    support.STATIC = H.DeviceStaticTensors()
    original = inspect.getsource(training.train_mini_batch)
    reference = original.replace("device='cuda'", 'device=logits.device')
    old = """    values_delta = torch.tensor(Delta.data, dtype=torch.float32, device=logits.device)
    indices_delta = torch.tensor([Delta.row, Delta.col], dtype=torch.int64, device=logits.device)
    Delta_sparse = torch.sparse_coo_tensor(indices_delta, values_delta, size=Delta.shape, dtype=torch.float32)"""
    assert old in reference
    cached = reference.replace(old, '    Delta_sparse, adj_sparse = _device_static(Delta, adj, logits.device)')
    cached = cached.replace('    adj_sparse = torch.tensor(adj, dtype=torch.float32).to_sparse()', '')
    training.__dict__['_device_static'] = support.STATIC.matrices
    exec(compile(cached, '<released-cached-cuda-step>', 'exec'), training.__dict__)
    training.get_IF = support.STATIC.fairness
    # train_teacher itself is not patched: its original CUDA selection now works.
    runner = inspect.getsource(support.run_seed)
    changes = [
        ('graph, labels, train, valid, test = released_data.load_data(name, seed=seed)', "graph, labels, train, valid, test = released_data.load_data(name, seed=seed)\n        graph = _SparseCUDAProxy(graph)\n        labels = labels.cuda()"),
        ("features = graph.ndata['feat']", "features = graph.ndata['feat'].cuda()"),
        ("model_t = released_models.Model(param, model_type='teacher')", "model_t = released_models.Model(param, model_type='teacher').cuda()\n        cpu_rng_after_teacher_init = torch.get_rng_state().clone()"),
        ('logits_t, _, teacher_accuracy, _, _, energy, teacher_if = teacher_result', 'logits_t, _, teacher_accuracy, _, _, energy, teacher_if = teacher_result\n        assert torch.equal(cpu_rng_after_teacher_init, torch.get_rng_state()), "Teacher changed CPU initialization RNG"'),
        ("student = released_models.Model(param, model_type='student')", "student = released_models.Model(param, model_type='student').cuda()"),
        ('torch_state = torch.random.get_rng_state()', 'torch_state = torch.random.get_rng_state()\n        cuda_state = torch.cuda.get_rng_state()'),
        ('torch.random.set_rng_state(torch_state)', 'torch.random.set_rng_state(torch_state)\n            torch.cuda.set_rng_state(cuda_state)'),
        ("device='cpu'", "device='cuda_with_torch_sparse_message_sum'"),
    ]
    for old_text, new_text in changes:
        assert runner.count(old_text) == 1, old_text
        runner = runner.replace(old_text, new_text)
    namespace = support.__dict__
    namespace['_SparseCUDAProxy'] = SparseCUDAProxy
    exec(compile(runner, '<released-source-cuda-launcher>', 'exec'), namespace)
    support.write_json(output / 'runtime_repairs.json', dict(original_train_step=original, executed_train_step=cached, executed_launcher=runner, graph_adapter=inspect.getsource(SparseCUDAProxy), teacher_training_loop='original released function unchanged', scope='Original DGL SAGEConv and original training loops; torch CUDA copy_u/sum compatibility primitive; static loss tensors and diagnostic Laplacian cached'))
    return reference


def report(rows, output):
    support.write_json(output / 'metrics.json', rows)
    lines = ['# FAITH CUDA reproduction with a message sum compatibility adapter', '', 'Original released DGL SAGEConv models and teacher/student training loops. The unavailable CUDA DGL graph copy_u/sum primitive is replaced by mathematically equivalent torch.sparse.mm on CUDA, with directed-edge, duplicate-edge, isolated-node, forward, and gradient checks against native DGL. Teacher and student matrix operations, dropout, and optimizer updates run on the A100. CPU parameter initialization and CUDA dropout use the separate streams of the source GPU execution.', '', '| Dataset | Method | Published main table % | Reproduction % | Difference pp | Seeds |', '|---|---|---:|---:|---:|---:|']
    for name in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == name]
        for method in ['SAGE', 'FAITH', 'KL_ablation']:
            values = [100*(r['teacher_accuracy'] if method == 'SAGE' else r['methods'][method]['accuracy']) for r in group]
            mean, std = float(np.mean(values)), float(np.std(values))
            paper = support.PAPER[name].get(method)
            target = f'{paper[0]:.2f} ± {paper[1]:.2f}' if paper else 'Not separately reported'
            delta = f'{mean-paper[0]:+.2f}' if paper else '—'
            lines.append(f'| {name} | {method} | {target} | {mean:.2f} ± {std:.2f} | {delta} | {len(values)} |')
    lines += ['', 'Mean ± population standard deviation over the five released cumulative seeds, with full 500-node validation masks. All FAITH coefficients and model settings come from the release. No hyperparameters, seeds, or checkpoints are selected by test performance. KL_ablation is not the published GLNN rerun.', '', 'This is a documented mathematical reproduction with a compatibility adapter, not the original binary environment. PyTorch 2.1.2 and DGL 2.1.0 differ from the README versions, and sparse reduction order can differ from native DGL CUDA. Numerical agreement and disagreement are reported explicitly. No inference-speed claim is made.', '', 'CPU-only and hybrid fallback runs are retained in ../released_reproduction/ and ../hybrid_reproduction/. They are runtime diagnostics, not alternate numbers to select per dataset. The CUDA primitive runtime was chosen for closer execution semantics: the teacher linear layers, Adam, and dropout all execute on GPU, with only a tested sum primitive substituted.', '', 'The FAITH release omits its GLNN baseline rerun configuration and checkpoints. Original GLNN CPF preprocessing differs from the public citation loader; the cross-protocol run in ../glnn_source_reference/ is diagnostic only. A numerical reproduction of the published GLNN row remains unverified.', '', '[FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705) · [Released code](https://github.com/shashivipul/FAITH)']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 45, 48, 52])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/cuda_reproduction')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Host CUDA access required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    graph_checks = verify_graph_primitive()
    raw_hashes = support.prepare_data()
    reference = install(args.output)
    checks = support.verify_repairs(reference)
    checks.update(graph_checks)
    support.write_json(args.output / 'checks.json', checks)
    support.write_json(args.output / 'environment.json', dict(torch=torch.__version__, dgl=dgl.__version__, gpu=torch.cuda.get_device_name(), source_sha=support.SOURCE_SHA, runtime='All model math on CUDA; torch sparse message sum replaces unavailable DGL CUDA primitive', raw_hashes=raw_hashes, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), support_sha256=hashlib.sha256((ROOT / 'reproduce.py').read_bytes()).hexdigest()))
    if args.check_only:
        print(json.dumps(checks), flush=True)
        return
    rows = []
    for name in args.datasets:
        for seed in args.seeds:
            rows.append(support.run_seed(name, seed, args.output, None))
            report(rows, args.output)
    support.write_json(args.output / 'verification.json', dict(completed_dataset_seed_runs=len(rows), expected=len(args.datasets)*len(args.seeds), seeds=args.seeds, full_validation_masks=True, teacher_cpu_rng_preserved=True, **checks))
    print(f'Completed CUDA reproduction: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
