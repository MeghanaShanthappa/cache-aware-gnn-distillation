"""Released FAITH with CPU DGL and CUDA dropout/student execution.

The CUDA bridge preserves the separate CPU/CUDA random streams used by the
authors' GPU run. Graph convolutions and teacher optimizer remain on CPU.
"""
import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
support_source = (ROOT / 'reproduce.py').read_text()
hide_cuda = "os.environ['CUDA_VISIBLE_DEVICES'] = ''"
assert support_source.count(hide_cuda) == 1
support = types.ModuleType('reproduction_support')
support.__file__ = str(ROOT / 'reproduce.py')
sys.modules[support.__name__] = support
exec(compile(support_source.replace(hide_cuda, ''), support.__file__, 'exec'), support.__dict__)

torch = support.torch
np = support.np
F = support.F
dgl = support.dgl


class CUDADropoutBridge(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        return F.dropout(x.to('cuda'), self.p, True).to(x.device)


class DeviceStaticTensors(support.StaticTensors):
    def matrices(self, delta, adjacency, device):
        if self.delta is None:
            values = torch.tensor(delta.data, dtype=torch.float32, device=device)
            indices = torch.tensor(np.stack([delta.row, delta.col]), dtype=torch.int64, device=device)
            self.delta = torch.sparse_coo_tensor(indices, values, size=delta.shape, dtype=torch.float32)
            # Source energy function transfers edge batches to logits.device.
            self.adjacency = torch.tensor(adjacency, dtype=torch.float32).to_sparse()
        return self.delta, self.adjacency


def install(output):
    training = support.released_training
    support.released_models.device = torch.device('cpu')  # teacher noise probe uses CPU DGL
    support.released_guide.F = F
    support.STATIC = DeviceStaticTensors()
    original = inspect.getsource(training.train_mini_batch)
    reference = original.replace("device='cuda'", 'device=logits.device')
    old = """    values_delta = torch.tensor(Delta.data, dtype=torch.float32, device=logits.device)
    indices_delta = torch.tensor([Delta.row, Delta.col], dtype=torch.int64, device=logits.device)
    Delta_sparse = torch.sparse_coo_tensor(indices_delta, values_delta, size=Delta.shape, dtype=torch.float32)"""
    assert old in reference
    cached = reference.replace(old, '    Delta_sparse, adj_sparse = _device_static(Delta, adj, logits.device)')
    cached = cached.replace('    adj_sparse = torch.tensor(adj, dtype=torch.float32).to_sparse()', '')
    training.__dict__['_device_static'] = support.STATIC.matrices
    exec(compile(cached, '<released-cached-device-step>', 'exec'), training.__dict__)
    training.get_IF = support.STATIC.fairness
    teacher_original = inspect.getsource(training.train_teacher)
    old_device = "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
    assert teacher_original.count(old_device) == 1
    teacher_cpu = teacher_original.replace(old_device, "device = torch.device('cpu')")
    exec(compile(teacher_cpu, '<released-teacher-cpu-dgl>', 'exec'), training.__dict__)

    # Reuse the same original-source launcher with explicit, auditable device edits.
    runner = inspect.getsource(support.run_seed)
    changes = [
        ("model_t = released_models.Model(param, model_type='teacher')", "model_t = released_models.Model(param, model_type='teacher')\n        model_t.encoder.dropout = _CUDADropoutBridge(param['dropout_t'])\n        cpu_rng_after_teacher_init = torch.get_rng_state().clone()"),
        ('logits_t, _, teacher_accuracy, _, _, energy, teacher_if = teacher_result', 'logits_t, _, teacher_accuracy, _, _, energy, teacher_if = teacher_result\n        assert torch.equal(cpu_rng_after_teacher_init, torch.get_rng_state()), "Teacher changed CPU RNG stream"'),
        ('kd_model.initialization(model_t, graph, features)', "kd_model.initialization(model_t, graph, features)\n        features, labels = features.cuda(), labels.cuda()\n        logits_t, energy = logits_t.cuda(), energy.cuda()"),
        ("student = released_models.Model(param, model_type='student')", "student = released_models.Model(param, model_type='student').cuda()"),
        ('torch_state = torch.random.get_rng_state()', 'torch_state = torch.random.get_rng_state()\n        cuda_state = torch.cuda.get_rng_state()'),
        ('torch.random.set_rng_state(torch_state)', 'torch.random.set_rng_state(torch_state)\n            torch.cuda.set_rng_state(cuda_state)'),
        ("device='cpu'", "device='cpu_dgl_cuda_dropout_and_student'"),
    ]
    for old_text, new_text in changes:
        assert runner.count(old_text) == 1, old_text
        runner = runner.replace(old_text, new_text)
    namespace = support.__dict__
    namespace['_CUDADropoutBridge'] = CUDADropoutBridge
    exec(compile(runner, '<released-source-hybrid-launcher>', 'exec'), namespace)
    support.write_json(output / 'runtime_repairs.json', dict(original_train_step=original, executed_train_step=cached, original_teacher_loop=teacher_original, executed_teacher_loop=teacher_cpu, executed_launcher=runner, bridge_source=inspect.getsource(CUDADropoutBridge), scope='CPU graph convolution and teacher optimizer; CUDA dropout and complete student training; original losses and configuration'))
    return reference


def verify(reference):
    checks = support.verify_repairs(reference)
    x = torch.ones((12, 8), requires_grad=True)
    bridge = CUDADropoutBridge(.4).train()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    out = bridge(x)
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert not torch.equal(cuda_state, torch.cuda.get_rng_state())
    out.sum().backward()
    torch.cuda.set_rng_state(cuda_state)
    ref_x = x.detach().clone().requires_grad_()
    ref = F.dropout(ref_x.cuda(), .4, True).cpu()
    ref.sum().backward()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    checks.update(cuda_dropout_output_and_gradient=True, cpu_rng_preserved_during_cuda_dropout=True)
    return checks


def report(results, output):
    support.write_json(output / 'metrics.json', results)
    text = ['# FAITH hybrid runtime reproduction', '', 'Released DGL models, loss formulas, full public validation masks, and cumulative seeds 42, 43, 45, 48, 52. Graph convolution and the teacher optimizer run on CPU because the official DGL CUDA wheel server was unavailable. Teacher dropout runs on CUDA through a transfer bridge, and student training runs fully on the A100. This preserves the separate CPU and CUDA random streams of the authors’ GPU initialization sequence; graph/teacher numerical computations still differ from a pure GPU run.', '', '| Dataset | Method | Main paper accuracy % | Reproduction accuracy % | Difference pp | Seeds |', '|---|---|---:|---:|---:|---:|']
    for name in sorted({r['dataset'] for r in results}):
        group = [r for r in results if r['dataset'] == name]
        for method in ['SAGE', 'FAITH', 'KL_ablation']:
            values = [100*(r['teacher_accuracy'] if method == 'SAGE' else r['methods'][method]['accuracy']) for r in group]
            mean, std = float(np.mean(values)), float(np.std(values))
            published = support.PAPER[name].get(method)
            target = f'{published[0]:.2f} ± {published[1]:.2f}' if published else 'Not separately reported'
            difference = f'{mean-published[0]:+.2f}' if published else '—'
            text.append(f'| {name} | {method} | {target} | {mean:.2f} ± {std:.2f} | {difference} | {len(values)} |')
    text += ['', 'Population standard deviations follow the released aggregation code. The same fixed settings and all five seeds are retained; no selection uses test accuracy. Full FAITH includes energy alignment and Oracle Similarity Alignment. KL_ablation retains FAITH tuning and is not the published GLNN rerun.', '', 'The CPU-only source rerun in ../released_reproduction/ is retained as a runtime diagnostic. It revealed a CiteSeer seed-48 collapse. CPU-only dropout advances the parameter-initialization RNG stream, unlike GPU dropout. This runtime explicitly preserves the separate streams without changing model widths, losses, coefficients, masks, or seed selection. It is a documented reproduction attempt, not a bitwise recreation of the authors’ unavailable GPU environment.', '', 'The published GLNN rerun configuration remains absent from the FAITH release; the original GLNN CPF pipeline differs from its citation loader. Cross-protocol GLNN diagnostics in ../glnn_source_reference/ must not be used as paper baselines.', '', 'Source code, checkpoints, logs, raw-data hashes, configuration and runtime repair records accompany this report. Main-paper and appendix discrepancies are described in ../released_reproduction/REPORT.md.', '', '[FAITH paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705) · [Authors’ code](https://github.com/shashivipul/FAITH)']
    (output / 'REPORT.md').write_text('\n'.join(text)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 45, 48, 52])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/hybrid_reproduction')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Host GPU access is required')
    torch.set_num_threads(8)
    args.output.mkdir(exist_ok=True, parents=True)
    raw_hashes = support.prepare_data()
    reference = install(args.output)
    checks = verify(reference)
    support.write_json(args.output / 'checks.json', checks)
    support.write_json(args.output / 'environment.json', dict(torch=torch.__version__, dgl=dgl.__version__, gpu=torch.cuda.get_device_name(), source_sha=support.SOURCE_SHA, raw_hashes=raw_hashes, runtime='CPU DGL with CUDA dropout and student', launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), support_sha256=hashlib.sha256((ROOT / 'reproduce.py').read_bytes()).hexdigest()))
    rows = []
    for name in args.datasets:
        for seed in args.seeds:
            rows.append(support.run_seed(name, seed, args.output, None))
            report(rows, args.output)
    support.write_json(args.output / 'verification.json', dict(completed_dataset_seed_runs=len(rows), expected=len(args.datasets)*len(args.seeds), seeds=args.seeds, full_validation_masks=True, teacher_cpu_rng_preserved=True, **checks))
    print(f'Completed hybrid runtime: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
