"""FAITH structural-loss adaptation and matched static-cache experiments.

The upstream fairness penalties are deliberately excluded. See README.md.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream'
METHODS = ['kd', 'structural']
POLICIES = ['none', 'frequency', 'disagreement', 'benefit']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def make_graph(edges, n, device):
    """Binary directed symmetric adjacency, exactly one loop per node.

    DGL SAGEConv(gcn) averages incoming neighbors AND self. Since the
    released graph already has self loops, the teacher counts self twice.
    """
    edges = edges[:, edges[0] != edges[1]]
    loop = torch.arange(n)
    edges = torch.cat([edges, torch.stack([loop, loop])], 1).to(device)
    adjacency = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device=device), (n, n)).coalesce()
    adjacency = torch.sparse_coo_tensor(adjacency.indices(), torch.ones_like(adjacency.values()), (n, n)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.equal(adjacency.indices(), adjacency.transpose(0, 1).coalesce().indices())
    self_again = torch.sparse_coo_tensor(torch.stack([loop, loop]).to(device), torch.ones(n, device=device), (n, n))
    mean_adj = (adjacency + self_again).coalesce()
    mean_adj = torch.sparse_coo_tensor(mean_adj.indices(), mean_adj.values() / (degree[mean_adj.indices()[0]] + 1), (n, n)).coalesce().to_sparse_csr()
    return adjacency, adjacency.to_sparse_csr(), mean_adj, degree


class Network(nn.Module):
    def __init__(self, inputs, classes, config, teacher=False):
        super().__init__()
        dims = [inputs] + [config['hidden_dim']] * (config['num_layers'] - 1) + [classes]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:])])
        self.teacher = teacher
        self.dropout = config['dropout_t' if teacher else 'dropout_s']
        if teacher:
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

    def forward(self, x, mean_adj=None):
        for i, layer in enumerate(self.layers):
            if self.teacher:
                x = torch.sparse.mm(mean_adj, x)
            x = layer(x)
            if i + 1 < len(self.layers):
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


def node_energy(logits, adjacency, degree):
    """sum_j A_ij ||z_i-z_j||^2 without an edge-by-class tensor."""
    norms = logits.square().sum(1)
    return degree * norms - 2 * (logits * torch.sparse.mm(adjacency, logits)).sum(1) + torch.sparse.mm(adjacency, norms[:, None]).flatten()


def structural_loss(logits, adjacency, degree, teacher_energy, edge_idx):
    # Upstream accumulates on both ends of a symmetric directed edge list.
    energy = 2 * node_energy(logits, adjacency, degree)
    return -F.cosine_similarity(energy[edge_idx[0]], teacher_energy[edge_idx[1]], dim=0)


@torch.no_grad()
def accuracy(logits, labels, mask):
    return (logits[mask].argmax(1) == labels[mask]).float().mean().item()


def fit(model, data, config, seed, teacher_logits=None, beta=0., epochs=500):
    seed_all(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01 if model.teacher else config['learning_rate'], weight_decay=config['weight_decay'])
    best_accuracy = -1.
    state = None
    best_epoch = -1
    stale = 0
    trace = []
    targets = None if teacher_logits is None else (teacher_logits / config['tau']).log_softmax(1)
    teacher_energy = None if not beta else node_energy(teacher_logits, data['adj'], data['degree']).detach()
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data['x'], data['mean_adj'])
        ce = F.cross_entropy(logits[data['train']], data['y'][data['train']])
        if teacher_logits is None:
            loss = ce
        else:
            kd = F.kl_div((logits / config['tau']).log_softmax(1), targets, reduction='batchmean', log_target=True)
            # Released FAITH uses no tau-squared multiplier.
            loss = config['lamb'] * ce + (1 - config['lamb']) * kd
            if beta:
                loss = loss + beta * structural_loss(logits, data['adj'], data['degree'], teacher_energy, data['edges'])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = accuracy(model(data['x'], data['mean_adj']), data['y'], data['model_val'])
        # Preserve upstream tie handling and patience, but reserve calibration labels.
        if validation >= best_accuracy:
            state = copy.deepcopy(model.state_dict())
            best_accuracy, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if epoch % 25 == 0 or stale >= 50 or epoch + 1 == epochs:
            trace.append({'epoch': epoch + 1, 'loss': loss.item(), 'validation_accuracy': validation})
        if stale >= 50:
            break
    model.load_state_dict(state)
    model.eval()
    return model, dict(epochs=epoch + 1, best_epoch=best_epoch + 1, best_validation_accuracy=best_accuracy, seconds=time.perf_counter() - start, trace=trace)


@torch.no_grad()
def benefit_features(teacher, student):
    t, s = teacher.softmax(1), student.softmax(1)
    k = t.shape[1]
    ht = -(t * t.clamp_min(1e-12).log()).sum(1) / math.log(k)
    hs = -(s * s.clamp_min(1e-12).log()).sum(1) / math.log(k)
    tt, ss = t.topk(2, dim=1).values, s.topk(2, dim=1).values
    agree = (t.argmax(1) == s.argmax(1)).float()
    kl = (t * (t.clamp_min(1e-12).log() - s.clamp_min(1e-12).log())).sum(1)
    return torch.stack([ht, hs, tt[:, 0], ss[:, 0], tt[:, 0] - tt[:, 1], ss[:, 0] - ss[:, 1], agree, kl, ht - hs], 1)


@torch.no_grad()
def fit_benefit(teacher, student, labels, calibration):
    """Fixed ridge regression of teacher correctness minus student correctness.

    Only reserved validation labels enter this fit. No test-label access.
    """
    features = benefit_features(teacher, student).double()
    mean = features[calibration].mean(0)
    scale = features[calibration].std(0).clamp_min(1e-6)
    design = (features - mean) / scale
    design = torch.cat([design, torch.ones((len(design), 1), device=design.device)], 1)
    target = ((teacher[calibration].argmax(1) == labels[calibration]).double() - (student[calibration].argmax(1) == labels[calibration]).double())
    xc = design[calibration]
    penalty = torch.eye(design.shape[1], device=design.device, dtype=torch.float64)
    penalty[-1, -1] = 0
    coef = torch.linalg.solve(xc.T @ xc + penalty, xc.T @ target)
    benefit = (design @ coef).clamp(-1, 1).float()
    # Equal argmax predictions have exactly zero classification improvement.
    benefit[teacher.argmax(1) == student.argmax(1)] = 0
    return benefit, dict(coefficients=coef.cpu().tolist(), mean=mean.cpu().tolist(), scale=scale.cpu().tolist(), samples=int(calibration.sum()), ridge=1., calibration_mean_benefit=target.mean().item())


@torch.no_grad()
def timed(fn, repetitions=40):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000)
    return statistics.median(values)


@torch.no_grad()
def evaluate(model, data, teacher, student, cache, popularity, query):
    n, c = teacher.shape
    mixed = student.clone()
    mixed[cache] = teacher[cache]
    mask = data['test']
    correct = (mixed[mask].argmax(1) == data['y'][mask]).float()
    weights = popularity[mask] / popularity[mask].sum()
    hit = torch.zeros(n, dtype=torch.bool, device=teacher.device)
    hit[cache] = True
    mapping = torch.full((n,), -1, dtype=torch.long, device=teacher.device)
    mapping[cache] = torch.arange(len(cache), device=teacher.device)
    payload = teacher[cache].contiguous()

    def serve():
        if not len(cache):
            return model(data['x'][query])
        slots = mapping[query]
        hits = slots >= 0
        output = torch.empty((len(query), c), device=teacher.device)
        output[hits] = payload[slots[hits]]
        output[~hits] = model(data['x'][query[~hits]])
        return output

    torch.testing.assert_close(serve(), mixed[query], atol=1e-5, rtol=1e-4)
    return dict(accuracy=correct.mean().item(), weighted_accuracy=(correct * weights).sum().item(), shifted_accuracy=(correct * (.5 * weights + .5 * weights.flip(0))).sum().item(), hit_rate=(hit[mask].float() * weights).sum().item(), query_batch_ms=timed(serve), cached_nodes=len(cache), payload_bytes=payload.numel() * 4, index_bytes=8 * n if len(cache) else 0, model_bytes=sum(p.numel() * p.element_size() for p in model.parameters()), routing_verified=True)


def load_data(name, device):
    dataset = Planetoid(str(ROOT / 'data'), name, transform=NormalizeFeatures())
    raw = dataset[0]
    coo, adj, mean, degree = make_graph(raw.edge_index, raw.num_nodes, device)
    result = {'x': raw.x.to(device), 'y': raw.y.to(device), 'train': raw.train_mask.to(device), 'test': raw.test_mask.to(device), 'adj': adj, 'mean_adj': mean, 'degree': degree, 'edges': coo.indices(), 'classes': dataset.num_classes}
    val = raw.val_mask.nonzero().flatten().numpy()
    val = np.random.default_rng(1729).permutation(val)
    for key, nodes in [('model_val', val[:len(val) // 2]), ('calibration', val[len(val) // 2:])]:
        mask = torch.zeros(raw.num_nodes, dtype=torch.bool, device=device)
        mask[torch.as_tensor(nodes, device=device)] = True
        result[key] = mask
    masks = ['train', 'model_val', 'calibration', 'test']
    assert all(not (result[a] & result[b]).any().item() for i, a in enumerate(masks) for b in masks[i + 1:])
    return result


def run_seed(name, seed, args, config):
    destination = args.output / name / f'seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset=name, seed=seed, config=config, beta=args.beta, epochs=args.epochs, budgets=args.budgets, source_sha=args.source_sha, protocol='structural_adaptation_v1', calibration_split_seed=1729, history_length=100000)
    complete = destination / 'complete.json'
    if complete.exists():
        assert json.loads((destination / 'manifest.json').read_text()) == manifest, 'Resume settings changed'
        return json.loads((destination / 'metrics.json').read_text())
    atomic_json(destination / 'manifest.json', manifest)
    data = load_data(name, 'cuda')
    atomic_json(destination / 'split.json', {k: data[k].nonzero().flatten().cpu().tolist() for k in ['train', 'model_val', 'calibration', 'test']})
    n = len(data['x'])
    seed_all(seed)
    teacher = Network(data['x'].shape[1], data['classes'], config, teacher=True).cuda()
    teacher, diagnostics = fit(teacher, data, config, seed, epochs=args.epochs)
    atomic_json(destination / 'teacher_trace.json', diagnostics)
    with torch.no_grad():
        teacher_logits = teacher(data['x'], data['mean_adj']).detach()
    teacher_accuracy = accuracy(teacher_logits, data['y'], data['test'])
    torch.save({'state': teacher.state_dict(), 'logits': teacher_logits.cpu()}, destination / 'teacher.pt')
    rng = np.random.default_rng(seed + 500)
    ranks = rng.permutation(np.arange(1, n + 1)).astype(float)
    pop = ranks ** -.8
    pop /= pop.sum()
    history = rng.choice(n, 100000, p=pop)
    counts = torch.as_tensor(np.bincount(history, minlength=n) + 1, dtype=torch.float, device='cuda')
    popularity = torch.as_tensor(pop, dtype=torch.float, device='cuda')
    test = data['test'].nonzero().flatten().cpu().numpy()
    query = torch.as_tensor(rng.choice(test, 1024, p=pop[test] / pop[test].sum()), device='cuda')
    seed_all(seed + 100)
    initial = Network(data['x'].shape[1], data['classes'], config).cuda()
    rows = []
    for method in METHODS:
        model, diagnostics = fit(copy.deepcopy(initial), data, config, seed + 200, teacher_logits, beta=args.beta if method == 'structural' else 0., epochs=args.epochs)
        atomic_json(destination / f'{method}_trace.json', diagnostics)
        with torch.no_grad():
            logits = model(data['x']).detach()
            kl = F.kl_div((logits / config['tau']).log_softmax(1), (teacher_logits / config['tau']).softmax(1), reduction='none').sum(1).clamp_min(0)
        benefit, calibration = fit_benefit(teacher_logits, logits, data['y'], data['calibration'])
        atomic_json(destination / f'{method}_calibration.json', calibration)
        torch.save({'state': model.state_dict(), 'logits': logits.cpu(), 'predicted_benefit': benefit.cpu()}, destination / f'{method}.pt')
        configs = [(0., 'none', torch.empty(0, dtype=torch.long, device='cuda'))]
        for budget in args.budgets:
            k = max(1, int(n * budget))
            configs += [(budget, 'frequency', counts.topk(k).indices), (budget, 'disagreement', (counts * kl).topk(k).indices)]
            scores = counts * benefit.clamp_min(0)
            candidates = scores.topk(k).indices
            # Fixed byte upper bound: empty/nonbeneficial slots need not be filled.
            selected = candidates[scores[candidates] > 0]
            configs.append((budget, 'benefit', selected))
        for budget, policy, cache in configs:
            row = dict(dataset=name, seed=seed, student=method, policy=policy, budget=budget, teacher_accuracy=teacher_accuracy, epochs=diagnostics['epochs'], best_epoch=diagnostics['best_epoch'])
            row.update(evaluate(model, data, teacher_logits, logits, cache, popularity, query))
            rows.append(row)
        print(json.dumps(dict(dataset=name, seed=seed, student=method, accuracy=accuracy(logits, data['y'], data['test']), teacher_accuracy=teacher_accuracy)), flush=True)
    atomic_json(destination / 'metrics.json', rows)
    atomic_json(complete, dict(rows=len(rows), split_disjointness=True))
    return rows


def write_report(rows, output):
    atomic_json(output / 'metrics.json', rows)
    with (output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ['# FAITH structural distillation and static cache pilot', '', 'This is a structural-only adaptation of the released FAITH code, not a reproduction of full FAITH. OSA and counterfactual fairness penalties are excluded. Results use an A100, synthetic Zipf queries, and the public Planetoid split with its validation set divided into model selection and cache calibration halves.', '', '## Request weighted accuracy at 10 percent node cache', '', 'Percent, mean ± sample standard deviation over training and workload seeds. No cache is the corresponding student.', '', '| Dataset | Student | No cache | Frequency | Disagreement | Predicted benefit |', '|---|---|---:|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in rows}):
        for method in METHODS:
            cells = []
            for policy in POLICIES:
                values = [r['weighted_accuracy'] * 100 for r in rows if r['dataset'] == dataset and r['student'] == method and r['policy'] == policy and r['budget'] == (0 if policy == 'none' else .1)]
                cells.append(f'{statistics.mean(values):.2f} ± {statistics.stdev(values) if len(values)>1 else 0:.2f}' if values else '—')
            text.append('| ' + ' | '.join([dataset, method] + cells) + ' |')
    text += ['', '## Accuracy and latency across every setting', '', '| Dataset | Student | Cache | Policy | Uniform accuracy % | Weighted accuracy % | Batch latency ms | Seeds |', '|---|---|---:|---|---:|---:|---:|---:|']
    for key in sorted({(r['dataset'], r['student'], r['budget'], r['policy']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['student'], r['budget'], r['policy']) == key]
        d, s, b, p = key
        text.append(f'| {d} | {s} | {b:.0%} | {p} | {statistics.mean(r["accuracy"] for r in group)*100:.2f} | {statistics.mean(r["weighted_accuracy"] for r in group)*100:.2f} | {statistics.mean(r["query_batch_ms"] for r in group):.3f} | {len(group)} |')
    text += ['', '## Interpretation limits', '', 'Teacher and student architectures, optimizer settings, temperature, and supervised mixing coefficient come from the released transductive SAGE configuration. The structural coefficient is 0.02 from the authors’ README. The sparse structural objective is checked against the released function in both value and gradient. Implementation and protocol differences are documented in README.md.', '', 'Validation labels are split with fixed seed 1729: 250 nodes for checkpoint selection and 250 for benefit estimation. Benefit estimation uses a fixed ridge coefficient of 1, with no test-label tuning. Benefit caching may leave slots empty when predicted improvement is nonpositive; all caches obey the same node-logit upper bound. Payload, dense index, and model bytes are in metrics.csv. Cache lookup timing includes miss-only MLP inference and excludes host transfer, network, and refresh.', '', 'All students have the same maximum 500-step budget and validation stopping rule; actual epochs differ and are recorded. Seeds change model/workload randomness, not the graph split. Five-seed comparisons are descriptive, with no significance claim. Earlier A40 results are not directly comparable because these runs use different architectures, training schedules, validation allocation, and hardware.']
    (output / 'REPORT.md').write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', type=float, default=.02)
    parser.add_argument('--budgets', nargs='+', type=float, default=[.01, .05, .1, .2])
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'citation_pilot')
    args = parser.parse_args()
    assert args.seeds > 0 and args.epochs > 0 and all(0 < b <= 1 for b in args.budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    args.source_sha = '94be4423aee12cdf4d4d098d369fe472aeedcbe7'
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution is required; run with approved host access')
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), source_sha=args.source_sha, protocol='structural_adaptation_v1', configuration_sha256=hashlib.sha256((UPSTREAM / 'param/best_parameters.json').read_bytes()).hexdigest())
    atomic_json(args.output / 'environment.json', environment)
    configs = json.loads((UPSTREAM / 'param/best_parameters.json').read_text())
    rows = []
    for name in args.datasets:
        config = configs[name.lower()]['tran']['SAGE']
        for seed in range(args.seeds):
            rows += run_seed(name, seed, args, config)
            write_report(rows, args.output)
    expected = len(args.datasets) * args.seeds * 2 * (1 + 3 * len(args.budgets))
    assert len(rows) == expected
    assert len({(r['dataset'], r['seed'], r['student'], r['policy'], r['budget']) for r in rows}) == expected
    assert all(r['routing_verified'] and 0 <= r['accuracy'] <= 1 and 0 <= r['weighted_accuracy'] <= 1 for r in rows)
    atomic_json(args.output / 'verification.json', dict(rows=expected, dataset_seed_runs=len(args.datasets)*args.seeds, routing='passed', coverage='passed', split_disjointness='passed', test_labels_used_for_selection=False))
    print(f'Completed and verified {expected} rows. Report: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
