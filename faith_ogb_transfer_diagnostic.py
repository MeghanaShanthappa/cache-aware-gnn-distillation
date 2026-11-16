"""Documented large-graph FAITH adaptation and matched cache study.

Retains released SAGE/MLP classes and energy/OSA objectives. Uses width 256,
binary undirected graphs, sparse energy algebra, and importance-sampled OSA.
This is not a reproduction of the published OGB row.
"""
import argparse
import contextlib
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path

import faith_cache as C

torch, np, F, sp = C.torch, C.np, C.F, C.support.sp
ROOT = C.ROOT
SEEDS = [42, 43, 45, 48, 52]
BUDGETS = [.01, .05, .1, .2]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def csr_tensor(a, device):
    return torch.sparse_csr_tensor(torch.as_tensor(a.indptr.astype(np.int64), device=device),
                                   torch.as_tensor(a.indices.astype(np.int64), device=device),
                                   torch.as_tensor(a.data, device=device), size=a.shape)


class CSRGraph:
    """Only the homogeneous gcn-aggregator graph primitive of DGL SAGEConv."""
    is_block = False

    def __init__(self, matrix, degree):
        self.adj, self.degree = matrix, degree
        self.srcdata = {}
        self.dstdata = self.srcdata
        self.constant_features = None
        self.constant_sum = None

    def num_edges(self):
        return self.adj.values().numel()

    def in_degrees(self):
        return self.degree

    @contextlib.contextmanager
    def local_scope(self):
        old = self.srcdata
        self.srcdata = {}
        self.dstdata = self.srcdata
        try:
            yield self
        finally:
            self.srcdata = old
            self.dstdata = old

    def update_all(self, message, reducer):
        assert message.name == 'copy_u' and reducer.name == 'sum'
        if self.srcdata['h'] is self.constant_features:
            self.dstdata['neigh'] = self.constant_sum
        else:
            self.dstdata['neigh'] = torch.sparse.mm(self.adj, self.srcdata['h'])


def pair_sample(a, count, seed):
    """Sample unordered pairs via uniformly sampled common-neighbor wedges.

    Pair inclusion probability per draw is common_neighbors / total_wedges.
    Returned weights are cosine similarity / probability (squared fairness)
    and sqrt(cosine similarity) / probability (source OSA).
    """
    rng = np.random.default_rng(seed)
    degree = np.diff(a.indptr).astype(np.int64)
    centers_mass = degree.astype(np.float64) * (degree - 1) / 2
    total = centers_mass.sum()
    centers = rng.choice(a.shape[0], count, p=centers_mass / total)
    first = (rng.random(count) * degree[centers]).astype(np.int64)
    second = (rng.random(count) * (degree[centers] - 1)).astype(np.int64)
    second += second >= first
    left = a.indices[a.indptr[centers] + first]
    right = a.indices[a.indptr[centers] + second]
    common = np.empty(count, dtype=np.float64)
    for start in range(0, count, 1024):
        sl = slice(start, start + 1024)
        common[sl] = np.asarray(a[left[sl]].multiply(a[right[sl]]).sum(1)).reshape(-1)
    assert np.all(common > 0) and np.all(left != right)
    denominator = np.sqrt(degree[left].astype(float) * degree[right])
    similarity = common / denominator
    probability = common / total
    return dict(left=left.astype(np.int64), right=right.astype(np.int64),
                osa_weight=np.sqrt(similarity) / probability,
                fairness_weight=similarity / probability,
                total_wedges=float(total), samples=count, seed=seed)


def device_pairs(pairs, device):
    return {k: torch.as_tensor(v, device=device, dtype=torch.long if k in ['left', 'right'] else torch.float32)
            if isinstance(v, np.ndarray) else v for k, v in pairs.items()}


def sampled_osa(z, pairs):
    difference = z[pairs['left']] - z[pairs['right']]
    return (torch.linalg.vector_norm(difference, dim=1) * pairs['osa_weight']).mean()


def sampled_unfairness(z, pairs):
    squared = (z[pairs['left']] - z[pairs['right']]).square().sum(1)
    return (squared * pairs['fairness_weight']).mean().item() / 1000


def node_energy(z, adjacency, degree):
    squared = z.square().sum(1)
    az = torch.sparse.mm(adjacency, z)
    return (degree * squared - 2 * (z * az).sum(1)
            + torch.sparse.mm(adjacency, squared[:, None]).squeeze(1)).clamp_min(0)


def energy_loss(z, teacher_energy, data):
    # Source student's energy accumulates to both endpoints; teacher's to source.
    student_energy = 2 * node_energy(z, data['adjacency'], data['degree'])
    numerator = student_energy @ data['adj_teacher_energy']
    ns = (data['degree'] * student_energy.square()).sum().sqrt().clamp_min(1e-8)
    nt = data['teacher_edge_norm']
    return -numerator / (ns * nt)


def objective(logits, teacher, teacher_energy, data, config, weights):
    train = data['train']
    ce = F.cross_entropy(logits[train], data['y'][train], reduction='none')
    ce = (ce * weights[train]).sum() / weights[train].sum()
    kd = F.kl_div((logits / config['tau']).log_softmax(1),
                  (teacher / config['tau']).log_softmax(1), reduction='none', log_target=True).sum(1)
    kd = (kd * weights).sum() / weights.sum()
    return config['lamb'] * ce + (1 - config['lamb']) * kd + config['beta'] * energy_loss(logits, teacher_energy, data) + config['gamma'] * sampled_osa(logits, data['pairs_train'])


def prepare(name, args):
    folder = args.data / name
    info = json.loads((folder / 'metadata.json').read_text())
    a = sp.load_npz(folder / 'adjacency.npz').tocsr()
    a.data.fill(1)  # Restore binary edges from previous symmetric normalization.
    a.sort_indices()
    assert (a != a.T).nnz == 0
    split = np.load(folder / 'splits.npz')
    valid = np.random.default_rng(1729).permutation(split['valid'])
    model_valid, calibration = np.array_split(valid, 2)
    assert not set(model_valid) & set(calibration)
    assert not set(valid) & set(split['test'])
    assert not set(split['train']) & set(split['test'])
    train_pairs = pair_sample(a, 32768, 281)
    eval_pairs = pair_sample(a, 65536, 733)
    cache = args.output / name
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / 'split.npz', train=split['train'], validation=model_valid,
                        calibration=calibration, test=split['test'])
    np.savez_compressed(cache / 'similarity_train_pairs.npz', **train_pairs)
    np.savez_compressed(cache / 'similarity_eval_pairs.npz', **eval_pairs)
    matrix = csr_tensor(a, 'cuda')
    degree = torch.as_tensor(np.diff(a.indptr).copy(), device='cuda', dtype=torch.float32)
    config = json.loads((ROOT / 'upstream/param/best_parameters.json').read_text())['ogbn-arxiv']['tran']['SAGE']
    config.update(dataset=name, hidden_dim=256, feat_dim=info['features'], label_dim=info['classes'],
                  beta=.02, gamma=1e-8 * 169343 / info['nodes'],
                  max_epoch=args.max_epochs, validation_every=5, patience=50, min_epochs=50)
    write(cache / 'config.json', config)
    metadata = dict(info, training_similarity_pairs=32768, evaluation_similarity_pairs=65536,
                    similarity_estimator='Unbiased importance estimator via uniform common-neighbor wedges; fixed train/evaluation samples are independent',
                    train_wedges=train_pairs['total_wedges'], model_validation_nodes=len(model_valid),
                    calibration_nodes=len(calibration), config=config,
                    deviations=['256 hidden units instead of released arxiv 2048',
                                'Products has no released settings; transfer arxiv settings with gamma scaled by 169343/N',
                                'Binary undirected adjacency with one self loop; SAGE adds its own self contribution',
                                'Full-batch training with sparse algebraic exact energy; importance-sampled OSA',
                                'Official validation split halved into model selection and independent cache calibration',
                                'Validate every five epochs; patience 50 epochs; maximum 500 epochs by default'])
    write(cache / 'data_protocol.json', metadata)
    data = dict(x=torch.as_tensor(np.load(folder / 'features.npy'), device='cuda'),
                y=torch.as_tensor(np.load(folder / 'labels.npy'), device='cuda'),
                train=torch.as_tensor(split['train'], device='cuda'),
                valid=torch.as_tensor(model_valid, device='cuda'),
                calibration=torch.as_tensor(calibration, device='cuda'),
                test=torch.as_tensor(split['test'], device='cuda'),
                adjacency=matrix, degree=degree, n=info['nodes'],
                pairs_train=device_pairs(train_pairs, 'cuda'), lap=device_pairs(eval_pairs, 'cuda'))
    data['graph'] = CSRGraph(matrix, degree)
    # Released first SAGE layer has no feature dropout and applies its linear
    # transform after aggregation (input width < hidden width). X has no grads.
    data['graph'].constant_features = data['x']
    data['graph'].constant_sum = torch.sparse.mm(matrix, data['x'])
    print(json.dumps(dict(dataset=name, phase='data_ready', nodes=info['nodes'], edges=a.nnz,
                          total_wedges=train_pairs['total_wedges'], gamma=config['gamma'])), flush=True)
    return data, config


@torch.no_grad()
def infer(model, data, teacher=False):
    model.eval()
    return model(data['graph'] if teacher else None, data['x']).detach()


def fit(model, data, config, path, teacher_logits=None, teacher_energy=None):
    if path.exists():
        saved = torch.load(path, map_location='cuda')
        model.load_state_dict(saved['state'])
        return saved['logits'].cuda(), saved['trace']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, last_best, best_state, trace = -1, 0, None, []
    teacher = teacher_logits is None
    weights = torch.ones(data['n'], device='cuda')
    began = time.perf_counter()
    for epoch in range(1, config['max_epoch'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(data['graph'] if teacher else None, data['x'])
        loss = F.cross_entropy(z[data['train']], data['y'][data['train']]) if teacher else objective(z, teacher_logits, teacher_energy, data, config, weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss at {path}, epoch {epoch}')
        loss.backward()
        optimizer.step()
        del z
        if epoch % config['validation_every'] == 0 or epoch == config['max_epoch']:
            with torch.no_grad():
                z = infer(model, data, teacher)
                val = (z[data['valid']].argmax(1) == data['y'][data['valid']]).float().mean().item()
            if val >= best:
                best, last_best = val, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            trace.append(dict(epoch=epoch, loss=loss.item(), validation_accuracy=val, best_epoch=last_best,
                              seconds=time.perf_counter() - began))
            if epoch % 25 == 0:
                print(json.dumps(dict(dataset=config['dataset'], seed=config['seed'], phase='teacher' if teacher else 'FAITH', **trace[-1])), flush=True)
            del z
            if epoch >= config['min_epochs'] and epoch - last_best >= config['patience']:
                break
    model.load_state_dict(best_state)
    z = infer(model, data, teacher)
    torch.save(dict(state=best_state, logits=z.cpu(), trace=trace, best_validation_accuracy=best,
                    best_epoch=last_best), path)
    write(path.with_suffix('.trace.json'), trace)
    print(json.dumps(dict(dataset=config['dataset'], seed=config['seed'], phase=path.stem + '_complete',
                          epochs=epoch, best_epoch=last_best, seconds=time.perf_counter() - began)), flush=True)
    return z, trace


def run_seed(name, seed, data, base_config, args):
    folder = args.output / name / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(base_config, seed=seed)
    manifest = dict(dataset=name, seed=seed, config=config, steps_per_phase=args.steps, phases=args.phases,
                    budgets=BUDGETS, cached_weight=.1, implementation_sha256=C.digest(Path(__file__)),
                    cache_implementation_sha256=C.digest(ROOT / 'faith_cache.py'),
                    upstream_models_sha256=C.digest(ROOT / 'upstream/src_trans/models.py'),
                    source_energy_sha256=C.digest(ROOT / 'upstream/src_trans/utils_guide.py'),
                    data_sha256={f: C.digest(args.data / name / f) for f in ['features.npy', 'labels.npy', 'adjacency.npz', 'splits.npz']},
                    split_sha256=C.digest(args.output / name / 'split.npz'),
                    similarity_sha256={f: C.digest(args.output / name / f) for f in ['similarity_train_pairs.npz', 'similarity_eval_pairs.npz']})
    if (folder / 'manifest.json').exists():
        assert json.loads((folder / 'manifest.json').read_text()) == manifest, 'Resume manifest differs'
    else:
        write(folder / 'manifest.json', manifest)
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'metrics.json').read_text())
    C.support.released_utils.set_seed(seed)
    model_t = C.support.released_models.Model(config, model_type='teacher').cuda()
    teacher, teacher_trace = fit(model_t, data, config, folder / 'teacher.pt')
    teacher_accuracy = (teacher[data['test']].argmax(1) == data['y'][data['test']]).float().mean().item()
    teacher_energy = node_energy(teacher, data['adjacency'], data['degree']).detach()
    data['adj_teacher_energy'] = torch.sparse.mm(data['adjacency'], teacher_energy[:, None]).squeeze(1)
    data['teacher_edge_norm'] = (data['degree'] * teacher_energy.square()).sum().sqrt().clamp_min(1e-8)
    del model_t
    # Separate initialization makes checkpoint resume independent of preceding RNG consumption.
    C.support.released_utils.set_seed(seed + 1000)
    initial = C.support.released_models.Model(config, model_type='student').cuda()
    z0, trace = fit(initial, data, config, folder / 'FAITH.pt', teacher, teacher_energy)
    rng = np.random.default_rng(seed + 500)
    pop = rng.permutation(np.arange(1, data['n'] + 1)).astype(float) ** -.8
    pop /= pop.sum()
    history = rng.choice(data['n'], max(100000, 10 * data['n']), p=pop)
    counts = torch.as_tensor(np.bincount(history, minlength=data['n']) + 1, dtype=torch.float32, device='cuda')
    popularity = torch.as_tensor(pop, dtype=torch.float32, device='cuda')
    test = data['test'].cpu().numpy()
    query = torch.as_tensor(rng.choice(test, 1024, p=pop[test] / pop[test].sum()), device='cuda')
    np.savez_compressed(folder / 'workload.npz', popularity=pop,
                        counts=counts.cpu().numpy(), query=query.cpu().numpy())
    rows = []

    def benefit_for(z, label):
        cal = data['calibration']
        fit_ = C.calibrate(teacher[cal], z[cal], data['y'][cal])
        write(folder / f'{label}_calibration.json', {k: v.cpu().tolist() for k, v in fit_.items()})
        return C.predict_benefit(teacher, z, fit_)

    def add(model, z, selected, variant, policy, budget, training_budget=0):
        metrics = C.evaluate(model, z, teacher, selected, data, popularity, query)
        cache_file = f'cache_{variant}_{policy}_{budget}_{training_budget}.npz'
        np.savez_compressed(folder / cache_file, selected=selected.cpu().numpy())
        rows.append(dict(dataset=name, seed=seed, variant=variant, policy=policy, budget=budget,
                         training_cache_budget=training_budget, teacher_accuracy=teacher_accuracy,
                         cache_file=cache_file,
                         additional_steps=0 if variant == 'frozen' else args.steps * args.phases, **metrics))

    empty = torch.empty(0, dtype=torch.long, device='cuda')

    def usual(model, z, variant):
        benefit = benefit_for(z, variant)
        add(model, z, empty, variant, 'none', 0)
        for b in BUDGETS:
            for policy in ['frequency', 'disagreement', 'benefit']:
                selected = C.select_cache(policy, teacher, z, counts, int(data['n'] * b), config['tau'], benefit)
                add(model, z, selected, variant, policy, b)

    def refine(label, budget=None):
        checkpoint = folder / f'{label}.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cuda')
            model = copy.deepcopy(initial)
            model.load_state_dict(saved['state'])
            model.eval()
            return model, saved['logits'].cuda()
        model, refinement_trace = C.continue_student(initial, teacher, teacher_energy, data, config,
                                                     counts, seed, args.steps, args.phases, budget)
        z = C.logits_of(model, data['x'])
        torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, logits=z.cpu(),
                        trace=refinement_trace), checkpoint)
        write(folder / f'{label}_trace.json', refinement_trace)
        return model, z

    usual(initial, z0, 'frozen')
    print(json.dumps(dict(dataset=name, seed=seed, phase='frozen_cache_complete')), flush=True)
    control, zc = refine('continued')
    usual(control, zc, 'continued')
    for b in BUDGETS:
        joint, zj = refine(f'joint_{b}', b)
        selected = C.select_cache('disagreement', teacher, zj, counts, int(data['n'] * b), config['tau'])
        benefit = benefit_for(zj, f'joint_{b}')
        add(joint, zj, empty, 'joint', 'none', 0, b)
        add(joint, zj, selected, 'joint', 'joint', b, b)
        add(control, zc, selected, 'continued', 'joint_cache_control', b, b)
        for policy in ['frequency', 'benefit']:
            selected_p = C.select_cache(policy, teacher, zj, counts, len(selected), config['tau'], benefit)
            add(joint, zj, selected_p, 'joint', policy, b, b)
        print(json.dumps(dict(dataset=name, seed=seed, phase='joint_cache_complete', budget=b)), flush=True)
        del joint, zj
    assert len(rows) == 46
    write(folder / 'metrics.json', rows)
    write(folder / 'complete.json', dict(rows=46, teacher_accuracy=teacher_accuracy,
                                        checkpoints_sha256={f: C.digest(folder / f) for f in ['teacher.pt', 'FAITH.pt', 'continued.pt'] + [f'joint_{b}.pt' for b in BUDGETS]}))
    print(json.dumps(dict(dataset=name, seed=seed, phase='complete', rows=46)), flush=True)
    return rows


def report(rows, output):
    write(output / 'metrics.json', rows)
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# OGB: scalable FAITH-based cache study', '',
             'A100 study with released SAGE/MLP model classes, width 256, full-graph energy alignment and importance-sampled OSA. This is a documented adaptation, not published FAITH reproduction. Products has no released configuration; arxiv settings transfer with gamma multiplied by 169343/N. All official test nodes are evaluated. Cache calibration uses half of official validation, disjoint from checkpoint selection.', '',
             '## Frozen student at 10% cache', '',
             'Request-weighted accuracy (%), mean ± population standard deviation. Synthetic Zipf(0.8) requests; all completed seeds are included. Uniform accuracy appears separately.', '',
             '| Dataset | Seeds | No cache | Frequency | Disagreement | Benefit |',
             '|---|---:|---:|---:|---:|---:|']
    def value(dataset, variant, policy, metric='weighted_accuracy', budget=.1):
        group = [r[metric] * 100 for r in rows if r['dataset'] == dataset and r['variant'] == variant and r['policy'] == policy and r['budget'] == (0 if policy == 'none' else budget)]
        return f'{statistics.mean(group):.2f} ± {statistics.pstdev(group):.2f}'
    for d in sorted({r['dataset'] for r in rows}):
        count = len({r['seed'] for r in rows if r['dataset'] == d})
        lines.append('| ' + ' | '.join([d, str(count)] + [value(d, 'frozen', p) for p in ['none', 'frequency', 'disagreement', 'benefit']]) + ' |')
    lines += ['', '## Uniform test accuracy (%)', '', '| Dataset | No cache | Frequency | Disagreement | Benefit |', '|---|---:|---:|---:|---:|']
    for d in sorted({r['dataset'] for r in rows}):
        lines.append('| ' + ' | '.join([d] + [value(d, 'frozen', p, 'accuracy') for p in ['none', 'frequency', 'disagreement', 'benefit']]) + ' |')
    lines += ['', '## Equal-update controls at 10% cache', '',
              '| Dataset | Continued no cache | Continued frequency | Continued disagreement | Joint | Continued using joint cache | Continued benefit | Joint benefit |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in sorted({r['dataset'] for r in rows}):
        settings = [('continued', 'none'), ('continued', 'frequency'), ('continued', 'disagreement'), ('joint', 'joint'), ('continued', 'joint_cache_control'), ('continued', 'benefit'), ('joint', 'benefit')]
        lines.append('| ' + ' | '.join([d] + [value(d, v, p) for v, p in settings]) + ' |')
    lines += ['', '## Protocol and limits', '',
              'Three-layer width-256 SAGE gcn teacher and released batch-normalized MLP student; dropout teacher 0.5/student 0.1, Adam lr 0.005, weight decay 0, lambda 0.2, tau 1 with no tau-squared multiplier. Beta 0.02; gamma 1e-8 on arxiv and 1e-8*169343/N on products. These choices were fixed before test evaluation. Arxiv released width is 2048, so width 256 is an explicit memory/runtime adaptation. Products settings are transferred, not author-specified.', '',
              'Maximum 500 epochs by default, validation every five, minimum 50, patience 50 epochs; validation ties replace the checkpoint. Model selection sees half of official validation; the other half calibrates a fixed ridge benefit regressor. Both are disjoint from train/test. Continued and joint refinement use the same initial weights, optimizer resets, dropout seeds and 150 added updates (three phases of 50); final refinement steps are evaluated without selecting by test results.', '',
              'Source energy cosine is computed by equivalent sparse matrix identities over every graph edge. OSA estimates the released sum of sqrt(adjacency-row cosine) times logit distance using 32,768 common-neighbor wedge samples with inverse pair-probability weights. Graph unfairness is independently estimated with 65,536 fixed evaluation samples; raw and probability values are both saved. This is a Monte Carlo graph metric, not exact full-graph paper unfairness. Fixed samples can introduce approximation and optimization error; seeds vary training/workload, not the fixed graph samples.', '',
              'Binary undirected adjacency has one self loop; the released SAGE gcn class adds its own self contribution. Products/arxiv features remain original. Source arxiv loader can retain reciprocal-edge multiplicity, whereas this extension deduplicates it. Teacher and MLP computation execute on CUDA; CSR message sums replace unavailable native DGL CUDA, with native-DGL value/gradient tests.', '',
              'Historical workload contains max(100000,10*N) Zipf(0.8) requests and additive-one count smoothing. Node cache bounds are 1%,5%,10%,20% of all graph nodes. Cache payload is 4*k*C bytes; lookup is 8*N bytes when nonempty, counted separately from student parameters. Benefit may underfill the budget. Teacher outputs are prepared offline. Test labels do not select caches, checkpoints or configurations.', '',
              'Latency includes lookup, output assembly and MLP inference only on misses, for 1,024 GPU-resident test requests, 10 warmups and median 40 synchronized calls. It excludes host/network transfer and offline construction/refresh. Query outputs and argmaxes must agree with canonical mixed predictions. Shared-device/order variability limits performance conclusions. Compare within this trial, not against earlier A40 or citation numbers.', '',
              'See per-dataset data_protocol.json, per-seed manifests, split.npz, fixed similarity samples, training traces and all metrics. This report updates only after complete seeds; all intended five seeds per dataset are required for the final study.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['ogbn-arxiv', 'ogbn-products'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--data', type=Path, default=ROOT / 'ogb-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/faith_ogb')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--phases', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Host GPU access required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    C.objective = objective
    C.unfairness = sampled_unfairness
    C.support.released_guide.F = F
    write(args.output / 'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
          implementation_sha256=C.digest(Path(__file__)), seeds=args.seeds, datasets=args.datasets,
          max_epochs=args.max_epochs, steps_per_phase=args.steps, phases=args.phases,
          study='FAITH-based scalable adaptation; not published OGB reproduction'))
    rows = []
    for name in args.datasets:
        data, config = prepare(name, args)
        for seed in args.seeds:
            rows.extend(run_seed(name, seed, data, config, args))
            report(rows, args.output)
        del data
        torch.cuda.empty_cache()
    assert len(rows) == len(args.datasets) * len(args.seeds) * 46
    write(args.output / 'verification.json', dict(rows=len(rows), runs=len(args.datasets) * len(args.seeds),
          routing='passed', calibration='separate from checkpoint selection and test',
          completed=True, adaptation=True))
    print(f'Completed {len(rows)} evaluations: {args.output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
