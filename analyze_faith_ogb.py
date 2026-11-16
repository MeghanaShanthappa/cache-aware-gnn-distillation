"""Independently verify OGB cache artifacts and reconstruct every quality row."""
import argparse
import hashlib
import json
import math
import statistics as S
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/faith_ogb'
DATA = ROOT / 'ogb-data'
SEEDS = [42, 43, 45, 48, 52]
BUDGETS = [.01, .05, .1, .2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['ogbn-arxiv', 'ogbn-products'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = [r for r in read(OUT / 'metrics.json') if r['dataset'] in args.datasets and r['seed'] in args.seeds]
    expected_rows = 46 * len(args.seeds) * len(args.datasets)
    assert len(rows) == expected_rows
    keys = [(r['dataset'], r['seed'], r['variant'], r['policy'], r['budget'], r['training_cache_budget']) for r in rows]
    assert len(set(keys)) == expected_rows
    source_hashes = dict(implementation_sha256=digest(ROOT / 'faith_ogb.py'),
                         cache_implementation_sha256=digest(ROOT / 'faith_cache.py'),
                         upstream_models_sha256=digest(ROOT / 'upstream/src_trans/models.py'),
                         source_energy_sha256=digest(ROOT / 'upstream/src_trans/utils_guide.py'))
    selection = read(OUT / 'validation_selection.json')
    selection_hash = digest(OUT / 'validation_selection.json')
    tuning_protocol = ROOT / 'results/faith_ogb_tuning/protocol.json'
    protocol = read(tuning_protocol)
    assert protocol['implementation_sha256'] == digest(ROOT / 'select_faith_ogb.py')
    assert protocol['training_implementation_sha256'] == source_hashes['implementation_sha256']
    for dataset in args.datasets:
        selected = selection[dataset]
        assert selected['criterion'] == 'model-validation accuracy only'
        assert selected['protocol_sha256'] == digest(tuning_protocol)
        candidates = selected['candidates']
        assert [r['multiplier'] for r in candidates] == [1., .1, .01, .001]
        assert all(r['gamma'] > 0 and r['updates'] == 150 for r in candidates)
        best = max(candidates, key=lambda r: (r['validation_accuracy'], r['gamma']))
        assert selected['selected_gamma'] == best['gamma']
        for candidate in candidates:
            folder = ROOT / 'results/faith_ogb_tuning' / dataset / f"multiplier{candidate['multiplier']}"
            trace = read(folder / 'FAITH.trace.json')
            assert trace[-1]['epoch'] == 150
            assert max(t['validation_accuracy'] for t in trace) == candidate['validation_accuracy']
            assert digest(folder / 'FAITH.pt') == candidate['checkpoint_sha256']
    checked = 0
    dataset_info = {}
    for dataset in args.datasets:
        folder = OUT / dataset
        metadata = read(DATA / dataset / 'metadata.json')
        n, classes = metadata['nodes'], metadata['classes']
        labels = np.load(DATA / dataset / 'labels.npy')
        split = np.load(folder / 'split.npz')
        original_split = np.load(DATA / dataset / 'splits.npz')
        masks = [set(split[k]) for k in ['train', 'validation', 'calibration', 'test']]
        assert all(not a & b for i, a in enumerate(masks) for b in masks[i+1:])
        assert masks[1] | masks[2] == set(original_split['valid'])
        np.testing.assert_array_equal(split['test'], original_split['test'])
        np.testing.assert_array_equal(split['train'], original_split['train'])
        test, valid = split['test'], split['validation']
        data_hashes = {f: digest(DATA / dataset / f) for f in ['features.npy', 'labels.npy', 'adjacency.npz', 'splits.npz']}
        pair_hashes = {f: digest(folder / f) for f in ['similarity_train_pairs.npz', 'similarity_eval_pairs.npz']}
        split_hash = digest(folder / 'split.npz')
        dataset_info[dataset] = dict(nodes=n, test_nodes=len(test), calibration_nodes=len(split['calibration']),
                                     validation_nodes=len(valid))
        for seed in args.seeds:
            run = folder / f'seed{seed}'
            manifest = read(run / 'manifest.json')
            assert all(manifest[k] == v for k, v in source_hashes.items())
            assert manifest['selected_coefficient_sha256'] == selection_hash
            assert manifest['config']['gamma_selection_sha256'] == selection_hash
            assert manifest['config']['gamma'] == selection[dataset]['selected_gamma']
            if manifest['reused_teacher_sha256'] is not None:
                original_teacher = ROOT / 'results/faith_ogb_transfer_diagnostic' / dataset / f'seed{seed}/teacher.pt'
                assert digest(original_teacher) == manifest['reused_teacher_sha256']
                assert digest(original_teacher.parent.parent / 'split.npz') == split_hash
            assert manifest['data_sha256'] == data_hashes
            assert manifest['similarity_sha256'] == pair_hashes and manifest['split_sha256'] == split_hash
            assert manifest['phases'] == 3 and manifest['steps_per_phase'] == 50
            assert manifest['cached_weight'] == .1 and manifest['budgets'] == BUDGETS
            completion = read(run / 'complete.json')
            assert completion['rows'] == 46
            for f, sha in completion['checkpoints_sha256'].items():
                assert digest(run / f) == sha
            local = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed]
            assert len(local) == 46 and local == read(run / 'metrics.json')
            workload = np.load(run / 'workload.npz')
            pop = workload['popularity']
            assert np.isclose(pop.sum(), 1) and np.all(pop > 0)
            assert np.all(workload['counts'] >= 1)
            assert int(workload['counts'].sum(dtype=np.float64)) == n + max(100000, 10 * n)
            assert set(workload['query']).issubset(masks[3])
            rng = np.random.default_rng(seed + 500)
            expected = rng.permutation(np.arange(1, n + 1)).astype(float) ** -.8
            expected /= expected.sum()
            np.testing.assert_array_equal(pop, expected)
            p = pop[test] / pop[test].sum()
            teacher = torch.load(run / 'teacher.pt', map_location='cpu')
            teacher_pred = teacher['logits'].argmax(1).numpy()
            for checkpoint in ['teacher.pt', 'FAITH.pt']:
                model = teacher if checkpoint == 'teacher.pt' else torch.load(run / checkpoint, map_location='cpu')
                correct = model['logits'][torch.as_tensor(valid)].argmax(1).numpy() == labels[valid]
                assert abs(correct.mean() - model['best_validation_accuracy']) < 2e-7
                trace = model['trace']
                assert max(t['validation_accuracy'] for t in trace) == model['best_validation_accuracy']
                assert model['best_epoch'] == max(t['epoch'] for t in trace if t['validation_accuracy'] == model['best_validation_accuracy'])
                assert len(trace) > 1 and all(math.isfinite(t['loss']) for t in trace)
            del model, teacher
            predictions = {}
            for label in ['FAITH', 'continued'] + [f'joint_{b}' for b in BUDGETS]:
                checkpoint = torch.load(run / f'{label}.pt', map_location='cpu')
                predictions[label] = checkpoint['logits'].argmax(1).numpy()
                if label != 'FAITH':
                    trace = checkpoint['trace']
                    assert len(trace) == 3 and sum(t['steps'] for t in trace) == 150
                    capacity = 0 if label == 'continued' else int(n * float(label.split('_')[1]))
                    assert all(t['cached_nodes'] == capacity and math.isfinite(t['last_loss']) for t in trace)
                del checkpoint
            model_bytes = local[0]['model_bytes']
            for r in local:
                assert all(math.isfinite(v) for v in r.values() if isinstance(v, (float, int)))
                assert r['routing_argmax_agreement'] == 1 and r['query_batch_ms'] > 0
                assert r['additional_steps'] == (0 if r['variant'] == 'frozen' else 150)
                selected = np.load(run / r['cache_file'])['selected']
                assert len(selected) == len(np.unique(selected)) == r['cached_nodes']
                assert len(selected) <= int(n * r['budget'])
                assert np.all((selected >= 0) & (selected < n))
                assert r['payload_bytes'] == 4 * len(selected) * classes
                assert r['index_bytes'] == (8 * n if len(selected) else 0)
                assert r['model_bytes'] == model_bytes
                label = 'FAITH' if r['variant'] == 'frozen' else 'continued' if r['variant'] == 'continued' else f"joint_{r['training_cache_budget']}"
                pred = predictions[label].copy()
                pred[selected] = teacher_pred[selected]
                correct = (pred[test] == labels[test]).astype(float)
                quality = dict(accuracy=correct.mean(), weighted_accuracy=correct @ p,
                               shifted_accuracy=correct @ (.5 * p + .5 * p[::-1]))
                hit = np.zeros(n, dtype=bool); hit[selected] = True
                quality['hit_rate'] = hit[test].astype(float) @ p
                for metric, value in quality.items():
                    assert abs(r[metric] - value) < 2e-6, (dataset, seed, r['variant'], r['policy'], metric, r[metric], value)
                teacher_accuracy = (teacher_pred[test] == labels[test]).mean()
                assert abs(r['teacher_accuracy'] - teacher_accuracy) < 2e-7
                checked += 1
            print(f'Independent checks passed: {dataset} seed {seed}', flush=True)
    exact = {}
    if (OUT / 'exact_fairness.json').exists():
        metadata = read(OUT / 'exact_fairness_verification.json')
        assert metadata['metrics_sha256'] == digest(OUT / 'metrics.json')
        assert metadata['implementation_sha256'] == digest(ROOT / 'exact_fairness_ogb.py')
        for r in read(OUT / 'exact_fairness.json'):
            key = tuple(r[k] for k in ['dataset', 'seed', 'variant', 'policy', 'budget', 'training_cache_budget'])
            assert key not in exact
            assert math.isfinite(r['exact_individual_unfairness']) and math.isfinite(r['exact_probability_unfairness'])
            exact[key] = r
        assert len(exact) == 460
    verification = dict(rows=checked, runs=len(args.datasets)*len(args.seeds), seeds=args.seeds, original_official_splits=True,
                        disjoint_training_validation_calibration_test=True,
                        independent_reconstruction_of_accuracy_weighted_shifted_and_hit_rate=True,
                        source_checkpoint_and_data_hashes=True, budgets_bytes_and_indices=True,
                        equal_refinement_steps=150, finite_metrics_and_routing_agreement=True,
                        positive_coefficient_selected_by_model_validation_only=True,
                        validation_selected_checkpoints=True,
                        similarity_metric='Exact sparse quadratic form supplement' if exact else 'Monte Carlo estimate, independent fixed evaluation sample',
                        datasets=dataset_info)
    full_seeds = args.seeds == SEEDS
    destination = OUT if len(args.datasets) == 2 else OUT / args.datasets[0]
    filename = 'independent_verification.json' if full_seeds else 'partial_independent_verification.json'
    (destination / filename).write_text(json.dumps(verification, indent=2) + '\n')
    if len(args.datasets) != 2 or not full_seeds:
        print(json.dumps(verification, indent=2))
        return

    def group(d, v, p, budget):
        return sorted([r for r in rows if r['dataset'] == d and r['variant'] == v and r['policy'] == p and r['budget'] == budget], key=lambda r: r['seed'])

    paired = []
    for d in dataset_info:
        for b in BUDGETS:
            pairs = [(f'frozen {p} vs no cache', group(d, 'frozen', p, b), group(d, 'frozen', 'none', 0)) for p in ['frequency', 'disagreement', 'benefit']]
            pairs += [('joint vs continued disagreement', group(d, 'joint', 'joint', b), group(d, 'continued', 'disagreement', b)),
                      ('joint vs continued same cache', group(d, 'joint', 'joint', b), group(d, 'continued', 'joint_cache_control', b)),
                      ('joint benefit vs continued benefit', group(d, 'joint', 'benefit', b), group(d, 'continued', 'benefit', b))]
            for name, treated, control in pairs:
                result = dict(dataset=d, budget=b, comparison=name)
                for metric in ['accuracy', 'weighted_accuracy', 'shifted_accuracy']:
                    values = [(a[metric] - b_[metric]) * 100 for a, b_ in zip(treated, control)]
                    result[metric + '_delta_pp'] = dict(mean=S.mean(values), population_std=S.pstdev(values),
                                                       wins=sum(x > 1e-5 for x in values), values=values)
                paired.append(result)
    (OUT / 'paired_comparisons.json').write_text(json.dumps(paired, indent=2) + '\n')
    lines = ['# Completed OGB FAITH-based cache extension', '',
             'At a 10% cache budget, benefit caching improves request-weighted and uniform accuracy on both OGB datasets. Joint refinement adds little over equally trained controls, while cache routing increases measured query latency. Exact graph unfairness depends on the dataset and policy.', '',
             'Both ogbn-arxiv and ogbn-products completed all five seeds and four budgets: 460 evaluations. Ten unit checks and CUDA equivalence checks passed; all 460 accuracy, request-weighted accuracy, shifted-workload accuracy and hit-rate rows were independently reconstructed from checkpoint predictions, saved cache indices and the workload. No failed or low-accuracy seed is excluded.', '',
             '## Accuracy at 10% cache', '',
             'Mean synthetic request-weighted accuracy (%), with mean uniform test accuracy in parentheses. These are two different metrics. The teacher column is uniform accuracy.', '',
             '| Dataset | Teacher | FAITH-based student | Frequency cache | Disagreement cache | Benefit cache |',
             '|---|---:|---:|---:|---:|---:|']
    for d in dataset_info:
        baseline = group(d, 'frozen', 'none', 0)
        cells = [d, f"{S.mean(r['teacher_accuracy'] for r in baseline)*100:.2f}"]
        for p in ['none', 'frequency', 'disagreement', 'benefit']:
            g = group(d, 'frozen', p, 0 if p == 'none' else .1)
            cells.append(f"{S.mean(r['weighted_accuracy'] for r in g)*100:.2f} ({S.mean(r['accuracy'] for r in g)*100:.2f})")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Validation-only coefficient selection', '',
              '| Dataset | Reference gamma | Selected gamma | Seed-42 validation accuracy at selection |',
              '|---|---:|---:|---:|']
    for d in dataset_info:
        chosen = selection[d]
        reference = chosen['candidates'][0]['gamma']
        best = next(c for c in chosen['candidates'] if c['gamma'] == chosen['selected_gamma'])
        lines.append(f"| {d} | {reference:.6g} | {chosen['selected_gamma']:.6g} | {best['validation_accuracy']*100:.2f}% |")
    lines += ['', 'Selection uses accuracy without a fairness constraint. A positive similarity coefficient retains OSA but does not establish the fairness improvement reported by the original paper. The exact graph metrics below assess the resulting tradeoff.', '']
    lines += ['', '## Joint refinement against matched controls', '',
              'Paired request-weighted percentage-point changes at 10%, mean ± population standard deviation (wins out of five seeds). Both students receive 150 updates from the same initial weights, optimizer resets and dropout seeds. Same-cache control holds cache indices fixed, isolating changing the student.', '',
              '| Dataset | Joint − continued disagreement | Joint − continued same cache | Joint benefit − continued benefit |',
              '|---|---:|---:|---:|']
    for d in dataset_info:
        cells = [d]
        for label in ['joint vs continued disagreement', 'joint vs continued same cache', 'joint benefit vs continued benefit']:
            stat = next(x for x in paired if x['dataset'] == d and x['budget'] == .1 and x['comparison'] == label)['weighted_accuracy_delta_pp']
            cells.append(f"{stat['mean']:+.2f} ± {stat['population_std']:.2f} ({stat['wins']}/5)")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Serving and graph smoothness', '',
              ('Means at 10% frequency caching. Unfairness is the exact full-graph adjacency-row cosine Laplacian quadratic form, divided by 1,000. A float64 sparse factorization avoids constructing all pairs. Probability unfairness applies it to softmax probabilities. Lower is better. The original sampled diagnostics are retained separately.' if exact else 'Means at 10% frequency caching. Unfairness estimates the released Laplacian metric divided by 1,000; probability unfairness applies it to softmax probabilities. Lower is better. Both use the same independent fixed graph sample across methods.'), '',
              '| Dataset | Batch latency ms, base → cache | Raw unfairness, base → cache | Probability unfairness, base → cache | Payload + index bytes |',
              '|---|---:|---:|---:|---:|']
    for d in dataset_info:
        base, cached = group(d, 'frozen', 'none', 0), group(d, 'frozen', 'frequency', .1)
        cells = [d]
        for metric in ['query_batch_ms', 'individual_unfairness', 'probability_unfairness']:
            def number(r):
                key = tuple(r[k] for k in ['dataset', 'seed', 'variant', 'policy', 'budget', 'training_cache_budget'])
                return exact[key]['exact_' + metric] if exact and metric != 'query_batch_ms' else r[metric]
            cells.append(f"{S.mean(number(r) for r in base):.3f} → {S.mean(number(r) for r in cached):.3f}")
        cells.append(f"{cached[0]['payload_bytes']:,} + {cached[0]['index_bytes']:,}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Scope and evidence', '',
              'This is a scalable FAITH-based adaptation, not reproduction of the published OGB result. It uses the released three-layer SAGE gcn and batch-normalized MLP classes, reduced width 256 instead of arxiv’s 2048, binary undirected edges, exact sparse energy and an importance-sampled OSA sum. Products lacks released settings. Gamma is selected from the source reference (size-scaled for products) times [1,0.1,0.01,0.001], with 150 updates per candidate on seed 42. Only model-validation accuracy selects a positive coefficient, then it is locked across all five final seeds. The failed direct-transfer experiment and its already-observed arxiv test results are preserved in ../faith_ogb_transfer_diagnostic; this follow-up is exploratory. The teacher/student gap and adaptations limit claims about improvements over published FAITH or stronger OGB reference models.', '',
              'The official validation split is divided with a fixed seed between checkpoint selection and cache-benefit calibration; test labels are used only for metrics. Benefit uses fixed ridge regression and can leave cache slots empty. Joint training downweights cached nodes to 0.1 in CE/KL while retaining energy and OSA. Joint ranking uses frequency times KL; the other policies use frequency or frequency times predicted benefit. All cache fractions are upper bounds over all graph nodes.', '',
              'Synthetic Zipf popularity is coupled with training seed, with five paired realizations. OSA uses a fixed graph-similarity training sample; variation across seeds does not estimate its sampling uncertainty. Original evaluation samples are independent. The exact quadratic fairness supplement removes evaluation-sampling error, while training OSA remains approximate. Accuracy gains on this workload are not evidence of deployment gains. No significance or state-of-the-art claim is made. Timing is a local GPU-resident 1,024-query microbenchmark; it includes routing and miss inference, excludes host/network/offline work, and is subject to shared-device and ordering variability.', '',
              'Memory columns count payload and lookup storage required by the serving policy, rather than peak allocator usage. The benchmark wrapper initializes an unused dense lookup even for an empty cache; the empty-cache query branch does not access it. Thus the recorded zero empty-cache lookup is a logical serving requirement, not its literal allocation in that wrapper.', '',
              'Earlier citation and A40 results use different architectures/protocols. They remain separate experiments. Full protocol and mean ± standard deviation tables: [REPORT.md](REPORT.md). All budgets and paired deltas: [paired_comparisons.json](paired_comparisons.json). Independent checks: [independent_verification.json](independent_verification.json). Per-seed files retain checkpoints, traces, calibration, workload and exact cache indices.', '']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines))
    from report_builder import write_current_report
    write_current_report()
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
