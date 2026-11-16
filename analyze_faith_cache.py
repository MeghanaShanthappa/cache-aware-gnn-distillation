"""Independent artifact checks and paired summaries for the completed cache trial."""
import hashlib
import json
import math
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/faith_cache'
BASE = ROOT / 'results/cuda_reproduction'
DATASETS = ['cora', 'citeseer', 'pubmed']
SEEDS = [42, 43, 45, 48, 52]
BUDGETS = [.01, .05, .1, .2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return dict(mean=stats.mean(values), population_std=stats.pstdev(values), values=values)


def main():
    rows = read(OUT / 'metrics.json')
    assert len(rows) == 690
    keys = [(r['dataset'], r['seed'], r['variant'], r['policy'], r['budget'], r['training_cache_budget']) for r in rows]
    assert len(set(keys)) == 690
    implementation = digest(ROOT / 'faith_cache.py')
    for dataset in DATASETS:
        for seed in SEEDS:
            target, source = OUT / dataset / f'seed{seed}', BASE / dataset / f'seed{seed}'
            manifest = read(target / 'manifest.json')
            checks = {
                'implementation_sha256': implementation,
                'model_source_sha256': digest(ROOT / 'upstream/src_trans/models.py'),
                'energy_source_sha256': digest(ROOT / 'upstream/src_trans/utils_guide.py'),
                'source_checkpoint_sha256': digest(source / 'FAITH.pt'),
                'teacher_checkpoint_sha256': digest(source / 'teacher.pt'),
                'source_config_sha256': digest(source / 'config.json'),
            }
            for key, value in checks.items():
                assert manifest[key] == value, (dataset, seed, key)
            assert manifest['cached_weight'] == .1
            assert manifest['phases'] == 3 and manifest['steps_per_phase'] == 50
            assert manifest['budgets'] == BUDGETS
            complete = read(target / 'complete.json')
            assert complete['rows'] == 46 and complete['source_baseline_accuracy_and_unfairness_match']
            local = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed]
            assert len(local) == 46 and local == read(target / 'metrics.json')
            split, config = read(source / 'split.json'), read(source / 'config.json')
            train, valid, test = (set(split[k]) for k in ['train', 'validation', 'test'])
            assert len(valid) == 500 and len(test) == 1000
            assert not (train & valid or train & test or valid & test)
            frozen = next(r for r in local if r['variant'] == 'frozen' and r['policy'] == 'none')
            original = read(source / 'metrics.json')['methods']['FAITH']
            assert abs(frozen['accuracy'] - original['accuracy']) < 1e-7
            assert math.isclose(frozen['individual_unfairness'], original['individual_unfairness'], rel_tol=1e-5, abs_tol=1e-5)
            for r in local:
                assert all(math.isfinite(v) for v in r.values() if isinstance(v, (float, int)))
                for key in ['accuracy', 'weighted_accuracy', 'shifted_accuracy', 'hit_rate']:
                    assert 0 <= r[key] <= 1
                assert r['routing_argmax_agreement'] == 1 and r['query_batch_ms'] > 0
                assert r['cached_nodes'] <= int(split['nodes'] * r['budget'])
                assert r['payload_bytes'] == 4 * r['cached_nodes'] * config['label_dim']
                assert r['index_bytes'] == (8 * split['nodes'] if r['cached_nodes'] else 0)
                assert r['model_bytes'] == frozen['model_bytes']
                assert r['additional_steps'] == (0 if r['variant'] == 'frozen' else 150)
            for label in ['continued'] + [f'joint_{b}' for b in BUDGETS]:
                trace = read(target / f'{label}_trace.json')
                assert len(trace) == 3 and sum(t['steps'] for t in trace) == 150
                assert all(math.isfinite(t['last_loss']) for t in trace)
                capacity = 0 if label == 'continued' else int(split['nodes'] * float(label.split('_')[1]))
                assert all(t['cached_nodes'] == capacity for t in trace)
    verification = dict(rows=690, runs=15, unique_conditions=True, source_hashes_unchanged=True,
                        original_baseline_metrics_match=True, cache_capacity_and_bytes=True,
                        finite_metrics=True, query_routing_agreement=True, matched_updates=150,
                        full_validation_masks_disjoint_from_test=True,
                        calibration='validation labels reused after checkpoint selection; exploratory',
                        failed_citeseer_seed45_retained=True, implementation_sha256=implementation)
    (OUT / 'independent_verification.json').write_text(json.dumps(verification, indent=2) + '\n')

    def group(dataset, variant, policy, budget):
        return sorted([r for r in rows if r['dataset'] == dataset and r['variant'] == variant
                       and r['policy'] == policy and r['budget'] == budget], key=lambda r: r['seed'])

    comparisons = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            pairs = [(f'frozen {p} vs no cache', group(dataset, 'frozen', p, budget), group(dataset, 'frozen', 'none', 0))
                     for p in ['frequency', 'disagreement', 'benefit']]
            pairs += [('joint vs continued disagreement', group(dataset, 'joint', 'joint', budget), group(dataset, 'continued', 'disagreement', budget)),
                      ('joint vs continued same cache', group(dataset, 'joint', 'joint', budget), group(dataset, 'continued', 'joint_cache_control', budget)),
                      ('joint benefit vs continued benefit', group(dataset, 'joint', 'benefit', budget), group(dataset, 'continued', 'benefit', budget))]
            for label, treated, control in pairs:
                assert len(treated) == len(control) == 5
                entry = dict(dataset=dataset, budget=budget, comparison=label)
                for metric in ['weighted_accuracy', 'accuracy', 'shifted_accuracy']:
                    delta = [(a[metric] - b[metric]) * 100 for a, b in zip(treated, control)]
                    entry[metric + '_delta_pp'] = summary(delta)
                    entry[metric + '_wins'] = sum(v > 1e-5 for v in delta)
                comparisons.append(entry)
    (OUT / 'paired_comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')

    lines = ['# Cache extension: completed A100 findings', '',
             'Static caching gives dataset-dependent accuracy gains; this joint refinement does not establish a consistent advantage. Hard teacher-logit replacement increases graph unfairness. The cache adds latency in this GPU microbenchmark.', '',
             'All 15 saved full-FAITH dataset/seed runs and four cache budgets (1%, 5%, 10%, 20%) were evaluated: 690 rows. The original checkpoints are unchanged. The unstable CiteSeer seed 45 is retained. These are extensions of our local reproduction, which did not reproduce the paper’s stability; they are not a published-baseline or state-of-the-art claim.', '',
             '## Frozen student, 10% cache upper bound', '',
             'Mean request-weighted accuracy (%), with paired changes in percentage points. Each policy uses the same five student/workload seeds. Uniform accuracy is separate from synthetic request-weighted accuracy.', '',
             '| Dataset | No cache | Frequency | Change | Disagreement | Change | Benefit | Change |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        baseline = stats.mean(r['weighted_accuracy'] for r in group(dataset, 'frozen', 'none', 0)) * 100
        cells = [dataset, f'{baseline:.2f}']
        for policy in ['frequency', 'disagreement', 'benefit']:
            value = stats.mean(r['weighted_accuracy'] for r in group(dataset, 'frozen', policy, .1)) * 100
            cells.extend([f'{value:.2f}', f'{value-baseline:+.2f}'])
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Benefit calibration uses the same 500 validation labels previously used for checkpoint selection; this result is exploratory. It may leave capacity empty. The three policies were fixed before evaluation; highlighting their dataset-specific outcomes does not define a deployable policy-selection rule.', '',
              '## Does joint training add value?', '',
              'Both continuation and joint refinement start from the same saved student and use 150 additional updates, matched optimizer resets and dropout seeds. Joint refinement alternates frequency-times-KL cache selection with downweighting cached nodes to 0.1 in CE/KL. Full FAITH energy and OSA losses remain intact. Final updates are evaluated without test-driven checkpoint selection.', '',
              '| Dataset | Joint − continued disagreement | Joint − continued using identical cache | Joint benefit − continued benefit |',
              '|---|---:|---:|---:|']
    for dataset in DATASETS:
        cells = [dataset]
        for label in ['joint vs continued disagreement', 'joint vs continued same cache', 'joint benefit vs continued benefit']:
            item = next(c for c in comparisons if c['dataset'] == dataset and c['budget'] == .1 and c['comparison'] == label)
            delta = item['weighted_accuracy_delta_pp']
            cells.append(f"{delta['mean']:+.2f} ± {delta['population_std']:.2f} ({item['weighted_accuracy_wins']}/5 wins)")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Values are paired request-weighted percentage-point differences, mean ± population standard deviation. Same-cache comparison isolates changing the student while holding routing decisions fixed. The improvements are not consistent across datasets; no statistical-significance claim is made.', '',
              '## Cost and fairness', '',
              'At 10% frequency caching, means across the five seeds:', '',
              '| Dataset | Uniform accuracy, base → cache | Batch latency ms, base → cache | Raw-logit unfairness, base → cache | Probability unfairness, base → cache | Payload + lookup bytes |',
              '|---|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        base, cached = group(dataset, 'frozen', 'none', 0), group(dataset, 'frozen', 'frequency', .1)
        cells = [dataset]
        for metric, scale in [('accuracy', 100), ('query_batch_ms', 1), ('individual_unfairness', 1), ('probability_unfairness', 1)]:
            a, b = [stats.mean(r[metric] for r in g) * scale for g in [base, cached]]
            cells.append(f'{a:.3f} → {b:.3f}')
        cells.append(f"{cached[0]['payload_bytes']:,} + {cached[0]['index_bytes']:,}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Lower unfairness is better. Raw-logit unfairness follows the release and is divided by 1,000; teacher/student logit scale differences affect it. Probability unfairness also rises, so the effect is not solely a logit-scale artifact. The failed low-accuracy student can have deceptively low unfairness.', '',
              'Latency is a local GPU-resident batch of 1,024 test queries: 10 warmups, median of 40 synchronized calls. Lookup, output assembly and miss-only MLP inference are included. Host transfer, network and offline cache construction are excluded. Timings are subject to shared-GPU and ordering variability; these measurements do not demonstrate a serving speedup. Dense index bytes are outside the nominal node-payload budget. No-cache entries allocate neither index nor payload.', '',
              '## Interpretation and audit', '',
              'The simplest positive result is frequency caching on Cora (+1.05 weighted points, +0.06 uniform points). CiteSeer benefit caching gains +3.85 weighted points and +1.34 uniform points, but includes the unstable base seed and reused validation calibration. PubMed does not improve at the headline 10% budget. Joint refinement is not supported as an improvement over simple independently trained cache baselines by this trial.', '',
              'Requests are synthetic Zipf(0.8), with 100,000 historical samples and additive-one count smoothing; real workload gains remain untested. Each seed couples one training realization with one workload realization. Cache ranking receives predictions and historical frequency; only benefit calibration receives validation correctness. Test correctness is used only for reporting. Uniform and shifted-popularity results, all budgets, actual memory, and per-seed values are retained rather than selecting a best test setting.', '',
              'The earlier CUDA reproduction used a forward/gradient-verified message-sum adapter because native DGL CUDA was unavailable. This extension uses the released MLP on CUDA and sparse equivalents of the graph objectives. Six unit tests check source-loss values, gradients and optimizer updates, sparse incidence equivalence, routing, and benefit constraints. Independent artifact checks verify all 690 unique rows, all 15 baseline metrics, source hashes, cache capacities/bytes, disjoint validation/test masks, finite metrics and equal 150-step refinement traces.', '',
              '- `REPORT.md`: full protocol and mean ± standard deviation headline tables.',
              '- `metrics.csv` / `metrics.json`: every setting, latency, unfairness, memory and routing agreement.',
              '- `paired_comparisons.json`: all budgets and five paired deltas per comparison.',
              '- `independent_verification.json`: independent completed-artifact checks.',
              '- Per-seed folders: checkpoint hashes, trained checkpoints, calibration coefficients and refinement traces.', '']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines))
    print(json.dumps(verification, indent=2))
    print('\n'.join(lines[:24]))


if __name__ == '__main__':
    main()
