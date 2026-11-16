# Cache extension: completed A100 findings

Static caching gives dataset-dependent accuracy gains; this joint refinement does not establish a consistent advantage. Hard teacher-logit replacement increases graph unfairness. The cache adds latency in this GPU microbenchmark.

All 15 saved full-FAITH dataset/seed runs and four cache budgets (1%, 5%, 10%, 20%) were evaluated: 690 rows. The original checkpoints are unchanged. The unstable CiteSeer seed 45 is retained. These are extensions of our local reproduction, which did not reproduce the paper’s stability; they are not a published-baseline or state-of-the-art claim.

## Frozen student, 10% cache upper bound

Mean request-weighted accuracy (%), with paired changes in percentage points. Each policy uses the same five student/workload seeds. Uniform accuracy is separate from synthetic request-weighted accuracy.

| Dataset | No cache | Frequency | Change | Disagreement | Change | Benefit | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 83.33 | 84.38 | +1.05 | 84.28 | +0.95 | 83.19 | -0.14 |
| citeseer | 59.94 | 60.47 | +0.53 | 59.92 | -0.02 | 63.79 | +3.85 |
| pubmed | 84.25 | 83.33 | -0.92 | 82.08 | -2.17 | 84.17 | -0.08 |

Benefit calibration uses the same 500 validation labels previously used for checkpoint selection; this result is exploratory. It may leave capacity empty. The three policies were fixed before evaluation; highlighting their dataset-specific outcomes does not define a deployable policy-selection rule.

## Does joint training add value?

Both continuation and joint refinement start from the same saved student and use 150 additional updates, matched optimizer resets and dropout seeds. Joint refinement alternates frequency-times-KL cache selection with downweighting cached nodes to 0.1 in CE/KL. Full FAITH energy and OSA losses remain intact. Final updates are evaluated without test-driven checkpoint selection.

| Dataset | Joint − continued disagreement | Joint − continued using identical cache | Joint benefit − continued benefit |
|---|---:|---:|---:|
| cora | +0.07 ± 0.36 (3/5 wins) | -0.06 ± 0.26 (2/5 wins) | -0.12 ± 2.41 (1/5 wins) |
| citeseer | +0.43 ± 0.90 (3/5 wins) | +0.43 ± 0.77 (3/5 wins) | +0.68 ± 1.00 (4/5 wins) |
| pubmed | -0.27 ± 0.14 (0/5 wins) | -0.11 ± 0.18 (1/5 wins) | +0.00 ± 0.37 (3/5 wins) |

Values are paired request-weighted percentage-point differences, mean ± population standard deviation. Same-cache comparison isolates changing the student while holding routing decisions fixed. The improvements are not consistent across datasets; no statistical-significance claim is made.

## Cost and fairness

At 10% frequency caching, means across the five seeds:

| Dataset | Uniform accuracy, base → cache | Batch latency ms, base → cache | Raw-logit unfairness, base → cache | Probability unfairness, base → cache | Payload + lookup bytes |
|---|---:|---:|---:|---:|---:|
| cora | 83.100 → 83.160 | 0.585 → 0.745 | 11.019 → 91.400 | 0.593 → 0.994 | 7,560 + 21,664 |
| citeseer | 63.640 → 64.480 | 0.662 → 0.851 | 2.704 → 5.616 | 0.117 → 0.164 | 7,968 + 26,616 |
| pubmed | 81.760 → 81.560 | 0.270 → 0.723 | 48.487 → 1016.641 | 6.073 → 10.438 | 23,652 + 157,736 |

Lower unfairness is better. Raw-logit unfairness follows the release and is divided by 1,000; teacher/student logit scale differences affect it. Probability unfairness also rises, so the effect is not solely a logit-scale artifact. The failed low-accuracy student can have deceptively low unfairness.

Latency is a local GPU-resident batch of 1,024 test queries: 10 warmups, median of 40 synchronized calls. Lookup, output assembly and miss-only MLP inference are included. Host transfer, network and offline cache construction are excluded. Timings are subject to shared-GPU and ordering variability; these measurements do not demonstrate a serving speedup. Dense index bytes are outside the nominal node-payload budget. No-cache entries allocate neither index nor payload.

## Interpretation and audit

The simplest positive result is frequency caching on Cora (+1.05 weighted points, +0.06 uniform points). CiteSeer benefit caching gains +3.85 weighted points and +1.34 uniform points, but includes the unstable base seed and reused validation calibration. PubMed does not improve at the headline 10% budget. Joint refinement is not supported as an improvement over simple independently trained cache baselines by this trial.

Requests are synthetic Zipf(0.8), with 100,000 historical samples and additive-one count smoothing; real workload gains remain untested. Each seed couples one training realization with one workload realization. Cache ranking receives predictions and historical frequency; only benefit calibration receives validation correctness. Test correctness is used only for reporting. Uniform and shifted-popularity results, all budgets, actual memory, and per-seed values are retained rather than selecting a best test setting.

The earlier CUDA reproduction used a forward/gradient-verified message-sum adapter because native DGL CUDA was unavailable. This extension uses the released MLP on CUDA and sparse equivalents of the graph objectives. Six unit tests check source-loss values, gradients and optimizer updates, sparse incidence equivalence, routing, and benefit constraints. Independent artifact checks verify all 690 unique rows, all 15 baseline metrics, source hashes, cache capacities/bytes, disjoint validation/test masks, finite metrics and equal 150-step refinement traces.

- `REPORT.md`: full protocol and mean ± standard deviation headline tables.
- `metrics.csv` / `metrics.json`: every setting, latency, unfairness, memory and routing agreement.
- `paired_comparisons.json`: all budgets and five paired deltas per comparison.
- `independent_verification.json`: independent completed-artifact checks.
- Per-seed folders: checkpoint hashes, trained checkpoints, calibration coefficients and refinement traces.
