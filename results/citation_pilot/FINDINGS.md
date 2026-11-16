# Findings from the FAITH structural loss pilot

These results isolate the structural loss from the released FAITH implementation. They do not evaluate full FAITH or its fairness objectives. All models and queries ran on the A100.

## Request weighted accuracy at a 10 percent cache upper bound

Percent, mean ± sample standard deviation across five seeds. Benefit caches may use less than the allowed capacity when predicted improvement is nonpositive.

| Dataset | KD without cache | Structural without cache | KD with benefit cache | Structural with benefit cache |
|---|---:|---:|---:|---:|
| Cora | 86.07 ± 2.14 | 85.22 ± 2.22 | 85.25 ± 2.19 | 84.84 ± 2.48 |
| CiteSeer | 71.89 ± 3.56 | 71.72 ± 2.79 | 72.46 ± 2.63 | 71.95 ± 3.55 |
| PubMed | 78.88 ± 3.35 | 78.00 ± 4.41 | 79.58 ± 5.24 | 78.23 ± 4.54 |

The structural student has lower mean request-weighted accuracy on all three datasets. Uniform accuracy changes differ: it rises from 83.18% to 83.54% on Cora and 71.72% to 71.82% on CiteSeer, while falling from 79.90% to 79.14% on PubMed. Structural loss therefore has no consistent benefit across the measured quality metrics.

The benefit cache adds 0.57 percentage points on CiteSeer and 0.70 points on PubMed to the ordinary KD student, but loses 0.82 points on Cora. These are descriptive seed means; gains are not established as statistically significant. At 10%, frequency caching reduces request-weighted accuracy for both students on all three datasets.

Cached inference remains slower in these local query measurements. For the ordinary KD student, mean latency changes from 0.499 to 0.788 ms on Cora, 0.635 to 0.928 ms on CiteSeer, and 0.334 to 0.552 ms on PubMed when using the benefit cache. Each batch contains 1,024 GPU-resident requests.

## Decision

Keep this implementation and its negative results as a structural-loss ablation and cache-policy pilot. It is insufficient evidence to adopt structural loss as the default student-training method or to claim a serving speedup. Before scaling this configuration to OGB, reproduce full FAITH with its omitted OSA objective or compare another stronger distillation baseline, and validate benefit calibration on independent splits.

The exact training settings, source revision, calibration protocol, limitations, per-seed results, and every cache budget are documented in README.md and results/citation_pilot/REPORT.md. verification.json records the completed consistency and scientific-correctness checks.
