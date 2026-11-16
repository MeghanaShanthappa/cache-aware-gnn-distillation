# FAITH structural distillation and static cache pilot

This is a structural-only adaptation of the released FAITH code, not a reproduction of full FAITH. OSA and counterfactual fairness penalties are excluded. Results use an A100, synthetic Zipf queries, and the public Planetoid split with its validation set divided into model selection and cache calibration halves.

## Request weighted accuracy at 10 percent node cache

Percent, mean ± sample standard deviation over training and workload seeds. No cache is the corresponding student.

| Dataset | Student | No cache | Frequency | Disagreement | Predicted benefit |
|---|---|---:|---:|---:|---:|
| Cora | kd | 30.90 ± 0.00 | 50.70 ± 0.00 | 50.72 ± 0.00 | 53.60 ± 0.00 |
| Cora | structural | 30.90 ± 0.00 | 50.70 ± 0.00 | 51.17 ± 0.00 | 52.52 ± 0.00 |

## Accuracy and latency across every setting

| Dataset | Student | Cache | Policy | Uniform accuracy % | Weighted accuracy % | Batch latency ms | Seeds |
|---|---|---:|---|---:|---:|---:|---:|
| Cora | kd | 0% | none | 31.90 | 30.90 | 0.821 | 1 |
| Cora | kd | 1% | benefit | 32.50 | 37.78 | 0.995 | 1 |
| Cora | kd | 1% | disagreement | 32.60 | 35.75 | 0.999 | 1 |
| Cora | kd | 1% | frequency | 32.20 | 35.06 | 1.062 | 1 |
| Cora | kd | 5% | benefit | 35.60 | 48.58 | 0.647 | 1 |
| Cora | kd | 5% | disagreement | 35.50 | 46.95 | 0.863 | 1 |
| Cora | kd | 5% | frequency | 33.60 | 43.94 | 0.872 | 1 |
| Cora | kd | 10% | benefit | 39.20 | 53.60 | 0.645 | 1 |
| Cora | kd | 10% | disagreement | 38.70 | 50.72 | 0.648 | 1 |
| Cora | kd | 10% | frequency | 36.40 | 50.70 | 0.608 | 1 |
| Cora | kd | 20% | benefit | 47.10 | 60.67 | 0.618 | 1 |
| Cora | kd | 20% | disagreement | 45.20 | 55.88 | 0.608 | 1 |
| Cora | kd | 20% | frequency | 41.30 | 57.15 | 0.572 | 1 |
| Cora | structural | 0% | none | 31.90 | 30.90 | 0.478 | 1 |
| Cora | structural | 1% | benefit | 32.60 | 38.63 | 0.736 | 1 |
| Cora | structural | 1% | disagreement | 32.80 | 36.31 | 0.735 | 1 |
| Cora | structural | 1% | frequency | 32.20 | 35.06 | 0.733 | 1 |
| Cora | structural | 5% | benefit | 35.70 | 48.43 | 0.657 | 1 |
| Cora | structural | 5% | disagreement | 35.50 | 46.49 | 0.658 | 1 |
| Cora | structural | 5% | frequency | 33.60 | 43.94 | 0.660 | 1 |
| Cora | structural | 10% | benefit | 38.70 | 52.52 | 0.657 | 1 |
| Cora | structural | 10% | disagreement | 38.90 | 51.17 | 0.665 | 1 |
| Cora | structural | 10% | frequency | 36.40 | 50.70 | 0.621 | 1 |
| Cora | structural | 20% | benefit | 46.80 | 59.75 | 0.625 | 1 |
| Cora | structural | 20% | disagreement | 45.90 | 58.04 | 0.622 | 1 |
| Cora | structural | 20% | frequency | 41.30 | 57.15 | 0.577 | 1 |

## Interpretation limits

Teacher and student architectures, optimizer settings, temperature, and supervised mixing coefficient come from the released transductive SAGE configuration. The structural coefficient is 0.02 from the authors’ README. The sparse structural objective is checked against the released function in both value and gradient. Implementation and protocol differences are documented in README.md.

Validation labels are split with fixed seed 1729: 250 nodes for checkpoint selection and 250 for benefit estimation. Benefit estimation uses a fixed ridge coefficient of 1, with no test-label tuning. Benefit caching may leave slots empty when predicted improvement is nonpositive; all caches obey the same node-logit upper bound. Payload, dense index, and model bytes are in metrics.csv. Cache lookup timing includes miss-only MLP inference and excludes host transfer, network, and refresh.

All students have the same maximum 500-step budget and validation stopping rule; actual epochs differ and are recorded. Seeds change model/workload randomness, not the graph split. Five-seed comparisons are descriptive, with no significance claim. Earlier A40 results are not directly comparable because these runs use different architectures, training schedules, validation allocation, and hardware.
