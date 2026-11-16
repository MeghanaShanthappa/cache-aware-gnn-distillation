# FAITH structural distillation and static cache pilot

This is a structural-only adaptation of the released FAITH code, not a reproduction of full FAITH. OSA and counterfactual fairness penalties are excluded. Results use an A100, synthetic Zipf queries, and the public Planetoid split with its validation set divided into model selection and cache calibration halves.

## Request weighted accuracy at 10 percent node cache

Percent, mean ± sample standard deviation over training and workload seeds. No cache is the corresponding student.

| Dataset | Student | No cache | Frequency | Disagreement | Predicted benefit |
|---|---|---:|---:|---:|---:|
| CiteSeer | kd | 71.89 ± 3.56 | 70.72 ± 3.88 | 70.90 ± 3.84 | 72.46 ± 2.63 |
| CiteSeer | structural | 71.72 ± 2.79 | 70.50 ± 3.83 | 70.31 ± 3.01 | 71.95 ± 3.55 |
| Cora | kd | 86.07 ± 2.14 | 84.96 ± 2.50 | 84.51 ± 2.38 | 85.25 ± 2.19 |
| Cora | structural | 85.22 ± 2.22 | 85.07 ± 2.73 | 84.49 ± 2.70 | 84.84 ± 2.48 |
| PubMed | kd | 78.88 ± 3.35 | 77.27 ± 4.86 | 76.29 ± 4.84 | 79.58 ± 5.24 |
| PubMed | structural | 78.00 ± 4.41 | 76.74 ± 4.70 | 75.82 ± 4.47 | 78.23 ± 4.54 |

## Accuracy and latency across every setting

| Dataset | Student | Cache | Policy | Uniform accuracy % | Weighted accuracy % | Batch latency ms | Seeds |
|---|---|---:|---|---:|---:|---:|---:|
| CiteSeer | kd | 0% | none | 71.72 | 71.89 | 0.635 | 5 |
| CiteSeer | kd | 1% | benefit | 71.86 | 72.36 | 0.922 | 5 |
| CiteSeer | kd | 1% | disagreement | 71.68 | 70.89 | 0.841 | 5 |
| CiteSeer | kd | 1% | frequency | 71.70 | 70.87 | 0.855 | 5 |
| CiteSeer | kd | 5% | benefit | 71.96 | 72.49 | 0.917 | 5 |
| CiteSeer | kd | 5% | disagreement | 71.72 | 70.93 | 0.774 | 5 |
| CiteSeer | kd | 5% | frequency | 71.70 | 70.70 | 0.731 | 5 |
| CiteSeer | kd | 10% | benefit | 71.94 | 72.46 | 0.928 | 5 |
| CiteSeer | kd | 10% | disagreement | 71.70 | 70.90 | 0.743 | 5 |
| CiteSeer | kd | 10% | frequency | 71.70 | 70.72 | 0.665 | 5 |
| CiteSeer | kd | 20% | benefit | 71.94 | 72.46 | 0.926 | 5 |
| CiteSeer | kd | 20% | disagreement | 71.36 | 70.34 | 0.671 | 5 |
| CiteSeer | kd | 20% | frequency | 71.42 | 70.37 | 0.585 | 5 |
| CiteSeer | structural | 0% | none | 71.82 | 71.72 | 0.655 | 5 |
| CiteSeer | structural | 1% | benefit | 71.78 | 71.49 | 1.024 | 5 |
| CiteSeer | structural | 1% | disagreement | 71.78 | 70.95 | 0.943 | 5 |
| CiteSeer | structural | 1% | frequency | 71.80 | 70.81 | 0.921 | 5 |
| CiteSeer | structural | 5% | benefit | 72.00 | 71.73 | 1.022 | 5 |
| CiteSeer | structural | 5% | disagreement | 71.78 | 70.71 | 0.875 | 5 |
| CiteSeer | structural | 5% | frequency | 71.72 | 70.78 | 0.834 | 5 |
| CiteSeer | structural | 10% | benefit | 71.94 | 71.95 | 1.022 | 5 |
| CiteSeer | structural | 10% | disagreement | 71.54 | 70.31 | 0.831 | 5 |
| CiteSeer | structural | 10% | frequency | 71.58 | 70.50 | 0.772 | 5 |
| CiteSeer | structural | 20% | benefit | 72.10 | 72.04 | 1.008 | 5 |
| CiteSeer | structural | 20% | disagreement | 71.48 | 70.32 | 0.773 | 5 |
| CiteSeer | structural | 20% | frequency | 71.46 | 70.33 | 0.683 | 5 |
| Cora | kd | 0% | none | 83.18 | 86.07 | 0.499 | 5 |
| Cora | kd | 1% | benefit | 83.16 | 85.33 | 0.797 | 5 |
| Cora | kd | 1% | disagreement | 83.08 | 84.68 | 0.802 | 5 |
| Cora | kd | 1% | frequency | 83.14 | 85.05 | 0.779 | 5 |
| Cora | kd | 5% | benefit | 83.02 | 85.22 | 0.789 | 5 |
| Cora | kd | 5% | disagreement | 83.02 | 84.67 | 0.750 | 5 |
| Cora | kd | 5% | frequency | 83.14 | 84.85 | 0.729 | 5 |
| Cora | kd | 10% | benefit | 83.08 | 85.25 | 0.788 | 5 |
| Cora | kd | 10% | disagreement | 82.74 | 84.51 | 0.731 | 5 |
| Cora | kd | 10% | frequency | 83.20 | 84.96 | 0.688 | 5 |
| Cora | kd | 20% | benefit | 83.08 | 85.25 | 0.791 | 5 |
| Cora | kd | 20% | disagreement | 82.26 | 84.30 | 0.694 | 5 |
| Cora | kd | 20% | frequency | 83.00 | 84.68 | 0.654 | 5 |
| Cora | structural | 0% | none | 83.54 | 85.22 | 0.553 | 5 |
| Cora | structural | 1% | benefit | 83.36 | 84.89 | 0.882 | 5 |
| Cora | structural | 1% | disagreement | 83.48 | 84.78 | 0.842 | 5 |
| Cora | structural | 1% | frequency | 83.52 | 85.03 | 0.813 | 5 |
| Cora | structural | 5% | benefit | 83.20 | 84.84 | 0.865 | 5 |
| Cora | structural | 5% | disagreement | 83.34 | 84.70 | 0.749 | 5 |
| Cora | structural | 5% | frequency | 83.58 | 85.15 | 0.733 | 5 |
| Cora | structural | 10% | benefit | 83.20 | 84.84 | 0.864 | 5 |
| Cora | structural | 10% | disagreement | 83.00 | 84.49 | 0.733 | 5 |
| Cora | structural | 10% | frequency | 83.54 | 85.07 | 0.688 | 5 |
| Cora | structural | 20% | benefit | 83.20 | 84.84 | 0.864 | 5 |
| Cora | structural | 20% | disagreement | 82.78 | 84.45 | 0.696 | 5 |
| Cora | structural | 20% | frequency | 83.36 | 84.83 | 0.651 | 5 |
| PubMed | kd | 0% | none | 79.90 | 78.88 | 0.334 | 5 |
| PubMed | kd | 1% | benefit | 79.84 | 79.53 | 0.553 | 5 |
| PubMed | kd | 1% | disagreement | 79.56 | 78.16 | 0.636 | 5 |
| PubMed | kd | 1% | frequency | 79.88 | 79.36 | 0.659 | 5 |
| PubMed | kd | 5% | benefit | 79.82 | 79.59 | 0.554 | 5 |
| PubMed | kd | 5% | disagreement | 78.24 | 76.92 | 0.632 | 5 |
| PubMed | kd | 5% | frequency | 79.60 | 77.86 | 0.632 | 5 |
| PubMed | kd | 10% | benefit | 79.80 | 79.58 | 0.552 | 5 |
| PubMed | kd | 10% | disagreement | 77.30 | 76.29 | 0.632 | 5 |
| PubMed | kd | 10% | frequency | 79.32 | 77.27 | 0.632 | 5 |
| PubMed | kd | 20% | benefit | 79.80 | 79.58 | 0.552 | 5 |
| PubMed | kd | 20% | disagreement | 76.84 | 75.88 | 0.632 | 5 |
| PubMed | kd | 20% | frequency | 79.26 | 77.24 | 0.632 | 5 |
| PubMed | structural | 0% | none | 79.14 | 78.00 | 0.372 | 5 |
| PubMed | structural | 1% | benefit | 79.34 | 78.37 | 0.557 | 5 |
| PubMed | structural | 1% | disagreement | 78.86 | 77.66 | 0.639 | 5 |
| PubMed | structural | 1% | frequency | 79.12 | 78.68 | 0.689 | 5 |
| PubMed | structural | 5% | benefit | 79.32 | 78.23 | 0.556 | 5 |
| PubMed | structural | 5% | disagreement | 77.62 | 76.45 | 0.634 | 5 |
| PubMed | structural | 5% | frequency | 78.86 | 77.27 | 0.635 | 5 |
| PubMed | structural | 10% | benefit | 79.32 | 78.23 | 0.555 | 5 |
| PubMed | structural | 10% | disagreement | 76.78 | 75.82 | 0.636 | 5 |
| PubMed | structural | 10% | frequency | 78.62 | 76.74 | 0.635 | 5 |
| PubMed | structural | 20% | benefit | 79.32 | 78.23 | 0.555 | 5 |
| PubMed | structural | 20% | disagreement | 76.52 | 75.79 | 0.634 | 5 |
| PubMed | structural | 20% | frequency | 78.72 | 76.93 | 0.635 | 5 |

## Interpretation limits

Teacher and student architectures, optimizer settings, temperature, and supervised mixing coefficient come from the released transductive SAGE configuration. The structural coefficient is 0.02 from the authors’ README. The sparse structural objective is checked against the released function in both value and gradient. Implementation and protocol differences are documented in README.md.

Validation labels are split with fixed seed 1729: 250 nodes for checkpoint selection and 250 for benefit estimation. Benefit estimation uses a fixed ridge coefficient of 1, with no test-label tuning. Benefit caching may leave slots empty when predicted improvement is nonpositive; all caches obey the same node-logit upper bound. Payload, dense index, and model bytes are in metrics.csv. Cache lookup timing includes miss-only MLP inference and excludes host transfer, network, and refresh.

All students have the same maximum 500-step budget and validation stopping rule; actual epochs differ and are recorded. Seeds change model/workload randomness, not the graph split. Five-seed comparisons are descriptive, with no significance claim. Earlier A40 results are not directly comparable because these runs use different architectures, training schedules, validation allocation, and hardware.
