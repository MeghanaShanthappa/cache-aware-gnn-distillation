# FAITH-based cache study

Completed: five datasets, five seeds each, four cache budgets (1%, 5%, 10%, 20%), and 1,150 evaluation rows. The citation reproduction and scalable OGB adaptation are separate protocols; dataset means are not pooled. All seeds are retained.

## Request-weighted accuracy at 10% cache (%)

| Dataset | Protocol | No cache | Frequency | Disagreement | Benefit |
|---|---|---:|---:|---:|---:|
| cora | Released citation configuration | 83.33 ± 2.64 | 84.38 ± 4.39 | 84.28 ± 4.13 | 83.19 ± 3.29 |
| citeseer | Released citation configuration | 59.94 ± 12.77 | 60.47 ± 6.77 | 59.92 ± 7.35 | 63.79 ± 7.43 |
| pubmed | Released citation configuration | 84.25 ± 1.58 | 83.33 ± 1.87 | 82.08 ± 2.09 | 84.17 ± 1.53 |
| ogbn-arxiv | Scalable adaptation | 65.05 ± 3.06 | 69.37 ± 2.58 | 70.39 ± 2.57 | 71.52 ± 2.42 |
| ogbn-products | Scalable adaptation | 68.73 ± 0.67 | 74.04 ± 0.41 | 75.11 ± 0.41 | 76.20 ± 0.39 |

Numbers use synthetic Zipf workloads and mean ± population standard deviation. Cache capacity counts all graph nodes; accuracy uses test requests. Benefit calibration reuses checkpoint-selection validation labels for citation datasets, while OGB reserves a separate validation half. Policy outcomes are descriptive; no best-per-dataset test policy is presented as a deployable selection rule.

Citation models use the original FAITH architecture/settings with a verified CUDA message-sum adapter, retaining the unstable CiteSeer seed. OGB uses width 256, binary graph preprocessing, sparse exact energy and sampled OSA. A positive OSA coefficient is selected using model-validation accuracy only, then locked for all seeds. The failed direct-transfer trial is retained; its arxiv test numbers were observed before this exploratory follow-up. These OGB runs are not a reproduction of the paper’s published result. Teacher/student gaps are reported in the OGB findings.

Joint-refinement comparisons use equal 150-update controls and an identical-cache control, reported separately in each findings file. Uniform accuracy, latency, fairness, all budgets and individual seeds remain available. Cache gains alone do not establish better training, faster serving or preserved fairness.

- [Citation findings](results/faith_cache/FINDINGS.md)
- [OGB findings](results/faith_ogb/FINDINGS.md)
- [OGB full protocol and results](results/faith_ogb/REPORT.md)
- [OGB exact full-graph fairness](results/faith_ogb/EXACT_FAIRNESS.md)
- [OGB independent verification](results/faith_ogb/independent_verification.json)
