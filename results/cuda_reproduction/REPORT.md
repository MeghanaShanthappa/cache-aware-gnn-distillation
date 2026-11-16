# FAITH CUDA reproduction with a message sum compatibility adapter

Original released DGL SAGEConv models and teacher/student training loops. The unavailable CUDA DGL graph copy_u/sum primitive is replaced by mathematically equivalent torch.sparse.mm on CUDA, with directed-edge, duplicate-edge, isolated-node, forward, and gradient checks against native DGL. Teacher and student matrix operations, dropout, and optimizer updates run on the A100. CPU parameter initialization and CUDA dropout use the separate streams of the source GPU execution.

| Dataset | Method | Published main table % | Reproduction % | Difference pp | Seeds |
|---|---|---:|---:|---:|---:|
| citeseer | SAGE | 70.86 ± 0.35 | 71.32 ± 0.92 | +0.46 | 5 |
| citeseer | FAITH | 74.10 ± 0.60 | 63.64 ± 18.94 | -10.46 | 5 |
| citeseer | KL_ablation | Not separately reported | 64.30 ± 16.55 | — | 5 |
| cora | SAGE | 82.18 ± 0.95 | 81.64 ± 0.76 | -0.54 | 5 |
| cora | FAITH | 83.96 ± 0.47 | 83.10 ± 0.68 | -0.86 | 5 |
| cora | KL_ablation | Not separately reported | 82.20 ± 1.22 | — | 5 |
| pubmed | SAGE | 77.38 ± 0.81 | 76.94 ± 0.55 | -0.44 | 5 |
| pubmed | FAITH | 81.53 ± 0.33 | 81.76 ± 0.53 | +0.23 | 5 |
| pubmed | KL_ablation | Not separately reported | 80.38 ± 0.12 | — | 5 |

Mean ± population standard deviation over the five released cumulative seeds, with full 500-node validation masks. All FAITH coefficients and model settings come from the release. No hyperparameters, seeds, or checkpoints are selected by test performance. KL_ablation is not the published GLNN rerun.

This is a documented mathematical reproduction with a compatibility adapter, not the original binary environment. PyTorch 2.1.2 and DGL 2.1.0 differ from the README versions, and sparse reduction order can differ from native DGL CUDA. Numerical agreement and disagreement are reported explicitly. No inference-speed claim is made.

CPU-only and hybrid fallback runs are retained in ../released_reproduction/ and ../hybrid_reproduction/. They are runtime diagnostics, not alternate numbers to select per dataset. The CUDA primitive runtime was chosen for closer execution semantics: the teacher linear layers, Adam, and dropout all execute on GPU, with only a tested sum primitive substituted.

The FAITH release omits its GLNN baseline rerun configuration and checkpoints. Original GLNN CPF preprocessing differs from the public citation loader; the cross-protocol run in ../glnn_source_reference/ is diagnostic only. A numerical reproduction of the published GLNN row remains unverified.

[FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705) · [Released code](https://github.com/shashivipul/FAITH)
