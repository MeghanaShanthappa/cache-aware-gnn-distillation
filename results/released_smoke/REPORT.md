# Released FAITH reproduction attempt

Uniform node classification accuracy (%), with the full original validation mask and no serving cache. Five seeds use the released cumulative schedule: 42, 43, 45, 48, 52. Population standard deviation matches the released aggregation code. These CPU runs retain the released DGL models and training loops, with recorded compatibility and static-tensor caching repairs.

| Dataset | Method | Published main table | Reproduction mean ± std | Difference pp | Runs |
|---|---|---:|---:|---:|---:|
| cora | SAGE | 82.18 ± 0.95 | 51.50 ± 0.00 | -30.68 | 1 |
| cora | FAITH | 83.96 ± 0.47 | 14.90 ± 0.00 | -69.06 | 1 |
| cora | KL_ablation | Not separately reported | 14.90 ± 0.00 | — | 1 |

## Scope and unresolved comparability

FAITH means the released Dirichlet energy and Oracle Similarity Alignment losses together, using beta/gamma from the README. FAITH-star counterfactual group fairness is a separate experiment and is excluded from this citation-graph comparison.

KL_ablation disables both extra FAITH losses but retains FAITH-tuned architectures and optimizer settings. It must not be labeled a reproduction of the GLNN row. The FAITH repository does not release its GLNN baseline training configuration or checkpoints. The original GLNN repository uses different citation-graph preprocessing and splits; its original-paper results are not the GLNN rerun in the FAITH table.

The main-table published GLNN targets are Cora 81.50 ± 1.21, CiteSeer 71.46 ± 1.79, PubMed 77.97 ± 3.35. The included FAITH appendix instead lists Cora GLNN std 2.23, CiteSeer std 3.16, PubMed mean 77.90; it also lists PubMed FAITH 81.72 rather than the main-table 81.53. These discrepancies are preserved rather than resolved by choosing a closer number.

The unavailable DGL CUDA wheel endpoint required CPU DGL 2.1.0 and PyTorch 2.1.2 rather than the README versions 0.6.1 and 1.6.0. NumPy 1.21.6, SciPy 1.7.3, and scikit-learn 1.0.2 match the release. Hardware/runtime differences prevent bitwise reproduction. The original Planetoid raw bytes were passed through DGL’s own citation loader; their hashes and filename-only Cora prefix change are recorded in environment.json.

Runtime repairs and their source text are in runtime_repairs.json. A complete original-vs-cached training step checks loss, gradients, and updated model parameters; the cached fairness diagnostic is checked against its released implementation. Checkpoint selection uses validation accuracy only; printed test-best fields from the source are not used as the reported result.

Sources: [FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705), [released FAITH code](https://github.com/shashivipul/FAITH), [original GLNN code](https://github.com/snap-research/graphless-neural-networks).
