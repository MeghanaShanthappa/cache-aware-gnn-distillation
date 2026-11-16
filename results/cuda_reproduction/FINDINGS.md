# FAITH reproduction findings

This rerun uses the released full FAITH models and training loops on Cora, CiteSeer, and PubMed. It restores the full validation masks, original configuration files, initialization order, and cumulative five-seed schedule. It includes both energy alignment and Oracle Similarity Alignment; the earlier cache pilot did not.

Teacher and student matrix operations, dropout, and Adam updates run on the A100. The original DGL SAGEConv class executes against a narrow compatibility adapter that implements its homogeneous copy_u/sum graph primitive with torch.sparse.mm on CUDA. Forward outputs and parameter/input gradients match native DGL on directed graphs, duplicate edges, isolated nodes, and both linear-before/after-aggregation cases. The original teacher training loop is unchanged. CPU initialization and CUDA dropout retain separate random streams, verified during each seed.

## Uniform test accuracy

Percent, mean ± population standard deviation, matching the release’s aggregation convention. Difference is reproduction minus published main-table mean, in percentage points.

| Dataset | Paper FAITH | Reproduction | Difference |
|---|---:|---:|---:|
| cora | 83.96 ± 0.47 | 83.10 ± 0.68 | -0.86 |
| citeseer | 74.10 ± 0.60 | 63.64 ± 18.94 | -10.46 |
| pubmed | 81.53 ± 0.33 | 81.76 ± 0.53 | +0.23 |

## Individual unfairness

Lower is better. Values use the released diagnostic divided by 1,000, matching these citation-graph table entries. These are distinct from group-fairness results for FAITH-star.

| Dataset | Paper FAITH | Reproduction |
|---|---:|---:|
| cora | 12.06 ± 1.30 | 11.02 ± 0.93 |
| citeseer | 1.78 ± 1.57 | 2.70 ± 2.33 |
| pubmed | 47.96 ± 3.38 | 48.49 ± 1.95 |

## Per seed stability

| Dataset | Seed | FAITH accuracy % | KL ablation accuracy % |
|---|---:|---:|---:|
| citeseer | 42 | 74.40 | 74.30 |
| citeseer | 43 | 73.20 | 73.10 |
| citeseer | 45 | 25.80 | 31.30 |
| citeseer | 48 | 73.40 | 70.20 |
| citeseer | 52 | 71.40 | 72.60 |
| cora | 42 | 83.10 | 80.50 |
| cora | 43 | 84.30 | 83.50 |
| cora | 45 | 82.20 | 83.70 |
| cora | 48 | 82.80 | 81.70 |
| cora | 52 | 83.10 | 81.60 |
| pubmed | 42 | 82.80 | 80.30 |
| pubmed | 43 | 81.60 | 80.60 |
| pubmed | 45 | 81.40 | 80.30 |
| pubmed | 48 | 81.60 | 80.30 |
| pubmed | 52 | 81.40 | 80.40 |

Every seed is retained in the reported mean and standard deviation. Severe low-accuracy runs are not removed. Individual unfairness must be interpreted together with predictive accuracy: nearly constant outputs can have low unfairness without being useful classifiers.

The CUDA CiteSeer seed-45 checkpoint predicts only 2 of 6 classes on its 1000 test nodes. Its test accuracy is 25.8%. The corresponding teacher reaches 72.7%, and the KL ablation also performs poorly. The full-graph native-DGL operator check has 100% teacher argmax agreement; it found no large operator error explaining this classifier failure. The underlying optimization failure is not isolated.

## Retained CPU diagnostic

| Dataset | CPU source rerun accuracy % | Primary CUDA accuracy % |
|---|---:|---:|
| cora | 84.12 ± 1.01 | 83.10 ± 0.68 |
| citeseer | 66.00 ± 15.95 | 63.64 ± 18.94 |
| pubmed | 81.48 ± 0.67 | 81.76 ± 0.53 |

All five seeds are retained in both runtimes. The failing CiteSeer seed changes between CPU-only and CUDA execution. These are runtime diagnostics; the best value is not selected separately for each dataset. The CUDA run is primary because it more closely preserves GPU model arithmetic and random-stream separation.

## What this establishes

The tables quantify numerical agreement rather than assume that executing the authors’ code guarantees a reproduced paper result. Differences in means and seed variability remain visible. Runs use DGL 2.1.0 model classes and PyTorch 2.1.2 with a CUDA message-sum compatibility adapter because the DGL CUDA wheel server was unavailable; they preserve the released NumPy 1.21.6, SciPy 1.7.3, and scikit-learn 1.0.2. The README’s original DGL/PyTorch environment was not recreated exactly. These results are not bitwise replications and contain no inference-speed claim.

Runtime repairs include missing definitions, device placement, reuse of graph-static tensors and a diagnostic Laplacian, and the separately validated CUDA message-sum adapter. An original-versus-cached optimizer-step check covers losses, gradients, and updated weights. Source hashes, all 500 validation nodes, seed coverage, configurations, and coefficients pass independent checks.

## Ordinary distillation remains unresolved

The KL-only loss ablation in REPORT.md retains FAITH’s architecture and tuning; it is not the published GLNN row. The FAITH release does not supply that baseline’s rerun configuration or checkpoints. An original GLNN configuration applied to FAITH’s differently processed data was preserved as a diagnostic in ../glnn_source_reference/STATUS.md, not accepted as a reproduced baseline. The original GLNN CPF data protocol, feature scale, model configuration, soft-target training set, and training schedule differ. A verified reproduction of the FAITH paper’s GLNN row therefore still requires the corresponding rerun configuration.

The included appendix disagrees with the main table on several values, including PubMed FAITH accuracy (81.72 versus 81.53) and GLNN means or standard deviations. Main-table targets are used consistently above; REPORT.md records these discrepancies.

Keep cache-policy development separate until the selected base method and its baseline protocol are established. These source reruns supersede the earlier structural-only pilot for assessing full FAITH accuracy, but do not establish an improvement over a verified GLNN baseline.

Sources: [FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705), [FAITH source](https://github.com/shashivipul/FAITH), [GLNN source](https://github.com/snap-research/graphless-neural-networks). Detailed source repairs, checkpoints, per-epoch logs, package lock, and comparison data are stored with this report.
