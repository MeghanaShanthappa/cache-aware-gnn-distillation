# FAITH reproduction findings

This rerun uses the released full FAITH models and training loops on Cora, CiteSeer, and PubMed. It restores the full validation masks, original configuration files, initialization order, and cumulative five-seed schedule. It includes both energy alignment and Oracle Similarity Alignment; the earlier cache pilot did not.

## Uniform test accuracy

Percent, mean ± population standard deviation, matching the release’s aggregation convention. Difference is reproduction minus published main-table mean, in percentage points.

| Dataset | Paper FAITH | Reproduction | Difference |
|---|---:|---:|---:|
| cora | 83.96 ± 0.47 | 84.12 ± 1.01 | +0.16 |
| citeseer | 74.10 ± 0.60 | 66.00 ± 15.95 | -8.10 |
| pubmed | 81.53 ± 0.33 | 81.48 ± 0.67 | -0.05 |

## Individual unfairness

Lower is better. Values use the released diagnostic divided by 1,000, matching these citation-graph table entries. These are distinct from group-fairness results for FAITH-star.

| Dataset | Paper FAITH | Reproduction |
|---|---:|---:|
| cora | 12.06 ± 1.30 | 11.34 ± 1.06 |
| citeseer | 1.78 ± 1.57 | 1.44 ± 0.94 |
| pubmed | 47.96 ± 3.38 | 50.05 ± 2.72 |

## What this establishes

The tables quantify numerical agreement rather than assume that executing the authors’ code guarantees a reproduced paper result. Differences in means and seed variability remain visible. Runs use CPU DGL 2.1.0 and PyTorch 2.1.2 because the DGL CUDA wheel server was unavailable; they preserve the released NumPy 1.21.6, SciPy 1.7.3, and scikit-learn 1.0.2. The README’s original DGL/PyTorch environment was not recreated exactly. These results are not bitwise replications and contain no inference-speed claim.

Compatibility repairs are limited to missing definitions, CPU tensor placement, and reuse of graph-static tensors. An original-versus-cached optimizer-step check covers losses, gradients, and updated weights. Source hashes, all 500 validation nodes, seed coverage, configurations, and coefficients pass independent checks.

## Ordinary distillation remains unresolved

The KL-only loss ablation in REPORT.md retains FAITH’s architecture and tuning; it is not the published GLNN row. The FAITH release does not supply that baseline’s rerun configuration or checkpoints. An original GLNN configuration applied to FAITH’s differently processed data was preserved as a diagnostic in ../glnn_source_reference/STATUS.md, not accepted as a reproduced baseline. The original GLNN CPF data protocol, feature scale, model configuration, soft-target training set, and training schedule differ. A verified reproduction of the FAITH paper’s GLNN row therefore still requires the corresponding rerun configuration.

The included appendix disagrees with the main table on several values, including PubMed FAITH accuracy (81.72 versus 81.53) and GLNN means or standard deviations. Main-table targets are used consistently above; REPORT.md records these discrepancies.

Keep cache-policy development separate until the selected base method and its baseline protocol are established. These source reruns supersede the earlier structural-only pilot for assessing full FAITH accuracy, but do not establish an improvement over a verified GLNN baseline.

Sources: [FAITH main paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705), [FAITH source](https://github.com/shashivipul/FAITH), [GLNN source](https://github.com/snap-research/graphless-neural-networks). Detailed source repairs, checkpoints, per-epoch logs, package lock, and comparison data are stored with this report.
