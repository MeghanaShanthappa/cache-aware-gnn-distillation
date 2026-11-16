# FAITH hybrid runtime reproduction

Released DGL models, loss formulas, full public validation masks, and cumulative seeds 42, 43, 45, 48, 52. Graph convolution and the teacher optimizer run on CPU because the official DGL CUDA wheel server was unavailable. Teacher dropout runs on CUDA through a transfer bridge, and student training runs fully on the A100. This preserves the separate CPU and CUDA random streams of the authors’ GPU initialization sequence; graph/teacher numerical computations still differ from a pure GPU run.

| Dataset | Method | Main paper accuracy % | Reproduction accuracy % | Difference pp | Seeds |
|---|---|---:|---:|---:|---:|
| citeseer | SAGE | 70.86 ± 0.35 | 71.32 ± 0.92 | +0.46 | 5 |
| citeseer | FAITH | 74.10 ± 0.60 | 63.80 ± 19.02 | -10.30 | 5 |
| citeseer | KL_ablation | Not separately reported | 64.48 ± 16.30 | — | 5 |
| cora | SAGE | 82.18 ± 0.95 | 81.64 ± 0.76 | -0.54 | 5 |
| cora | FAITH | 83.96 ± 0.47 | 83.08 ± 0.90 | -0.88 | 5 |
| cora | KL_ablation | Not separately reported | 83.02 ± 0.78 | — | 5 |

Population standard deviations follow the released aggregation code. The same fixed settings and all five seeds are retained; no selection uses test accuracy. Full FAITH includes energy alignment and Oracle Similarity Alignment. KL_ablation retains FAITH tuning and is not the published GLNN rerun.

The CPU-only source rerun in ../released_reproduction/ is retained as a runtime diagnostic. It revealed a CiteSeer seed-48 collapse. CPU-only dropout advances the parameter-initialization RNG stream, unlike GPU dropout. This runtime explicitly preserves the separate streams without changing model widths, losses, coefficients, masks, or seed selection. It is a documented reproduction attempt, not a bitwise recreation of the authors’ unavailable GPU environment.

The published GLNN rerun configuration remains absent from the FAITH release; the original GLNN CPF pipeline differs from its citation loader. Cross-protocol GLNN diagnostics in ../glnn_source_reference/ must not be used as paper baselines.

Source code, checkpoints, logs, raw-data hashes, configuration and runtime repair records accompany this report. Main-paper and appendix discrepancies are described in ../released_reproduction/REPORT.md.

[FAITH paper](https://ojs.aaai.org/index.php/AAAI/article/download/39744/43705) · [Authors’ code](https://github.com/shashivipul/FAITH)
