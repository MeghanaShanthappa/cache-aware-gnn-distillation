# Cache augmentation of the saved full FAITH students

All five reproduction seeds are retained, including the low-accuracy CiteSeer seed. Teacher and frozen student checkpoints are unchanged. Evaluation uses the A100, static teacher-logit caches, four node-cache upper bounds, and synthetic Zipf requests. Results compare our local FAITH reproduction, not the paper’s unverified GLNN baseline.

## Frozen FAITH at 10 percent cache

Request-weighted accuracy in percent, mean ± population standard deviation across five training/workload seeds. No cache uses the same checkpoints and test popularity.

| Dataset | No cache | Frequency | Disagreement | Predicted benefit |
|---|---:|---:|---:|---:|
| citeseer | 59.94 ± 12.77 | 60.47 ± 6.77 | 59.92 ± 7.35 | 63.79 ± 7.43 |
| cora | 83.33 ± 2.64 | 84.38 ± 4.39 | 84.28 ± 4.13 | 83.19 ± 3.29 |
| pubmed | 84.25 ± 1.58 | 83.33 ± 1.87 | 82.08 ± 2.09 | 84.17 ± 1.53 |

Uniform test accuracy in percent:

| Dataset | No cache | Frequency | Disagreement | Predicted benefit |
|---|---:|---:|---:|---:|
| citeseer | 63.64 ± 18.94 | 64.48 ± 16.87 | 64.46 ± 16.90 | 64.98 ± 15.97 |
| cora | 83.10 ± 0.68 | 83.16 ± 0.67 | 83.10 ± 0.41 | 82.96 ± 0.81 |
| pubmed | 81.76 ± 0.53 | 81.56 ± 0.61 | 79.76 ± 0.52 | 81.66 ± 0.49 |

## Compute matched refinement comparison

Independent continuation and joint refinement both add 3 phases of 50 full-graph Adam updates, with matched optimizer resets, initial weights, dropout seeds, and final-step evaluation. Joint selection uses frequency times KL; selected nodes receive 0.1 weights in CE/KL only. The complete FAITH energy and OSA losses remain unchanged. No continuation checkpoint is selected by test accuracy.

| Dataset | Continued no cache | Continued frequency | Continued disagreement | Continued benefit | Joint cache | Joint benefit | Continued with joint cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | 60.55 ± 12.82 | 60.18 ± 7.01 | 59.99 ± 7.46 | 61.36 ± 6.60 | 60.42 ± 7.24 | 62.04 ± 6.42 | 59.99 ± 7.29 |
| cora | 84.31 ± 3.02 | 84.45 ± 4.38 | 84.34 ± 4.47 | 83.85 ± 3.37 | 84.41 ± 4.16 | 83.73 ± 1.47 | 84.47 ± 4.40 |
| pubmed | 84.29 ± 1.32 | 83.35 ± 1.88 | 81.85 ± 2.06 | 84.27 ± 1.31 | 81.58 ± 2.07 | 84.27 ± 1.28 | 81.69 ± 2.19 |

## Protocol and limits

Cache budgets are 1%, 5%, 10%, and 20% of all nodes. Float32 teacher logits use 4*k*C bytes and the dense int64 lookup uses 8*N bytes. The student parameters are reported separately. Predicted-benefit caches may leave slots empty if estimated benefit is nonpositive. Frequency uses 100,000 historical Zipf(0.8) requests with additive-one count smoothing. Cache selection uses all graph nodes, while accuracy and query timing use public test nodes.

Benefit calibration is fixed ridge regression (penalty 1, unpenalized intercept), targeting teacher correctness minus student correctness on the 500 validation nodes. Features are prediction entropy, confidence, margins, agreement, and KL. These validation labels already selected the saved checkpoints; calibration is not an independently held-out experiment. Test labels never select a cache, checkpoint, or loss setting. No graph features or true nodewise test benefit enter cache scores.

Reported paired accuracy uses the same synthetic popularity in every method for each seed. Uniform accuracy and a half-reversed popularity shift are also saved. Values are descriptive, with no significance claim. A strong improvement on the failed CiteSeer seed is recovery of an unstable base student, not proof of a general cache-training advantage.

Individual unfairness uses the released graph-similarity Laplacian and full mixed logits, divided by 1,000. A second metric uses class probabilities to expose raw-logit scale effects. Hard teacher overrides do not automatically preserve FAITH’s fairness properties; quality and unfairness must be judged jointly.

Timing uses 1,024 GPU-resident requests, 10 warmups, median of 40 synchronized calls. It includes lookup, output assembly, and miss-only student inference; it excludes host transfer, network, cache refresh, and offline teacher/cache construction. Query outputs and argmaxes are checked against quality evaluation. These are local microbenchmarks, not end-to-end serving measurements.

The primary source-reproduction limitation remains: a verified CUDA message-sum adapter replaces unavailable native CUDA DGL. The serving MLP is the actual released student. Saved reproduction files remain untouched. The source FAITH node losses are reweighted only during our refinement; its graph energy and similarity objectives are kept intact.

All budgets, latency, unfairness, actual memory, refinement traces, calibration coefficients, and source hashes are saved with the metrics.
