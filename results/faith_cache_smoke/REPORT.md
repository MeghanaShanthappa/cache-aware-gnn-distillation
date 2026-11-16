# Cache augmentation of the saved full FAITH students

All five reproduction seeds are retained, including the low-accuracy CiteSeer seed. Teacher and frozen student checkpoints are unchanged. Evaluation uses the A100, static teacher-logit caches, four node-cache upper bounds, and synthetic Zipf requests. Results compare our local FAITH reproduction, not the paper’s unverified GLNN baseline.

## Frozen FAITH at 10 percent cache

Request-weighted accuracy in percent, mean ± population standard deviation across five training/workload seeds. No cache uses the same checkpoints and test popularity.

| Dataset | No cache | Frequency | Disagreement | Predicted benefit |
|---|---:|---:|---:|---:|
| cora | 85.93 ± 0.00 | 86.50 ± 0.00 | 86.48 ± 0.00 | 85.93 ± 0.00 |

Uniform test accuracy in percent:

| Dataset | No cache | Frequency | Disagreement | Predicted benefit |
|---|---:|---:|---:|---:|
| cora | 83.10 ± 0.00 | 83.30 ± 0.00 | 83.20 ± 0.00 | 83.10 ± 0.00 |

## Compute matched refinement comparison

Independent continuation and joint refinement both add three phases of 50 full-graph Adam updates, with matched optimizer resets, initial weights, dropout seeds, and final-step evaluation. Joint selection uses frequency times KL; selected nodes receive 0.1 weights in CE/KL only. The complete FAITH energy and OSA losses remain unchanged. No continuation checkpoint is selected by test accuracy.

| Dataset | Continued no cache | Continued frequency | Continued disagreement | Continued benefit | Joint cache | Joint benefit | Continued with joint cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 58.19 ± 0.00 | 72.71 ± 0.00 | 75.33 ± 0.00 | 77.06 ± 0.00 | 70.35 ± 0.00 | 72.77 ± 0.00 | 74.90 ± 0.00 |

## Protocol and limits

Cache budgets are 1%, 5%, 10%, and 20% of all nodes. Float32 teacher logits use 4*k*C bytes and the dense int64 lookup uses 8*N bytes. The student parameters are reported separately. Predicted-benefit caches may leave slots empty if estimated benefit is nonpositive. Frequency uses 100,000 historical Zipf(0.8) requests with additive-one count smoothing. Cache selection uses all graph nodes, while accuracy and query timing use public test nodes.

Benefit calibration is fixed ridge regression (penalty 1, unpenalized intercept), targeting teacher correctness minus student correctness on the 500 validation nodes. Features are prediction entropy, confidence, margins, agreement, and KL. These validation labels already selected the saved checkpoints; calibration is not an independently held-out experiment. Test labels never select a cache, checkpoint, or loss setting. No graph features or true nodewise test benefit enter cache scores.

Reported paired accuracy uses the same synthetic popularity in every method for each seed. Uniform accuracy and a half-reversed popularity shift are also saved. Values are descriptive, with no significance claim. A strong improvement on the failed CiteSeer seed is recovery of an unstable base student, not proof of a general cache-training advantage.

Individual unfairness uses the released graph-similarity Laplacian and full mixed logits, divided by 1,000. A second metric uses class probabilities to expose raw-logit scale effects. Hard teacher overrides do not automatically preserve FAITH’s fairness properties; quality and unfairness must be judged jointly.

Timing uses 1,024 GPU-resident requests, 10 warmups, median of 40 synchronized calls. It includes lookup, output assembly, and miss-only student inference; it excludes host transfer, network, cache refresh, and offline teacher/cache construction. Query outputs and argmaxes are checked against quality evaluation. These are local microbenchmarks, not end-to-end serving measurements.

The primary source-reproduction limitation remains: a verified CUDA message-sum adapter replaces unavailable native CUDA DGL. The serving MLP is the actual released student. Saved reproduction files remain untouched. The sourceFAITH pure node losses are reweighted only during our refinement; its graph energy and similarity objectives are kept intact.

All budgets, latency, unfairness, actual memory, refinement traces, calibration coefficients, and source hashes are saved with the metrics.
