# Completed OGB FAITH-based cache extension

At a 10% cache budget, benefit caching improves request-weighted and uniform accuracy on both OGB datasets. Joint refinement adds little over equally trained controls, while cache routing increases measured query latency. Exact graph unfairness depends on the dataset and policy.

Both ogbn-arxiv and ogbn-products completed all five seeds and four budgets: 460 evaluations. Ten unit checks and CUDA equivalence checks passed; all 460 accuracy, request-weighted accuracy, shifted-workload accuracy and hit-rate rows were independently reconstructed from checkpoint predictions, saved cache indices and the workload. No failed or low-accuracy seed is excluded.

## Accuracy at 10% cache

Mean synthetic request-weighted accuracy (%), with mean uniform test accuracy in parentheses. These are two different metrics. The teacher column is uniform accuracy.

| Dataset | Teacher | FAITH-based student | Frequency cache | Disagreement cache | Benefit cache |
|---|---:|---:|---:|---:|---:|
| ogbn-arxiv | 70.82 | 65.05 (63.42) | 69.37 (64.21) | 70.39 (65.91) | 71.52 (67.81) |
| ogbn-products | 77.59 | 68.73 (68.40) | 74.04 (69.31) | 75.11 (70.83) | 76.20 (72.85) |

## Validation-only coefficient selection

| Dataset | Reference gamma | Selected gamma | Seed-42 validation accuracy at selection |
|---|---:|---:|---:|
| ogbn-arxiv | 1e-08 | 1e-11 | 61.19% |
| ogbn-products | 6.9147e-10 | 6.9147e-13 | 75.95% |

Selection uses accuracy without a fairness constraint. A positive similarity coefficient retains OSA but does not establish the fairness improvement reported by the original paper. The exact graph metrics below assess the resulting tradeoff.


## Joint refinement against matched controls

Paired request-weighted percentage-point changes at 10%, mean ± population standard deviation (wins out of five seeds). Both students receive 150 updates from the same initial weights, optimizer resets and dropout seeds. Same-cache control holds cache indices fixed, isolating changing the student.

| Dataset | Joint − continued disagreement | Joint − continued same cache | Joint benefit − continued benefit |
|---|---:|---:|---:|
| ogbn-arxiv | +0.12 ± 0.04 (5/5) | +0.13 ± 0.03 (5/5) | -0.00 ± 0.07 (2/5) |
| ogbn-products | -0.04 ± 0.03 (0/5) | +0.04 ± 0.03 (4/5) | +0.01 ± 0.02 (3/5) |

## Serving and graph smoothness

Means at 10% frequency caching. Unfairness is the exact full-graph adjacency-row cosine Laplacian quadratic form, divided by 1,000. A float64 sparse factorization avoids constructing all pairs. Probability unfairness applies it to softmax probabilities. Lower is better. The original sampled diagnostics are retained separately.

| Dataset | Batch latency ms, base → cache | Raw unfairness, base → cache | Probability unfairness, base → cache | Payload + index bytes |
|---|---:|---:|---:|---:|
| ogbn-arxiv | 0.451 → 0.845 | 3093150.610 → 3107898.889 | 8184.009 → 8356.494 | 2,709,440 + 1,354,744 |
| ogbn-products | 0.450 → 0.851 | 53054392.522 → 74469847.973 | 58715.918 → 61674.686 | 46,041,576 + 19,592,232 |

## Scope and evidence

This is a scalable FAITH-based adaptation, not reproduction of the published OGB result. It uses the released three-layer SAGE gcn and batch-normalized MLP classes, reduced width 256 instead of arxiv’s 2048, binary undirected edges, exact sparse energy and an importance-sampled OSA sum. Products lacks released settings. Gamma is selected from the source reference (size-scaled for products) times [1,0.1,0.01,0.001], with 150 updates per candidate on seed 42. Only model-validation accuracy selects a positive coefficient, then it is locked across all five final seeds. The failed direct-transfer experiment and its already-observed arxiv test results are preserved in ../faith_ogb_transfer_diagnostic; this follow-up is exploratory. The teacher/student gap and adaptations limit claims about improvements over published FAITH or stronger OGB reference models.

The official validation split is divided with a fixed seed between checkpoint selection and cache-benefit calibration; test labels are used only for metrics. Benefit uses fixed ridge regression and can leave cache slots empty. Joint training downweights cached nodes to 0.1 in CE/KL while retaining energy and OSA. Joint ranking uses frequency times KL; the other policies use frequency or frequency times predicted benefit. All cache fractions are upper bounds over all graph nodes.

Synthetic Zipf popularity is coupled with training seed, with five paired realizations. OSA uses a fixed graph-similarity training sample; variation across seeds does not estimate its sampling uncertainty. Original evaluation samples are independent. The exact quadratic fairness supplement removes evaluation-sampling error, while training OSA remains approximate. Accuracy gains on this workload are not evidence of deployment gains. No significance or state-of-the-art claim is made. Timing is a local GPU-resident 1,024-query microbenchmark; it includes routing and miss inference, excludes host/network/offline work, and is subject to shared-device and ordering variability.

Memory columns count payload and lookup storage required by the serving policy, rather than peak allocator usage. The benchmark wrapper initializes an unused dense lookup even for an empty cache; the empty-cache query branch does not access it. Thus the recorded zero empty-cache lookup is a logical serving requirement, not its literal allocation in that wrapper.

Earlier citation and A40 results use different architectures/protocols. They remain separate experiments. Full protocol and mean ± standard deviation tables: [REPORT.md](REPORT.md). All budgets and paired deltas: [paired_comparisons.json](paired_comparisons.json). Independent checks: [independent_verification.json](independent_verification.json). Per-seed files retain checkpoints, traces, calibration, workload and exact cache indices.
