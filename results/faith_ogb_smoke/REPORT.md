# OGB: scalable FAITH-based cache study

A100 study with released SAGE/MLP model classes, width 256, full-graph energy alignment and importance-sampled OSA. This is a documented adaptation, not published FAITH reproduction. Products has no released configuration; arxiv settings transfer with gamma multiplied by 169343/N. All official test nodes are evaluated. Cache calibration uses half of official validation, disjoint from checkpoint selection.

## Frozen student at 10% cache

Request-weighted accuracy (%), mean ± population standard deviation. Synthetic Zipf(0.8) requests; all completed seeds are included. Uniform accuracy appears separately.

| Dataset | Seeds | No cache | Frequency | Disagreement | Benefit |
|---|---:|---:|---:|---:|---:|
| ogbn-arxiv | 1 | 11.49 ± 0.00 | 7.81 ± 0.00 | 7.97 ± 0.00 | 13.62 ± 0.00 |
| ogbn-products | 1 | 25.93 ± 0.00 | 27.97 ± 0.00 | 27.74 ± 0.00 | 29.00 ± 0.00 |

## Uniform test accuracy (%)

| Dataset | No cache | Frequency | Disagreement | Benefit |
|---|---:|---:|---:|---:|
| ogbn-arxiv | 10.49 ± 0.00 | 10.03 ± 0.00 | 10.17 ± 0.00 | 12.05 ± 0.00 |
| ogbn-products | 26.98 ± 0.00 | 27.23 ± 0.00 | 27.17 ± 0.00 | 29.58 ± 0.00 |

## Equal-update controls at 10% cache

| Dataset | Continued no cache | Continued frequency | Continued disagreement | Joint | Continued using joint cache | Continued benefit | Joint benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | 22.30 ± 0.00 | 12.52 ± 0.00 | 12.62 ± 0.00 | 12.46 ± 0.00 | 12.62 ± 0.00 | 21.91 ± 0.00 | 21.62 ± 0.00 |
| ogbn-products | 26.48 ± 0.00 | 28.22 ± 0.00 | 27.95 ± 0.00 | 28.01 ± 0.00 | 27.95 ± 0.00 | 29.00 ± 0.00 | 29.00 ± 0.00 |

## Protocol and limits

Three-layer width-256 SAGE gcn teacher and released batch-normalized MLP student; dropout teacher 0.5/student 0.1, Adam lr 0.005, weight decay 0, lambda 0.2, tau 1 with no tau-squared multiplier. Beta 0.02; gamma 1e-8 on arxiv and 1e-8*169343/N on products. These choices were fixed before test evaluation. Arxiv released width is 2048, so width 256 is an explicit memory/runtime adaptation. Products settings are transferred, not author-specified.

Maximum 500 epochs by default, validation every five, minimum 50, patience 50 epochs; validation ties replace the checkpoint. Model selection sees half of official validation; the other half calibrates a fixed ridge benefit regressor. Both are disjoint from train/test. Continued and joint refinement use the same initial weights, optimizer resets, dropout seeds and 150 added updates (three phases of 50); final refinement steps are evaluated without selecting by test results.

Source energy cosine is computed by equivalent sparse matrix identities over every graph edge. OSA estimates the released sum of sqrt(adjacency-row cosine) times logit distance using 32,768 common-neighbor wedge samples with inverse pair-probability weights. Graph unfairness is independently estimated with 65,536 fixed evaluation samples; raw and probability values are both saved. This is a Monte Carlo graph metric, not exact full-graph paper unfairness. Fixed samples can introduce approximation and optimization error; seeds vary training/workload, not the fixed graph samples.

Binary undirected adjacency has one self loop; the released SAGE gcn class adds its own self contribution. Products/arxiv features remain original. Source arxiv loader can retain reciprocal-edge multiplicity, whereas this extension deduplicates it. Teacher and MLP computation execute on CUDA; CSR message sums replace unavailable native DGL CUDA, with native-DGL value/gradient tests.

Historical workload contains max(100000,10*N) Zipf(0.8) requests and additive-one count smoothing. Node cache bounds are 1%,5%,10%,20% of all graph nodes. Cache payload is 4*k*C bytes; lookup is 8*N bytes when nonempty, counted separately from student parameters. Benefit may underfill the budget. Teacher outputs are prepared offline. Test labels do not select caches, checkpoints or configurations.

Latency includes lookup, output assembly and MLP inference only on misses, for 1,024 GPU-resident test requests, 10 warmups and median 40 synchronized calls. It excludes host/network transfer and offline construction/refresh. Query outputs and argmaxes must agree with canonical mixed predictions. Shared-device/order variability limits performance conclusions. Compare within this trial, not against earlier A40 or citation numbers.

See per-dataset data_protocol.json, per-seed manifests, split.npz, fixed similarity samples, training traces and all metrics. This report updates only after complete seeds; all intended five seeds per dataset are required for the final study.
