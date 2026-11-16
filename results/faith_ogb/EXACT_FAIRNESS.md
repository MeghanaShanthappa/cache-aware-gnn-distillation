# Exact full-graph unfairness supplement

The quadratic Laplacian metric admits exact sparse factorization even though the nonquadratic OSA training objective requires sampling. These values cover every graph node and cosine-similar pair. They supplement the sampled diagnostics in the original metrics without changing training, cache selection, accuracy or source artifacts. Float64 arithmetic reduces cancellation. Lower is better; values are divided by 1,000.

| Dataset | Policy at 10% | Raw-logit unfairness | Probability unfairness |
|---|---|---:|---:|
| ogbn-arxiv | none | 3093150.610 | 8184.009 |
| ogbn-arxiv | frequency | 3107898.889 | 8356.494 |
| ogbn-arxiv | disagreement | 2985565.304 | 8102.145 |
| ogbn-arxiv | benefit | 2990150.994 | 8125.269 |
| ogbn-products | none | 53054392.522 | 58715.918 |
| ogbn-products | frequency | 74469847.973 | 61674.686 |
| ogbn-products | disagreement | 66568034.342 | 59214.551 |
| ogbn-products | benefit | 67183524.428 | 60208.193 |
