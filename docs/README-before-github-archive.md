# FAITH structural distillation and cache evaluation

## OGB large-graph extension

Both datasets completed five seeds and four cache budgets: 460 evaluations.
At 10% cache, validation-calibrated benefit caching raises mean request-weighted
accuracy from 65.05% to 71.52% on arxiv and 68.73% to 76.20% on products.
Uniform accuracy rises from 63.42% to 67.81% and 68.40% to 72.85%, respectively.
Joint refinement adds little over equal-update controls. Cache routing increases
the local query-batch latency; exact graph unfairness varies by dataset/policy.
See the [combined five-dataset study](FAITH_CACHE_STUDY.md),
[OGB findings](results/faith_ogb/FINDINGS.md), and
[exact fairness supplement](results/faith_ogb/EXACT_FAIRNESS.md).

`faith_ogb.py` extends the FAITH-based cache study to ogbn-arxiv and
ogbn-products on the A100. It uses the released three-layer SAGE gcn and
batch-normalized MLP classes, with 256 hidden units, full-graph sparse energy
alignment and importance-sampled Oracle Similarity Alignment. This is a
documented adaptation: released arxiv width is 2048, and products has no
released model configuration. Products transfers arxiv settings with reference
gamma scaled by 169343/N. A failed direct transfer is retained in
`results/faith_ogb_transfer_diagnostic/`. A positive gamma is subsequently
selected from reference multiples [1, 0.1, 0.01, 0.001] using only model-validation
accuracy, with 150 updates per candidate on seed 42, then locked for all final
seeds. This is an exploratory follow-up: initial arxiv test results had already
been observed. Binary adjacency deduplicates reciprocal edges.

Official train/test splits are retained. A fixed half of official validation
selects model checkpoints; the other half calibrates cache benefit. The
training/evaluation graph-similarity samples are independent, with 32,768 and
65,536 pairs. Initial graph-unfairness diagnostics are Monte Carlo estimates;
`exact_fairness_ogb.py` supplements them with the exact full-graph quadratic
metric using sparse factorization in float64. Sparse-energy
values and gradients match the released edge loss, and the similarity
estimator matches exact OSA values and gradients in expectation.

```sh
.venv-reproduction/bin/python prepare_ogb_data.py --root ogb-data
.venv-reproduction/bin/python -m unittest test_faith_ogb test_faith_cache -v
.venv-reproduction/bin/python select_faith_ogb.py
.venv-reproduction/bin/python faith_ogb.py
.venv-reproduction/bin/python exact_fairness_ogb.py
.venv-reproduction/bin/python analyze_faith_ogb.py
```

The default study has five seeds and four budgets per dataset, with 460 cache
evaluation rows. Both independent continuation and joint refinement receive
150 added updates. Host GPU access is required for training. Completed seeds
resume only when source, data, sampling, split and configuration manifests
match. Per-seed checkpoints, traces, workload, calibration coefficients and
exact cache indices are retained in `results/faith_ogb/`. Its report updates
after complete seeds. The independent analyzer reconstructs every quality
and hit-rate row from saved predictions, caches and workloads. These results
are separate from the earlier A40 study and the citation reproduction.

## Cache extension of the full FAITH reproduction

The completed A100 study adds static teacher-logit caches and alternating
cache/student refinement to all 15 saved CUDA reproduction runs. Four node-cache
upper bounds (1%, 5%, 10%, 20%) produce 690 evaluations. Source checkpoints
remain unchanged; the failed CiteSeer seed is retained.

At a 10% budget, frequency caching changes mean request-weighted accuracy by
+1.05 points on Cora, +0.53 on CiteSeer, and -0.92 on PubMed. Validation-calibrated
benefit caching gives +3.85 points on CiteSeer, but calibration reuses the
checkpoint-selection validation labels and is exploratory. Joint refinement
does not consistently beat equal-update continuation controls. Frequency
caching increases measured latency and both graph-unfairness metrics.

The method keeps the full FAITH energy and OSA objectives. Joint refinement
alternates cache selection by historical frequency times teacher/student KL
with downweighting cached nodes to 0.1 in CE/KL. Independent continuation and
joint variants both receive 150 updates from the same initial checkpoint,
with matched optimizer resets and dropout seeds. Test labels do not select
caches or checkpoints. This extends the local reproduction; it does not resolve
the discrepancy with the paper's published stability.

```sh
.venv-reproduction/bin/python -m unittest test_faith_cache -v
.venv-reproduction/bin/python faith_cache.py
python3 analyze_faith_cache.py
```

Run `faith_cache.py` with host GPU access. It resumes only completed seeds with
matching manifests. Results, limits, paired comparisons, and independent checks
are in [the findings](results/faith_cache/FINDINGS.md),
[the protocol report](results/faith_cache/REPORT.md), and
`results/faith_cache/independent_verification.json`. The earlier structural pilot
below uses a different calibration split and is a separate experiment.

## Released code reproduction

`reproduce_cuda.py` is the primary CUDA reproduction attempt. It executes the
released DGL SAGEConv model and original teacher/student training loops on the
A100. Because native CUDA DGL cannot be downloaded, a narrow graph adapter
supplies only the homogeneous `copy_u` plus `sum` message primitive through
`torch.sparse.mm`. Independent checks compare forward outputs and parameter/input
gradients with native DGL on directed edges, duplicate edges, isolated nodes,
and both orders of feature transformation and aggregation. CUDA dropout, linear
layers, and Adam preserve the source GPU random-stream structure.

```sh
.venv-reproduction/bin/python reproduce_cuda.py --check-only
.venv-reproduction/bin/python reproduce_cuda.py
.venv-reproduction/bin/python analyze_reproduction.py --runtime cuda
```

Results are in `results/cuda_reproduction/`. This is a mathematical reproduction
with a recorded compatibility adapter, not a bitwise recreation of the authors'
unavailable binary environment. All settings and seeds remain fixed. Sparse
reduction ordering and framework versions can still affect numerical outcomes.

The complete CUDA attempt contains 15 dataset/seed runs. Its main comparison
and independent checks are in `results/cuda_reproduction/FINDINGS.md` and
`independent_verification.json`. The published GLNN baseline remains unverified;
the KL ablation is kept separately and is not relabeled as GLNN.

`reproduce_hybrid.py` is a retained runtime diagnostic. It keeps DGL graph convolution and the teacher optimizer
on CPU, bridges teacher dropout through CUDA, and runs the entire student on
the A100. This preserves the separation between CPU parameter-initialization
randomness and CUDA dropout randomness. The CPU-only run below exposed a
CiteSeer seed-48 collapse; the same fixed hyperparameters and seed with this
runtime did not collapse. All five original seeds are rerun, without choosing
seeds or coefficients by their test performance.

```sh
.venv-reproduction/bin/python check_reproduction_cuda.py
.venv-reproduction/bin/python reproduce_hybrid.py
```

Use approved host GPU execution. Results are in `results/hybrid_reproduction/`.
This intermediate diagnostic stopped after all five Cora and CiteSeer seeds;
PubMed was not run in this mode. The primary CUDA attempt covers all three
datasets and all five seeds. Completed fallback checkpoints are preserved.
Source-step loss, gradient, and optimizer-update equivalence are checked on
CUDA, and the dropout bridge is checked for identical outputs/gradients and
preservation of the CPU RNG stream. CPU teacher matrix arithmetic still differs
from a pure CUDA DGL run, so this is not a bitwise reproduction. Losses, model
configurations, full validation masks, and source training loops are preserved.

`reproduce.py` is a separate reproduction attempt using the actual released DGL
models and teacher/student training loops. It does not import the earlier
`experiment.py` port. Run it with:

```sh
.venv-reproduction/bin/python reproduce.py
```

It restores all 500 validation nodes, the released cumulative seed schedule
(42, 43, 45, 48, 52), the teacher-noise initialization before student creation,
and full FAITH's energy and Oracle Similarity Alignment losses. Coefficients
come from the released README. Group-fairness FAITH-star is a different method
and is not part of the citation-graph table being reproduced.

The DGL CUDA download endpoint was unavailable. A separate CPU DGL 2.1.0 and
PyTorch 2.1.2 environment runs the authors' models with the released NumPy,
SciPy, and scikit-learn versions. The launcher records minimal runtime repairs:
missing coefficient/import definitions, hardcoded CUDA placement, and caching
of unchanged graph tensors and a diagnostic Laplacian. An original-versus-cached
optimizer step verifies loss values, gradients, and updated weights. Source
files in `upstream/` remain unchanged. Cached Planetoid raw bytes are passed
through DGL's original citation processor; the Cora prefix is renamed only to
match its `cora_v2` convention. Input hashes are recorded.

Results are saved in `results/released_reproduction/`. `FAITH` is compared with
the main paper's transductive accuracy, with the appendix's differing numbers
noted explicitly. `KL_ablation` disables FAITH's two penalties under the same
configuration; it is **not** the paper's GLNN baseline. The release does not
contain that baseline's run configuration or checkpoints. The original GLNN
repository is preserved in `glnn-upstream/` for auditing, but its original CPF
data protocol differs from FAITH's public citation split.

## Earlier structural pilot

This experiment tests whether the structural distillation component of FAITH
improves a feature-only student and whether static teacher-logit caching adds
value. It is a **structural-only adaptation**, not a reproduction of full FAITH.
It excludes Oracle Similarity Alignment and counterfactual fairness penalties.
No fairness outcome is claimed.

## Sources and implementation

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/39744
- Authors' code: https://github.com/shashivipul/FAITH
- Inspected revision: `94be4423aee12cdf4d4d098d369fe472aeedcbe7`.
- Upstream files are preserved unchanged in `upstream/`.
- `experiment.py` contains the modern PyTorch implementation.
- `test_equivalence.py` checks the released structural loss and parameter gradients,
  the SAGE aggregation function and gradients, and calibration label isolation.

The released entry point references undefined `beta` and `gamma` variables,
depends on PyTorch 1.6/DGL 0.6.1, repeatedly materializes dense adjacency and
similarity matrices, and has inconsistent alternate-mode return signatures.
The README's beta=0.02 is used explicitly; gamma=0 excludes fairness alignment.
The released structural function is negative cosine similarity between student
node energies indexed by edge sources and teacher energies indexed by edge
destinations. Student energy accumulates at both ends of the symmetric directed
edge list; teacher energy accumulates at sources. This factor-of-two difference
cancels in cosine similarity. The port preserves it and checks the actual
released loss, rather than substituting a different interpretation of the paper.

Energy is evaluated with a sparse algebraic identity:

`E_i = degree_i * ||z_i||² - 2 z_i·(A Z)_i + (A ||Z||²)_i`.

Teacher and student depth, width, dropout, learning rate, weight decay,
temperature, and supervised-loss mixing are read from the authors' transductive
SAGE configuration. The SAGE teacher uses the GCN aggregator. With the released
self-looped graph, this averages graph neighbors and an additional self term.
Temperature scaling follows the release, including the absence of tau² scaling.

## Initial evaluation

Datasets: Cora, CiteSeer, PubMed. Five training/workload seeds per dataset.
The public Planetoid split and normalized features correspond to the released
citation-graph loader. The original validation set is divided with fixed seed
1729 into 250 checkpoint-selection nodes and 250 cache-calibration nodes.
Training, model validation, calibration, and test masks are disjoint and saved.
Original unlabeled nodes remain available for transductive soft-target training.

Students compared:

1. KD: the released supervised-plus-KL mixture without structural/fairness loss.
2. Structural: identical settings with structural loss coefficient 0.02.

Both students start from identical weights and use the same dropout seeds,
maximum 500 epochs, and validation-accuracy checkpoint rule with patience 50.
Actual epochs can differ. This is a matched maximum-budget comparison rather
than a claim that every model receives exactly the same number of updates.
The release refreshes an edge list during training; its all-ones sampling mask
retains all edges, so this adaptation keeps the equivalent static edge list.

Each student is evaluated without caching and with frequency, disagreement,
and predicted-benefit caches at 1%, 5%, 10%, 20% of all nodes. Frequency is based
on 100,000 synthetic Zipf(0.8) requests, with additive-one smoothed counts.
Disagreement uses frequency times KL at the configured temperature.

Benefit estimation fits fixed ridge regression (penalty 1, unpenalized intercept)
on reserved calibration nodes. Its target is teacher correctness minus student
correctness. Inputs are teacher/student entropy, confidence, margins, agreement,
KL, and entropy difference. All predictions are computed offline; no benefit
model or student is evaluated on cache hits at query time. Cache score is
historical frequency times nonnegative predicted improvement. Agreeing teacher
and student argmaxes have exactly zero improvement. Nonpositive candidates are
excluded, so this policy can leave capacity unused. All policies obey the same
node-logit storage upper bound; actual bytes are reported. The classifier, dense
int64 index, and float32 logit payload are counted separately.

Test labels are used only for final accuracy metrics. Model checkpoint selection
and benefit calibration use separate validation halves. Test-popularity queries
are drawn only after the synthetic workload is fixed; cache policies never
receive test correctness. A label-isolation test changes all noncalibration
labels and confirms that benefit scores do not change.

Accuracy is reported uniformly across test nodes, weighted by test popularity,
and under a fixed half-original/half-reversed popularity shift. Timing uses
1,024 GPU-resident test queries, 10 warmups, median of 40 synchronized calls.
It includes mapping, assembling outputs, and MLP computation on misses; it
excludes host transfer, network, and refresh. Query-path outputs are checked
against whole-graph mixed predictions for every evaluation setting.

## Running

Use the isolated `.venv`, with approved host GPU execution in this app:

```sh
.venv/bin/python -m unittest test_equivalence -v
.venv/bin/python experiment.py
```

The program resumes only completed seeds whose manifests match the requested
settings. Per-seed checkpoints, logits, traces, split indices, calibration
coefficients, metrics, and a combined report are saved in
`results/citation_pilot/`. A complete run has 390 rows: 3 datasets × 5 seeds ×
2 students × (1 no-cache setting + 3 policies × 4 budgets).

## Interpretation

This initial experiment does not reproduce the authors' published FAITH numbers:
it isolates structural loss, reserves part of validation for calibration, uses
a sparse PyTorch port and modern dependencies, and does not evaluate fairness.
Earlier A40 experiments used different architectures and training budgets;
their absolute latency and accuracy are not direct controls for this A100 run.
Decide whether to scale to Amazon and OGB from matched results, retaining
unsuccessful outcomes and simple frequency caching as primary controls.
