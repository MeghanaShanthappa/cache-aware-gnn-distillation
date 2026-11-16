# Reproducing and inspecting the studies

## Restore the archived project

Clone the repository with its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/mshanthappa-hub/cache-aware-gnn-distillation.git
cd cache-aware-gnn-distillation
```

The release is private like the repository. Authenticate GitHub CLI with an account that has repository access, then download all archive parts and restore them:

```bash
gh release download research-snapshot-2026-11-16 \
  --repo mshanthappa-hub/cache-aware-gnn-distillation \
  --dir ../gnn-distillation-assets
python3 tools/restore_artifacts.py --assets ../gnn-distillation-assets
python3 tools/verify_archive.py --results
```

Restoration verifies archive part checksums and rejects unsafe archive paths. It restores original files at their project-relative paths. It does not execute saved Python files or load PyTorch checkpoints.

## Reconstruct derived inputs

The citation inputs are restored exactly by the citation archive. For OGB inputs, `prepare_ogb_data.py` extracts the preserved official ZIPs and rebuilds feature, label, split, and normalized adjacency files:

```bash
python prepare_ogb_data.py --root ogb-data --datasets ogbn-arxiv ogbn-products
```

Extract the published GIANT-XRT arrays from the preserved feature archives:

```bash
python3 tools/extract_giant_inputs.py
```

The products teacher regenerates three-hop feature propagation and nine-hop training-label propagation when its preprocessing cache is absent. To reconstruct this cache without training, use the experiment environment with CUDA access:

```bash
python -c 'import leaderboard_common as C; import leaderboard_teacher as T; C.setup(); T.products_preprocess()'
```

`artifacts/input-files.json` records the exact original hashes, including all derived files. GPU/runtime changes can affect regenerated floating-point arrays; the archived outputs and checkpoints remain the authoritative recorded runs.

## Runtime separation

The original experiments use Python 3.10 and PyTorch 2.1.2+cu118 on an NVIDIA A100. `requirements.reproduction.lock.txt` records the source-reproduction environment. The earlier structural pilot has a separate lock file and must not be pooled with the main studies.

The FAITH reproduction used CPU DGL 2.1.0 with a verified CUDA sparse message-sum adapter. The stronger-teacher study used a separate native CUDA DGL 1.1.2 installation. Do not replace the FAITH environment with the stronger-teacher runtime and treat the result as the same execution.

The measured stronger-teacher environment and official native package URL/checksum are in `results/leaderboard_cache/environment.json`. Native DGL was extracted into `leaderboard-upstream/dgl-conda`, and the NVIDIA CUDA runtime libraries were installed under `leaderboard-upstream/gpu-packages`. Use these library directories and PyTorch's `torch/lib` in `LD_LIBRARY_PATH` for the AGDN teacher. The experiment code selects the isolated DGL package through `leaderboard_common.py`.

## Inspect and regenerate reports

The current consolidated report is `FAITH_CACHE_STUDY.md`; its authoring template is `docs/report-template.md`. `report_builder.py` regenerates the report from the original saved metrics, verifies the older reference hashes, and keeps a sibling user-facing mirror in sync if it exists.

```bash
python report_builder.py
python analyze_leaderboard.py
```

`analyze_leaderboard.py` independently reconstructs all 26 stronger-teacher quality rows from saved prediction vectors, cache indices, and workloads. It does not retrain models. Its report is `results/leaderboard_cache/REPORT.md`.

GPU checks can additionally restore student weights and compare every saved prediction:

```bash
python verify_leaderboard.py
python -m unittest test_leaderboard_cache -v
```

Source and archive hashes are checked against the recorded executed snapshots when the current authoring code has changed after the original run.

## Run new experiments separately

The archived results are complete. The existing runners skip completed work. To run a genuinely new study, select a new output directory and preserve the archived results rather than overwriting them. Inspect each script's `--help` for its output and configuration arguments.

Main entry points:

- `reproduce_cuda.py`: released FAITH citation reproduction through the verified CUDA adapter.
- `faith_cache.py`: static caches and compute-matched joint refinement for citation datasets.
- `select_faith_ogb.py`, `faith_ogb.py`: documented OGB FAITH adaptation and positive-coefficient selection.
- `exact_fairness_ogb.py`: exact quadratic fairness supplement.
- `leaderboard_teacher.py`: stronger graph teachers.
- `leaderboard_distill.py`: ordinary distillation and all three static cache policies.
- `run_leaderboard_study.py`: sequential study controller with concurrency-aware student checkpoints and cache timing in the current worker scripts.

The GIANT study's seed 42 results are preliminary. The FAITH study uses five seeds and retains the failed CiteSeer seed. Smoke tests, failed transfers, tuning trials, and the earlier products loader run are preserved for audit and excluded from the primary comparisons.
