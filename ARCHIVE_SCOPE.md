# Archive scope

This repository is a separate snapshot of the cache-aware GNN distillation project. It does not modify or replace the WalkLM project.

| Material | Saved location |
|---|---|
| Experiment and analysis code, current report, documentation, requirements, tables, logs, verification records, plots, and source snapshots | Git repository |
| All experiment outputs, including every checkpoint, NumPy prediction file, workload, split, cache selection, tuning run, smoke test, failed trial, and partial run | `all-experiment-results` release archive |
| Original citation-graph inputs and their local processed forms | `citation-inputs` release archive |
| Original official OGB arxiv and products ZIP archives | `original-ogb-inputs` release archive |
| Original GIANT-XRT arxiv and products archives, including published embeddings | `original-giant-arxiv` and `original-giant-products` release archives |
| Upstream research implementations | Four unmodified, pinned Git submodules; revisions in `artifacts/upstream-revisions.json` |
| Every saved result file and its exact checksum | `artifacts/result-files.json` |
| Every input file, including derived NumPy arrays and propagated features, and its exact checksum | `artifacts/input-files.json` |
| Release archive parts, byte sizes, SHA-256 checksums, and restoration groups | `artifacts/release-manifest.json` |

The large archives are split into parts smaller than GitHub's per-asset limit. Download every part of a selected group before restoring it. Parts are concatenated in numeric order and decompressed as one Zstandard stream. The restore tool verifies each part's SHA-256 before extracting.

Python virtual environments, interpreter installations, package caches, compiled bytecode, Matplotlib font caches, and installed native libraries are rebuildable runtime material. Requirements and measured runtime records are preserved instead of copying these installations into Git.

Extracted OGB CSVs, normalized graph matrices, extracted GIANT NumPy embeddings, and propagated products feature tables are reproducible copies or transformations of the preserved original input archives. They are not duplicated in the release. Their original hashes remain in `input-files.json`, and reconstruction instructions are in `docs/REPRODUCING.md`.

All reported results come from the original completed runs. Packaging does not rerun training, select new checkpoints, change model weights, or remove unsuccessful runs. Small result files in Git are byte-for-byte copies; release extraction restores the complete result directory.

Historical absolute paths and process IDs in provenance files are retained as evidence of the original execution environment. They are not portable installation paths or evidence that a process is still running.
