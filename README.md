# Cache-aware GNN distillation

Research code, technical reports, complete experiment outputs, and verification records for combining a feature-only distilled student with a limited cache of offline graph-teacher predictions.

**Read the [technical report](FAITH_CACHE_STUDY.md).** It contains the architecture, Amazon products walkthrough, teacher/student comparisons, the three cache policies, actual cache use, and separate results with and without FAITH.

This is a separate project from [ld-walklm](https://github.com/mshanthappa-hub/ld-walklm). The repository and its research-snapshot release are private backups.

## Main result from the stronger-teacher follow-up

Standard OGB test accuracy (%), seed 42, with a **10% maximum node-cache capacity**. Every cache policy uses the same frozen student and teacher. These single-seed observations are preliminary.

| Dataset | Teacher | Distilled student | Frequency cache | Disagreement cache | Benefit cache |
|---|---:|---:|---:|---:|---:|
| ogbn-arxiv | 76.32 | 76.07 | 76.09 | 76.14 | 76.32 |
| ogbn-products | 86.43 | 85.98 | 86.03 | 86.43 | 86.44 |

The combined student-and-cache system recovered approximately teacher-level accuracy. The student weights and standalone accuracy did not change. Benefit caching actually stored predictions for **4.0% of arxiv nodes** and **3.2% of products nodes**, leaving unused capacity when predicted benefit was nonpositive.

The arxiv teacher is GIANT-XRT + AGDN + BoT + self-distillation; the products teacher is GIANT-XRT + SAGN + SCR. Both were selected from the top ten in the official OGB leaderboard snapshot checked on 16 November 2026. Students are three-layer, width-256 MLPs using precomputed GIANT-XRT features and ordinary CE/KL distillation. This follow-up does not include FAITH objectives.

Cache routing increased measured latency from **0.50 to 1.32 ms** on arxiv and **0.42 to 1.32 ms** on products for 1,024 GPU-resident requests. Request-weighted accuracy decreased slightly on arxiv and improved on products; those synthetic-workload metrics are supplementary. The study does not establish a serving-speed improvement or a new leaderboard result.

- [Detailed stronger-teacher results](results/leaderboard_cache/REPORT.md)
- [All 26 evaluation rows](results/leaderboard_cache/metrics.csv)
- [Independent quality reconstruction](results/leaderboard_cache/independent_verification.json)
- [Restored-checkpoint verification](results/leaderboard_cache/checkpoint_verification.json)
- [Accuracy by cache capacity](results/leaderboard_cache/cache-accuracy.png)

## Earlier studies and experimental boundaries

The main FAITH-based study contains **1,150 evaluation rows**: 690 citation-graph rows and 460 OGB rows, retaining all five seeds and the failed CiteSeer seed. The OGB implementation is a documented adaptation using reduced width and sampled similarity training. Earlier ordinary-distillation A40 references use different models and inputs. Comparisons between these studies do not isolate FAITH's effect.

- [Five-seed OGB FAITH findings](results/faith_ogb/FINDINGS.md)
- [Citation cache findings](results/faith_cache/FINDINGS.md)
- [Exact OGB fairness supplement](results/faith_ogb/EXACT_FAIRNESS.md)
- [Older reference provenance](results/legacy_no_faith_reference/provenance.json)
- [Original research README](docs/README-before-github-archive.md)

Completed, diagnostic, tuning, smoke, failed-transfer, and partial results are all retained. They are labelled separately and are not pooled into the main tables.

## What is saved

Code, reports, metrics, logs, plots, configurations, checksums, and source snapshots are in Git. Large model checkpoints, prediction arrays, cache indices, workloads, and original dataset/embedding archives are attached to the private **research-snapshot-2026-11-16** release.

See [archive scope](ARCHIVE_SCOPE.md), [the complete result inventory](artifacts/result-files.json), [the input inventory](artifacts/input-files.json), and [the release manifest](artifacts/release-manifest.json). Extracted input copies and propagated features are reconstructed from the preserved originals; their exact original hashes are recorded. Python environments and package caches are rebuilt from the preserved runtime records.

## Restore and reproduce

```bash
git clone --recurse-submodules https://github.com/mshanthappa-hub/cache-aware-gnn-distillation.git
cd cache-aware-gnn-distillation
gh release download research-snapshot-2026-11-16 \
  --repo mshanthappa-hub/cache-aware-gnn-distillation \
  --dir ../gnn-distillation-assets
python3 tools/restore_artifacts.py --assets ../gnn-distillation-assets
python3 tools/verify_archive.py --results
```

GitHub authentication is required for the private repository and release. Restoration requires `zstd` on the local machine. Full result restoration requires space for approximately 28 GB of original experiment outputs, in addition to downloaded compressed archives.

Follow [reproduction instructions](docs/REPRODUCING.md) for input reconstruction, Python/runtime separation, report generation, and GPU checks. The completed recorded metrics can be inspected without rerunning training. Preserve the original result directories when starting new experiments.

## Dependencies and provenance

Four upstream implementations are pinned as Git submodules. [Upstream revisions](artifacts/upstream-revisions.json), original licenses, and [third-party notices](THIRD_PARTY_NOTICES.md) preserve attribution. The private archive does not grant a new license to source, datasets, embeddings, or models.

Primary sources: [FAITH](https://ojs.aaai.org/index.php/AAAI/article/view/39744), [GLNN](https://arxiv.org/abs/2110.08727), [OGB](https://ogb.stanford.edu/docs/nodeprop/), [GIANT-XRT](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt), [AGDN](https://github.com/skepsun/Adaptive-Graph-Diffusion-Networks), and [SCR](https://github.com/THUDM/SCR).
