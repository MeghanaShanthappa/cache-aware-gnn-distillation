# Failed direct transfer diagnostic

This initial width-256 experiment used released arxiv gamma and size-scaled transfer to products. All five arxiv seeds completed; the first products FAITH student peaked at validation epoch 5 and stopped at epoch 55 near 36% validation accuracy. Cache refinement was interrupted. Test numbers from completed arxiv trials were observed, but they do not choose the next configuration.

Retained here: original source snapshot (../../faith_ogb_transfer_diagnostic.py), all completed checkpoints and metrics, and partial products artifacts. This is a failed-transfer diagnostic, not the final OGB study. The follow-up selects a positive OSA coefficient only by model-validation accuracy over a predeclared multiplicative grid, then locks it for all five seeds. Cache calibration uses a disjoint validation half.
