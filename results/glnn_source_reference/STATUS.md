# GLNN configuration diagnostic

These runs apply the original GLNN student routine and YAML configuration to FAITH teacher checkpoints and FAITH public citation data. They are **not a paper baseline reproduction** and must not be used in a main benchmark comparison.

The original GLNN citation protocol uses CPF graphs, raw bag-of-words feature scales, and random per-class train/validation splits. The FAITH DGL citation loader uses row-normalized features and public masks. Applying the original GLNN optimizer configuration to this different input protocol produced poor Cora validation and test accuracy. The source exposes several unresolved differences: feature scale, student width and depth, soft-target node set, and number of optimizer steps. The experiment does not isolate a single cause. No settings were tuned against the test results.

The valid released-FAITH reproduction attempt, including the original architecture and full validation masks, is in ../released_reproduction/REPORT.md. The released FAITH code contains no configuration or checkpoints establishing how its published GLNN row was rerun.
