# Optional research menu

This is a menu, not a fixed recipe. Select from it only when EDA, diagnostics, ancestry, and experiment memory support the choice.

- Rich gated FM or field-weighted FM with train-fitted categorical vocabularies and compact numerical buckets.
- Candidate-aware, strictly chronological multi-timescale history residuals, preferably anchored to a working parent rather than replacing it.
- Duration-relative auxiliary supervision such as RAD, D2Q/D2Co, or censored watch-time modeling, implemented inside the existing training loop.
- DIN-lite or multi-task DIN only when cheaper history features show signal and the design fits the CPU/memory budget.
- Compact DCN-V2 residuals, LightGCN, or CatBoost as diversity branches when their cost and failure history justify exploration.
- BCE with a small RankNet or LambdaLoss term in the same update; pairwise-only fine-tuning is a recorded dead end.
- Conservative out-of-fold or multi-exit ensembles that regenerate every component in the candidate program and use label-free fixed blending.

Favor, without forcing, the evidence-backed progression: rich FM/FwFM core; candidate-aware history; duration-relative auxiliary; metric-aligned refinement; DIN-lite if affordable; conservative ensemble; then DCN-V2, LightGCN, or CatBoost diversity. The scheduler may reorder this when durable evidence supports doing so.
