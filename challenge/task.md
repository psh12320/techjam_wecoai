# KuaiRand-Pure autonomous recommender research task

Build a CPU-practical model for **within-user ranking over logged impressions**.
The target is the native binary `long_view` column. The immutable organizer
metrics are `GAUC` and `nDCG@5`; `primary = (GAUC + nDCG@5) / 2`, and higher is
better. The published validation baseline is GAUC `0.6674`, nDCG@5 `0.5357`,
primary `0.6016`.

## Hard protocol

- Use `input/train.csv` for fitting and `input/valid.csv` for public validation.
- Never look for, infer, or access dates after 2022-04-28. No hidden-test data is
  present in the agent workspace.
- `long_view` in validation is a label for evaluation/early stopping only. It
  must never be included in model inputs or copied/encoded into predictions.
- Current-row engagement outcomes (`is_click`, `is_like`, `play_time_ms`, etc.)
  are unavailable at serving time. They may be auxiliary **training targets or
  historical aggregates**, but never direct validation features.
- Static `video_features_basic_pure.csv` and `user_features_pure.csv` are allowed.
  `video_features_statistic_pure.csv` is excluded because its monthly aggregate
  window risks future leakage.
- Preserve validation row order. Write exactly one file at
  `working/validation_predictions.csv` with header `row_id,score`, one row per
  `input/valid.csv` row, row IDs `0..124908`, finite numeric scores only.
- Only the external deterministic evaluator's metric is authoritative.
- Complete within 15 minutes per candidate on CPU. Use a fixed seed.

## Research priorities

Start from the supplied official FM seed. Propose one atomic change per branch.
Prioritize ranking-aligned losses, out-of-time residual learning, personalized
history/sequence features, leakage-safe multi-task history, and conservative
hybrids. Do not retry known dead ends: simply increasing FM embedding dimension,
or adding coarse static user/item fields without personalized interactions.

The final candidate must improve both GAUC and nDCG@5, not merely trade one for
the other. Explain the hypothesis in 3–5 sentences, implement it fully, print
useful progress, and always write the required prediction file.
