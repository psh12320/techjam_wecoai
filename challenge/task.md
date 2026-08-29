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

## Experiment memory

- Reproduced five-field FM: GAUC 0.66713, nDCG@5 0.53580, primary 0.60147.
- Frozen champion: 80% three-seed rich field-gated FM, 10% history LambdaRank,
  and 10% user-duration RAD. It reaches GAUC 0.67105, nDCG@5 0.53801,
  primary 0.60453: deltas +0.00392/+0.00221/+0.00306 over the reproduced FM.
- The robust core is a three-seed field-gated FM over 13 fields, with
  train-fitted vocabularies and explicit unknown slots. By itself it reaches
  GAUC 0.67012, nDCG@5 0.53728, primary 0.60370.
- Its gains have the same direction for both component metrics on rolling
  2022-04-15–17 and 2022-04-18–21 validation folds. Future changes must beat
  this core and repeat those checks, not just overfit the public week.
- Standalone history LambdaRank and RAD are weaker and remain blend-only.
- Naive hard-negative BPR, coarse sequence SVD, broad static LightGBM, and
  indiscriminately adding all metadata fields to ordinary FM did not win.
- Never use unobserved items as negatives. Pair only logged long-view and
  short-view impressions from the same user.
- Candidate features must be the train/validation column intersection. Sort
  rows so every ranking group is contiguous and use the supplied evaluator.

High-priority atomic improvements supported by KuaiRand-specific research:

1. Preserve the FM score and learn a small residual with BCE plus a weak
   within-user RankNet/LambdaLoss@5 term.
2. Add watch-time auxiliary supervision through Relative Advantage Debiasing,
   D2Q/D2Co, or censored Counterfactual Watch Time. Play time is a training
   target only; blend raw interest logits with native-label FM.
3. Improve the proven field-gated core with field-aware pair weights or a
   small residual, rather than indiscriminately adding more metadata fields.
4. Add candidate-aware, strictly historical DIN-lite affinity and recency.

Promote only candidates improving both GAUC and nDCG@5. Prefer a small,
complementary residual or hybrid over replacing the strong FM anchor.
