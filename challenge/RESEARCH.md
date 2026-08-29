# Evidence-backed experiment queue

This queue translates primary recommender research into bounded KuaiRand-Pure
experiments. The official validation metrics and leakage policy in `task.md`
remain authoritative.

## Promotion gate

A branch becomes a championship candidate only when it:

1. raises both GAUC and nDCG@5 over the reproduced FM;
2. exceeds the current champion's primary score;
3. has the same improvement direction on at least two rolling train cutoffs;
4. remains positive across three seeds before final submission; and
5. reports wall time, API tokens/cost, failures, and manual interventions.

Weak standalone models may survive as blend-only candidates only when their
out-of-time residuals improve both component metrics.

## Priority 1: watch-time relative-interest auxiliaries

- [Relative Advantage Debiasing](https://doi.org/10.1609/aaai.v40i18.38555):
  predict train-only midrank CDF targets by video and by user×duration bin.
  Rank-normalize and conservatively blend with native-label FM.
- [D²Co](https://arxiv.org/abs/2308.08120): fit duration-conditioned watch-time
  modes and regress the corrected continuous-interest target.
- [D2Q](https://arxiv.org/abs/2206.06003) and
  [Watch-Time Gain](https://arxiv.org/abs/2208.05190): cheap duration-percentile
  and duration-standardized target ablations.
- [Counterfactual Watch Time](https://arxiv.org/abs/2406.07932): use its
  censored likelihood on an FM backbone, with play time strictly a training
  target. Rank raw interest logits, not duration-clipped predicted watch time.

RAD is first because the paper evaluates essentially the same chronological
KuaiRand-Pure split. Pure IPS/DR is deferred: the hidden benchmark uses
standard-policy impressions, `long_view` is observed for every displayed row,
and the dataset has no identifiable display-position propensity.

## Priority 2: FM-preserving metric residual

Use chronological OOF FM logits and learn `score = FM + eta * residual` with a
small residual capacity. Train with BCE plus a weak within-user
[RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
and [LambdaLoss@5](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/)
term. Pair only observed positive/short-view impressions for the same user;
never sample unexposed items. Cap pairs per user and promote only if both
metrics improve.

## Priority 3: rich field-aware interactions

The baseline ignores `user_features_pure.csv` and all but one tag. Test a
ladder—augmented FM, field-gated FM, then
[FwFM](https://arxiv.org/abs/1806.03514) or
[FFM](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)—over:

- all available tags and safe video/music/upload fields;
- user activity, registration, follow/fan/friend, and profile buckets;
- explicit 3/7/13/18/21/32/48/69/95/125/175/250/400-second duration bands;
- tab×duration, user×tag, and user×scenario interactions.

Do not promote a small single-validation gain without rolling folds: ordinary
FM with every coarse field already regressed.

## Priority 4: candidate-aware history

Implement a DIN-lite screen using the last 5/20/100 strictly earlier events:
positive/negative FM-embedding similarity, author/tag matches, and 1/3/7-day
recency decay. If fixed attention helps, move to a small
[DIN](https://arxiv.org/abs/1706.06978) residual. The final model must use
train-cutoff history for all validation rows; validation outcomes are never
rolled forward.

## Priority 5: multitask and graph follow-ups

- Shared FM embeddings with small auxiliary click, like, and corrected
  watch-interest heads; discard auxiliary heads at inference.
- FM-initialized [LightGCN](https://arxiv.org/abs/2002.02126) using only logged
  positive-versus-exposed-short-view comparisons.
- Sparse top-K item/author/tag neighborhoods before considering full neural
  sequence models.

## Validation cautions

- Public validation has a median of only four impressions per user and many
  all-negative/all-positive groups; report paired user bootstrap uncertainty.
- Target statistics must be frozen at chronological cutoffs. Training rows with
  growing within-period histories are not distribution-matched to a validation
  set whose history is frozen at the train boundary.
- Calibration is rank-invariant and is useful only for cross-model fusion.
- Ranking groups must be contiguous, and final scoring always uses the shipped
  organizer evaluator rather than a model library's default nDCG/AUC.
