# Hard constraints

- Begin from the immutable organizer FM seed. Every candidate must be AIDE-generated and self-contained; never import prior candidate code or load saved predictions, checkpoints, weights, or hidden-test artifacts.
- Optimize public-validation `GAUC`, `nDCG@5`, and `primary = (GAUC + nDCG@5) / 2`. A candidate wins the development gate only when it strictly exceeds `0.6710518008586268`, `0.5380142516919405`, and `0.6045330262752837`, respectively. A primary-only trade-off fails.
- Fit model state only from `train.csv`. Public-validation labels are evaluator feedback, never features, target encodings, history state, graph edges, or training examples.
- Serving-time validation columns are `user_id, video_id, date, hourmin, time_ms, duration_ms, tab`. Current-row engagement outcomes are unavailable. Training outcomes may be auxiliary targets or strictly earlier history only.
- A history event is eligible only when its timestamp is strictly earlier than the candidate. Emit features for every equal-user/equal-time group before updating state; never roll validation outcomes into later validation rows.
- Fit vocabularies, bins, normalizers, target statistics, and graph structures on the applicable training prefix only. Pairwise negatives must be observed short-view impressions for the same user.
- `video_features_statistic_pure.csv` is forbidden. Preserve validation order and write exactly `./working/validation_predictions.csv` with `row_id,score`, row IDs `0..124908`, and finite scores.
- Read the seed from `AIDE_SEED` (default `0`) and use it for every source of randomness. Search acceptance is a single deterministic seed-0 execution; do not train or ensemble multiple random seeds.
- Candidate execution is offline and credential-free. Read only `./input`, write only `./working`, use at most four CPU threads, stay below 3 GB RAM, and finish within 900 seconds.
- Use the organizer evaluator if progress metrics are needed. The external deterministic evaluator remains authoritative.
- Make one bounded, attributable scientific change per node. Reuse working structure from the parent, state a falsification condition, and abandon or repair failures without changing the assigned hypothesis during repair.
