# KuaiRand-Pure autonomous recommender research task

You are the model-building research agent. Begin with the organizer's immutable
FM program, generate a complete executable candidate at every iteration,
evaluate it through the trusted deterministic judge, learn from the campaign
journal, and make one bounded scientific improvement at a time.

This file defines the benchmark. The prompt supplies five separate evidence
sections: Hard constraints, Experiment memory, Optional research menu, EDA evidence,
and Literature evidence. Treat the menu as optional and let measured
evidence determine the next model family.

## Objective

The binary target is `long_view`. Higher is better for all metrics:

- `GAUC`
- `nDCG@5`
- `primary = (GAUC + nDCG@5) / 2`

The organizer FM ancestry root currently evaluates at approximately GAUC
`0.667133`, nDCG@5 `0.535805`, primary `0.601469`. First reproduce a strong
rich-FM direction independently, then improve based on recorded evidence.

A development candidate passes only when one AIDE-generated, deterministic,
full-fidelity seed-0 execution strictly exceeds all three reference values:

- GAUC `0.6710518008586268`
- nDCG@5 `0.5380142516919405`
- primary `0.6045330262752837`

A primary-only trade-off is not accepted. A competition-ready result also
requires a fresh frozen-prompt `clean` campaign with valid hashes, zero manual interventions,
and a complete lifecycle ledger.

## Exact input contract

Only these public files are available under `./input`:

- `train.csv`: 1,141,112 rows, dates 2022-04-08 through 2022-04-21
- `valid.csv`: 124,909 rows, dates 2022-04-22 through 2022-04-28
- `video_features_basic_pure.csv`
- `user_features_pure.csv`
- organizer `baseline.py`, `data.py`, and `evaluate.py`
- a label-free manifest

`train.csv` columns:

`user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow,
is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms,
profile_stay_time, comment_stay_time, is_profile_enter, tab`

`valid.csv` columns:

`user_id, video_id, date, hourmin, time_ms, long_view, duration_ms, tab`

Static video columns:

`video_id, author_id, video_type, upload_dt, upload_type, visible_status,
video_duration, server_width, server_height, music_id, music_type, tag`

Static user columns:

`user_id, user_active_degree, is_lowactive_period, is_live_streamer,
is_video_author, follow_user_num, follow_user_num_range, fans_user_num,
fans_user_num_range, friend_user_num, friend_user_num_range, register_days,
register_days_range, onehot_feat0..onehot_feat17`

## Serving-time and chronology contract

At serving time, only the validation context and static metadata are available.
Current-row engagement outcomes such as click, like, play time, profile stay,
and `long_view` are unavailable. Training outcomes may be used as auxiliary
targets or as history only when their event timestamp is strictly earlier than
the candidate timestamp.

For equal-user/equal-time groups, emit every feature before updating any state.
Never roll validation outcomes into later validation rows. The deterministic
last-three-training-days internal holdout uses a strict event-time boundary, not
a date-only mask: compute `boundary_time_ms = min(time_ms where date >= 20220419)`
and split `time_ms < boundary_time_ms` versus `time_ms >= boundary_time_ms`, keeping
all equal boundary timestamps in holdout. On the supplied training data this must
produce boundary `1650295266482`, prefix rows `1079102`, and holdout rows `62010`;
assert those values so a date-only April 19-21 split cannot silently pass. Fit
preprocessing and auxiliary state only on that training prefix. For the final
public-validation fit, rebuild those
objects using all and only `train.csv`; do not strand later-training categories
as unknown or unintentionally multiply prefix exposure. Public validation
remains external research feedback and is never used for fitting.

`video_features_statistic_pure.csv` is forbidden because its aggregation window
can leak future information. Pairwise negatives must be observed short-view
impressions from the same user, not unobserved catalog samples.

## Candidate contract

Each response must begin with the requested structured candidate card and then
contain one complete self-contained Python program. The card records exact
lineage, cited EDA/literature evidence, one scientific change, features, losses,
targeted metric, expected effects, runtime/memory, risks, abort criteria,
falsification condition, and internal-validation fidelity.

For `improve` and `refine`, preserve the selected parent's working backbone and
make one localized scientific change; do not replace the entire program. Risky
additions need a conservative residual/gate/blend, a same-split parent comparison,
and a parent-relative abort condition. A debug repair must preserve the parent's
scientific family and hypothesis. Do not import code from
`challenge/`, prior runs, reports, or frozen solutions, and never load previous
predictions, checkpoints, or weights.

Write exactly `./working/validation_predictions.csv` with header `row_id,score`,
row IDs `0..124908` in original validation order, and finite scores. Read the
campaign seed from `AIDE_SEED` and use it for every random generator. Train and
evaluate exactly once at seed 0; do not run or ensemble random seeds inside a
candidate.

Candidate execution is offline and credential-free. Use no more than four CPU
threads, less than 3 GB RAM, and 900 seconds. The external deterministic
evaluator and its aggregate diagnostics are authoritative.

## Autonomous loop

1. Reproduce the organizer FM ancestry root.
2. Read the bounded EDA, literature, experiment-memory, and current-journal evidence.
3. Choose a model family and Pareto-compatible parent using both metric components,
   diversity, runtime/API cost, failure history, and lineage compatibility.
4. State one falsifiable atomic hypothesis and implement it completely.
5. Evaluate once at seed 0 and record hashes, metrics, diagnostics, cost, and recovery.
6. Repair a bounded implementation failure without changing its science, or abandon
   the branch and select another evidence-backed family.
7. Continue until the official convergence, 50-iteration, six-hour, or cost limit.
