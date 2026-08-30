# KuaiRand-Pure autonomous recommender research task

You are the model-building research agent, not a generic code assistant. Starting
from the organizer's supplied FM program, generate a complete executable
candidate at every iteration, evaluate it, learn from the journal, and improve
within-user ranking over logged impressions.

The native binary target is `long_view`. The immutable organizer metrics are
`GAUC` and `nDCG@5`; `primary = (GAUC + nDCG@5) / 2`, and higher is better.

## Ownership and score target

- Before prompt development, the best score actually produced by an
  AIDE-generated node was only
  GAUC `0.667133`, nDCG@5 `0.535805`, primary `0.601469`.
- Experiment memory from prompt-development campaigns: AIDE independently
  generated a one-seed PyTorch 13-field gated FM reaching GAUC `0.669560`,
  nDCG@5 `0.536572`, primary `0.603066`. A simple per-field-pair FwFM variant
  regressed to GAUC `0.668860`, nDCG@5 `0.536742`, primary `0.602801`. Preserve
  the successful rich-FM structure and favor candidate-aware history or
  duration supervision next; do not load prior code, predictions, or a
  checkpoint.
- A later AIDE-generated candidate-aware chronological history residual reached
  GAUC `0.669442`, nDCG@5 `0.537444`, primary `0.603443`. This was only
  `0.000026` short of resetting the organizer's `0.002` convergence counter.
  Preserve strict train-only chronology and target the remaining GAUC gap with
  duration-relative supervision or a weak metric-aligned refinement.
- The successful history node used coarse strictly-past train-only fields:
  prior exposure and prior long-view bins, candidate video/author/primary-tag
  seen flags, and one-day exposure/long-view bins. Reuse those findings rather
  than inventing a broad history rewrite. In pandas, never access a leading-
  underscore helper column through a named `itertuples()` attribute; use
  positional tuples, arrays, or a helper name without a leading underscore.
- The newest zero-intervention development campaign independently generated a
  rich FM at GAUC `0.670553`, nDCG@5 `0.537588`, primary `0.604071`, then a
  strict-history child at GAUC `0.670702`, nDCG@5 `0.537969`, primary
  `0.604335`. The remaining gaps are only about `0.000350` GAUC, `0.000045`
  nDCG@5, and `0.000198` primary. Preserve this strong geometry; GAUC is now
  the limiting component.
- A weak RankNet/Lambda refinement from that child regressed to GAUC `0.670371`,
  nDCG@5 `0.537396`, primary `0.603883`. Do not repeat that implementation or
  simply increase its pairwise weight. A DIN-lite child failed only because its
  attention tensor had width 208 while the first linear layer expected 224;
  this is an implementation dead end, not negative scientific evidence.
- Earlier evidence found that a coarse within-user-normalized 50/50 blend of
  independently trained rich and history models raised GAUC to `0.671110`.
  Because it was trained outside the current AIDE node it does not count, but
  it makes an AIDE-generated conservative ensemble the highest-utility next
  branch. Train both members inside the generated program; never load their
  saved predictions or checkpoints.
- AIDE has now tested that ensemble autonomously. In its own campaign it reached
  GAUC `0.670102`, nDCG@5 `0.537281`, primary `0.603691`: better than that
  campaign's two members, but worse than the strongest history node. Record it
  as valid complementary evidence, not the next experiment to repeat.
- AIDE also tested DIN-lite under the official resources. The initial node and
  two repairs each timed out at 900 seconds; the final bounded repair exceeded
  3 GB and was autonomously abandoned. Treat this specific sequential-attention
  implementation as too expensive on CPU. Do not retry DIN until a cheaper
  static/history diversity branch has evidence.
- A CatBoost-focused development campaign was previously starved by the
  organizer's three-valid-non-improvement patience rule before CatBoost was
  reached. That campaign treated a rich-FM node above primary `0.6020` as
  sufficient ancestry coverage and scheduled CatBoost early; the later result
  below supersedes that old ordering.
- The focused campaign then generated an 18-field gated FM adding `music_id`,
  `music_type`, `video_type`, `upload_type`, and an aspect-ratio bucket. It
  reached GAUC `0.670800`, nDCG@5 `0.537667`, primary `0.604233`; the remaining
  gaps are only about `0.000252`, `0.000347`, and `0.000300`. Preserve those
  static fields when the parent supplies them. CatBoost exceeded memory, and
  its repairs never produced a legal score, so do not prioritize it again.
- The next AIDE campaign produced a compact DCN residual at GAUC `0.671131`,
  nDCG@5 `0.537527`, primary `0.604329`, then a strict-history child at GAUC
  `0.671493`, nDCG@5 `0.537692`, primary `0.604593`. The latter already beats
  champion GAUC and primary; it misses only nDCG@5 by about `0.000322`. Preserve
  this rich -> DCN -> history structure when the journal parent contains it.
- The following focused campaign independently produced a weaker rich parent,
  then a compact DCN at GAUC `0.670815`, nDCG@5 `0.537716`, primary `0.604266`.
  Adding more coarse history fields (seen music and duration-bucket match) then
  regressed GAUC to `0.670390` without improving nDCG. Under the organizer's
  three-valid-non-improvement rule this starved the planned metric-aligned
  branch. Do not repeat those extra history flags. The subsequent campaign
  moved the assigned narrow metric-aligned repair before history so it could be
  evaluated; its result is recorded next.
- That repair has now been tested. AIDE used one RankNet-only fine-tuning pass
  and a 0.9/0.1 within-user-normalized parent/tuned blend; it regressed to GAUC
  `0.671006`, nDCG@5 `0.537568`, primary `0.604287` from a DCN parent at
  `0.671068 / 0.537598 / 0.604333`. The apparent pairwise coefficient did not
  anchor training because pairwise loss was the only data loss in each update.
  Do not repeat this recipe or another pairwise-only pass. A future metric loss
  must mix BCE and a detached top-5 pair loss in the same update at low learning
  rate, but it is lower priority than the proven narrow history branch.
- API-free prompt-development diagnostics on the legal AIDE-generated DCN and
  history scores found complementary errors. A fixed within-user normalized
  `0.51 history + 0.39 DCN + 0.10 rich` blend reached approximately GAUC
  `0.671914`, nDCG@5 `0.538024`, primary `0.604969`. A tiny train-derived
  tag/user-author calibration also beat all three old champion components, but
  topped out around primary `0.604786`. These are research findings, not
  artifacts: generated programs may not load those predictions or checkpoints.
  The promising self-contained experiment is a single shared nested multi-exit
  model that regenerates all exits from train in one run.
- A stricter-milestone campaign then exposed why rich-FM reproduction varied.
  Its first rich node changed duration/aspect buckets and reached only primary
  `0.602819`; a refinement changed to a train-only checkpoint split, clamped
  gates, and regressed to `0.591425`. Its DCN child reached only `0.603887` and
  the campaign correctly converged. By contrast, the earlier legal 18-field
  rich node at `0.604233` used the exact transforms and optimization recipe
  frozen below. Treat any change to those mechanics as a new experiment, not
  reproduction, and do not make it before the `0.6035` milestone.
- With those mechanics frozen, the next zero-intervention development campaign
  produced a rich node at `0.670555 / 0.537538 / 0.604047`, a compact DCN at
  `0.671571 / 0.538012 / 0.604791`, and a narrow strict-history node at
  `0.671497 / 0.538101 / 0.604799`. The history node beat all three reference
  metrics on seed 0 and reproduced exactly on a seed-0 rerun. Seeds 1 and 2
  reached `0.604546` and `0.604197`; the three-seed mean was only GAUC
  `0.671134`, nDCG@5 `0.537895`, primary `0.604514`, so robustness was correctly
  rejected. Preserve this deterministic lineage but target a materially wider
  nDCG margin.
- The shared rich/DCN/history multi-exit ensemble was also generated and tested
  in that campaign. It regressed to `0.671424 / 0.537936 / 0.604680`; do not
  repeat that auxiliary-exit-loss implementation. Separate development
  diagnostics instead found that a fixed label-free per-user rank blend of
  `0.45 history + 0.45 DCN + 0.10 RAD-video` reached about GAUC `0.672082`,
  nDCG@5 `0.538263`, primary `0.605173`. No generated candidate may load those
  arrays: it must independently train the RAD component from the public train
  split and regenerate every score in one self-contained program.
- The next development campaign reproduced the frozen rich FM exactly at
  `0.670555 / 0.537538 / 0.604047`, but its DCN and history descendants drifted
  from the successful executed architecture and reached only primary
  `0.604096` and `0.603737`. Four attempts to append an independent
  LightGBM RAD-video stage each hit the 900-second timeout. Do not repeat a
  second model, refit stage, or LightGBM duration branch. The correction is to
  reproduce the exact vector-cross DCN and eight-feature history mechanics
  below, then add at most one tiny duration-relative head inside the same
  minibatch loop.
- The following zero-intervention v18 development campaign independently
  generated the corrected lineage. Its repaired DCN reached
  `0.671580 / 0.537920 / 0.604750`; its strict-history child reached
  `0.671552 / 0.537676 / 0.604614`; and its detached RAD rank-blend child
  reached `0.672141 / 0.538207 / 0.605174` on seed 0. The same generated RAD
  program reached `0.604614` and `0.604365` primary on seeds 1 and 2. Its
  three-seed mean was only `0.671459 / 0.537976 / 0.604718`, so it was
  correctly rejected. API-free component diagnostics tested fixed legal blend
  weights and found no weight-only blend that beat all three reference metrics
  on every seed. Preserve the architecture and fixed `0.45/0.45/0.10` emitted
  blend; the next atomic correction is checkpoint alignment, not weight search.
- The zero-intervention v19 development campaign showed that checkpoint
  alignment alone was insufficient. Its history node reached
  `0.671472 / 0.538095 / 0.604784` on seed 0 and reproduced exactly, but seeds
  1 and 2 reached only primary `0.604617` and `0.604216`; the three-seed mean
  was `0.671165 / 0.537913 / 0.604539`. The checkpoint-aligned RAD child reached
  only `0.671559 / 0.537945 / 0.604752` on seed 0. Preserve the fixed blend and
  checkpoint correction for later, but first make the single history-node
  learning-rate stabilization below. Do not search blend weights or add a new
  model in the same experiment.
- A separately orchestrated reference reached GAUC `0.671052`, nDCG@5
  `0.538014`, primary `0.604533`. It is evidence, not an artifact you may load.
- Your candidate counts as a breakthrough only when its generated code exceeds
  **all three** reference values. A primary trade-off is not acceptable.
- Never import, read, reconstruct from saved predictions, or call code from
  `challenge/`, `reports/`, prior `runs/`, or any champion checkpoint. Your
  ancestry must remain the organizer FM seed plus your own generated code.

## Exact data contract

Only these files exist under `./input`:

- `train.csv`: 1,141,112 rows dated 2022-04-08 through 2022-04-21.
- `valid.csv`: 124,909 rows dated 2022-04-22 through 2022-04-28.
- `video_features_basic_pure.csv`, `user_features_pure.csv`.
- Organizer `baseline.py`, `data.py`, and `evaluate.py`.
- A label-free manifest. Hidden-test rows and champion artifacts are absent.

Exact `train.csv` columns:

`user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow,
is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms,
profile_stay_time, comment_stay_time, is_profile_enter, tab`

Exact `valid.csv` columns:

`user_id, video_id, date, hourmin, time_ms, long_view, duration_ms, tab`

Static video columns:

`video_id, author_id, video_type, upload_dt, upload_type, visible_status,
video_duration, server_width, server_height, music_id, music_type, tag`

Static user columns:

`user_id, user_active_degree, is_lowactive_period, is_live_streamer,
is_video_author, follow_user_num, follow_user_num_range, fans_user_num,
fans_user_num_range, friend_user_num, friend_user_num_range, register_days,
register_days_range, onehot_feat0..onehot_feat17`

## Hard protocol

- Fit only on `train.csv`; public validation labels are for evaluation and early
  stopping, never direct features or target encodings.
- Current-row engagement outcomes (`is_click`, `is_like`, `play_time_ms`, and
  related fields) do not exist at serving time. They may be auxiliary training
  targets or strictly earlier history values only.
- A historical event is eligible only when its timestamp is strictly less than
  the candidate timestamp. Exclude equal timestamps. Never roll validation
  outcomes forward into later validation rows.
- `time_ms` is already the event timestamp; do not add `date * 1e9` to it. When
  generating training history, emit features for every row in an equal-user,
  equal-time group before updating state for that group, so equal-time labels
  cannot leak through row order.
- Fit vocabularies, duration bins, normalization, target statistics, graphs,
  and OOF features on the relevant training prefix only.
- Pairwise negatives must be short-view impressions actually exposed to the
  same user; never sample unobserved catalog items.
- `video_features_statistic_pure.csv` is forbidden because its aggregation
  window risks future leakage.
- Preserve validation order. Write exactly
  `./working/validation_predictions.csv` with header `row_id,score`, row IDs
  `0..124908`, and finite scores.
- Read the base seed from `AIDE_SEED` with default `0`. Use it for every source
  of randomness so the exact program can be confirmed across seeds.
- Use at most four CPU threads, less than 3 GB peak memory, and 15 minutes.
- The external deterministic evaluator is authoritative.
- If candidate code imports the organizer evaluator for progress logging, add
  `./input` to `sys.path` **before** `from evaluate import evaluate`. Do not
  implement a substitute metric. The required prediction file is still the
  only interface used for scoring.

## First milestone: independently reproduce rich FM

Before speculative branches, implement a CPU-efficient rich field-gated FM
that preserves the organizer FM mechanics. The prior evidence-backed starting
configuration is deliberately stated as research memory, not reusable code:

- fields: `user_id, video_id, author_id, tab, duration_bucket,
  duration_rule_band, tab_duration_cross, hour, primary_tag, tag_2,
  user_active_degree, follow_user_num_range, register_days_range`;
- evidence-backed static extension: retain the core above and add `music_id`,
  `music_type`, `video_type`, `upload_type`, and a coarse aspect-ratio bucket
  when they fit the resource budget. This 18-field form has been materially
  stronger than settling for the 13-field milestone alone;
- use this exact field order:
  `user_id, video_id, author_id, tab, duration_bucket, duration_rule_band,
  tab_duration_cross, hour, primary_tag, tag_2, user_active_degree,
  follow_user_num_range, register_days_range, music_id, music_type, video_type,
  upload_type, aspect_ratio_bucket`;
- exact duration-bucket millisecond edges:
  `[0, 3000, 7000, 12000, 20000, 35000, 60000, 120000, 1000000000]`,
  using right-sided insertion minus one;
- exact duration-rule bands in seconds: `<=8`, `(8,18]`, `(18,35]`,
  `(35,70]`, and `>70`;
- parse tags after replacing `|` by `,` and removing spaces; use the first two
  nonempty tags with an unknown fallback;
- exact aspect-ratio bands: missing/invalid bucket, `<0.8`, `[0.8,1.2)`,
  `[1.2,1.8)`, and `>=1.8`;
- train-fitted uncapped per-field vocabularies with one unknown slot. Prefer a
  single offset embedding table for all fields so initialization and FM
  arithmetic match the strongest legal node;
- FM dimension 16, batch 8192, at most 40 epochs, patience 4;
- use `torch.optim.Adam(lr=0.001, weight_decay=1e-6)` and add only gate L2
  `1e-3` to BCE. Do not replace optimizer weight decay with a separately
  summed all-embedding penalty in the reproduction node;
- first five unconstrained field gates initialized to 1.0 and all other gates
  to 0.1. Do not clamp, sigmoid, or otherwise reparameterize them;
- initialize embedding weights from normal standard deviation `0.01`, linear
  weights to zero, seed every RNG from `AIDE_SEED`, and early-stop/checkpoint on
  full public-validation primary each epoch. Public validation is explicitly
  permitted for feedback and checkpoint selection; it is never a training
  feature or target-encoding source;
- patience 4. During search, train **one seed only** (`AIDE_SEED`) and emit that
  seed's predictions; never train or average seeds 0/1/2 inside a search node.
  The generic runner performs the three-seed confirmation only after a node has
  crossed the champion. Prefer vectorized PyTorch sparse embeddings over Python
  loops or repeated `numpy.add.at` updates, and reserve at least two minutes of
  the 15-minute timeout for prediction writing and deterministic review.

This direction must reach primary `0.6035+` before portfolio expansion so it
resets the organizer convergence counter and leaves room for the evidence-backed
DCN, history, and ensemble sequence. Implement it yourself;
do not attempt another generic historical LightGBM branch before establishing
this milestone.

## Research portfolio after the milestone

Choose the assigned family when it is scientifically viable; otherwise explain
why a different atomic experiment has higher expected utility.

1. `fwfm`: one regularized scalar per field pair over the rich FM.
2. `history_residual`: candidate-aware last 5/20/64 histories, 6-hour and
   1/3/7-day recency, video/author/tag/music/duration matches, positives and
   exposed short views.
3. `duration_auxiliary`: RAD, D2Q/D2Co, or censored CWM supervision. Current
   play time is a training target only; rank predicted interest, not clipped
   predicted watch time.
4. `metric_aligned`: weak same-user RankNet or LambdaLoss@5 combined with BCE.
5. `din_lite`: last-32 candidate-conditioned attention; add dense click and
   duration auxiliary heads only after single-task DIN works. For the first
   viable node, preserve the rich/history FM logit as an anchor, use the last
   16 strictly earlier train events, 16-dimensional candidate/history
   embeddings, and attention inputs consisting of history, candidate,
   difference, product, and log-age. Derive the first linear layer's input
   width from the tensors actually concatenated instead of hard-coding 224.
   Use a small residual coefficient initialized near 0.05 and bounded below
   0.30; single-task long-view BCE only, AdamW near 5e-4, dropout 0.1, gradient
   clipping 5, and no duration/click head in the same node.
6. `ensemble`: conservative OOF combination using within-user normalization
   and fixed/coarse weights selected without fine-searching public validation.
   The preferred atomic experiment is to train the rich and corrected-history
   members independently in one execution, within-user z-normalize their
   outputs, and use the preregistered 50/50 blend.
7. `dcn_v2`, `lightgcn`, or `catboost`: bounded diversity branches only after
   the higher-probability directions have evidence. CatBoost is now the
   preferred unexplored branch: train it from public train rows inside the
   candidate, use only serving-time/static plus strict-history features, set
   `thread_count=4`, bound depth/iterations for the 15-minute limit, and combine
   it conservatively with a freshly trained FM score. If the FM score is used
   as a training feature, obtain it out-of-fold; otherwise use a preregistered
   small blend rather than leaking in-sample FM predictions.

For the current focused portfolio, `dcn_v2` is the preferred post-rich branch.
The family name is retained for the ledger, but exact reproduction must use the
executed v16 architecture rather than interpreting the old prose as a low-rank
DCN-V2 matrix. Keep the FM logit as an anchor. Flatten the 18 gated
16-dimensional field embeddings into `x0` of width 288. Apply two standard
vector-weight cross layers
`x_next = x0 * sum(x_current * w, dim=1, keepdim=True) + b + x_current`,
with both `w` and `b` initialized to zero. In parallel, apply a
`288 -> 128 -> 64` ReLU tower with dropout `0.1` after both hidden layers.
Concatenate the 288-wide cross output and the 64-wide tower output, project it
to one residual logit, and add it to the FM logit through
`0.30 * sigmoid(raw_scale)`. Initialize `raw_scale` to
`-2.9444389791664403`: sigmoid itself starts near `0.05`, so the actual
multiplier starts at `0.015`, not `0.05`. Xavier-initialize tower/residual
weights and zero their biases. Use the rich reproduction's Adam, BCE plus gate
L2, batching, and full-validation checkpoint protocol unchanged. The successful
node reached `0.6715706862 / 0.5380115527 / 0.6047911194` at epoch 8 with an
actual residual multiplier near `0.01834`. Do not add history, auxiliary heads,
or another model in this node. Bound the generated implementation at 12 epochs
with early-stopping patience 4 and vectorize categorical encoding with
`Series.map(...).fillna(0)` rather than Python row loops. This is the verified
runtime-preserving autonomous repair: it completed in about 412 seconds,
whereas the equivalent 40-epoch attempt exceeded the 900-second limit.

Immediately after a valid DCN node, prioritize the narrow `history_residual`
that previously improved both metrics. Retrain the same rich-FM plus exact DCN
from scratch and add exactly eight train-standardized values in this order:
cumulative prior-exposure count bin, cumulative prior-long-view count bin,
time-since-most-recent-long-view bin, seen-candidate-video flag, seen-author
flag, seen-primary-tag flag, trailing-24-hour exposure count bin, and
trailing-24-hour long-view count bin. Map counts `0`, `1`, `2..3`, `4..7`,
`8..15`, `16..31`, and `32+` to `0..6`; map recent long-view age `<6h` to 3,
`6..24h` to 2, `1..3d` to 1, and older/no prior to 0. Standardize using train
means and standard deviations only. Feed the eight values through
`8 -> 32 -> 1` with ReLU and add that residual atop the DCN exit through its
own `0.30 * sigmoid(raw_scale)` gate, again initialized with raw value
`-2.9444389791664403` for an actual starting multiplier of `0.015`. Use only
final-history BCE plus the existing gate L2; do not add auxiliary-exit losses.
The successful node reached `0.6714967417 / 0.5381014351 / 0.6047990884` at
epoch 5, with DCN and history multipliers near `0.01813` and `0.02289`. Score
and retain the FM, DCN, and final exits for later legal in-program blending.
Do not add the regressive seen-music or duration-bucket-match flags. Use strict
equal-`(user_id,time_ms)` buffering, freeze train state for validation, and
never roll validation outcomes forward.

Atomic v20 stabilization: preserve this history-residual program exactly,
including all 18 categorical fields, eight strict-history values,
FM/DCN/history heads, initializations, BCE plus the existing gate penalty,
Adam, batch sizes, and primary-based best-checkpoint/early-stop patience 4.
Change only learning-rate control. Attach
`ReduceLROnPlateau(mode="max", factor=0.25, patience=0, threshold=1e-5,
threshold_mode="abs", cooldown=0, min_lr=2.5e-4)` to the existing Adam and
step it once after each full validation-primary evaluation. The first epoch
that fails to improve primary by more than `1e-5` must reduce every Adam
parameter-group learning rate from `1e-3` to `2.5e-4` exactly once and retain
that rate thereafter. Preserve optimizer moments, do not reset `bad_epochs`,
and log the learning rate each epoch. Make no other model, feature, loss,
regularization, initialization, seed, checkpoint, or serving change in this
node.

After a valid narrow-history node, prioritize one `duration_auxiliary` atomic
experiment using RAD-video as a diversity component. Preserve the exact
history model, its BCE path, and its returned final-history and DCN logits.
From `train.csv` only, construct once, vectorially, the float32 RAD target as
the midrank percentile of `play_time_ms` within each `video_id`:
`(average_rank - 0.5) / group_size`; fill missing play time with zero and use
`0.5` for a singleton video. `play_time_ms` is a training target only and must
never enter validation features.

Add only one tiny RAD head inside the existing PyTorch program and minibatch
loop. Feed it detached already-computed representations, preferably the
64-wide DCN MLP output concatenated with the eight standardized history
values, then one linear `72 -> 1` projection. Detachment is mandatory: RAD
gradients may update the tiny head but must not alter the FM, DCN, or history
parameters. In the same optimizer step add
`0.10 * SmoothL1(sigmoid(rad_logit), rad_target, beta=0.10)` to the unchanged
main BCE and gate L2. There must be one forward/backward pass per batch, one
checkpoint, no second optimizer, no second training phase, no LightGBM, and no
independent model. Compute the target once rather than once per epoch. Keep
batch 8192, validation batch 32768, `num_workers=0`, four Torch threads, and
reserve at least 120 seconds for final evaluation and writing.

At inference, convert the history logits, DCN logits, and RAD prediction to
deterministic average percentile ranks separately within each validation user,
using row order to break unresolved ties. Emit exactly
`0.45*rank(history) + 0.45*rank(dcn) + 0.10*rank(RAD-video)`. The weights are
fixed experiment memory, not a validation search. Do not add RAD-user,
pairwise loss, new history fields, or any other change in this node. Do not load
ancestor predictions, checkpoints, prior-run arrays, or repository scripts;
the full generated program must train and predict every component itself.

During every validation pass, collect the history, DCN, and RAD outputs,
convert each to the same deterministic within-user average percentile ranks,
form the exact fixed `0.45/0.45/0.10` prediction that will be written, and
checkpoint on the exact emitted blend's public-validation primary. Use minimum
improvement `1e-5` and patience 4. Do not select on history-only logits. This is
the only scientific change in this node: do not alter the architecture,
features, losses, optimizer, initializations, 12-epoch cap, blend weights, or
rank convention while correcting checkpoint selection.

The previously tested shared `ensemble` may be revisited only with new evidence;
its first auxiliary-exit implementation regressed both metrics.

Only after those branches have evidence may `metric_aligned` return as a narrow
top-5 repair. Preserve the parent BCE checkpoint/prediction; sample only
same-user exposed long-view versus short-view train pairs, at most four pairs
per user and at most 100,000 pairs. Use logits directly and add either RankNet
or detached Lambda@5 weighting with coefficient `0.03..0.05` for one short
fine-tuning epoch. Pairwise loss must be mixed with BCE in the same optimizer
updates; a coefficient on pairwise-only loss is not an anchor. Produce a
conservative preregistered blend dominated by the
untuned parent (for example 0.9 parent + 0.1 tuned after within-user
normalization). Do not add a new architecture, hard-negative mining, or a large
pairwise weight. The goal is nDCG@5 `+0.0004` while keeping GAUC above
`0.671052`.

Known dead ends: naive hard-negative BPR, coarse sequence SVD, broad static
LightGBM, blindly adding all metadata to ordinary FM, and changing only the FM
embedding dimension. Do not repeat them.

## Iteration behavior

- Make one bounded, attributable change relative to the supplied parent.
- Reuse the parent's correct loader, evaluator, output contract, and successful
  model structure instead of rewriting everything unnecessarily.
- Debug a failed implementation without changing its scientific hypothesis.
- Prefer the candidate with the highest probability of improving both metrics,
  then expected minimum component gain, ranking diversity, runtime, and risk.
- A weak standalone model survives only if a preregistered small OOF blend
  improves both metrics.
- Print useful progress, but always finish and write the prediction file.
