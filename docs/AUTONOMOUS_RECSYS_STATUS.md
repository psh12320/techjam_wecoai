# Autonomous KuaiRand Recommender Research: Status and Handoff

Last audited: 2026-08-30 (Asia/Singapore), after the GPT-5.6 Sol v23 smoke and v24 six-iteration development dry-run. All metrics below were read from repository journals or deterministic reports. No result is described as competition-ready unless a fresh `clean` campaign says so.

## Current verdict

The benchmark-specific AIDE harness is functional and has produced several valid single-seed improvements over the old reference champion. The strongest AIDE-generated single run is the v18 RAD/duration node at GAUC `0.6721409447313641`, nDCG@5 `0.5382066966715908`, primary `0.6051738207014774`. The v19 strict-history node reached GAUC `0.6714723459645795`, nDCG@5 `0.5380948554927942`, primary `0.6047836007286869`.

Neither result is a fresh frozen-prompt clean-run acceptance. Under the pre-reset three-seed rule, both were rejected. Every manifest-backed campaign found in this audit is `development`, not `clean`. The most recent v20 campaign is explicitly **interrupted and incomplete**: it has no `resource_summary.json` or `run_completed` event, and its event ledger records a user-directed interruption.

The durable API tracker at [`challenge/private/api_spend.json`](../challenge/private/api_spend.json) reports exactly:

- cumulative estimated cost: **`$10.243473`**;
- requests: `96`;
- input tokens: `951212`;
- output tokens: `502675`;
- next notification boundary: `$20`.

This tracker is machine-private and ignored by Git. The amount above was read from the tracker, not reconstructed from conversation history.

## Post-checkpoint reset implementation

Phase 1 was committed and pushed to `origin/techjam` as `04a8d32` (`checkpoint: autonomous KuaiRand AIDE research harness`). After that push, the reset was implemented without a paid API call:

- acceptance is now exactly one full-fidelity deterministic seed-0 execution; seeds 1 and 2, robust-mean thresholds, confirmation reservations, and the confirmation CLI were removed;
- `task.md` is a thin benchmark contract, while hard constraints, ledger-derived experiment memory, the optional method menu, EDA evidence, and literature evidence are separate bounded prompt sections;
- candidate cards now require exact parent ID/code hash, EDA and literature citation IDs, one scientific change, hypothesis, features/losses, targeted metric effects, runtime/memory, risks, abort/falsification criteria, fidelity, and the internal-validation protocol;
- the real label-safe EDA found train long-view rate `0.33662`, cold-user rows `1.593%`, user-tag prior support `77.44%`, median/p90 validation history `51/141`, median/p90 candidates per user `4/12`, median watch ratio `0.1026`, completion share `15.23%`, and hour-distribution TVD `0.0422`;
- nine frozen primary-source literature notes cover FwFM, DIN, D2Q, D2Co, censored watch time, LambdaLoss, DCN-V2, LightGCN, and conservative ensemble selection; generated programs remain offline and receive citations, not external code or weights;
- the scheduler now maintains a GAUC/nDCG/primary Pareto frontier and scores family-parent pairs using weaker-component gain, floor preservation, exploration, prediction diversity, runtime/API cost, timeouts/failures, parent-dominated/near-duplicate/public-tuning penalties, and lineage compatibility;
- internal validation is anchored on the last three labeled training days but split at a strict event-time boundary because KuaiRand date labels overlap slightly in epoch time; all equal timestamps go to holdout;
- the trusted reviewer writes aggregate-only segment/top-5/diversity diagnostics and no row-level validation labels or predictions.

An API-free v21 baseline-only campaign completed with `0` requests and `$0` incremental cost. It reproduced GAUC `0.6671326321610643`, nDCG@5 `0.5358048805448538`, primary `0.601468756352959`, wrote valid aggregate diagnostics, recorded prompt hash `8618a8a433ad2461d55eb110f86ff4ad3d1095ff7cc7a038ca49c5b40dc2e046`, and used internal-validation hash `6fac0c1b9244fc9f31cf3c46e2201e45e5fff17e7ac64a108c492c902b0d759a`. This verifies the reset harness, not a winning clean candidate.

The paid GPT-5.6 Sol v21 smoke (`techjam-aide-v21-sol-smoke`) completed with two AIDE requests, zero manual interventions, and `$0.37668` incremental spend. Both generated candidates were valid, card-complete, lineage-bound rich-FM programs, but both regressed: node `236f4cbc597c44dc8e8edadf5d27eae0` scored GAUC `0.6454378172167383`, nDCG@5 `0.5261356050437236`, primary `0.585786711130231`; node `bd07139641ac4ba9ab6ad761187e2c31` scored `0.6359918380674021`, `0.5218392646763755`, `0.5789155513718889`. The run is a successful harness smoke and a rejected scientific result.

The audit found that v21 forced two near-duplicate wholesale rich-FM replacements, retained prefix-only vocabularies for the final fit, discarded the proven optimizer/checkpoint recipe, and changed multiple mechanisms simultaneously. Prompt/scheduler v22 now records these nodes as parent-dominated memory, requires a single declared change scope plus preserved parent components, distinguishes internal-prefix preprocessing from an all-training final refit, penalizes parent-dominated families, and permits a second forced rich-FM refinement only when the first attempt improves both metric components. The v22 experiment-memory hash is `0d306458efc85afd6bd27e5d61922c106d41a2e8849144ec5b66dff8b4094e2a`.

The v22 smoke (`techjam-aide-v22-sol-smoke`) spent `$0.371544` over two requests and produced no evaluated generated candidate. Its first card was valid and parent-preserving, but the code passed a CSR one-hot matrix to the organizer `baseline.FM` integer-index API and failed with `IndexError` after five seconds. GPT-5.6 Sol's second response correctly diagnosed the representation bug, but rephrased the locked scientific metadata and was rejected before compute. V23 now publishes the dense-integer organizer API contract, gives debug prompts the exact locked card values to copy verbatim, validates hypothesis and change scope as well as family/change, and freezes both v22 failures into experiment memory hash `7441d07d95782c585266441d14891eadd2caf99071a0c1478b85639afc977ddc`.

The v23 smoke (`techjam-aide-v23-sol-smoke`) spent `$0.393064` over two requests and completed with zero manual interventions. Its first AIDE-generated rich-FM node (`8e7f3fe7d59d4d148c34937032602d4c`) scored GAUC `0.6670841339220689`, nDCG@5 `0.5359480054028259`, primary `0.6015160696624474`: a primary gain of `0.0000473133094884` over the organizer FM, with nDCG improving by `0.0001431248579721` but GAUC regressing by `0.0000484982389954`. The second history-residual node (`5b39a9ae0baf41358e4e0ea42c04211f`) correctly disabled its user-primary-tag posterior residual after the internal parent gate failed and reproduced the first node exactly. This validates non-regressive autonomous fallback, but neither node passes the both-component champion gate. Independent audit found that both programs used a date-only internal split with `695` rows misplaced relative to the canonical strict timestamp boundary. V24 now requires and asserts boundary `1650295266482`, prefix rows `1079102`, and holdout rows `62010`. It also reserves the two newest evaluated descendants in experiment-memory selection and compacts all 16 selected summaries into the actual prompt, preventing the next paid run from omitting the v23 tradeoff/fallback. The corrected memory hash is `1d069625f4c483484cad76a37699d636572d81bbbdc7915a6a55f5eee6e5131e`; the v24 six-iteration development prompt hash is `6035c5eccecdba4801976b345b7967579dcfeb67f89f20f104ee7a632f28a9d4`.

## Competition objective and official constraints

The task is within-user ranking over KuaiRand-Pure logged impressions. The native target is `long_view`. The deterministic metrics are GAUC and nDCG@5, with:

`primary = (GAUC + nDCG@5) / 2`

The current quality gate is strict improvement on the same generated prediction over all three reference values:

| Metric | Required floor |
|---|---:|
| GAUC | `0.6710518008586268` |
| nDCG@5 | `0.5380142516919405` |
| Primary | `0.6045330262752837` |

The organizer's published validation baseline is GAUC `0.6674`, nDCG@5 `0.5357`, primary `0.6016`. The exact organizer FM seed as executed by this harness is GAUC `0.6671326321610643`, nDCG@5 `0.5358048805448538`, primary `0.601468756352959`.

Official and repository-enforced limits are:

- train on the public training split and use public validation only for permitted evaluation/feedback; hidden test is unavailable during development;
- no external training data, benchmark-test outcomes, pretrained benchmark weights, prior predictions, champion checkpoints, or champion implementation code;
- begin ancestry from the organizer FM seed and generate complete, self-contained candidate programs;
- at most 50 iterations, six hours wall-clock, and the organizer's stop rule of three consecutive iterations without a validation-primary improvement greater than `0.002`;
- at most four CPU threads, less than 3 GB peak memory, and 15 minutes (`900` seconds) per candidate;
- emit exactly `working/validation_predictions.csv` with columns `row_id,score`, row IDs `0..124908` in original order, and finite scores;
- hidden-test ranking is computed once from the final designated submission.

The shipped evaluator and detailed benchmark contract use GAUC and nDCG@5; stale references elsewhere to other metrics are not authoritative.

## Generic AIDE versus this research agent

Generic AIDE is a code-tree search engine: an LLM drafts, improves, or debugs programs; executions become journal nodes; the search policy chooses a parent; and good or failed branches remain in the tree.

This repository wraps that generic mechanism in a KuaiRand-specific research protocol:

- the organizer FM is executed locally as the immutable root before any paid iteration;
- all draft, improve, and debug prompts receive the benchmark description, exact schema, data overview, installed environment, resource limits, prediction contract, experiment memory, and a scheduler assignment;
- candidates must return a structured candidate card before their complete Python program;
- a portfolio scheduler tracks model-family attempts/successes, failure/repeat penalties, expected signal priors, and required rich-FM -> DCN -> history -> duration lineage;
- generated predictions are scored by the organizer-compatible deterministic evaluator, not by an LLM reviewer;
- failures are fed back to bounded autonomous debug attempts without changing the scientific family;
- every trial is recorded with lineage, hashes, metrics, resource/cost data, scheduler decisions, and recovery state;
- untrusted candidate code runs in a restricted local process with no network and no child-process creation.

The current v22 scheduler is Pareto-, diversity-, component-, failure-, and API-cost-aware. Hard constraints, factual experiment memory, EDA, literature, and the optional research menu are separate hashed prompt sections.

## Repository architecture

| Path | Role |
|---|---|
| [`challenge/task.md`](../challenge/task.md) | Benchmark prompt, exact schema/protocol, research memory, model menu, and current prescriptive recipes. |
| [`challenge/run_aide_research.py`](../challenge/run_aide_research.py) | Campaign runner: dry-run gate, paid budgets, FM root, scheduling, execution, deterministic review, confirmations, manifests, ledgers, summaries, and acceptance. |
| [`challenge/agent_seed.py`](../challenge/agent_seed.py) | Immutable organizer-FM ancestry program used for node zero. |
| [`challenge/prepare_agent_data.py`](../challenge/prepare_agent_data.py) | Builds the candidate-visible public bundle and the evaluator-only validation index outside candidate input. |
| [`challenge/techjam_recsys/aide_portfolio.py`](../challenge/techjam_recsys/aide_portfolio.py) | Prompt version, candidate-card parser, source safety checks, family normalization, utility scheduler, and lineage selection. |
| [`challenge/techjam_recsys/aide_reviewer.py`](../challenge/techjam_recsys/aide_reviewer.py) | Validates the prediction contract, calls deterministic metrics, records candidate artifacts, and reports component deltas. |
| [`challenge/techjam_recsys/protocol.py`](../challenge/techjam_recsys/protocol.py) | Metric thresholds, official convergence logic, hash-chained `TrialRecord` ledger, intervention count, and champion selection. |
| [`challenge/techjam_recsys/campaign_safety.py`](../challenge/techjam_recsys/campaign_safety.py) | Candidate-input allowlist, source/input/dependency/evaluator fingerprints, campaign manifest, and append-only lifecycle/intervention evidence. |
| [`aide/interpreter.py`](../aide/interpreter.py) | Generated-code process, filesystem audit hook, network/process denial, timeout, memory enforcement, and descendant cleanup. |
| [`aide/agent.py`](../aide/agent.py) | Generic draft/improve/debug prompt flow; this fork includes environment and data overview in all three prompt types. |
| [`challenge/techjam_recsys/metrics.py`](../challenge/techjam_recsys/metrics.py) | Organizer-compatible GAUC/nDCG evaluation and within-user rank normalization. |
| [`challenge/requirements.txt`](../challenge/requirements.txt) | Deterministic model/evaluation dependencies. |
| [`challenge/requirements-agent.txt`](../challenge/requirements-agent.txt) | Paid-agent, PyTorch, CatBoost, and process-monitoring dependencies. |
| [`challenge/tests`](../challenge/tests) | Static tests for prompts/scheduler, safety, protocol/ledger, and submission behavior. |

Generated data, private evaluator state, API accounting, and run artifacts live under ignored `challenge/agent_data/`, `challenge/private/`, and `challenge/runs/` directories.

## Changes made to AIDE

### Benchmark, schema, and environment prompts

`KuaiRandAgent` injects the real row counts, columns, task semantics, serving-time exclusions, chronology rules, prediction contract, four-thread/3-GB/900-second limits, and installed-package versions. Generic `aide/agent.py` now adds environment and data overview to draft, improve, and debug prompts rather than drafts alone. Code output is allowed up to 10,000 tokens per call by default so complete neural candidates are less likely to be truncated.

### Structured candidate cards

The response must begin with a `<candidate_spec>` JSON object. The parser persists normalized model family, declared family, features, losses, hyperparameters, runtime estimate, risks, expected metric effects, parse status, and assigned family. These values appear in the experiment ledger instead of every node being labeled generically as `improve`.

The v22 card additionally requires one `change_scope` and a list of `preserved_parent_components`, along with the parent code hash, EDA/literature IDs, exact falsifier, memory estimate, abort criteria, and explicitly targeted metric.

### Portfolio and lineage scheduler

The current deterministic scheduler:

- attempts one parent-relative rich-FM milestone before portfolio expansion, allowing one immediate refinement only after both components improve;
- tracks attempts, parent improvements, parent-dominated results, valid nodes, runtime, cost, diversity, failures, and timeouts per family;
- combines family priors, exploration, Pareto/component evidence, floor preservation, parent compatibility, and regression/failure/repeat/cost penalties;
- autonomously repairs the most recent debuggable leaf up to the configured depth;
- chooses a Pareto-compatible parent while allowing evidence-backed family re-entry after cooldown.

### Deterministic evaluator

Candidates never receive validation labels through their input bundle. The evaluator-only index is stored under `challenge/private/evaluator/`. The reviewer checks exact headers, row count/order, finite scores, and obvious label copying, then calls the organizer-compatible metric implementation. Evaluation makes no API call.

### Experiment ledger and hashes

Each JSONL trial records model family, structured configuration, hypothesis, parent trial, code diff, node ID, code SHA-256, prompt version/hash, campaign manifest hash, source/input/dependency hashes, seed, exact metrics, error and recovery state, scheduler action/utility/alternatives, LLM tokens, end-to-end wall time, bounded candidate execution time, manual-intervention count, prediction artifact IDs/hashes, and exit status. Records are append-only and hash chained through `previous_record_sha256` and `record_sha256`.

### Autonomous repair

Failed candidates become debuggable leaves. The scheduler prioritizes a bounded repair while requiring the same scientific hypothesis and family. The ledger distinguishes `repair_pending`, `repair_failed`, `promoted_after_repair`, and `abandoned`. This recovered, for example, the v18 DCN timeout and the fourth v20 history attempt.

### Sandbox and leakage protection

Candidate input must match a fixed public allowlist. Source validation blocks imports and references associated with the repository, champion artifacts, network clients, process creation, dynamic execution, absolute/parent paths, and host CPU-count discovery. At runtime, a Python audit hook restricts reads/writes to the candidate workspace, denies socket operations and child-process creation, and permits writing only candidate work products. Campaign code, input, dependencies, and private evaluator are fingerprinted and checked for drift during a run.

These controls are defense in depth, not a formal proof of non-leakage. A final review of the generated winner remains necessary.

### Runtime and API cost tracking

The runner separately records end-to-end trial latency and interpreter-reported execution time capped at the 900-second trial limit. It enforces a six-hour wall cap, 3-GB resident-memory cap, and descendant cleanup. Paid execution requires `--execute`, explicit current pricing, token ceilings, and a run-dollar ceiling. Cost is appended to the durable private tracker after each completed API response. Notifications occur only at cumulative `$10` multiples.

### Clean-campaign evidence

A campaign manifest hashes the prompt-determining source, copied input, dependency pins, and evaluator. The append-only event ledger records exactly one start, a completion, and any human intervention. Under the current pre-reset code, clean acceptance additionally requires `campaign_mode=clean`, zero intervention events, valid matching lifecycle evidence, and a successful three-seed confirmation. Phase 2 will replace that last condition with the requested deterministic seed-0 single-run gate.

No audited campaign has `clean_campaign_accepted=true`.

## Exact data schema and leakage rules

Candidate-visible files are `train.csv`, `valid.csv`, `video_features_basic_pure.csv`, `user_features_pure.csv`, organizer `baseline.py`, `data.py`, `evaluate.py`, and a label-free manifest. Hidden-test rows and champion artifacts are absent.

### `train.csv`

`1,141,112` rows, 2022-04-08 through 2022-04-21:

```text
user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow,
is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms,
profile_stay_time, comment_stay_time, is_profile_enter, tab
```

### `valid.csv`

`124,909` rows, 2022-04-22 through 2022-04-28:

```text
user_id, video_id, date, hourmin, time_ms, long_view, duration_ms, tab
```

### Static video table

```text
video_id, author_id, video_type, upload_dt, upload_type, visible_status,
video_duration, server_width, server_height, music_id, music_type, tag
```

### Static user table

```text
user_id, user_active_degree, is_lowactive_period, is_live_streamer,
is_video_author, follow_user_num, follow_user_num_range, fans_user_num,
fans_user_num_range, friend_user_num, friend_user_num_range, register_days,
register_days_range, onehot_feat0..onehot_feat17
```

### Serving-time and chronological rules

- `long_view` is the native label, not a feature.
- Current-row outcomes such as click, like, follow, comment, forward, hate, play time, profile/comment stay, and profile entry are unavailable when ranking. They may be training-only auxiliary targets or strictly earlier historical values.
- Validation outcomes are evaluation-only: never model features, target encodings, rolling history, or graph edges.
- An event can enter history only when its `time_ms` is strictly smaller than the candidate's. Equal-user/equal-time rows must all be featurized before state is updated.
- `time_ms` is already the timestamp; do not combine it with `date * 1e9`.
- Fit vocabularies, bins, normalization, target statistics, graphs, and OOF features only on the relevant training prefix.
- Pairwise negatives must be exposed same-user short-view impressions, not unobserved catalog items.
- `video_features_statistic_pure.csv` is forbidden because its aggregation window can leak future information.
- Preserve validation row order and never access hidden outcomes or frozen champion artifacts.

## Installed environment and resource limits

The audited virtual environment reports Python `3.12.13`. Direct dependencies are pinned as follows.

Deterministic stack:

```text
numpy 1.26.4; pandas 2.2.3; scipy 1.14.1; scikit-learn 1.5.2;
lightgbm 4.5.0; optuna 4.1.0; omegaconf 2.3.0;
python-dotenv 1.0.1; pytest 8.3.4
```

Agent/research stack:

```text
black 25.1.0; funcy 2.0; humanize 4.13.0; jsonschema 4.25.1;
openai 1.109.1; anthropic 0.67.0; rich 14.1.0;
dataclasses-json 0.6.7; loguru 0.7.3; shutup 0.2.0; tqdm 4.67.1;
coolname 2.2.0; igraph 0.11.9; genson 1.3.0; requests 2.32.5;
backoff 2.2.1; torch 2.8.0; catboost 1.2.8; psutil 7.0.0
```

Candidate limits remain four CPU threads, less than 3 GB resident memory, no GPU requirement, no network, no subprocesses, and 900 seconds per execution.

## Metric history from actual ledgers

All AIDE rows below come from `journal.json`/`iterations.jsonl` under the named ignored run directory. “Single-run gate” means the one prediction beat all three old champion values. It does not mean clean acceptance.

| Campaign / node | Family or event | GAUC | nDCG@5 | Primary | Outcome |
|---|---|---:|---:|---:|---|
| All recent runs / organizer root | Official FM seed | `0.6671326321610643` | `0.5358048805448538` | `0.6014687563529590` | Reproduced immutable ancestry root. |
| `2-mustard-hyrax-of-growth` / `4f14aac...` | Early 13-field rich gated FM | `0.6695595095664563` | `0.5365718834126034` | `0.6030656964895298` | Valid improvement. |
| same run / `3a5d381...` | FwFM | `0.6688600485351314` | `0.5367422721304138` | `0.6028011603327725` | Regressed primary. |
| v16 `2-logical-cooperative-eagle` / `8376985...` | Exact 18-field rich FM | `0.6705554934560686` | `0.5375380665015315` | `0.6040467799788001` | Stable rich-FM reproduction. |
| v16 / `2c66f30...` | DCN residual | `0.6715706861689237` | `0.5380115527226895` | `0.6047911194458067` | Primary/GAUC above reference; nDCG short by `0.000002699`. |
| v16 / `944d723...` | Eight-feature strict history | `0.6714967417306275` | `0.5381014350560500` | `0.6047990883933387` | AIDE single-run gate passed. |
| v16 / `187d31c...` | Shared multi-exit ensemble | `0.6714236901946460` | `0.5379358382438919` | `0.6046797642192690` | Regressed both components versus history. |
| v17 `2-opal-gharial-of-fortitude` / steps 4-7 | Independent duration/RAD variants | — | — | — | Four consecutive `TimeoutError` failures at `900` seconds. |
| v17 / step 8 | LightGCN branch | — | — | — | `RuntimeError` after `16.88` seconds. |
| v18b `2-warm-mellow-corgi` / step 2 | Exact DCN first attempt | — | — | — | `TimeoutError` at `900` seconds. |
| v18b / `ef3129a...` | Autonomous DCN repair | `0.6715803104644023` | `0.5379195897807865` | `0.6047499501225944` | Finished in `412.33` seconds; nDCG below floor. |
| v18b / `bb8429d...` | Strict history | `0.6715519937294574` | `0.5376755854403260` | `0.6046137895848918` | Primary/GAUC above floor, nDCG regressed. |
| v18b / `9ed3ba4...` | Detached RAD duration head, fixed rank blend | `0.6721409447313641` | `0.5382066966715908` | `0.6051738207014774` | Strongest AIDE single-run gate pass. Development only. |
| v19b `2-ruddy-cockatoo-of-felicity` / `5bce29f...` | DCN | `0.6715210330306979` | `0.5379143487438849` | `0.6047176908872913` | nDCG below floor. |
| v19b / `50169ce...` | Checkpoint-aligned strict history | `0.6714723459645795` | `0.5380948554927942` | `0.6047836007286869` | AIDE single-run gate passed. Development only. |
| v19b / `2c01acb...` | Checkpoint-aligned RAD blend | `0.6715592190904085` | `0.5379451163156858` | `0.6047521677030472` | nDCG below floor; worse than history parent. |
| v20 `2-greedy-speedy-tiger` / `e1d3b05...` | DCN | `0.6715040504075095` | `0.5379995795156094` | `0.6047518149615594` | nDCG narrowly below floor. Partial campaign. |
| v20 / steps 3-5 | History + learning-rate stabilization attempts | — | — | — | Three `TimeoutError` failures at `900` seconds. |
| v20 / `7727ad5...` | Fourth autonomous history repair | `0.6711103869450264` | `0.5381204403553140` | `0.6046154136501702` | Single-run gate passed, but seed-0 replay fell to primary `0.6038752986969963`; campaign interrupted. |

Additional actual failure evidence:

- v9 DIN-lite: three `900`-second timeouts followed by a `MemoryLimitError` at 3 GB; no valid DIN score.
- v11 CatBoost: initial `MemoryLimitError`; three repairs were rejected by candidate source policy; no legal CatBoost score.
- v12 LightGCN: one timeout, then a valid but weak `0.6696418534393576 / 0.5374640200919603 / 0.6035529367656589`.
- v14 metric-aligned RankNet-only refinement: `0.6710062815366977 / 0.5375683978112750 / 0.6042873396739863`, below its DCN parent.
- v15 rich-FM mechanics drift: a train-checkpoint/clamped-gate refinement collapsed to primary `0.5914252032306777`.

## Single-run breakthroughs, old multi-seed rejections, and clean acceptance

### AIDE-generated single-run breakthroughs

- v16 strict history: primary `0.6047990883933387`, with both components above their reference floors.
- v18 RAD: primary `0.6051738207014774`, the strongest current one-run node.
- v19 strict history: primary `0.6047836007286869`.
- v20 repaired history: primary `0.6046154136501702`, but the interrupted campaign's seed-0 replay did not reproduce it.

These are development findings. They demonstrate that AIDE can generate a winning prediction in a run, not that the frozen autonomous system is accepted.

### Historical three-seed rejections

The current code still performs seeds `0,1,2` after a breakthrough when confirmation is requested. This automation is scheduled for removal in Phase 2; it remains historical evidence only.

| Campaign / candidate | Seed-0 primary | Seed-1 primary | Seed-2 primary | Mean GAUC | Mean nDCG@5 | Mean primary | Historical decision |
|---|---:|---:|---:|---:|---:|---:|---|
| v16 strict history | `0.6047990883933387` | `0.6045460727123331` | `0.6041974444084679` | `0.6711337711939555` | `0.5378946324821378` | `0.6045142018380466` | Rejected. |
| v18 RAD | `0.6051738207014774` | `0.6046139752229123` | `0.6043647459590120` | `0.6714588305377123` | `0.5379761973845556` | `0.6047175139611339` | Rejected. |
| v19 history | `0.6047836007286869` | `0.6046166725662081` | `0.6042161579504128` | `0.6711648748494420` | `0.5379127459807630` | `0.6045388104151026` | Rejected. |

### Clean-run acceptance

There is no clean-run acceptance. v16, v18b, and v19b have valid zero-human-intervention lifecycle evidence, but they are development campaigns and their old confirmation policy rejected them. v20 is not valid lifecycle evidence: it has one `run_started` event, one human interruption event, no `run_completed` event, and no resource summary.

## v20 interruption record

The interrupted run is `techjam-aide-portfolio-dev-v20` in `challenge/runs/aide/aide_logs/2-greedy-speedy-tiger/`.

- prompt SHA-256: `b42a74d4b84741f50e90fd8a77129b2da4582b6fc3ef5c3abf76b7d2b8cc60b8`;
- manifest SHA-256: `cdf2a94e0422d0720c688f10147eed35f32d9669d28414f66ccadaad62267e59`;
- preserved trial records: `8`;
- paid requests during the partial run: `6`;
- partial-run tokens: `88854` input and `34091` output;
- partial-run estimated cost: `$0.7335`;
- durable cumulative cost moved from `$8.368685` to `$9.102185`;
- event ledger action: `campaign_interrupted`, reason `User-directed reset before repository checkpoint`, status `incomplete`.

The per-trial records show zero intervention at the time they were written; the human interruption event was appended afterward. There is no final summary, no clean evidence, and no accepted result. Do not resume or represent this campaign as successful.

## Known dead ends and implementation failures

- Changing rich-FM mechanics while claiming reproduction created large score drift; exact transforms, optimizer, gating, and checkpoint semantics matter.
- The tested FwFM extension regressed primary.
- Pairwise-only/RankNet refinements reduced both component metrics; future metric alignment must retain BCE and use a small bounded residual or blend.
- The first shared multi-exit ensemble regressed. Fixed-weight component diagnostics are research evidence only; generated candidates may not load those arrays.
- Independent LightGBM/RAD stages repeatedly timed out. Duration supervision should remain a tiny detached head in the existing training loop.
- DIN-lite exceeded the CPU-time and memory budget in its tested form.
- CatBoost has not yet produced a legal score because of memory and policy failures; it needs a much smaller, genuinely bounded design.
- LightGCN timed out once and its repaired score was materially below the strong FM/DCN/history lineage.
- Adding extra coarse history flags such as music/duration matching regressed GAUC in v13.
- Broad historical LightGBM, hard-negative BPR, sequence SVD, and naive metadata expansion were weak deterministic local branches and should not displace the proven backbone without new EDA evidence.
- v20 exposed a runtime/determinism problem: three stabilized-history variants timed out, the repair passed once, and the immediate seed-0 replay did not reproduce the score.
- v21 GPT-5.6 Sol produced two valid but globally regressive rich-FM rewrites; executable code is not evidence of scientific success, and these nodes are now recorded as parent-dominated dead ends.
- Internal-holdout preprocessing must be rebuilt on all training rows for final fitting. Keeping prefix-only vocabularies caused `2.575%` of validation users to remain unknown despite legal later-training observations.

## Setup and exact commands

Run these from the repository root in PowerShell. Never put an API key on the command line; paid runs load the ignored `.env.local`.

### Environment and public data preparation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r challenge\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r challenge\requirements-agent.txt

# After KuaiRand-Pure is present under the starter kit:
.\.venv\Scripts\python.exe challenge\prepare_agent_data.py
.\.venv\Scripts\python.exe challenge\reproduce_baseline.py --seeds 0
.\.venv\Scripts\python.exe -m pytest tests challenge\tests -q
```

### Static paid-run dry run (no API call)

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --steps 2 --model gpt-5.6-sol --max-output-tokens-per-call 8000 `
  --max-input-tokens 35000 --max-output-tokens 16000 `
  --max-run-usd 0.50 --input-usd-per-million 4 `
  --output-usd-per-million 20 --campaign-mode development
```

### API-free baseline and diagnostics smoke

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --baseline-only --steps 1 --campaign-mode development `
  --run-id techjam-aide-v21-baseline-static
```

### Two-iteration smoke campaign (`$0.50` ceiling)

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --execute --steps 2 --model gpt-5.6-sol --max-output-tokens-per-call 8000 `
  --max-input-tokens 35000 --max-output-tokens 16000 `
  --max-run-usd 0.50 --input-usd-per-million 4 `
  --output-usd-per-million 20 --campaign-mode development `
  --run-id techjam-aide-smoke-<version>
```

### Bounded development campaign (`$2` ceiling)

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --execute --steps 6 --model gpt-5.6-sol --max-output-tokens-per-call 10000 `
  --max-input-tokens 120000 --max-output-tokens 60000 `
  --max-run-usd 2 --input-usd-per-million 4 `
  --output-usd-per-million 20 --campaign-mode development `
  --run-id techjam-aide-portfolio-dev-<version>
```

Every generated candidate is executed exactly once with fixed seed `0`. The removed `--confirm-on-breakthrough` option is rejected by the parser.

### Fresh clean campaign (`$5` ceiling)

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --execute --steps 46 --model gpt-5.6-sol --max-output-tokens-per-call 10000 `
  --max-input-tokens 250000 --max-output-tokens 200000 `
  --max-run-usd 5 --input-usd-per-million 4 `
  --output-usd-per-million 20 --campaign-mode clean `
  --run-id techjam-aide-portfolio-clean-<version>
```

The clean command starts from the organizer seed and uses only the frozen prompt, experiment memory, EDA summary, literature manifest, scheduler configuration, dependency lock, evaluator, and safety manifest. It never performs online literature lookup or multi-seed confirmation.

## Current limitations and prioritized next work

1. **Run the bounded six-iteration v24 GPT-5.6 Sol development campaign.** Its static dry-run is valid, contains all 16 frozen experiment summaries, and its worst-case API envelope is `$1.68`, within the `$2` development ceiling.
2. **Stabilize the fastest rich-FM/DCN/history backbone.** Reproduce it comfortably under timeout and checkpoint on the exact emitted prediction. Resolve v20's seed-0 replay mismatch before adding complexity.
3. **Prioritize the literature/EDA-backed duration branch.** Preserve the successful RAD direction, then compare one cached D2Q auxiliary inside the same training loop; do not add a second model/refit stage.
4. **Build an nDCG@5 specialist.** Preserve the GAUC backbone, then test a tiny same-user BCE-dominant Lambda@5 residual or conservative normalized blend.
5. **Use multi-fidelity only for search efficiency.** Screen variants on the internal chronological split, but require full fidelity for any accepted node and record both manifests.
6. **Freeze and prove autonomy.** After development evidence stabilizes, run a fresh zero-intervention clean campaign and accept only an AIDE-generated seed-0 candidate that beats every component floor.

## Commit-safety warning

Do **not** commit API keys, `.env`/`.env.local`, credentials, raw datasets, validation/test predictions, evaluator label arrays, checkpoints, model weights, NumPy prediction arrays, or bulky run artifacts. The repository `.gitignore` already excludes the main locations and extensions, including:

```text
.env* (except .env.example), .secrets/, secrets/, *.pem, *.key,
kuairand-starter-kit/KuaiRand-Pure/, challenge/agent_data/,
challenge/private/, challenge/runs/, challenge/artifacts/,
data/, datasets/, input/, predictions/, checkpoints/,
*.npy, *.npz, *.parquet, *.pt, *.pth, *.ckpt
```

Before every commit, inspect `git status` and the staged diff. Run artifacts cited in this handoff are intentionally local evidence paths and should remain ignored.

## Evidence index and uncertainties

Primary evidence used for this audit:

- thresholds and protocol: [`challenge/techjam_recsys/protocol.py`](../challenge/techjam_recsys/protocol.py), [`challenge/task.md`](../challenge/task.md);
- schema and chronology: [`challenge/task.md`](../challenge/task.md), [`challenge/prepare_agent_data.py`](../challenge/prepare_agent_data.py);
- v16 ledger: `challenge/runs/aide/aide_logs/2-logical-cooperative-eagle/`;
- v17 ledger: `challenge/runs/aide/aide_logs/2-opal-gharial-of-fortitude/`;
- v18b ledger and summary: `challenge/runs/aide/aide_logs/2-warm-mellow-corgi/`;
- v19b ledger and summary: `challenge/runs/aide/aide_logs/2-ruddy-cockatoo-of-felicity/`;
- interrupted v20 journal, hash-chained iterations, manifest, and event ledger: `challenge/runs/aide/aide_logs/2-greedy-speedy-tiger/`;
- current durable spend: `challenge/private/api_spend.json`;
- older AIDE failure evidence: `challenge/runs/aide/aide_logs/2-urchin-of-heavenly-intensity/`, `2-meteoric-bronze-caracal/`, `2-heretic-skua-of-serendipity/`, and `2-fuzzy-imposing-loon/`;
- deterministic non-AIDE branches and old orchestrated reference: [`challenge/README.md`](../challenge/README.md), [`challenge/reports/champion-v3.json`](../challenge/reports/champion-v3.json).

Known uncertainty: several older pre-manifest AIDE directories do not carry a canonical `run_id` or complete resource summary. Their node metrics and failures are still preserved in `journal.json`, but only manifest-backed v8+ campaigns should be used for clean-campaign provenance. v20 is intentionally incomplete and must not be reconstructed into a successful summary.
