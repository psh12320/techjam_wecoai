# TechJam 2026: Autonomous KuaiRand Research Agent

This fork turns WecoAI AIDE into a constrained recommender-system research
agent for the required KuaiRand-Pure benchmark. It preserves AIDE's code-tree
search while replacing its LLM-based metric review with the organizer's exact,
deterministic evaluator.

## What wins

The required task is **within-user ranking over logged impressions** with native
`long_view` as the binary label. The only ranking metrics are:

| Metric | Published validation baseline | Published hidden-test baseline |
|---|---:|---:|
| GAUC | 0.6674 | 0.6610 |
| nDCG@5 | 0.5357 | 0.5282 |
| Primary = mean | 0.6016 | 0.5946 |

The technical score uses the **absolute improvement** in both component metrics
on the final hidden-test submission. Peak intermediate performance is not the
submission rule: the run stops after three consecutive iterations without a
`> 0.002` validation-primary improvement, at 50 iterations, or at six hours.
The validation-best checkpoint at convergence is submitted once.

Overall judging is 35% Technical Execution, 20% Innovation & Problem Insight,
20% Impact & Relevance (including autonomy/manual interventions), 15%
Feasibility & Practicality (LLM tokens and wall-clock after clearing baseline),
and 10% final presentation. This is why the system logs hypotheses, diffs,
metrics, failures/recoveries, tokens, time, GPU-hours, and interventions rather
than optimizing a leaderboard number alone.

The stale sentence in the supplied brief mentioning `NDCG@10 / Recall@50`
conflicts with the shipped evaluator and the detailed benchmark section. The
executable contract—GAUC and nDCG@5—is authoritative.

## Current verified state

| Branch | Validation GAUC | nDCG@5 | Primary | Decision |
|---|---:|---:|---:|---|
| Official FM, reproduced | 0.66713 | 0.53580 | 0.60147 | immutable baseline |
| Ordered-history LightGBM | 0.65275 | 0.52961 | 0.59118 | reject |
| LambdaRank history model | 0.66234 | 0.53386 | 0.59810 | blend-only |
| Hard-negative BPR FM | 0.66716 | 0.53557 | 0.60136 | reject |
| FM + sequence SVD | 0.66714 | 0.53623 | 0.60169 | reject |
| Out-of-time FM residual ranker | 0.66761 | 0.53620 | 0.60190 | below best rung |
| Train-only rich gated FM, 3-seed ensemble | 0.67012 | 0.53728 | 0.60370 | robust core |
| 80% rich + 10% LambdaRank + 10% RAD-U | **0.67105** | **0.53801** | **0.60453** | frozen champion |

The frozen champion improves GAUC by `+0.00392`, nDCG@5 by `+0.00221`, and
primary by `+0.00306` over the exact reproduced baseline. The rich core alone
clears the organizer's `0.002` primary-improvement gate, and its direction is
positive for both metrics on two earlier rolling cutoffs. The ensemble weights
come from a deliberately coarse 0.1 simplex grid rather than a fragile fine
search. See `reports/champion-v3.json` for exact metrics and provenance.

Four paid GPT-5.4 requests have used 7,357 input and 18,640 output tokens, for a
durably recorded estimated total of `$0.2979925` at the verified uncached
rates. No `$10` notification boundary has been crossed. The paid agent confirmed
the leakage-safe history/LambdaRank direction but did not beat the deterministic
local branches; all failures and recoveries remain in the run ledger. API cost
is written immediately after every completed response.

## Architecture

1. **Immutable protocol layer** — organizer evaluator, fixed temporal split,
   strict row alignment, finite-score checks, no dates after 2022-04-28.
2. **Baseline seed** — the exact five-field NumPy FM is executed before paid
   search and becomes the root of AIDE's solution tree.
3. **Deterministic reviewer** — candidates write aligned validation scores;
   Python computes GAUC/nDCG@5. This removes one LLM call per iteration.
4. **Research tree** — AIDE drafts, improves, and debugs atomic code changes.
   Good and failed nodes remain branchable/auditable.
5. **Champion policy** — prefer validation-primary best candidates that beat
   both components; use within-user normalization for conservative hybrids.
6. **Budget/convergence controller** — hard 50-iteration/six-hour limits,
   official epsilon/patience, per-model token accounting, conservative run
   ceilings, durable cumulative spend, and notifications at each $10 boundary.

## Winning research agenda

The next high-value search order is deliberate:

1. **FM-preserving residual improvements.** Keep the strong latent interaction
   score and learn small, out-of-time corrections; do not replace FM with a
   weaker tabular model.
2. **Sequence interest.** Add DIN-style attention or recent-positive candidate
   affinity using only historical events. Static coarse fields alone were
   already shown to be redundant.
3. **Metric-aligned training.** Test listwise softmax/LambdaLoss and carefully
   weighted pair sampling. The naive BPR branch is logged as a dead end, not a
   reason to abandon ranking losses.
4. **Leakage-safe multi-task learning.** Use click/like/hate/watch time only as
   auxiliary training targets or past aggregates. Never use current-row
   outcomes at prediction time.
5. **Duration/censoring.** Add a modern censored-watch-time auxiliary head based
   on CWM's insight, while keeping native `long_view` and official metrics.
6. **Temporal robustness.** Select on the official validation score but require
   consistent gains on rolling late-train folds to reduce leaderboard drift.
7. **Conservative hybrid.** Blend only complementary out-of-time predictions;
   optimize primary with a constraint that both GAUC and nDCG@5 beat baseline.
8. **Bonus datasets only after Pure clears baseline materially.** KuaiRand-1K
   and 27K add bonus points but should not consume the critical path early.

## Reproduce locally

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r challenge\requirements.txt
```

The paid AIDE loop has a separate optional runtime so baseline reproduction
does not install API clients or unrelated packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r challenge\requirements-agent.txt
```

Download and verify KuaiRand-Pure:

```powershell
Invoke-WebRequest `
  -Uri https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz `
  -OutFile kuairand-starter-kit\KuaiRand-Pure.tar.gz
Get-FileHash -Algorithm MD5 kuairand-starter-kit\KuaiRand-Pure.tar.gz
tar -xzf kuairand-starter-kit\KuaiRand-Pure.tar.gz `
  -C kuairand-starter-kit
```

Expected MD5: `0820331067a3784d9691136f772b35a7`.

Reproduce the official validation baseline without loading test labels:

```powershell
.\.venv\Scripts\python.exe challenge\reproduce_baseline.py --seeds 0
```

Prepare AIDE's development-only input and run all tests:

```powershell
.\.venv\Scripts\python.exe challenge\prepare_agent_data.py
.\.venv\Scripts\python.exe -m pytest tests challenge\tests -q
```

Run deterministic local branches (no LLM/API spend):

```powershell
.\.venv\Scripts\python.exe challenge\run_portfolio.py
.\.venv\Scripts\python.exe challenge\run_pairwise_fm.py
.\.venv\Scripts\python.exe challenge\run_enriched_fm.py `
  --variant rich_lite --gated --seed 0 --epochs 12 --patience 3 `
  --batch-size 4096
```

The enriched encoder fits vocabularies on training rows only and maps every
future-only category to one explicit unknown slot. This prevents the earlier
transductive random-embedding artifact.

Inspect a paid-run dry run; this never calls the API:

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --steps 1 --model gpt-5.4 --max-output-tokens-per-call 6000
```

Paid execution still requires explicit `--execute` and current verified
per-million-token prices. It has conservative per-run token/dollar ceilings and
persists cost immediately after every completed request. The operator is
notified whenever cumulative estimated API cost crosses another $10 boundary.
Do not put the API key on the command line; the runner loads ignored
`.env.local`.

Build the frozen 80/10/10 hidden-test submission after validation is locked:

```powershell
.\.venv\Scripts\python.exe challenge\train_submission.py --jobs 4
```

This fits through 2022-04-28 and reads only IDs, timestamp/context, duration,
tab, and static metadata for 2022-04-29 through 2022-05-08. Hidden
`long_view`, watch time, and engagement columns are never loaded into the score
frame. The generated 170,588-row CSV and component arrays live under ignored
`challenge/runs/submission/`.

## Fork workflow

After creating your GitHub fork, keep WecoAI as `upstream` and your fork as
`origin`:

```powershell
git remote rename origin upstream  # only if origin currently points to WecoAI
git remote add origin https://github.com/<you>/aideml.git
git push -u origin techjam
```

This workspace tracks `upstream/main` on branch `techjam` and pushes to
`https://github.com/psh12320/techjam_wecoai.git` as `origin`.

## Submission checklist

- Freeze the validation-best checkpoint at convergence, not the last model.
- Retrain using the pre-declared train+validation policy; never tune on hidden
  test feedback.
- Generate `row_id,user_id,video_id,score` in exact organizer row order.
- Run `submit.py --check` and reject NaN/Inf, gaps, duplicates, or misalignment.
- Report GAUC, nDCG@5, primary, and absolute deltas over official baseline.
- Include the JSONL iteration ledger and count manual interventions.
- Report LLM input/output tokens, agent wall-clock, iterations, and GPU-hours.
- Document leakage controls, limitations, and failed/recovered experiments.
- Prepare a concise demo: problem → autonomous loop → tree/logs → metric lift →
  cost → final reproducible submission.
