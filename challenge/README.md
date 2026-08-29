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
| Official FM, reproduced | 0.66713 | 0.53580 | 0.60147 | baseline champion |
| Ordered-history LightGBM | 0.65275 | 0.52961 | 0.59118 | reject |
| LambdaRank history model | 0.66234 | 0.53386 | 0.59810 | blend-only |
| FM + LambdaRank z-blend | 0.66839 | 0.53669 | 0.60254 | current best rung |
| Hard-negative BPR FM | 0.66716 | 0.53557 | 0.60136 | reject |
| FM + sequence SVD | 0.66714 | 0.53623 | 0.60169 | reject |
| Out-of-time FM residual ranker | 0.66761 | 0.53620 | 0.60190 | below best rung |
| AIDE history LambdaRank draft | — | — | — | schema failure; logged |

The current best improves both components but its `+0.00094` primary lift over
the reproduced FM is below the organizer's meaningful `0.002` threshold. It is
not yet a winning final model. Failed branches remain in the logs because
recovery evidence is part of the judging rubric.

The first paid AIDE iteration used one GPT-5.4 request (1,867 input and 3,654
output tokens, approximately $0.0595 at the verified uncached rates). It
proposed leakage-safe history aggregates plus LambdaRank, independently
confirming the local portfolio direction, but selected a training-only column
that validation does not contain. The failure is retained as agent evidence.
The search policy now treats the reproduced FM as its sole draft, so future
approved iterations must improve or debug an evaluated node instead of paying
to restart from scratch.

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
   official epsilon/patience, per-model token accounting, and a required
   one-run spending approval.

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
```

Inspect a paid-run dry run; this never calls the API:

```powershell
.\.venv\Scripts\python.exe challenge\run_aide_research.py `
  --steps 1 --model gpt-5.4 --max-output-tokens-per-call 6000
```

Paid execution is deliberately blocked unless every one-run approval field is
provided. The operator must review current official pricing and explicitly set
an approval ID, dollar ceiling, input/output token ceilings, and the verified
per-million-token prices. The runner refuses an envelope whose conservative
uncached estimate exceeds the approved dollar amount. Do not put the API key
on the command line; the runner loads ignored `.env.local`.

## Fork workflow

After creating your GitHub fork, keep WecoAI as `upstream` and your fork as
`origin`:

```powershell
git remote rename origin upstream  # only if origin currently points to WecoAI
git remote add origin https://github.com/<you>/aideml.git
git push -u origin techjam
```

This workspace already tracks `upstream/main` on branch `techjam`; only the
personal `origin` URL is missing.

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
