# Final Collection: Eight Equal Repetitions, $120 Ceiling

October 18, 2026. The owner rejected the assistant's expanded 32-repetition
allocation. It is withdrawn, not a requirement from Raghav. The active code and
[contract](../final_study/contract.json) now specify **896 planned attempts** and
a **$120 maximum additional main-study allowance**. No main generation has run.

## Contract v6 (October 19)

Contract `final-instruction-study-v6-block-order-partial-analysis`, SHA-256
`04100ffe593c3887cabdd7cc74921bb0c65d2f078e0b1fe001cde7b56bdd9fab`, supersedes the
unexecuted v5 (`17c2750f…`). It responds to the
[October 19 pre-launch review](PR_REVIEW_OCT19.md) and the team's own check of it.
**Unchanged:** 896 attempts, eight equal repetitions (seeds 44000–44007), the $120
ceiling, models, settings, prompts (the rendered catalog is byte-identical), the
global parser, transport recovery, the price window and all paid evidence.

| Change | Why |
| --- | --- |
| **Block-major execution order.** All 112 cells of seed 44000 run (shuffled within the block), then 44001, and so on. | Under v5's single shuffle across seeds, a stop at 90% or 95% left zero complete primary contrasts. Now a 90% stop leaves seven complete blocks. Provider drift over time aligns with blocks, which the paired analysis already removes. The historical Hindi check keeps its original order. |
| **Predeclared partial-study rule** (analysis v2). A block is dropped from a contrast only if one of its eight cells (four models × two arms) lacks an endpoint; inference uses the remaining blocks if at least six remain. Fewer than six still suppresses inference. | Under v5, one invalid US-English global reply suppressed all six global hypotheses. A complete study gives exactly the v1 results (tested). |
| **`analyze --stage main`** replays every saved receipt offline and writes `analysis_summary.json`, `primary_contrasts.json/.csv`, `missingness_report.json` and `analysis_rows.json`, labelled `COMPLETE` or `PARTIAL_STUDY`. | A stopped run previously had no supported reporting path. |
| **Local write retries extended** to ten, about 54 s in total. | 1.9 s was shorter than typical antivirus or sync-client holds. |
| **Journal reconciliation in `recover`** (and `--auto-recover`). If a lock outlasts the retries after a reply was received and charged, `recover` rewrites that one request journal from the unchanged ledger, only when the transport receipt matches and the journal is the intent record or the exact pre-parse digest. Logged in `journal_reconciliations.json`; at most 30. A plain relaunch still refuses. | Previously this state had no supported repair; V6 and V8 each needed a one-off script for it. |
| **Contradiction removed.** The embedded analysis no longer says automatic recovery is "not approved"; transport is governed by the contract's `transport_policy`. | The frozen contract stated two policies. |

**What the fallback assumes.** Dropping an incomplete block assumes its missingness
is unrelated to that contrast's outcome, given schedule and condition. That can be
false: failures may depend on language, model or graph size. Every result therefore
also reports the excluded blocks, each missing cell, the [0, 1]-bounded missing-data
extremes over all eight blocks, and failure counts by method, condition and model.
Read a PARTIAL result together with those, never as a completed study.

**Run it so that it finishes by October 25.** The price window (verified October 18,
valid through October 25) is checked before every request. A run still going on
October 26 stops, and refreshing prices creates a new contract and output folder,
which cannot continue the stopped run. At 26.3 serial API hours (39.5 with a 50%
allowance) plus recovery waits, **launch no later than the morning of October 23**.
Do not edit the dates to extend the window: refresh them only after re-checking the
official prices, then `freeze` again before any main attempt exists.

**Before launch, on the owner machine:**

1. Rotate the exposed API key and configure the new one locally. Rotation is still
   not recorded anywhere; record the date, never the key.
2. Move the whole workspace and its private ledger together to a folder outside
   OneDrive or any other sync client (for example `C:\research\UA_Captsone_SocN`),
   and never run two copies. Sync clients are the most likely cause of persistent locks.
3. Disable sleep and hibernation for the duration of the run.
4. Run the offline suite, `preflight`, `verify --stage hindi_check` and `status`.
   `status` must report `READY_FOR_MAIN_EXECUTION` (it needs the private Hindi
   ledger and the key, so it cannot be checked on a public clone).
5. Record the professor's approval of this analysis plan in
   [TEAM_APPROVAL.md](TEAM_APPROVAL.md), or the explicit decision to proceed
   without it. Neither is recorded today.

## Exact Scope

| Dimension | Fixed selection |
| --- | --- |
| Models | GPT-4.1, GPT-5.6-Luna, GPT-6-Luna, GPT-6-Sol |
| Methods | Global, local, sequential, iterative |
| Country comparison | US, India, Japan, Brazil; instructions in English |
| Instruction comparison | English, Hindi, Japanese, Portuguese; country fixed to US |
| Unique conditions | Seven: the two comparisons share US/English |
| Repetitions | Eight for **every** model/method/condition, seeds 44000-44007 |
| Roster | The same 50 fictional adults in every attempt |
| Total | 4 models x 4 methods x 7 conditions x 8 repetitions = **896 attempts** |
| Per method / per model | **224 attempts each** |

No global-only increase, omitted planned condition, Spanish condition, or full
4-country by 4-language cross-product. The 896 rows are prospectively fixed,
not selected after looking at favorable results.

## Cost And Completion

The saved V8 usage forecast is **$102.44275142** and **26.30 serial API hours**
(39.46 hours with a 50% time allowance), leaving about **$17.56** under the cap.
This estimates new collection usage, not an invoice or a completion guarantee.
Prior calibration is already completed and excluded. There is no automatic
$128 contingency allowance.

The code rejects allowances above $120, including the old $520 command, before
credential lookup or provider access. The ledger reserves a conservative maximum
before each request. Charges plus unresolved reservations share one cap across
every model, method and condition. Resume cannot reset the allowance or rebuy a
received reply. SDK retries remain zero.

**Transport recovery (retained in contract v5, October 19).** About 100,000 requests at the
V8 rate of two provider 5xx errors per ~11,600 means several interruptions are
expected. `recover` handles exactly one kind: an attempt that got **no saved
reply** after a connection error, timeout, 408/409/429 or 5xx. It keeps the
original full reservation against the same $120 cap, records the retirement in
`transport_recoveries.json`, and permits **one** separately reserved replacement
of that identical request. A failed replacement, non-retryable 4xx, a saved reply, a
parse/settlement error or more than 30 retirements still stops for review.
Received replies are never bought again. `--auto-recover` applies the same rule
automatically, waits 30–120 s first, and waits (free, up to an hour) when the
provider's model check fails before any paid request.

Retirement is durable in SQLite before its journal and recovery log are written.
If the process stops between those writes, `recover` or the same launch command
finishes only the missing writes from that existing ledger record. It preserves
the original timestamp, authorization and full reservation. A conflicting journal
or log stops execution; reconciliation never silently overwrites conflicting data.
Reconciliation itself can be interrupted and restarted without authorizing a
second replacement. The current contract supersedes v4 only before any main
stage exists; old paid observations keep their original contracts.

**Windows file-lock protection (October 19).** Saving an intent, received reply,
request checkpoint, or runner JSON report retries only `PermissionError`, up to
ten times after the first write attempt (v6; v5 had five). The waits are 0.05,
0.1, 0.25, 0.5, 1, 2, 5, 10, 15 and 20 seconds, about 54 s in total. Each attempt
saves the same data using an atomic, flushed replacement;
it does not call the model again or rerun parsing. Both inherited writer aliases
are covered, including the post-reply and post-parse checkpoints. The aliases
are restored when launch exits, so historical workflows are not modified.

A persistent lock after all eleven attempts, disk-full error or other I/O failure
still stops execution. If that stop leaves one request journal stale after a
received reply, `recover` (or `--auto-recover`) reconciles only that journal from
the unchanged ledger, under the narrow conditions in the Contract v6 section.
This is not automatic repair of arbitrary damaged evidence. Do not use transport recovery to repurchase a received reply, delete a
journal, or change the contract to bypass such a stop. Prefer a nonsynced working
folder for collection; move the complete canonical workspace and ledger together,
and never execute two copies. No antivirus exclusion is required by this repair.

**No partial run is presented as a complete study.** All 896 attempts stay in the
manifest. A cap or uncertain-transport stop is `STOPPED`/`PARTIAL`, not success.
All terminal attempts with any invalid graphs are `COMPLETE_WITH_FAILURES`;
failures stay in the denominator. No fabricated graphs or rerolling until success.
Full collection is the target, but future provider behavior can exhaust the cap.
Invoices may also include unrelated account usage or taxes; this ceiling applies
to this runner's conservative token/rate accounting.

October 18 checked rates and execution settings are unchanged. Price freshness
expires after October 25 and then blocks new calls, so the whole run must
**finish** by October 25, not merely start (see Contract v6 above). The recorded GPT-4.1 shutdown
date is January 14, 2027; no automatic model substitution is allowed.

## Analysis And Claims

The [analysis adapter](../final_study/analysis.py) reuses reviewed calculations
with an explicitly validated eight-schedule specification. The historical
32-schedule analysis and its evidence remain unchanged.

- Global's primary endpoint is density. The quota-constrained methods use
  average local clustering over all 50 nodes; their density is descriptive.
- Average the four fixed models equally within each schedule. **Eight schedules
  are eight planning units, not 32** by counting model rows separately.
- Six baseline contrasts per method give 24 primary hypotheses. Paired-block
  t diagnostics, Holm adjustment and Bonferroni simultaneous intervals retain
  their stated independence/distributional assumptions and family size.
- Missing models never become three-model averages or zero-valued graphs.
  A schedule block with any missing cell is dropped from the affected comparison
  only. With all eight blocks the result is `COMPLETE_ASSUMPTIONS_REQUIRED`; with
  six or seven it is `PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED` (t on the remaining
  blocks); with fewer than six, inference is suppressed. Missing cells, excluded
  blocks and missing-data bounds are always reported.
- Homophily dimensions, age assortativity, model-specific contrasts and other
  topology summaries remain descriptive, not causal demographic dominance.

Eight repetitions offer less precision than 32. No minimum effect, power,
interval width, significant result or conference acceptance is guaranteed.
Claims concern **seven named instruction contracts**, one fictional roster and
four fixed models, not representative populations or translation-independent
language effects. AI-assisted translation review is not human certification.

## Preserved Paid Evidence

The original contract and four source files are archived byte-for-byte under
`final_study/history/v1/`. All 104 V8 observations remain intact: 100 valid graphs
and four invalid replies, with four empty graphs among the valid results.
Calibration is not reused as confirmation.

Eight received Hindi global checks retain their original contract:
`5886a306a053579ed95f47011a8a41acd893c8ee75480cd25968c64658790607`.
They cost $0.02740295 conservatively and took 30.626 API seconds. All eight parsed;
one was valid empty. Their raw replies, duplicate flags, graphs, PNGs, request
identities and private accounting remain unchanged.

The bridge validates archived hashes and unchanged prompt catalog/runtime/settings,
then replays the original replies. It does not relabel them under the new contract.
Paid Hindi regeneration is disabled here. The unchanged parser validates every
line before collapsing exact undirected duplicates; invalid IDs, self-links and
malformed lines remain failures.

## Run And Inspect

Use the pinned environment. `UA_Captsone_SocN` is the only paid workspace;
`UA_Captsone_publish` and GitHub hold public artifacts, not another spendable
ledger. Never put credentials in Git, shell arguments, notebooks or comments.

```powershell
# Offline: all 112 combinations and the generated 896-row manifest.
python -X utf8 -B -m final_study preflight

# Offline public replay of existing Hindi observations; no new purchases.
python -X utf8 -B -m final_study verify --stage hindi_check

# Canonical workspace: also verifies private accounting and local credentials.
python -X utf8 -B -m final_study status

# Only after the owner's main launch instruction. Never above $120.
# --auto-recover retires lost-transport attempts and resumes (see above).
python -X utf8 -B -m final_study launch --stage main --approve-usd 120 --approval-reference owner-eight-repetitions-120 --auto-recover

# Manual alternative after a lost-transport stop, then rerun the same launch.
python -X utf8 -B -m final_study recover --stage main --approval-reference owner-eight-repetitions-120

# After collection: independently replay the complete attempt inventory.
python -X utf8 -B -m final_study verify --stage main

# Complete or stopped run: offline replay, then the primary analysis.
# A stopped run is reported as PARTIAL_STUDY under the predeclared fallback.
python -X utf8 -B -m final_study analyze --stage main
```

The exact main command resumes saved evidence. Changing its cap/reference,
missing ledgers, stale prices, altered receipts or uncertain responses stops
execution. `recover` may first finish evidence for an already authorized
retirement. It exits with `RecoveryRefused` rather than authorizing a replacement
for other failures. Never delete evidence to bypass a stop.

Main outputs live in `outputs/final_study/<current-contract-id>/main/`. Successful
observations have a receipt, `.adj` and `.png`; failures have no fabricated graph.
Completed collection writes `network_metrics.csv`, `analysis_rows.json`, and
`primary_contrasts.json/.csv`. Manifest and analysis counts derive from code.

See [verification evidence](FINAL_COLLECTION_VERIFICATION.md). Test fixtures
are not LLM observations. The simulator is unchanged by this correction.
