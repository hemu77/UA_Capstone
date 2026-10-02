# Revised Calibration Runner

October 2, 2026. This is an engineering calibration implementation, not main-study
authorization or a certificate of research validity. Preparing or testing the
runner does not grant permission. The owner subsequently approved `extended104`
with a **$10 additional ceiling**, separately from the earlier $1 wording probe.

## Scope

**Paid execution is complete: 104/104, $7.854865725 of $10, zero unresolved new
requests.** Read [the results and main-study hold](CALIBRATION_V6_RESULTS.md).
The 22/26 empty global graphs require a scientific task decision, not silent reruns.
Latest suite: 142 tests passed; notebook: 45 cells handled, 32 archival bodies skipped.

The runner uses the same 50 fictional adults, four models (`gpt-4.1`,
`gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol`) and all four methods. Countries are US,
India, Japan and Brazil. Instruction languages are English, Hindi, Japanese and
Brazilian Portuguese. Persona fields and values remain English.

There are seven settings: four countries in English, plus US in each of the other
three languages. This is not a full country-by-language factorial.

| Scope | Networks | Purpose |
| --- | ---: | --- |
| `core68` | 68 | Luna: 7 settings x 4 methods x 2 repeats = 56; other 3 models: US-English x 4 methods x 1 repeat = 12. |
| Additional multilingual coverage | 36 | Other 3 models x Hindi/Japanese/Portuguese x 4 methods x 1 repeat, fixed US. |
| `extended104` | 104 | Core plus coverage. Every model/method/language combination is exercised at least once. |

Both scopes were prepared offline. The selected paid scope is `extended104`;
the extra 36 were explicitly approved, not silently purchased with approval
for 68. These small counts check behavior and costs; they are not powered tests
of language equivalence. None is automatically pooled into the main study.

## What Changed

- The live adapter now uses the revised singular/plural, country-context and
  whole-network instructions that were prepared separately from V5.
- Actor order, nomination counts and candidate display have separate seeded
  streams. Schedules are saved, not inferred from a seed label alone.
- Early sequential decisions retain local instructions during corrections;
  subsequent sequential decisions include current degree.
- Global `NONE` can represent a genuinely empty network. Sparse, disconnected
  and zero-edge graphs are not rejected for looking unusual. Undefined metrics
  remain undefined, not zero.
- Recoverable global corrections must preserve the initial valid pair set.
  Local/iterative corrections retain the requested count and eligibility rules.
- At most three parsing attempts are allowed. A received truncated reply, a
  transport ambiguity, an exhausted cap or exhausted corrections stops the run.
  There is no automatic paid replacement.

## Spending and Recovery

`paid_study.Budget` and `PaidCaller` remain the shared billing implementation.
They are unchanged. The original private SQLite ledger is required; this runner
does not create a new real ledger to escape historical spending.

Authorization records the contract, selected scope, historical ledger row hashes,
starting balance and an additional finite allowance. The cumulative ceiling
includes prior spending. Reauthorizing cannot reset this anchor. A key is not read
and a client is not opened until the gates pass. The API base URL is explicitly
OpenAI's endpoint, not an ambient environment override.

Before each request, a durable intent file is written. After a response and after
parsing, the journal binds that request to the exact ledger row. Missing or changed
rows stop execution before a new client opens. This covers partial graphs as well
as completed receipts. An intent without a ledger row is ambiguous after a crash:
it stops for inspection rather than assuming the request was free. A confirmed
pre-transport budget rejection removes only its own unused intent.

Completed graphs can be reconstructed from saved requests, without the API.
Receipt verification checks prompts, response parsing, schedule, event replay,
adjacency, metrics and artifact hashes. When private billing is available, it also
checks receipt-to-ledger equality and that no charged attempt is omitted.

## Evidence Separation

All new files live under `outputs/calibration_v6/<contract-hash>/`:

| Path | Meaning |
| --- | --- |
| `contract.json`, `planned_cells.json` | Exact candidate configuration and scope. |
| `authorization_template.json` | Unapproved example; never used as approval. |
| `preflight.json`, `offline/` | Labelled fixture graphs, PNGs and replay records; zero real model calls. |
| `authorization.json` | Created only after an explicit, separately approved paid scope and cap. |
| `requests/` | Write-ahead intents and ledger-row integrity checkpoints. |
| `runs/` | Paid adjacency, PNG and full response/event receipts. |
| `network_metrics.csv`, `cost_stats.csv`, `summary.json` | Verified calibration summaries and receipt-based usage; partial counts remain partial. |
| `inspection.json`, `inspection_networks.csv`, `decision_compliance.csv` | Key-free receipt replay, metrics, initial-decision failure denominators and provenance. |
| `hypothetical_main_forecasts.csv` | Unchanged-prompt planning scenarios; never spending authority or confirmed designs. |
| `checkpoint_recovery.json`, `checkpoint_original.json` | One inspected parse-only journal repair with original evidence retained. |
| `io_adapter/` | Versioned source snapshots and bounded local-file retry audit; no API retries. |
| `last_stop.json` | Sanitized failure class and retained accounting state. |

The old 68 V5 graphs, source files and protocol remain unchanged. The completed
72-call wording screen remains separately frozen. The viewer continues to show
its existing observed dataset; fixture graphs are never exported into it.

## Run Offline

From the repository root with the existing pinned environment:

```powershell
.\.venv\Scripts\python.exe -B calibration_v6.py --prepare
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_calibration_v6.py
.\.venv\Scripts\python.exe -B calibration_v6.py --preflight
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

`--preflight` uses the real graph engine with deterministic labelled replies and
blocks socket connections. Integration tests use the installed SDK with an
intercepted transport and temporary ledgers. Neither is evidence of model behavior.

## Recorded Paid Approval

- [x] Owner approved `extended104` and $10 additional usage on October 2.
- [x] Model access and official model prices checked on October 2.
- [x] Contract/preflight and regression results reviewed before payment.
- [x] API credential supplied only to the child process, not repository files.
  This does not establish that previously exposed keys were rotated; rotate them.
- [x] Small paid prefix inspected before continuing the authorized scope.

The execution owner's commands, only after those decisions, are:

```powershell
# Replace placeholders only with the actual approved values; do not run now.
.\.venv\Scripts\python.exe -B calibration_v6.py --authorize --scope <core68-or-extended104> --additional-usd <approved-amount> --approval-reference <decision-id> --prices-verified-on <YYYY-MM-DD>
.\.venv\Scripts\python.exe -B calibration_v6.py --execute --limit 1
```

`--limit N` selects the first N cells of the authorized scope, not N additional
graphs. Existing completed cells are verified and skipped; partial cells replay
the same request identities and use received cached responses. Raising the limit
does not raise the dollar ceiling or expand the authorized scope.

Passing these checks makes the runner ready for a bounded paid calibration after
approval. It does not approve a main-study repetition count, meaningful effect,
equivalence margin or inferential method. Those scientific decisions remain in
[the statistical decision](STATISTICAL_ANALYSIS_DECISION.md).

## Pre-Payment Verification Record (Historical)

- [x] Version `5fb3715550db` completed **104 full-roster offline fixtures**, with
  API construction and socket connections disabled. All 104 JSON records,
  adjacency replays, PNG hashes/formats, schedules and CSV metric rows verified.
  Files: `outputs/calibration_v6/5fb3715550db/`.
- [x] Full Python suite: **128 tests passed in 165.155 seconds**, including 12
  new calibration tests. Existing pandas/NumPy deprecation warnings remain.
- [x] SDK-intercepted generation and exact cached replay across all four models
  and all four methods. These responses are fixtures, not API measurements.
- [x] Paid-path prefix execution tested against a temporary ledger; repeating a
  completed prefix does not create an API client or add charges.
- [x] Missing historical/partial rows, modified metrics, unjournaled requests,
  exhausted caps, stale contracts, uncertain outcomes and abandoned replacement
  paths stop instead of silently buying replacement responses.
- [x] Bounded read-only reviewer reports no remaining material spending, resume
  or evidence-integrity findings after fixes. This is an engineering review.
- [x] Existing viewer tests: **24 passed**. No fixture data was exported into the
  viewer and no viewer source was changed.
- [x] Maintained notebook verified: 43 code cells handled, including 32 intentionally
  skipped archival bodies; no API calls. The source notebook was not rewritten.
- [x] Before payment, frozen V5/probe contracts and the $6.481868475 ledger
  balance were unchanged. The offline preparation itself added $0 API usage.
- [x] At that checkpoint there were no real paid receipts or authorization.
  Subsequent authorization is excluded from Git; the unapproved template is shareable.

The review caught JSON tuple/list mismatch during receipt replay and an inherited
replacement path for abandoned requests. Both have reproducing regression tests.
The shared frozen caller was not modified to fix these new-runner boundaries.

## Local Checkpoint Recovery

A Windows `PermissionError` interrupted an atomic journal replacement after a
response had already been received and parsed. The exact ledger response existed;
this was not a missing API reply. One reviewed recovery verified the 45-request
cached prefix and the parse-only hash transition, preserved the old checkpoint,
then reconciled that one journal without changing any ledger row or buying a reply.
The file-locking process was not identified, so no specific OS service is blamed.

`calibration_v6_io.py` then wrapped only active journal-file writes with at most
five bounded `PermissionError` retries of identical bytes. It records source
snapshots and retry events. It never retries network transport or changes the
frozen graph engine, prompts, budget, request identity or parsing policy.
`recover_calibration_v6_checkpoint.py` is a one-time inspected recovery, not a
routine command or permission to repair different checkpoints.

Offline verification after collection (no key, original ledger not needed):

```powershell
.\.venv\Scripts\python.exe -B inspect_calibration_v6.py
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
```

The inspector refuses mismatched contracts, altered receipts/artifacts and
duplicate paid request IDs. Final billing reconciliation additionally requires
the original private ledger. The public report is not a provider invoice.
