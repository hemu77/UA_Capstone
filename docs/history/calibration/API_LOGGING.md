# API Diagnostics

Implemented September 30, 2026. This adds observability, not paid collection or
automatic recovery. The existing uncertain request and all spending remain unchanged.

## What Is Recorded

The existing SQLite ledger at `outputs/revision_budget_v1/budget.sqlite` remains
the single durable record. A separate logging service is unnecessary.

- Before dispatch: the existing SHA-256 ledger request ID is saved as
  `client_request_id` and sent as `X-Client-Request-Id`.
- After a reply: `diagnostics` records the server's `x-request-id`, HTTP status,
  UTC timestamp, monotonic elapsed milliseconds and selected numeric rate-limit,
  processing-time and retry-delay headers. Existing usage/cost records are unchanged.
- On failure: the same client ID, available response metadata, exception class,
  safe machine error code and up to four cause-class names are retained. Stages
  distinguish transport, response parsing and local settlement failures.
- The export omits prompts, reply text, error messages/bodies, URLs, authorization
  headers, cookies and arbitrary headers. Metadata values are bounded and filtered.
  The underlying research ledger still contains its original prompts/replies;
  **do not share the raw database as a diagnostic log**.

For example, a future connection failure may show `APIConnectionError` caused
by `ConnectError`, with our client ID but no server ID or HTTP status. An HTTP
429 can show the server ID and `rate_limit_exceeded`. Neither automatically
proves whether a charge occurred. Reservations remain retained and SDK automatic
retries remain disabled in both paid runners.

OpenAI recommends request-ID logging and documents client IDs for investigating
requests whose responses were lost. A client ID is a correlation identifier,
not a promise of idempotency, a receipt or permission to retry:
[official request-ID guidance](https://developers.openai.com/api/reference/overview#debugging-requests).
See also [official error guidance](https://developers.openai.com/api/docs/guides/error-codes).

## Read Logs Without API Calls

From the repository root:

```powershell
# Unresolved and explicitly abandoned requests only; reads SQLite in read-only mode.
rtk proxy .\.venv\Scripts\python.exe -B paid_study.py --diagnostics

# One request, including a successful request if its ledger ID is supplied.
rtk proxy .\.venv\Scripts\python.exe -B paid_study.py --diagnostics --request-id 659cc446cedb665c4735858c9bd74d243ed79ecf4a076cbb3f5a9574a8c2d9a9

# Offline SDK/transport and regression checks.
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

The historical failed request above predates this feature. Its `client_request_id`
and `diagnostics` are correctly `null`; we cannot invent or retroactively send
them. It retains its original timestamp, error type and reservation. The ledger
ID alone was not previously sent to OpenAI as a tracing header.

The installed OpenAI Python SDK is 1.43.1. The implementation uses its public
`with_raw_response.create()` and `parse()` APIs to capture HTTP metadata. Tests
exercise that installed SDK through an in-memory HTTPX transport; no provider
requests or credentials are needed for those tests.

At the original logging-only stage, all 84 tests passed. Those checks covered success, HTTP 429, connection
failure, timeout, missing usage, safe export, legacy missing metadata and cache
reuse. The shared ledger then stood at $4.77526755 with the original uncertain
$0.000969375 reservation unchanged. Existing pandas/NumPy deprecation warnings
remain nonfatal. This was a focused local review, not a full security scan.

## Frozen Research Boundary

This edit changes `paid_study.py`, which is part of the frozen generation-source
fingerprint. Existing calibration reports are historical snapshots, not reviews
of this new source. The original file is preserved at
`outputs/revision896_retry_v5_preflight/source_snapshot/paid_study.py`, SHA-256
`ba2146c8b611b0b68ad0a35356f7361acf15e956b68590373f6d9029da599e2b`.

No graph, receipt, ledger status or charge was rewritten to fit the new code.
No calibration was regenerated or re-authorized. Source-bound generation and
fresh re-analysis require an explicit compatibility/review decision first;
do not bypass the stale-hash guard or relabel earlier graphs. The 896-run study
remains blocked. Logging cannot replace recovery review or provider reconciliation.

## Reviewed Calibration Recovery

The later recovery review explicitly accepted the two wrapper changes, not a
general override of frozen source checks. `source_compatibility.json` binds the
original generation contract to the exact executing source, protocol, roster and
runtime. The original wrapper files and authorization remain archived. Request
bodies and parsed SDK replies were compared offline for all four models.

One failed V5 request was explicitly retired with its full $0.000969375
reservation permanently retained and billing still unverified. Its replacement
has a distinct deterministic request ID and a single-use authorization. Restart
reuses that replacement if received; it cannot pay for it twice. Another lost
reply stops again. The six original receipts were not rewritten, and offline
replay reused all 194 preceding replies without a paid call.

New receipts include actual execution hashes and retained-request links, both
verified against the shared ledger. No separate ledger, refund, automatic retry,
budget reset or main-study authorization was introduced. All 88 tests and the
896-cell offline preflight passed before resuming calibration.

Calibration subsequently completed 68/68. The cumulative ledger is $6.473344225,
including $0.017300875 in two retained unknown-billing reservations. See
[CALIBRATION_RESULTS.md](CALIBRATION_RESULTS.md) for the final accounting and
estimates. Both the calibration authorization and main-study gate are now closed.
