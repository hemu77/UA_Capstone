# Final Collection Verification

October 18, 2026. Contract:
`5886a306a053579ed95f47011a8a41acd893c8ee75480cd25968c64658790607`.

## Offline Evidence

- **226/226 tests passed**, 494.290 seconds, with socket connections disabled.
  The 16 new tests exercise the final runner through mocked HTTP responses;
  they are software tests, not synthetic observations passed off as LLM data.
- **112/112 engine combinations passed** preflight in both the canonical work
  checkout and publishing checkout. The same frozen contract verified in both.
- **104/104 historical observations replayed**, with all **327 public files
  unchanged**. Historical failures and valid empty graphs remain distinct.
- **301 rendered prompt cases** cover all seven study settings, global/local/
  sequential/add/drop operations and nomination counts 1 through 20.
- The generated manifest contains **3,584 distinct planned attempts**. Live
  Hindi-check seeds do not overlap the 32 main schedule seeds.

## Review Fixes

Two bounded GPT-6.1-Sol read-only reviews covered statistical interpretation and
execution safeguards. No professor approval or human translation validation was
invented. The execution review found four concrete defects in the new code,
which were repaired before live use:

1. Readiness originally checked public graphs without private billing evidence.
   It now checks the original authorization, SQLite ledger, request journals,
   transport records and complete terminal receipt inventory too.
2. A cached parser-decision mismatch used a repairable exception type. It now
   raises a non-retryable integrity error, so corruption cannot trigger another
   paid correction. The regression first reproduced that unwanted call, then
   passed after the repair.
3. A completed Hindi gate could hide a missing local key. Fresh status now
   reports `LOCAL_CREDENTIAL_REQUIRED`; main authorization cannot pass it.
4. A fresh checkout tried to acquire a lock in a nonexistent directory. The
   output directory is now created before holding the OS lock. Its regression
   failed before the fix and passed afterward.

Additional tests cover received-prefix resume without changing old ledger rows,
crash after settlement but before parsing, duplicate-only normalization with raw
response preservation, terminal invalid/truncated responses, no paid resend on
HTTP 520, orphan transport evidence, lost ledgers, changed caps, budget stops
before transport and invalidation of stale readiness.

One intermediate test run failed because assertions were accidentally placed
in the wrong test method. That test-only error was corrected; the reported
226-pass run is after the fix. Earlier intermediate results are not used as
final clearance.

## Live Evidence And Scope

**Eight of eight actual Hindi global observations passed**, two per model,
with 50 nodes in each. Seven are nonempty; one GPT-5.6-Luna response is a valid
empty graph. Three replies had one redundant undirected pair each; normalization
preserved their raw text and required no extra model requests. All eight PNGs
passed image integrity checks and a nonempty GPT-6-Sol plot was visually inspected.
The public replay command passed again after the live process exited.

Conservative usage was **$0.02740295** under the separate $1 allowance; recorded
API time was **30.626 seconds**. This is token/rate accounting, not a reconciled
provider invoice. The main stage has no authorization file, ledger, or attempts.
The live process reported `READY_FOR_MAIN_EXECUTION` with `main_authorized=false`.
The key was provided through hidden input, used only in process memory, and
removed when the process ended. Future status/launch checks require a local key.

See the live-check summary under `outputs/final_study/5886a306a053/hindi_check/`
and the current [execution handoff](FINAL_COLLECTION.md). `verify --stage
hindi_check` independently rebuilds graphs from the received responses.
Readiness is recomputed; a saved status file is not spending authorization.

**Not run:** the 3,584-attempt main collection. No simulator or deployment changes
are part of this patch. Passing tests and a small live check cannot guarantee
research validity, provider reliability, billing totals, or conference acceptance.
