# Final Collection Verification

## Restart-Safety Repair (October 18)

Current candidate contract:
`be0e2e5740060bad8694bdc39d10ee2bcd60c50fc67717499395ca0bd490516e`.

- Fixed publication of five exact pinned source files. Their code text was
  unchanged; normalization of mixed CRLF/LF bytes had broken fresh-clone checks.
  No historical source pins, paid receipts or graphs were changed.
- Added ledger-based reconciliation for crashes before/after the recovery
  journal and log writes. Recovery timestamps, reservations and original rows
  stay unchanged. Conflicting journals/logs are rejected rather than overwritten.
- Added actual-checkout contract validation without mocked hashes, migration
  refusal after any prior main stage, all four file-write crash boundaries,
  both manual recovery/direct launch, repeated reconciliation interruptions,
  the recovery-limit boundary and conflicting-evidence checks.
- Red evidence: the new crash-boundary test failed in six subcases against v3.
  After repair, all 34 focused tests pass with outbound sockets blocked.
- Exact committed checkout `e1d9106` independently passed all 34 focused tests
  in 34.376 seconds, 112 preflight cases and eight Hindi public replays. No
  source-hash override was used. All 68 frozen source blobs were also checked
  directly from the Git index before commit, including the five preserved files.
- The independent historical pin check confirms all 58 original source files
  and all 327 historical public artifacts remain byte-identical.
- Current offline preflight passes 112 combinations. Public replay verifies
  all eight original Hindi observations, including the one valid empty graph.
- The plan is still 896 attempts, eight per combination and 224 per method;
  the ceiling is still $120. Main collection has not started; new API spend $0.
- Canonical private accounting passes and the prompt catalog still matches the
  paid Hindi gate. Status is `LOCAL_CREDENTIAL_REQUIRED`: no key is configured,
  and no main-stage directory or spending authorization exists.
- Official model pages were rechecked October 18: per-million input/output
  rates remain [GPT-4.1: $2/$8](https://developers.openai.com/api/docs/models/gpt-4.1),
  [GPT-5.6-Luna: $0.20/$1.20](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
  [GPT-6-Luna: $0.10/$0.50](https://developers.openai.com/api/docs/models/gpt-6-luna),
  and [GPT-6-Sol: $2/$10](https://developers.openai.com/api/docs/models/gpt-6-sol).
  The existing conservative 1.25 input multiplier and $120 ceiling are unchanged.

Full-suite, independent review and committed-checkout results will be recorded
here before release. This candidate section is not yet a final clearance.

## Historical Budget Correction

October 18, 2026. Current contract:
`9d7c929fbff5a41865b741f4c0e053d8c72f00e8135e0213ae2ed6618fc37ecc`.

- Generation and analysis specify eight equal schedules, 896 attempts, 112
  combinations with eight repetitions each, and 224 attempts per method/model.
- The new failing tests reproduced `3584 != 896` and showed an oversized
  allowance reached credential lookup. After correction, allocation and cap
  tests pass; allowances above $120 fail before credentials or transport.
- Original Hindi evidence is archived/pinned and publicly replayed: eight valid
  graphs, one empty. Existing private accounting also verifies without a key.
  Paid Hindi regeneration and old-allocation main authorization are rejected.
- Checks cover eight-block inference, 24 comparisons, complete-denominator
  requirements, missing-model suppression, immutable historical analysis,
  archive/prompt tampering, direct/reopened budgets and partial/failure statuses.
- The bounded GPT-6.1-Sol reviewer independently checked all eight receipts,
  unchanged ledger rows and AST-identical prompt/parser adapter. A stale test
  that tried to purchase Hindi fixtures was updated to test read-only readiness.
- Current engine preflight passes all 112 combinations. Main has not started;
  this correction uses no model-generation API calls. A local credential is
  required at eventual launch, not another calibration purchase.

**Historical local regression: 232/232 tests passed in 446.722 seconds**, with outbound
socket connections blocked. This includes 22 current final-runner tests. No
paid API requests were made. The original publishing-checkout claim was not
sufficient: the subsequently inspected Git tree contained five normalized files
that failed the frozen hashes. The restart-safety section above supersedes that
publication claim; a clean committed checkout must pass without hash overrides.
The generated manifest has 896 unique cells and exactly eight per combination.
The notebook remains valid JSON with 65 cells; only its obsolete allocation
explanation changed. Preserved CRLF is checked without rewriting evidence bytes.

The bounded security diff review `adc192c5-9244-4fb5-9c42-98f1c9bbaeb8`
completed with no findings in its 11 source/config/report files. This is not a
whole-repository guarantee or live billing verification. The current readiness
result is `LOCAL_CREDENTIAL_REQUIRED`, with all preserved live evidence and
private accounting verified; 896 attempts, $120 ceiling, main not launched.

The following section is the prior version's evidence, not the current
allocation. See [the corrected handoff](FINAL_COLLECTION.md).

## Historical Launch Package

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
