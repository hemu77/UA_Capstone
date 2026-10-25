# October 17 Verification Record

Scope: the offline post-review repair package, not paid main collection.
Baseline: `9f016999f25ac224aeb60f5e5df6bd51e47bc208` on `r1/social-netw`.

## Results

- **210/210 offline tests passed**, final full run in **415.887 seconds**.
  The process removed `OPENAI_API_KEY` and blocked socket connections before
  unittest discovery. Provider calls in tests use local fixtures/mocks.
- **26/26 focused tests passed** across `test_precollection` and
  `test_inspect_calibration_v6`; 19 tests are in the new pre-collection module.
- **104/104 saved observations replayed** in both the publishing and research
  checkouts: 100 valid graphs, four retained failures, 11,634 received responses.
  All **327 pinned public calibration files** were unchanged. Private journals
  were excluded from public copying, not deleted or treated as missing graphs.
- The candidate parser preserved all **22 valid global edge sets**. It interpreted
  three duplicate-pair failure replies offline; the invalid-ID reply stayed
  invalid. Historical outcomes remain unchanged.
- Notebook JSON/schema validated; the new audit-loading cell executed. All
  **63 original cell source bodies** are unchanged; two cells were added.
- The new denominator PNG was visually inspected. Historical PNG integrity was
  verified, not claimed as a new aesthetic or scientific validation.
- **88 local documentation links** resolved. Staged credentials/private-path
  pattern scan found zero hits. CRLF-aware `git diff --cached --check` passed.
- New model-generation spend: **$0**. No paid runner or simulator was launched.

The final source/artifact pin manifest SHA-256 is
`822006626d7642e9afb81b6e700b59635a59ed2bdd2052524c3b307868075fd0`.
Use `precollection_audit.verified_report()` to verify the report, attempt state
and derivative file bindings rather than trusting this prose alone.

## Review Fixes Made Before Publishing

Two bounded read-only Ultra lanes checked execution/test risks and the
fixed-panel statistical implementation. The implementation was tightened after
review: OS-held locks replaced crash-prone sentinel locking; a separate attempt
state rejects stale reports after report-file deletion/write failures; the
complete analysis specification is hash-bound; precision output uses its declared
planning target. Blank-line tolerance matches V8 and is explicitly recorded.

An intermediate 208-test run found one real regression in the test harness:
the historical V6 notebook test selected the final two cells by position.
It now selects exactly one tagged loader/display cell. Its original corrupt-report
rejection and no-display assertions remain intact. The final 210-test run passed.

The reviewer confirmed no remaining material statistical-correctness finding in
its bounded updated scope, and confirmed the notebook regression was not weakened.
The independent pins protect accidental change, not a malicious actor who can
replace both code and manifest. The formal security-plugin snapshot is partial
because implementation changed after it opened; it is not final-commit security
certification. Local source review and the staged secret scan are separate checks.

## Not Claimed

No new live prompt validation, billing-invoice reconciliation, credential
rotation, full fresh-environment installation, Linux lock execution, simulator
browser regression or scientific/publication guarantee. Final contract adoption,
targeted changed-prompt checks and main execution approval remain in
[the current handoff](PRECOLLECTION_REVIEW.md).
