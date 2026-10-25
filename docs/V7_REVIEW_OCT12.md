# V7 Independent Review and Regression Evidence

October 12, 2026. **Historical pre-execution review record.** Paid V7 diagnostics
subsequently ran and exposed a nonempty-event verification defect and persistent
empty-global behavior. See the [execution/recovery record](V7_EXECUTION_NOTES.md)
for the repair, later tests and current findings; the earlier clearance below
does not certify the subsequent experiment's scientific validity.
Read the [repair plan](REPAIR_V7_PLAN.md) for scientific scope and
[runner guide](REPAIR_V7_RUNNER.md) for commands and stage gates.

## What the Reviewer Checked

One user-authorized, read-only **GPT-6.1-Sol** agent, medium reasoning, reviewed
the versioned prompt candidate, 288-entry catalog, diagnostic design,
historical isolation, request identity, cached replay and spending safeguards.
It separately reviewed the final processing-tier change and returned **no
concrete issue**, conditional on completed regressions and refreshed contracts.
Both engineering conditions below are now met.

This was a bounded review of the V7 change, not a new audit of every historical
experiment. Generated manifests were checked for provenance, scope and agreement
with code, not manually reviewed line by line. The reviewer made no edits or
model-generation API calls. This was not a formal security certification.

| Reviewed item | Exact SHA-256 |
| --- | --- |
| `repair_v7.py` | `b99a97a548af0c51ae5090ea6b177227a9855fda27f2bcf8e0f67dc71b14bbf1` |
| `revision_v7_prompts.py` | `06a2fa7d617a31ca87e1e8a115496587618e5cbb1e56c0bfa6fe0fc62a531766` |
| Canonical rendered prompt catalog | `f0c95393b6ca4b9150e42280b34671f9d1a66c46e46f2fe62c15bbdddb5bc96b` |

Catalog and contract hashes use the repository's canonical JSON digest, not a
raw JSON file-byte hash. Source hashes are file-byte SHA-256 values.

## One Accounting Limitation Repaired

The inherited caller omitted the processing tier, although estimates assume
Standard. OpenAI documents that an omitted tier defaults to `auto`, which can
inherit project configuration; the response reports the tier actually used.
[Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

V7 now sends `service_tier="default"` explicitly and binds it into the contract
and request identity. It saves the returned tier and parsed response in private
`transport/` audits before settling accounting. Missing or non-Standard tiers
stop execution with the reservation retained. Resume checks those audits against
cached replies; it cannot silently purchase a replacement for missing evidence.

The old V6 caller and original receipts remain unchanged. No evidence established
that historical requests were actually billed on a more expensive tier. This
fix removes an assumption for future V7 requests; it does not rewrite old charges.
The serial adapter reuses the frozen caller rather than duplicating its billing
implementation; it is not designed for concurrent requests.

## Verification Results

- [x] Full Python suite: **161 tests passed in 213.026 seconds**, outbound
  `socket.connect` and `connect_ex` blocked. No live provider calls.
- [x] Focused V7 runner suite: **11 tests passed**. The wire-tier assertion was
  reproduced failing before the fix, then passing after it.
- [x] Unknown and non-Standard tier fixtures stop after one attempt, preserve
  their audit/reservation and make no second request on replay.
- [x] Removing a private audit blocks paid resume. Public graph reconstruction
  remains possible without billing files and does not certify their contents.
- [x] Rebuilt all **64 + 96 + 104 = 264 offline fixture cells**, each retaining
  50 nodes; every report records `paid_requests=0`.
- [x] All **11 frozen V6 source hashes** match the saved V6 contract.
- [x] Earlier V5/V6 evidence and the superseded V7 offline preparations are
  preserved in place. No old observation is relabelled as a new result.

Current version-2 offline contracts:

| Stage | Contract prefix | Meaning |
| --- | --- | --- |
| Global diagnostic | [`bbd4c3c82c66`](../outputs/repair_v7/global_probe/bbd4c3c82c66/preflight.json) | 64 fixtures; no model responses |
| Local wording diagnostic | [`5ce210921f76`](../outputs/repair_v7/wording_probe/5ce210921f76/preflight.json) | 96 fixtures; upstream paid evidence still absent |
| Calibration illustration | [`1bec1e6b1a52`](../outputs/repair_v7/calibration/1bec1e6b1a52/preflight.json) | 104 fixtures using `a0_b0`/reference; not an accepted recipe |

## Pricing Check and Remaining Gates

Official Standard short-context list rates checked October 12, USD per million
tokens. These match the runner's recorded base rates:

| Model | Input | Output | Official source |
| --- | ---: | ---: | --- |
| `gpt-4.1` | $2.00 | $8.00 | [Model page](https://developers.openai.com/api/docs/models/gpt-4.1) |
| `gpt-5.6-luna` | $0.20 | $1.20 | [Model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| `gpt-6-luna` | $0.10 | $0.50 | [Pricing](https://developers.openai.com/api/docs/pricing) |
| `gpt-6-sol` | $2.00 | $10.00 | [Model page](https://developers.openai.com/api/docs/models/gpt-6-sol) |

Local accounting adds a 1.25 input multiplier rather than assuming cache savings.
The global first-attempt reservation sum remains **$3.2392444** under its $5
stage limit. Later $10/$15 ceilings are conditional stop limits, not completion
guarantees. The historical planning sum remains about $17.09 for all three
stages, not a quote; actual changed-prompt usage is unmeasured.

- [ ] Configure a fresh local key and confirm this account can access all four
  models. Recheck prices if execution occurs after the recorded date.
- [ ] Run and inspect the 64 global diagnostic outcomes first. Accept or revise
  the intended task on semantic grounds, not whichever produces denser graphs.
- [ ] Review subsequent wording results before choosing a calibration recipe.
  Every stage keeps its separate approval/evidence gates.
- [ ] Obtain team/professor decisions on estimands, meaningful effects,
  multiplicity, failure handling and allocation before main collection.

Human/native-speaker bilingual validation remains absent and explicitly waived.
Machine review and passing tests do not guarantee equivalent model behavior,
adequate power or conference acceptance. A dispatched request may be billed even
when local validation rejects its reply; reconcile retained reservations with
provider usage before treating local accounting as an invoice-level cost.
