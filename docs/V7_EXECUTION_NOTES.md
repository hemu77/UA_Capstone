# V7 Paid Repair Execution Record

October 12, 2026. The owner requested paid regeneration after the V7 offline
review and asked for a review comment on PR #1. This record starts before paid
execution. Results will be added only from saved, verified model receipts.

## Boundaries Recorded Before Generation

- Preserve V5/V6 evidence and all V7 attempts. Valid empty graphs are observations,
  not failures to replace. Do not overwrite old source, plots or receipts.
- Run the reviewed 64-graph global diagnostic first under its $5 cap. Wording
  diagnostics (96/$10) and calibration (104/$15) remain conditional on inspecting
  their upstream results. Main collection is not authorized.
- Keep the reviewed four models, roster of 50 fictional adults and schedules.
  No substitution based on model cost, availability or attractive results.
- Use the supplied credential only in the execution process, never in files,
  Git, command-line arguments or reports. It remains exposed in prior chat and
  should be rotated; this run does not assert that a fresh key was configured.
- Pin Standard processing, preserve returned-tier audits, and stop on uncertain
  charges. A cached received reply is replayed, not purchased again.

## Task Selection Rule Recorded Before Seeing Outcomes

The preferred candidate for subsequent exploratory calibration is `a0_b0`
with `reference` local wording: describe the entire network, allow any number
of friends, and specify the response grammar once without redundant empty-graph
reminders. This is a semantic choice, not a choice of the densest, cheapest or
most statistically significant arm. It is not yet an accepted calibration recipe.

Inspect all arms and preserve their outcomes. Evidence of task misunderstanding,
repeated output-format failures, truncation or degenerate output must be reported
and addressed before scaling. If the preferred candidate remains uninformative,
do not silently pick a better-looking arm or relabel a redesign as confirmation.
The wording screen checks robustness within the local method; it does not prove
human-equivalent translation or generalize to every generation method.

## Earlier Issues the Execution Must Not Repeat

| Earlier issue | Required safeguard |
| --- | --- |
| Portuguese country-context grammar | Exact reviewed V7 composed sentences and catalog hashes, not ad hoc replacements during a run. |
| Global empty-output concentration and matching interpretation | Preserve all global arms, edge counts and empty results; inspect instruction interpretation before calibration. |
| Quotas mechanically constrain density | Treat density as a design/compliance check for quota-based methods, not unconstrained social preference. |
| Candidate-list order confounded with persona ID | Retain the frozen independently seeded candidate orders and matched schedules. |
| Review and experiment versions diverged | Bind execution to exact source, catalog, roster and contract hashes. |
| Processing tier inherited from project settings | Explicit Standard request and matching private response audit. |
| Interrupted writes or uncertain billing | Preserve ledger, journals and reservations; no automatic replacement calls or budget reset. |
| Historical results mixed with fresh evidence | Separate versioned namespaces and explicit fixture/diagnostic/calibration labels. |
| Passing tests presented as research certification | Report software checks, behavioral evidence and remaining statistical decisions separately. |

## Live Status

The initial cumulative ledger
contained 24,790 attempts, $14.3367342 charged/reserved and zero unresolved rows.
These are historical local-accounting totals, not new charges or invoice proof.

### Verification Stop After Three Replies

The first live attempt received three valid replies, costing $0.010156 in local
conservative accounting. The two empty graphs verified. The third graph was
nonempty and exposed a software defect: replay compared in-memory edge tuples
directly with JSON lists. Its graph, metrics, decisions and schedule reproduced;
normalizing only the event representation made the event records identical.

The prior 161-test suite covered empty paid round trips but missed this nonempty
serialization path. That was a test gap, not an invalid model answer. The new
nonempty fixture failed at the same verifier before the repair, then passed after
it. It executes 64 mocked nonempty graphs and resumes with no additional requests.
Separate fixtures cover all four methods and reject real event tampering.

`repair_v7_checked.py` provides a separately fingerprinted verification adapter.
It uses the same `paid.clean_numbers` representation normalization already used
by the V6 verifier. Generation, prompts, identities, receipts, ledger and caps
remain frozen. A public `verification_adapter.json` records the exact deviation.
Use this entry point to execute or resume V7, not the original frozen CLI.

An authorized GPT-6.1-Sol read-only reviewer independently replayed the three
receipts and checked their transport evidence. The parent additionally used a
read-only SQLite connection and confirmed identical ledger-row digests before
and after replay, zero unresolved attempts and zero recovery API calls.

| Recovery evidence | Value |
| --- | --- |
| Frozen runner SHA-256 | `b99a97a548af0c51ae5090ea6b177227a9855fda27f2bcf8e0f67dc71b14bbf1` |
| Verification adapter SHA-256 | `75a200aaf4d6c12c105b8a4774726a11e497e59ef3e1f94a729308f5dc2ffa5e` |
| Canonical ledger-row digest at recovery | `8856b2348baed954af88692f3fe4525d9c081a2e211195b15f25446cc26e20a0` |

The original failing verifier is retained for the frozen source contract, not
presented as repaired in place. The adapter does not waive any graph, request,
metric, artifact, spending or service-tier checks.

The final offline suite passed **164 tests in 242.542 seconds** with outbound
socket connections blocked. Paid execution then resumed using the original
authorization and the three cached replies, not a reset budget or replacement
experiment.

### Completed 64-Graph Clause Screen

All 64 graphs passed receipt replay, graph/event/metric reconstruction, artifact
hash checks and private transport/ledger matching. The conservative charge was
**$0.1178608**; summed API response time was **95.938 seconds**. These totals
include the first three replies, not an additional purchase of them.

| Model | Graphs | Empty | Nonempty |
| --- | ---: | ---: | ---: |
| GPT-4.1 | 16 | 16 | 0 |
| GPT-5.6-Luna | 16 | 16 | 0 |
| GPT-6-Luna | 16 | 16 | 0 |
| GPT-6-Sol | 16 | 3 | 13 |

**Decision: the global task is not cleared for recalibration.** Removing the
suspect clauses did not resolve the concentration of empty outputs. The valid
empty graphs are preserved, not dropped from analysis or called parser errors.
The earlier assumption that a clause change would be sufficient was not supported.

Source: [complete 64-run summary](../outputs/repair_v7/global_probe/bbd4c3c82c66/summary.json).
Its canonical SHA-256 is
`9a4a6585730109984133d7292e18e1fdac3840c3cc6674e63a2cf75a867b2412`.

### Eight-Graph Task-Interpretation Extension

After inspecting the 64 results, the owner explicitly approved eight additional
US-English global graphs: four models x two repetitions. This extension uses
the **remaining original $5 allowance**, not a new $5 budget. Its authorization
retains the original $14.3367342 ledger baseline and $19.3367342 cumulative ceiling,
with the completed diagnostic and authorization hashes bound to the extension.

The new hypothesis is that the instruction needs to distinguish generating a
hypothetical network from recovering known relationships. This is a hypothesis,
not an established explanation of the empty results. The original
[Stanford paper, Figure C3](https://cs.stanford.edu/people/jure/pubs/llms-social-networks-aaai25.pdf)
explicitly frames the task as creating a network; this motivates checking that
distinction, not assuming a successful fix.

The extension prepends these exact sentences to the `a0_b0` reference prompt:

> Construct a hypothetical friendship network for a simulation using the fictional people below. Generate a plausible realization of friendships, rather than recovering relationships already recorded in the candidate data.

It still permits `NONE`, isolated people, disconnected components and any number
of friends. There is no forced edge count, connectedness target or example graph.
Models, personas, parsing, sampling settings and matched seeds 31000/31001 stay
unchanged. The new hypothesis, source hash and output directory remain separate.
Two observations per model can detect an obvious behavioral problem, not establish
reliability, effect significance or translation equivalence. The reference runs
occurred earlier; this is not a contemporaneously randomized comparison.

Use `global_task_probe.py preflight` for the offline fixture check. Its paid
`authorize` and `execute` commands require exact-version review and the original
allowance; they cannot grant a fresh stage budget. Do not use `repair_v7.py approve`
to re-anchor or substitute this extension for the original screen.

The bounded GPT-6.1-Sol recovery reviewer found an orphan-transport guard missing
from the extension's authorization path. The parent also identified stale
completion status when origin approval evidence failed validation. Both were
reproduced offline and repaired before any extension payment. Regressions check
that surviving transport evidence blocks reauthorization, and that invalid
origin evidence or a shifted cap clears completion with zero additional requests.
The final four extension tests passed in 9.370 seconds. The reviewer confirmed
both fixes and exact source/contract isolation; it did not certify research validity.

| Extension identity | SHA-256 |
| --- | --- |
| `global_task_probe.py` | `f3a87d7148bbd41fa1b9a8fe6cb0b4bed9315de930796d27c49a5526a17bd06a` |
| Contract | `669a46f811a557406f4331c0caf3b7fea60eefc5a56a88c3d1b0244fe1121f35` |
| Rendered English catalog | `620c48a879f60b47b449b554e5f68923c30f7ccaa068902ab3956b231cda231d` |

Earlier extension directories containing only preflight fixtures are superseded
preparations, not paid observations. Only the reviewed contract above is active.

### Terminal Response Failure, Not a Replacement Graph

The extension's first five replies produced valid nonempty graphs. Reply six,
GPT-4.1 repetition 1, was received and billed but failed strict parsing with
`Duplicate friendship choices are not allowed`. It repeated the undirected
friendships 19--35 and 38--45. Reversed ID order alone is not the parser failure:
the existing parser treats friendships as undirected. The initial live progress
description attributed this to ordering; inspection of the recorded parse error
corrected that diagnosis before any recovery or report conclusion.

`finish_global_task_probe.py` is a separate, fingerprinted continuation for the
same eight first responses. It retains the malformed reply under `failures/`,
does not create a graph or PNG for it, and does not purchase a replacement.
Only `DiagnosticInvalid` is retained as a terminal observation. Missing evidence,
uncertain transport, altered authorization and other exceptions still stop.
The original `summary.json` remains `STOPPED`; `completion.json` separately
accounts for all attempted responses, valid graphs, failures, cost and API time.

The new regression reproduces the six-response stop, refuses offline completion
with two missing responses, finishes only the remaining two, then replays all
eight without further requests or changed ledger rows. It asserts seven graphs,
one failure, and preservation of the original stop record. These are mocked
software checks; observed outcomes are reported separately below.

## Observed Extension Results and Decision

**All eight approved first responses are complete: seven valid nonempty graphs
and one retained invalid response.** All graphs have the full 50-person roster.
No replacement responses were purchased; no graph was manufactured for the failure.

| Model | Valid graphs / attempts | Edges in valid graphs, repetitions 0 then 1 | Invalid replies |
| --- | --- | --- | --- |
| GPT-4.1 | 1 / 2 | 54; repetition 1 invalid | 1 duplicate-edge reply |
| GPT-5.6-Luna | 2 / 2 | 81, 122 | 0 |
| GPT-6-Luna | 2 / 2 | 77, 73 | 0 |
| GPT-6-Sol | 2 / 2 | 80, 77 | 0 |

The extension cost **$0.0243403**, including the invalid response, and used
**33.092 seconds** of summed API response time. Together, both V7 diagnostics
cost **$0.1422011**: 72 first responses, 71 graph artifacts and one invalid
observation. This is not 72 valid graphs or a completed recalibration.
Combined summed API time was **129.030 seconds**, excluding local verification,
review and recovery time. These conservative usage-based amounts are not an
invoice reconciliation or a reliable forecast for the larger multilingual study.

Evidence: [extension completion report](../outputs/repair_v7/global_task_probe/669a46f811a5/completion.json),
[eight-attempt CSV](../outputs/repair_v7/global_task_probe/669a46f811a5/summary.csv),
[64-run CSV](../outputs/repair_v7/global_probe/bbd4c3c82c66/summary.csv), and
[16-condition clause summary](../outputs/repair_v7/global_probe/bbd4c3c82c66/condition_summary.csv).
Failure receipts are in the extension's `failures/` directory, not `runs/`.

**Interpretation:** explicit hypothetical construction is a promising task
clarification in this small English diagnostic. It does not prove the cause of
all historical emptiness, cross-language robustness, low failure rates, real
population validity, or a statistically powered research result. The duplicate
reply must remain in the denominator. Selecting only the seven successful graphs
and claiming eight successful trials would misrepresent these observations.

### Recovery and Verification Evidence

The GPT-6.1-Sol reviewer found three continuation risks before the last two paid
calls: surviving orphan receipts, write-capable cached failure replay, and an
unbound original stop report. All were repaired and regression-checked before
continuing. The final reviewer reported no further findings in that bounded scope.
The recovery source SHA-256 is
`04c68ddb67c26403590cf086882c5ef8ba26bade6822924ce22cad298c2d1f58`.
Its sidecar binds the original stopped-summary digest
`18dd08ef31592359f6956d77bf1dba1119ee73f40e19f3938808f7f85a782724`.
Neither the paid probe's source nor its contract changed during this recovery.

The final `--verify-only` run blocked outbound sockets and reproduced all eight
observations, with every ledger row unchanged. The ledger ended at 24,862
attempts and $14.4789353 cumulative charged/reserved: exactly 72 new attempts and
$0.1422011 above the pre-V7 baseline. There were no unresolved rows. The two
historical abandoned rows were preserved, not silently deleted or reused.
All 11 V6 frozen-source hashes still matched. Representative original and
extension PNGs are visual checks only; graph correctness comes from receipt
replay, adjacency and metric checks, not appearance.

Final verification of the completed repair:

- **169 offline tests passed in 353.643 seconds**, with outbound sockets blocked.
- The clean publishing checkout independently replayed **71 public graph receipts
  and one terminal failure**, with API-client creation and sockets blocked.
- All 238 staged source/document/artifact paths were compared byte-for-byte with
  the working repository. Private authorization, transport, request-journal and
  billing files were excluded; credential-pattern scanning found no matches.
- Source, Markdown, JSON and CSV whitespace checks passed. NetworkX's generated
  `.adj` comment line contains trailing whitespace; original hashed graph files
  were deliberately not reformatted to satisfy a cosmetic check.

### What Remains Before Recalibration

- [x] Preserve the 64 clause-screen observations and the separate eight task-probe attempts.
- [x] Keep the invalid reply and its cost, without creating an empty or repaired graph for it.
- [x] Complete cached replay with no new paid requests and unchanged ledger rows.
- [x] Keep V5/V6 historical artifacts and source hashes intact.
- [ ] Review the exact hypothetical-construction wording for Hindi, Japanese and
  Brazilian Portuguese before applying it outside English. Human bilingual
  validation remains unavailable and is not claimed.
- [ ] Predefine duplicate/format-failure handling for the next protocol. Options
  such as deterministic canonicalization need explicit prospective rules and
  separately reported compliance, not an ad hoc change to make this reply pass.
- [ ] Re-freeze and review the revised diagnostic/calibration contract, then
  inspect multilingual behavior before scaling. The old 104-run recipe must
  not be launched unchanged on the strength of this eight-call screen.
- [ ] Complete the conditional local wording screen and fresh calibration, then
  update cost/runtime and precision estimates from that exact accepted design.
- [ ] Obtain team/professor agreement on primary outcomes, meaningful effect
  sizes, repetition allocation, multiplicity and missingness before main spending.

No 96-graph wording batch, 104-graph recalibration or main-study generation has
been run during this V7 repair. Unused budget is not permission to bypass these
gates or expand the approved eight-call extension.

## Recheck Locally

Offline regressions require the repository dependencies, not an API key:

```powershell
python -B -m unittest tests.test_repair_v7 tests.test_global_task_probe
```

The owner can replay the extension against the original private ledger, with
network access explicitly blocked and no key required:

```powershell
python -B finish_global_task_probe.py --verify-only
```

That second command needs the private ledger and transport audits, intentionally
excluded from GitHub. Team members can inspect public prompt receipts, failure
evidence, graph artifacts and summaries; public artifacts alone do not certify
provider billing. Do not create substitute private authorization files to bypass
this boundary or run paid commands merely to reproduce a review.
