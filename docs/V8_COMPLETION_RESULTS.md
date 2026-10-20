# V8 Completion And Review Evidence

October 13, 2026. This report supersedes the 101/104 checkpoint for current
calibration status. It does not overwrite historical results or authorize main
collection. Read [the plain-language handoff](V8_TEAM_HANDOFF.md) first.

## Completed Calibration

| Outcome | Count |
| --- | ---: |
| Planned and completed observations | 104 |
| Valid graphs, all with the same 50-person roster | 100 |
| Valid nonempty graphs | 96 |
| Valid empty graphs, all global | 4 |
| Terminal invalid responses, all global | 4 |
| Valid local / sequential / iterative graphs | 26 / 26 / 26 |
| Valid global graphs | 22 of 26 |
| Received API responses | 11,634 |

An observation is a planned attempt, not necessarily a usable graph. A valid
`NONE` response is a graph containing all 50 people and no ties. A malformed
answer is a missing graph, not an empty graph. Undefined homophily for an empty
graph must not be converted into evidence of no homophily.

Global failures are preserved in cells 023, 072 and 089 (duplicate friendships)
and 088 (an ID outside the roster). No malformed graph was repaired by dropping
duplicates or choosing edges on the model's behalf. Missingness is 4/26 global
attempts, or 15.38%, versus 0/78 for the other methods. This is a substantive
limitation, not a passing-test loophole.

Global empty graphs are cells 022, 053, 069 and 080, all GPT-6-Luna. Relative to
V6's 22/26 empty global graphs, the new batch contains 4/26 empty and 4/26 invalid
global observations. This is descriptive version comparison, not an unbiased
causal estimate of prompt improvement or evidence of human realism.

The replay records 11,625 first decisions, 11 initial parse failures and nine
correction requests. Seven initial failures occur among 11,599 non-global first
decisions; all those graphs eventually complete. The four global failures are
terminal. Decisions within a graph are not independent statistical repetitions.

## Accounting And Recovery

| Item | USD |
| --- | ---: |
| Calibration received-token usage, conservative pricing | 7.889144775 |
| Original HTTP-500 reservation retained | 0.001172 |
| Original HTTP-520 reservation retained | 0.000971625 |
| Calibration total usage plus reservations | **7.8912884** |
| Original approved calibration allowance | 15.00 |
| New usage after the 101-observation checkpoint | 0.086225 |

The two uncertain provider responses were not assumed free. Each replacement
needed its own exact-request approval. Recovery reused 236 and 139 saved replies
respectively; it did not resend those prefixes. Original authorization anchors,
historical response rows and frozen generation sources stayed unchanged.

Both replacements succeeded. The second replacement's received-token estimate
was $0.000041. No other automatic replacement or main-study request was sent.
Conservative estimates use frozen rates and an input uplift; they are not a
reconciled provider invoice or a claim that both failed calls were actually billed.

## Main-Study Planning, Not Authorization

| Fresh design | Attempted graphs | Estimated USD | USD with 25% contingency | Serial API hours | Hours with 50% allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global/local/sequential/iterative: eight repetitions each | 896 | 102.44 | 128.05 | 26.30 | 39.46 |
| Global: 32; local/sequential/iterative: eight each | 1,568 | 104.49 | 130.61 | 27.03 | 40.54 |

These scenarios cover four models and seven settings: four countries with
English fixed, plus US framing in Hindi, Japanese and Brazilian Portuguese.
They are not a full four-country-by-four-language factorial. No calibration
reuse is assumed. The larger-global allocation adds about $2.04, but its
scientific value depends on predefined estimands, missingness handling,
precision targets and multiplicity, not just a larger graph count.

Non-US conditions for the three non-Luna-6 models were not directly calibrated.
Costs/time for 288 or 504 projected attempts respectively transfer the
GPT-6-Luna country/US ratios to those models. Only two Luna repetitions support
those ratios. Contingencies are buffers, not confidence intervals or guaranteed
upper bounds. Estimates include the observed failed-attempt costs; they do not
promise 896 or 1,568 valid graphs. A future runner still needs an explicit cap.

The completed calibration accumulated 12,096.272 seconds (3.360 hours) of
received-request duration. Its 11.314-hour first-to-last-response interval also
includes interruptions and review gaps. Main-study API-hour projections exclude
offline verification, review and other operational overhead; they are not
wall-clock delivery promises.

## Separate Hindi Probe: Candidate Not Adopted

The approved four-cell GPT-4.1 US/Hindi local probe completed at **$0.30618 of
$1**, with 123 received requests and 100.563 seconds of recorded API duration.
It compares original `alternative_2` with an appended one-line-ID clarification,
using two fresh matched seeds (34000 and 34001), not the selected `reference`.

| Arm | Completed valid graphs | Terminal failures | Observed first decisions | Initial failures | Corrections | USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original alternative 2 | 0/2 | 2 | 26 | 3 | 5 | 0.0734185 |
| Clarified alternative 2 | 1/2 | 1 | 85 | 5 | 7 | 0.2327615 |

All three terminal failures exhausted corrections with `Response must be one
line`. The clarified arm also produced a multiline friendship-pair answer,
despite the added instruction. Its only valid graph needed 54 requests for 50
decisions, including four corrections. The candidate therefore **does not pass
a reliability screen and is not adopted**. No original failure is replaced.

These tiny, truncated denominators are not 100 decisions per arm. Early failure
ends observation, and decisions are correlated within graphs. A lower observed
first-failure proportion cannot establish improvement. No causal effectiveness,
human bilingual equivalence, or successful Hindi fix is claimed. A future
alternative-format hypothesis would need a separately versioned protocol and
new authorization; unused probe allowance is not permission for more cells.
The frozen calibration's selected `reference` instructions remain unchanged.

Private cached verification and public key-free replay both passed, with earlier
ledger rows unchanged. Evidence:
[`outputs/repair_v7/hindi_candidate_probe/9a643535816c`](../outputs/repair_v7/hindi_candidate_probe/9a643535816c).
Contract: `9a643535816c13df8ae3d201a65764b4d7c31b1eb5379e963fc3d9e438856745`.
Summary: `9de66358fe3e0718fe1cf1000e1e9c51d0218954dffa036fc6d3a65816214507`.

New usage in this final continuation was **$0.392405**: $0.086225 to finish the
calibration after the checkpoint, plus $0.30618 for this separate diagnostic.
The already retained uncertain reservations were not charged again locally.

## Checks And Evidence

- [x] All 104 observations replayed against the private ledger with network
  access blocked; existing ledger rows unchanged.
- [x] All 104 observations passed public key-free replay, including failure
  outcomes, request identity, usage calculations, graph edges, metrics and events.
- [x] Saved adjacency files retain all 50 IDs with no self-links. PNG integrity
  and source metadata checked; the newly completed global PNG visually inspected.
- [x] Both retained reservations appear in the final atomic inspection report.
- [x] 191 offline regressions passed in 455.082 seconds; the final two recovery
  tests passed in 6.182 seconds; seven publishing-checkout tests passed in 20.057 seconds.
- [x] A bounded GPT-6.1-Sol reviewer checked exact-request recovery safeguards.
- [x] Complete and verify the separately authorized Hindi probe; retain all
  three failures and record the decision not to adopt the candidate.
- [x] Replay both stages again in the clean publishing checkout: 108 observations
  with network blocked, no API key and no private ledger. Results match.
- [x] Bounded read-only reviewer checked final counts, costs, forecast assumptions
  and the Hindi non-adoption decision; no remaining material claim error found.
- [x] Check all 52 changed/new publication files for blocked private paths,
  credential patterns, oversized files and broken relative Markdown links.
- [ ] Professor/team scientific decisions and explicit main-study authorization.

The new regression first reproduced stale clearance: tampering with recovery
evidence could leave a previous success label on disk. The second adapter now
invalidates the exact report under the workflow lock before contract/recovery
validation. Nonce and contract-evidence failures leave `VALIDATION_PENDING`.
Publication of success and retained-cost evidence is atomic. This change affects
verification, not prompts or generated friendships.

Calibration source contract:
`176533db70415212b5b53e72afd3ba1400dc50ba2c6847f235d19ae65202a3b8`.
Completed summary hash:
`ac632885755a801cd6891c1e318d5df67db01a41260a81690cf3fc0d4dd78f55`.
Second recovery adapter hash:
`518bdea90d03b4d7b1c5c86f46ce9548f5f64b64959d3b244efb4be8aaed0fe4`.

Public evidence directory:
[`outputs/repair_v7/calibration/176533db7041`](../outputs/repair_v7/calibration/176533db7041).
Its `inspection.json`, `inspection_networks.csv`, `condition_summary.csv`,
`decision_compliance.csv`, `hypothetical_main_forecasts.csv`, and
`detectable_effect_scenarios.csv` contain the machine-readable results.
The `runs/` and `failures/` directories preserve the source observations.
`checkpoint_inspection.json` is the prior partial-state audit, not final clearance.
Private journals, transport diagnostics, authorization files and ledger snapshots
are not part of the public handoff.

## Reproduce Without Spending

Use the repository environment and frozen sources; do not upgrade dependencies
and silently reinterpret a changed contract. The public commands need neither
an API key nor the private ledger:

```powershell
python -B repair_v8_transport_second.py inspect
python -B probe_hindi_candidate.py inspect
python -B -m unittest tests.test_repair_v8_transport_second
```

`python -B repair_v8_transport_second.py verify` additionally checks the private
ledger when it is available. The published repository intentionally omits it.
No `execute` command is required for team review. Main collection and simulator
redeployment remain outside this change. Human bilingual validation is absent;
AI review, valid graphs and passing tests do not guarantee research validity or
conference acceptance.
