# V8 Wording Screen and Recalibration

## Final Status, October 13

The checkpoints below are preserved history. Current status is **104/104
calibration observations completed and verified: 100 valid graphs, four global
failures**, with four valid empty global graphs. Received usage is $7.889144775;
both retained transport reservations total $0.002143625, giving **$7.8912884 of
the original $15 cap**. No main-study call was made.

The owner separately approved the exact HTTP-520 replacement. The new
`repair_v8_transport_second.py` adapter validates both recovery records, excludes
only their original attempts from logical replay, and includes both reservations
in spending. The 139 saved responses were reused. Existing ledger rows and all
frozen generation sources remained unchanged. Source SHA:
`518bdea90d03b4d7b1c5c86f46ce9548f5f64b64959d3b244efb4be8aaed0fe4`.

The separately approved Hindi diagnostic also completed and verified: **four
observations, one valid graph, three failures, $0.30618**. Original alternative 2
failed both repetitions; its clarified version failed one. It is **not adopted**.
This does not replace or modify the calibration's prospectively fixed reference.

The final [results and forecasts](V8_COMPLETION_RESULTS.md) and
[plain-language handoff](V8_TEAM_HANDOFF.md) supersede older incomplete status.
191 offline tests passed in 455.082 seconds; final recovery tests passed 2/2,
and seven focused publishing checks passed in 20.057 seconds. A failing test
exposed stale success after invalid recovery evidence; the new adapter now clears
the exact report under lock before dependency validation and publishes final
clearance/cost metadata together. Both full stages passed private and public
offline replay with unchanged ledger rows.

Fresh main projections are $102.44/26.30 serial API hours for 896 attempts, or
$104.49/27.03 hours for 1,568. Cost contingencies are $128.05/$130.61. These are
not authorized budgets, guaranteed valid-graph counts, invoice totals, or
scientific approval. The simulator remains V6; no deployment is part of this work.

```powershell
# Public, key-free replay of current evidence (no private ledger required).
python -B repair_v8_transport_second.py inspect
python -B probe_hindi_candidate.py inspect
# Additional private-ledger checks for the execution owner.
python -B repair_v8_transport_second.py verify
python -B probe_hindi_candidate.py verify
```

Use the second recovery adapter for current calibration. Earlier checkpoint
commands below document their historical state and are not the current workflow.

## Decision Before Spending

The owner authorized the revised 96-graph wording screen within its original
$10 allowance and the following 104-cell recalibration within its original $15
allowance. Main-study generation is not authorized. These are graph-cell counts,
not API-call counts: a local graph normally requires 50 first responses, and
iterative generation uses additional rounds. Caps include correction attempts.

V8 preserves every V6/V7 source and observation. It uses a new source-bound
contract in the existing stage directories, with the same canonical stage
authorization file. Changing the source cannot grant another allowance.

## What Changed and Why

- The local wording screen no longer requires the global experiment to pass.
  Its question is whether local nomination phrasing changes compliance or ties;
  global network construction is a different task. This removes an artificial
  dependency, not a failed observation.
- Global recalibration now explicitly asks for a hypothetical simulated network,
  rather than leaving it unclear whether relationships must already appear in
  the candidate data. The exact English, Hindi, Japanese and Brazilian Portuguese
  instructions are saved in the rendered prompt catalog.
- `NONE` remains a valid empty graph. Isolates and disconnected components remain
  allowed. No edge count, minimum density or connectedness target is imposed.
- The strict parser is unchanged. A malformed global response is a terminal
  failed observation with its cost retained. It does not become an empty graph,
  deduplicated graph or replacement request. Local tasks retain the existing
  maximum of three attempts per decision. Exhausting those attempts produces a
  failed graph cell, not invented remaining decisions.
- Complete and partial saved response prefixes replay read-only. Missing ledger
  rows, orphan transport/failure evidence, changed prompts, and inconsistent
  artifacts block spending. Unexpected transport failures still stop for
  reconciliation; uncertainty is not permission to repurchase.

## Prospective Scope

| Stage | Scope | Interpretation |
| --- | --- | --- |
| Wording | 4 models x 4 US instruction languages x 3 local phrasings x 2 repetitions = 96 graph cells | Wording sensitivity and compliance; not a powered language-effect test |
| Recalibration | Same 104-cell coverage as V6, with 26 cells per method | Fresh exploratory calibration of the accepted task wording |

Models: `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol`.
Languages: English, Hindi, Japanese, Brazilian Portuguese. Countries: US,
India, Japan, Brazil. The seven recalibration settings are four English country
framings plus US-Hindi, US-Japanese and US-Portuguese. This is **not** a full
country-by-language factorial. Each graph uses the same 50 fictional adults.
Candidate-order randomization, nomination quotas and independent RNG streams
remain unchanged from the reviewed engine.

Reference local wording is selected before viewing this screen's outcomes.
Alternative phrasings remain diagnostic evidence; we do not choose the variant
with the most attractive homophily, density or significance. Global uses the
`a0_b0` clauses plus explicit hypothetical construction. The earlier eight-call
English result motivated this clarification, so this revision is openly
post-diagnostic, not claimed as an untouched preregistered hypothesis.

Nomination quotas still constrain density in local/sequential/iterative methods.
Treat density there as a design/compliance descriptor, not unrestricted model
sociability. Homophily, selected identities and suitably controlled topology
are relevant conditional outcomes. Nodes and edges within a graph are not
independent experimental repetitions.

## Language Review

The implementation agent authored the new construction prefixes. A separate
GPT-6.1-Sol read-only reviewer checked their meaning and their composition with
the existing prompts. Its initial judgment found no substantive mismatch:
each asks for plausible hypothetical friendships instead of recovering recorded
relationships, without requiring nonempty output. This is AI semantic review,
not human/native-speaker validation or proof of behavioral equivalence.

## Commands and Historical Boundaries

Use `repair_v8.py` for these two prospective stages. The old V7 paid entry points
are retained for provenance, **not supported entry points for new wording or
calibration authorization**. In particular, frozen V7 does not know V8's failed-
cell artifact format. V8 checks surviving failure receipts across older contract
folders before authorization; do not bypass that check by invoking old scripts
or deleting authorization files.

```powershell
python -B repair_v8.py preflight wording_probe
python -B repair_v8.py preflight calibration
python -B -m unittest tests.test_repair_v8
```

Paid execution additionally needs the exact reviewed catalog/contract, current
pricing evidence, original stage permission and a locally available credential.
Calibration also needs the completed wording summary and a recorded decision
bound to its hash. Do not fabricate review flags to bypass these checks.

```powershell
python -B repair_v8.py approve wording_probe
python -B repair_v8_io.py execute wording_probe
python -B repair_v8.py verify wording_probe
# Only after interpreting the completed wording screen:
python -B repair_v8.py approve calibration
python -B repair_v8_io.py execute calibration
python -B repair_v8.py verify calibration
```

`verify` blocks outbound sockets and requires cached evidence. Private ledger,
request journal, transport audit and authorization files are intentionally not
published. Public graph replay is distinct from private billing reconciliation.

## Price Basis

Standard short-context input/output rates per million tokens were checked on
October 12 and rechecked unchanged October 13, 2026: GPT-4.1 $2/$8,
GPT-5.6-Luna $0.20/$1.20, GPT-6-Luna $0.10/$0.50,
GPT-6-Sol $2/$10. The existing conservative estimator applies a 1.25 input
multiplier, does not assume cache discounts, and verifies the returned processing
tier. Sources: [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[GPT-5.6-Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[GPT-6-Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[GPT-6-Sol](https://developers.openai.com/api/docs/models/gpt-6-sol).

Historical cost planning is roughly $6 for wording and $8 for recalibration,
not a quote or completion guarantee. The runner reserves each request's upper
bound before sending it and cannot exceed the original stage ceiling. Actual
usage, failures and timing must be reported after execution. No main-study
authorization follows automatically from a passing calibration.

## Execution Status

The paid wording screen is complete. The 104-cell recalibration is currently
stopped at **101 observations (97 valid graphs, four global failures)** after
a second provider interruption, HTTP 520 (details below),
under the original $15 allowance, anchored at cumulative ledger usage $20.425088
and a ceiling of $35.425088. Main collection remains unauthorized. The
runner was reviewed and passed 176 offline regressions in 294.999 seconds before
payment. The final paid-runner SHA-256 is
`bba9050afadcdd6866495474c22ba81756f25010ad498472a128184adfd3d474`.

- Wording contract: `026a883076d535553285b80395c83f55755f8a651da0a854a31631dfe8759f13`.
- Calibration contract: `176533db70415212b5b53e72afd3ba1400dc50ba2c6847f235d19ae65202a3b8`.
- Rendered catalog: `829dd92230317bb9ffb86665b0bef854640ace7715f06ca880794a20eb747dec`.

The original wording allowance is anchored at cumulative ledger usage
$14.4789353, with a cumulative ceiling of $24.4789353. This is a $10 stage
allowance, not permission to spend $24.48 more. No old authorization was reset.

The bounded reviewer caught three issues before their results were relied on:
surviving failure-only evidence at authorization, stale inspector clearance
after failed validation, and a graph-count variable overwritten by a Boolean
in wording-pair analysis. Guards and targeted regressions were added. The last
issue reproduced as a failing integer-type assertion, then passed after the
variable fix. The two analysis repairs do not change the paid runner or prompts.

### Local Checkpoint Interruption

After 22 completed wording graphs and 23 received responses of the next graph,
Windows rejected replacement of one local request checkpoint (`WinError 5`).
The ledger contained a successful received/parsed response and zero unresolved
requests. The external file holder is unknown. This was not a provider timeout
or permission to buy another response.

`repair_v8_io.py` reuses the existing reviewed local-write retry helper. It
retries only the identical request-journal file write on `PermissionError`,
at most five times, logs those events, and snapshots its source and helpers.
It never retries an API request. The paid generation source and contracts stay
unchanged. The recovery path verifies an exact before/after checkpoint hash,
proves only the parse-result addition differs, replays the 23 saved responses
with network access blocked, and uses a read-only database connection.

The original V6 I/O adapter should have been composed into this new runner from
the start. This omission has been recorded in the existing project learning;
the supported paid command above now includes that adapter. Original checkpoint
bytes and recovery evidence remain preserved.

### Completed Wording Screen

All 96 planned cells were attempted: **94 valid graphs and two terminal failures**,
not 96 successful graphs. The 4,757 requests cost **$5.9461527** conservatively
within the original $10 allowance. API duration totaled 4,931.134 seconds
(1.370 hours); observed elapsed time was 1.915 hours including local work and
the checkpoint interruption. Private cached verification and public key-free
replay both passed; verification bought no responses and left ledger rows unchanged.

| Wording | Attempted cells | Valid graphs | First decisions observed | Initial failures | Correction requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 32 | 32 | 1,600 | 1 | 1 |
| Alternative 1 | 32 | 32 | 1,600 | 0 | 0 |
| Alternative 2 | 32 | 30 | 1,548 | 6 | 8 |

The reference had one corrected Portuguese GPT-6-Luna response; Hindi reference
had zero initial failures in 400 decisions. Both terminal failures were
GPT-4.1 Hindi alternative 2. Its observed decision denominator is truncated
because exhausted corrections stop a graph; missing decisions are not successes.

Mean matched edge Jaccard was 0.646 for alternative 1 (32 pairs) and 0.662 for
alternative 2 (30 valid pairs). These are descriptive sensitivity checks, not
causal wording effects: there are only two repetitions and no unchanged-prompt
resampling control. Invalid pairs remain missing rather than assigned zero overlap.

The calibration reference was fixed before this screen, not selected afterward
for better topology. This result supports proceeding with its operational
calibration, not accepting every alternative or clearing the main study.
Machine-readable evidence lives under
`outputs/repair_v7/wording_probe/026a883076d5/` (`inspection.json`,
`condition_summary.csv`, `decision_compliance.csv`, `wording_sensitivity.csv`).

The full offline regression suite, including the separate Hindi candidate/probe,
passed **184 tests in 461.988 seconds**, with outbound connections blocked.
The candidate also passed a five-test combined prompt/contract check. Neither
software tests nor this review certify research validity or guarantee conference acceptance.

### Separate Hindi Candidate

The screen exposed two terminal Hindi `alternative_2` GPT-4.1 failures. A
[separate offline candidate](HINDI_WORDING_CANDIDATE.md) clarifies single-line
nominee IDs versus friendship pairs. Three focused tests pass, and bounded
AI review found no blocking issue. No frozen prompt, parser, saved failure or
prospective calibration reference changed. The owner subsequently authorized a
separate four-graph, $1 live probe. It is gated on completed, verified calibration
and has not run yet; offline success is not proof of resolved live failures.

### Provider Interruption, October 13

After 63 completed observations (62 valid graphs and one retained global
duplicate-edge failure), the provider returned HTTP 500 during a GPT-6-Luna
US/Japanese iterative cell, repetition 1. That cell has 236 saved received
replies followed by one uncertain request; no complete graph is claimed for it.
The runner stopped without an automatic retry. Its additional calibration
accounting/reservations at the stop totaled $5.3276665, within the original $15.

The uncertain request retains its full $0.001172 reservation, not a zero-cost
assumption. Its HTTP status, server/client request identifiers and elapsed time
are recorded privately. The owner authorized one exact replacement; it succeeded.
`repair_v8_transport.py` preserves the original attempt and full reservation,
uses a fresh deterministic replacement identity, and replays the 236 saved replies.
Frozen prompt and generation sources were not edited. The separate Hindi probe
stays gated on completed and verified calibration. Main remains unauthorized.

### Subsequent HTTP-520 Checkpoint

Continuation reached 101/104 observations: **97 valid graphs and four terminal
global failures**. The global failures are duplicate friendships in cells 023,
072 and 089, and an out-of-roster ID in cell 088. They are not empty graphs or
deduplicated successes. Another provider HTTP 520 interrupted cell 101
(GPT-6-Luna iterative, Japan/English, repetition 1) at ordinal 139. All 139
received replies are retained; cells 102 and 103 remain unattempted.

The unresolved request retains **$0.000971625**. Total calibration accounting is
**$7.8050634**: $7.802919775 received-response usage, $0.001172 for the earlier
retired request, and $0.000971625 for the new unresolved request. The original
budget anchor and ceiling are unchanged. A separate exact replacement decision
has been requested; the earlier one-request permission is not reused.

The latest full offline suite passed **189 tests in 442.480 seconds**. Four
focused recovery tests additionally cover cached resume, another lost response,
insufficient budget, orphan evidence, post-commit interruption, preserved earlier
rows and atomic publication of verification plus retained-cost evidence.
Five focused tests also passed in the clean publishing checkout in 7.752 seconds.
Its public terminal receipts replayed 101 observations without a key or private
ledger. The private checkpoint audit verified those observations plus the 139
received prefix replies, with unchanged ledger rows and no API calls.

The read-only `inspect_repair_v8_checkpoint.py` checks the stopped observations
and incomplete prefix without opening an API client or modifying the ledger.
Its separate `checkpoint_inspection.json` never grants whole-stage clearance.
Final main cost/runtime forecasts and the Hindi probe remain pending completion.
See [the plain-language handoff](V8_TEAM_HANDOFF.md) for review priorities.

```powershell
# Offline partial-state audit; private ledger required, no spending.
python -B inspect_repair_v8_checkpoint.py
# Full verification/inspection remain blocked while HTTP-520 is unresolved.
python -B repair_v8_transport.py verify
python -B repair_v8_transport.py inspect
```

The paid command is not a generic retry button. A second recovery must be
separately authorized and reviewed. OpenAI's [error guidance](https://developers.openai.com/api/docs/guides/error-codes)
describes transient server errors; it does not reconcile whether a particular
lost response was billed. Retaining the reservation is conservative local
accounting, not a claim of a verified provider charge.
