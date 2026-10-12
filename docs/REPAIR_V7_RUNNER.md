# V7 Repair Implementation and Execution Gates

October 12, 2026. **Software implemented; paid V7 generation not started.**
The owner approved staged repairs following the PR review. No main-study batch
is approved. Read [scope, regeneration map and costs](REPAIR_V7_PLAN.md) first.

## What Changed, in Plain Language

- Portuguese country context now uses complete grammatical sentences, including
  retry instructions. Other reference-language text is unchanged.
- Four global treatments isolate two instruction clauses. All allow the same
  outputs, including a valid empty graph; a parser difference cannot masquerade
  as an effect of the clause. There is no instruction to force extra edges.
- Three local wordings per language allow a narrow wording-sensitivity screen.
  All use the same fictional adults, output IDs and exact nomination quotas.
  English, Hindi, Japanese and Brazilian Portuguese are included, not Spanish.
- Existing V6 generation, scheduling, parsing, plotting and payment code is
  reused through a serial adapter. Its frozen source files are not edited.
- Historical V5/V6 results stay in their original locations. We do not delete
  empty graphs, overwrite receipts, or relabel old runs as repaired evidence.

All prompts are machine-authored. The implementation agent's composition check
is not independent bilingual validation. A separate GPT-6.1-Sol reviewer has now
checked the exact catalog and final runner; see [review evidence](V7_REVIEW_OCT12.md).
The owner waived unavailable human review, not disclosure of that limitation.

## Stages and Boundaries

| Stage | Planned coverage | Additional limit | Advancement |
| --- | --- | --- | --- |
| `global_probe` | 64 graphs: four models x four clause arms x four repetitions, US-English | $5 | Exact-version independent review and current prices first |
| `wording_probe` | 96 local graphs: four models x four US instruction languages x three wordings x two repetitions | $10 | Reviewed global results and a new exact contract |
| `calibration` | 104 graphs, same coverage as V6, all freshly generated | $15 | Both diagnostic stages accepted; reviewed recipe required |

Every graph includes the same 50 adults. Seeds are scheduling labels, not
population sizes. All four global arms share each repetition's display order.
Local actors, quotas and candidate display order are matched between wordings
and models. Provider output remains stochastic; schedules are not a guarantee
of identical model samples. The 264 runs are diagnostics/calibration, not 264
confirmation observations.

The calibration interface currently permits the reference local wording only.
If diagnostics show that another wording is needed, revise and review a new
all-method catalog rather than silently substituting a local-only phrase.
The offline `a0_b0` calibration recipe is an illustration, not a selected winner.
Choose a task on semantic grounds, not because an arm produces more edges.

## Safety and Evidence

- A single cumulative SQLite ledger is reused. Reservations occur before the
  request. No hidden SDK retries; no transport-failure replacement.
- V7 explicitly sends `service_tier="default"` (Standard) in the actual SDK
  request. The returned tier and parsed response are saved privately before
  accounting settlement. Missing or non-Standard tier stops the stage and keeps
  its reservation; a response already dispatched may still be billed.
- Each stage gets one authorization outside its version directory. Changing
  source hashes or losing that file cannot silently grant another allowance.
- Received replies are cached by complete contract, cell, settings and messages.
  Resume replays them without another model call. Missing journal/ledger entries,
  unresolved charges and historical-ledger changes block further spending.
- Global parsing or truncation failure stops the diagnostic, preserving the
  charged reply. Local tasks retain the frozen maximum of two correction calls.
  Corrections are counted and charged; they are not independent observations.
- Valid `NONE` yields a 50-node, zero-edge graph and is retained. This is distinct
  from a failed run. Offline fixtures never become paid observed-network receipts.
- A completed run is reconstructed from its saved replies, then checked against
  adjacency, events, nomination decisions, schedule, metrics and artifact hashes.
- Paid resume also requires a matching private transport audit, including tier,
  response identity, text and usage. Public graph replay deliberately does not
  need that private file; it checks graph evidence, not provider billing.
- `summary.json` becomes `VALIDATION_PENDING` before a recheck, and `STOPPED` if
  execution fails. Only successful full verification restores completion status.
  Reports include tokens, conservative charges, API time and correction counts.
- The previous stage's accepted summary hash must match. No automatic promotion
  from a successful software check to scientific acceptance or main collection.

The legacy ledger field `phase="main"` is an accounting bucket only. All V7
contracts and reports explicitly set `main_authorized=false`; this runner has
no main-study command. Keys, authorizations, private reviews, request journals
and transport audits are excluded from publication. Billing reconciliation is still needed
before claiming invoice-level costs.

## Checks Completed Offline

Prepared artifacts for this source version:

| Stage | Contract | Fixture count |
| --- | --- | --- |
| Global probe | [`bbd4c3c82c66`](../outputs/repair_v7/global_probe/bbd4c3c82c66/preflight.json) | 64 |
| Wording probe | [`5ce210921f76`](../outputs/repair_v7/wording_probe/5ce210921f76/preflight.json) | 96 |
| Illustrative calibration | [`1bec1e6b1a52`](../outputs/repair_v7/calibration/1bec1e6b1a52/preflight.json) | 104 |

These version-2 contracts bind the requested processing tier. The earlier
`683f09663490`, `eb523e1644bb` and `2b16f9469ae0` directories are retained as
superseded offline preparations, not observed results or active contracts.

All **161 tests passed in 213.026 seconds** with outbound socket connections
blocked. The 11 V7 runner tests passed separately as well. They use temporary
ledgers and intercepted SDK transport, never the owner's key or billing ledger.
Exact-message pricing, post-import source drift and actual SDK tier selection
were each observed failing before their respective fixes, then passing.

- [x] All 64 global, 96 wording and 104 illustrative calibration fixture cells
  complete without a network connection or a paid request.
- [x] A 288-entry prompt/retry catalog covers all 16 language-country combinations
  and global/local/sequential/iterative-add/iterative-drop operations.
- [x] Mock SDK tests retain empty graphs, resume without new calls, block missing
  charges, preserve caps across versions, and stop before unaffordable requests.
- [x] Timeout, invalid global reply and output truncation are not bought again.
- [x] Standard is present in intercepted request JSON; missing/unexpected returned
  tiers retain one reservation, and missing audit files block paid resume.
- [x] Frozen V6 source hashes still match its saved contract.
- [x] Independent GPT-6.1-Sol review of the final new code and prompt catalog.
- [x] Standard-tier rate values checked against official sources on October 12.
- [ ] Actual provider account access and a fresh locally configured API key.
- [ ] Paid diagnostic results, accepted final wording and new calibration results.
- [ ] Team/professor approval of estimands, effect sizes, multiplicity, missingness
  policy and repetition allocation before main collection.

These are engineering checks, not a guarantee of semantic equivalence, reliable
effect estimates, conference acceptance, or zero failed paid requests.

## Reproduce and Run

From the repository root with dependencies installed:

```powershell
python -B -m unittest discover -s tests -p test_revision_v7_prompts.py
python -B -m unittest discover -s tests -p test_repair_v7.py
python -B repair_v7.py preflight global_probe
python -B repair_v7.py preflight wording_probe
```

Those commands are offline. `prepare` writes the exact source-bound contract,
prompt catalog and planned cells. `preflight` also creates a fixture report and
a cost-reservation calculation. Neither grants permission or uses a real key.
The contract changes if source, runtime, roster, recipe or rates change.

Record the independent review in a private `review.json` inside the exact
contract directory. It must bind `contract_sha256`, set the three reviewed
checks (`prompt_semantics_reviewed`, `diagnostic_design_reviewed`,
`spending_safeguards_reviewed`), disclose `human_bilingual_validation=false`, and
provide a traceable `review_reference`. Match `rates`, today's `prices_verified_on`
and supporting `price_evidence`. The global-probe record now binds the completed
review and October 12 rate check; this private file is intentionally not in Git.
Recheck prices on a later execution date. Do not set flags merely to bypass a gate.

Only after those checks and a fresh local `OPENAI_API_KEY`:

```powershell
python -B repair_v7.py approve global_probe
python -B repair_v7.py execute global_probe
```

Resume with `execute`, never a new authorization. For a stopped request, inspect
the local summary and ledger; do not delete evidence to force a retry. Later
stages additionally need `upstream_review.json` containing the next contract
hash, an acceptance rationale, and each required prior stage's 12-character
contract ID and accepted summary hash. Calibration also takes
`--recipe path/to/reviewed_recipe.json` with `global_arm` and `wording`.
Passing these gates does not start a stage automatically.

The public simulator stays on V6 until reviewed real V7 data are explicitly
exported. No fabricated simulation, new network PNG or paid result was published
as part of this offline repair.
