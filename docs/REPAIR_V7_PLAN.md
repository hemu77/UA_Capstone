# V7 Repair Plan, Regeneration Map and Budget

October 12, 2026. **Owner approved the staged repair plan. Offline runner implemented;
no V7 API calls made.** Stage ceilings are $5 / $10 / $15, conditional on the gates
below. This is not main-study approval. GPT-6.1-Sol exact-version review and
October 12 Standard-price checks are [recorded](V7_REVIEW_OCT12.md); a fresh local
key and actual provider-access checks remain pending. See [execution and checks](REPAIR_V7_RUNNER.md).
This is the active next-step plan following [the PR response](PR_REVIEW_OCT12.md).
V5/V6 remain historical exploratory evidence, not failed files to delete.

## 1. What Needs Fixing

| Problem | Repair | Done means |
| --- | --- | --- |
| Portuguese context grammar | Isolated candidate with full country-specific context sentences, used for normal and retry instructions. | Exact composed strings checked across four countries, all five prompt operations and singular/plural cases; automated exact-version review completed, human validation remains absent. |
| Mostly-empty global outputs | Compare two suspect clauses in a controlled 2-by-2 experiment, with identical roster/order/settings across arms. | Inspect compliance, edges, isolates and all empty outcomes. Decide what task is intended; do not select wording solely because it creates denser or significant results. |
| Language versus one particular wording | Three variants total per instruction language, including English paraphrases; keep persona data and output semantics fixed. | Independently authored/reviewed variants, provenance, robustness comparisons, and explicit disclosure that no human bilingual review occurred. |
| Homophily promised but not primary | Recommend RQ1/RQ4 as confirmatory candidates; RQ2/RQ3 remain reported descriptively. Include a justified homophily endpoint if the primary question promises it. | Team approves exact outcomes, meaningful raw-unit effects and hypothesis families before new confirmation data. No selecting the best-looking calibration outcome and reporting it as prespecified. |
| Weak power / too many tests | Recalculate precision and power for the approved estimands and families. | Repetition counts justified under sensitivity scenarios. No automatic eight, 32 or pooled n=32; shared seeds remain blocks and models remain fixed configurations. |
| Empty graphs and failed requests confused | Keep valid empty graphs, distinguish missing metrics from missing runs, retain all attempts. | No replacing an empty graph with a nicer result; no assertion that transport/truncation failures are outcome-independent. |
| Documentation drift | Current status has one repair-plan entry point; retain dated reports as history. | README, team checklist and release notes point here; claims and checkboxes follow evidence rather than anticipated work. |

**Implemented now:** `revision_v7_prompts.py` repairs Portuguese context,
defines four global-clause arms and three local wordings per language, and keeps
the frozen V6 source intact. `repair_v7.py` provides prepare, offline preflight,
review-gated authorization and budgeted execution. A 288-entry catalog covers
all countries, languages, operations, cardinalities and supported retry wording.
Alternative local phrasings are AI-authored by the implementation agent, not
independently authored human translations. GPT-6.1-Sol independently reviewed
the exact catalog. That machine review does not establish human-equivalent
translation quality or behavioral equivalence across languages.

All global arms explicitly permit `NONE` with the same response grammar. Factor
A varies the no-required-count clause; factor B varies an additional reminder
about empty networks. This tests clause salience while keeping parsing constant,
not a literal V5/V6 reproduction or a causal diagnosis of all previous changes.

## 2. What Must Be Regenerated

The 104 V6 graphs contain 26 global and 20 Portuguese graphs; five are both.
Therefore **26 + 20 - 5 = 41** graphs have prompts affected if we change global
instructions and Portuguese context only. The other 63 do not need to be thrown
away. All 104 retain value for diagnosis and historical comparison.

- **Offline reanalysis:** reuse all saved graphs, recompute metrics/null controls
  and precision scenarios with no model API charges. Mark results exploratory.
- **Global diagnostic:** 64 new graphs, all 50 personas. Four clause arms x four
  models x four repetitions, initially US-English. This isolates those clauses,
  not multilingual generality. Preserve a common empty-output representation and
  distinguish output-format failures from edge-count effects. Freeze the exact
  arms and outcome interpretation before calling models.
- **Wording diagnostic:** 96 new local graphs, all 50 personas. Four models x
  four US instruction languages x three wordings x two repetitions. Local limits
  cost while testing wording sensitivity. Two repetitions are a diagnostic screen,
  not evidence of adequate inferential power. No porting of its conclusions to
  other methods without the later method checks. All three wordings, including the
  reference, are generated fresh with matched schedules and a recorded time window.
- **Accepted-version calibration:** recommend a fresh 104-graph set matching the
  previous coverage only after the preceding stages are accepted. This avoids a
  nominally new calibration built from 41 fresh runs and 63 older runs. Preserve
  the reference wording selected by a prespecified rule; all rejected/alternative
  wording outputs stay visible as diagnostic evidence, not silently deleted.
- **Main study:** new confirmation sample after statistical/design approval.
  Neither diagnostic nor calibration graphs automatically count toward it.
  Total main-study graph count is not yet fixed; 896 and 1,568 are historical scenarios.

This recommendation totals **264 diagnostic/calibration graphs**, not 264
confirmation graphs. Stages are conditional, not a bulk order. If a probe fails,
stop there; do not pay for the later stages. If the global task changes materially
or additional roster/wording coverage is required, revise both scope and budget.

The four models remain `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol`.
The seven settings remain four English country framings (US, India, Japan, Brazil)
plus US-Hindi, US-Japanese and US-Brazilian-Portuguese. This is not a full
country-by-language factorial. Model labels and access/prices must be verified
against the actual configured provider before payment, not inferred from the
assistant model used to review code.

## 3. Cost and Runtime: Planning, Not a Quote

Source: [`cost_stats.csv`](../outputs/calibration_v6/5fb3715550db/cost_stats.csv)
and recorded rates in `paid_study.py`. Values below use historical conservative
accounting, not a fresh provider invoice. The rate values were checked against
official Standard-tier pricing on October 12; [sources and limitations](V7_REVIEW_OCT12.md).
V7 now explicitly requests Standard and records the returned processing tier.

| Stage | Cost evidence / assumption | Proposed separate stop limit |
| --- | --- | --- |
| Offline fixes, tests, saved-data analysis | $0 model-generation API cost | No API calls |
| 64 global clause probes | $3.2392 sum of per-request reservations from exact prepared prompts, historical rates and the configured 8,192-output-token cap | $5; invalid/truncated/uncertain replies still stop the stage |
| 96 local wording graphs | $5.9912 at historical per-model means; 50% allowance gives $8.9867 | $10 |
| 104 accepted-version calibration graphs | Previous comparable set cost $7.8549; revised global replies may be longer. 50% on the historical total alone gives $11.7823, not a worst-case bound | $15 |
| Combined staged permissions | Mixed historical/envelope planning sum about $17.09, with uncertain changed-prompt costs | $30 total across conditional stages; no automatic advancement |

**Global arithmetic now comes from code:** each input reservation uses the UTF-8
byte count plus the same framing allowance as the frozen paid caller. Charge
reservation is `(input_bound * input_rate * 1.25 + 8192 * output_rate) / 1e6`.
The 64 exact prepared requests sum to $3.2392444 at historical rates. This replaces
the previous hypothetical 12,000-output-token scenario. No SDK or parsing retry
is permitted for global diagnostics; a malformed answer is retained and stops
the stage. An empty `NONE` answer is valid and does not trigger a replacement.

For local wording and calibration, the sums of per-call maximum reservations are
$60.6778 and $91.7382, respectively, under offline fixture histories and with no
correction requests. These are **not predicted bills**: actual replies release
unused reservation amounts after each request. They do show why a $10/$15 stop
cannot guarantee completion. Calibration's illustration uses arm `a0_b0`, not
an accepted recipe; different real graph histories can change prompt sizes.
Recheck prices if execution occurs on a later date; actual account access still
needs checking before generation. A Standard-tier reservation is not an invoice.

**Wording arithmetic:** each model contributes 24 local graphs. Sum of the four
historical per-model local means is $0.2496316; multiply by 24 = $5.9912.
Recorded rates already include conservative-accounting behavior; do not call
this an official OpenAI list-price quote. Extra wording length or failures can
increase usage, and machine-assisted translation calls would require separate
accounting if made through the paid study provider.

Historical serial API time was **1.20 hours** for an extrapolated 96-local-graph
screen and **2.81 hours** for the completed 104 calibration. With 50% time margin,
these become approximately 1.80 and 4.22 API hours, excluding global probe changes,
rate limits, inspection, rendering and human/team decision time. No reliable
global runtime estimate follows from predominantly one-token empty outputs.

**Main-study cost is not finalized.** Previous 896-network projection was $102.16
or $127.70 with 25% allowance, under the old prompt/allocation assumptions.
It must not be reused as a promise for the repaired study. Reprice after probe
usage, outcomes and repetitions are accepted; do not charge main collection now.

## 4. Organization Without Losing Evidence

Keep existing files in place to preserve their paths and hashes. New files follow
the existing repository pattern instead of moving old artifacts:

```text
revision_v7_prompts.py                  # separately versioned prompt treatments
repair_v7.py                           # gated diagnostics and calibration runner
tests/test_revision_v7_prompts.py        # grammar and historical-source checks
tests/test_repair_v7.py                 # offline spending/replay/stage checks
docs/REPAIR_V7_RUNNER.md                # commands, verification, pending gates
outputs/calibration_v6/5fb3715550db/     # preserved historical receipts/results
outputs/repair_v7/global_probe/<hash>/  # prepared contract/catalog/fixture report
outputs/repair_v7/wording_probe/<hash>/ # prepared; no observed graphs yet
outputs/repair_v7/calibration/<hash>/   # illustrative recipe; not accepted
```

Every future experiment needs exact prompt/variant IDs, source/roster/contract
hashes, model settings, stage, schedule, request identity, usage and run status.
Use a unique namespace per contract; reuse the durable shared billing ledger and
request cache rather than starting a ledger that forgets prior reservations.
Public exports remain allowlisted and omit keys, private journals and billing files.
The simulator must not silently mix V5, V6, probes and accepted V7 results.

## 5. Gates Against Repeating These Errors

- [x] Preserve frozen V6 sources and verify their hashes against the saved contract.
- [x] Add Portuguese composition tests for four countries, all prompt operations,
  counts 1/2/8 and retry text; verify other-language output is unchanged.
- [x] Fail if the inherited context sentence disappears or appears twice; no
  silent string-repair success. Tests check grammar targets, not semantic validity.
- [x] Produce the 288-entry prompt/retry catalog before any paid call. Automated
  bilingual review must still cover that catalog hash, not an earlier version.
- [x] One authorized GPT-6.1-Sol read-only reviewer inspected the exact candidate,
  global diagnostic design, historical isolation and spending safeguards. The
  [bounded review](V7_REVIEW_OCT12.md) is not human validation or approval of
  main-study statistical design.
- [x] Compose all five prompt operations across all 16 country-language combinations
  offline, and exercise all 264 planned diagnostic/calibration fixture cells;
  paid calibration coverage remains the seven declared settings.
- [ ] Review empty-outcome rates, degree-one concentration, isolates, invalid
  responses and retry rates by language/model, not merely successful file counts.
  A warning triggers review, never deletion/replacement of valid observations.
- [x] Test interruption/resume, source drift, duplicate requests, uncertain charges
  and bounded corrections in the new offline runner integration. A changed
  version cannot reset the per-stage allowance; lost evidence blocks repayment.
- [ ] Prespecify estimands, effects, multiplicity, allocation and missingness rules.
  Exercise inference on synthetic null and alternative data before freeze;
  valid coverage alone is insufficient if intervals are uninformative.
- [ ] Confirm four-language semantics and practical output interpretation across
  all four models before broad collection. Claims remain fixed-roster sensitivity.
- [x] Run the full offline Python regression suite and all three V7 fixture
  preflights. Finite conditional stage permissions are recorded above; no
  checkbox grants main collection or replaces a live provider check.

What these gates offer is smaller, detectable failures and traceable decisions,
not a guarantee of zero defects, zero failed paid requests, research validity or
conference acceptance. No new paid stage is ready to run merely because this
plan and the first offline candidate exist.
