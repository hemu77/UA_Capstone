# V7 Repair Plan, Regeneration Map and Budget

October 12, 2026. **Offline repairs only. No new API spending authorized.**
This is the active next-step plan following [the PR response](PR_REVIEW_OCT12.md).
V5/V6 remain historical exploratory evidence, not failed files to delete.

## 1. What Needs Fixing

| Problem | Repair | Done means |
| --- | --- | --- |
| Portuguese context grammar | Isolated candidate with full country-specific context sentences, used for normal and retry instructions. | Exact composed strings checked across four countries, all five prompt operations and singular/plural cases; later exact-version semantic review still required. |
| Mostly-empty global outputs | Compare two suspect clauses in a controlled 2-by-2 experiment, with identical roster/order/settings across arms. | Inspect compliance, edges, isolates and all empty outcomes. Decide what task is intended; do not select wording solely because it creates denser or significant results. |
| Language versus one particular wording | Three variants total per instruction language, including English paraphrases; keep persona data and output semantics fixed. | Independently authored/reviewed variants, provenance, robustness comparisons, and explicit disclosure that no human bilingual review occurred. |
| Homophily promised but not primary | Recommend RQ1/RQ4 as confirmatory candidates; RQ2/RQ3 remain reported descriptively. Include a justified homophily endpoint if the primary question promises it. | Team approves exact outcomes, meaningful raw-unit effects and hypothesis families before new confirmation data. No selecting the best-looking calibration outcome and reporting it as prespecified. |
| Weak power / too many tests | Recalculate precision and power for the approved estimands and families. | Repetition counts justified under sensitivity scenarios. No automatic eight, 32 or pooled n=32; shared seeds remain blocks and models remain fixed configurations. |
| Empty graphs and failed requests confused | Keep valid empty graphs, distinguish missing metrics from missing runs, retain all attempts. | No replacing an empty graph with a nicer result; no assertion that transport/truncation failures are outcome-independent. |
| Documentation drift | Current status has one repair-plan entry point; retain dated reports as history. | README, team checklist and release notes point here; claims and checkboxes follow evidence rather than anticipated work. |

**Implemented now:** `revision_v7_prompts.py` corrects only Portuguese context
wording, including retries. It delegates argument validation to V6, rejects
unexpected template drift, and changes no V6 source. It has no paid execution
entry point. The global wording is intentionally still V6: this module is not a
completed V7 study, independent translation, or authorization to generate.

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
accounting, not a fresh provider invoice or verified current pricing.

| Stage | Cost evidence / assumption | Proposed separate stop limit |
| --- | --- | --- |
| Offline fixes, tests, saved-data analysis | $0 model-generation API cost | No API calls |
| 64 global clause probes | $4.8312 scenario envelope for one attempt per graph at the token assumptions below; not a bound on retries | $5; can stop before all 64 finish |
| 96 local wording graphs | $5.9912 at historical per-model means; 50% allowance gives $8.9867 | $10 |
| 104 accepted-version calibration graphs | Previous comparable set cost $7.8549; revised global replies may be longer. 50% on the historical total alone gives $11.7823, not a worst-case bound | $15 |
| Combined staged permissions | Historical/envelope planning sum about $18.68, with uncertain changed-prompt costs | Up to $30 only if each stage is separately approved |

**Global arithmetic:** 16 graphs/model; assume 1,200 input and 12,000 output tokens
per first attempt. Historical input rates sum to $4.30/million tokens across the
four models; output rates sum to $19.70/million. Then
`16 * (1200 * 4.30 + 12000 * 19.70) / 1e6 * 1.25 = $4.8312`.
These are planning token assumptions, not configured or verified final caps.
Three such attempts per graph would cost $14.4936; therefore the $5 cap must
reserve before every request and stop, not promise completion. First-response
diagnostics should not be erased by retries. Actual prompt lengths, provider
rates, maximum outputs and repair policy must be priced before authorization.

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
revision_v7_prompts.py                  # offline Portuguese candidate, exists
tests/test_revision_v7_prompts.py        # scoped regression checks, exists
docs/REPAIR_V7_PLAN.md                   # this active plan, exists
outputs/calibration_v6/5fb3715550db/     # preserved historical receipts/results
outputs/global_probe_v7/<contract>/    # proposed; not generated
outputs/wording_variants_v7/<contract>/ # proposed; not generated
outputs/calibration_v7/<contract>/     # proposed; not generated
stats/revision_v7/<contract>/          # proposed; derived analysis only
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
- [ ] Produce the exact full prompt/retry catalog before any paid call. Automated
  bilingual review must cover that catalog hash, not an earlier version.
- [ ] Have one bounded read-only reviewer independently inspect the new candidate,
  global ablation and statistical design. Request permission before dispatch;
  use GPT-6.1-Sol as requested, but do not treat model review as human validation.
- [ ] Test all five prompt operations across all 16 country-language combinations
  offline; paid study coverage remains the seven declared settings.
- [ ] Review empty-outcome rates, degree-one concentration, isolates, invalid
  responses and retry rates by language/model, not merely successful file counts.
  A warning triggers review, never deletion/replacement of valid observations.
- [ ] Test interruption/resume, source drift, duplicate requests, uncertain charges
  and bounded corrections before each new paid-runner integration.
- [ ] Prespecify estimands, effects, multiplicity, allocation and missingness rules.
  Exercise inference on synthetic null and alternative data before freeze;
  valid coverage alone is insufficient if intervals are uninformative.
- [ ] Confirm four-language semantics and practical output interpretation across
  all four models before broad collection. Claims remain fixed-roster sensitivity.
- [ ] Run relevant regressions and notebook/provenance checks, then obtain finite
  stage-specific spending approval. No checkbox substitutes for that approval.

What these gates offer is smaller, detectable failures and traceable decisions,
not a guarantee of zero defects, zero failed paid requests, research validity or
conference acceptance. No new paid stage is ready to run merely because this
plan and the first offline candidate exist.
