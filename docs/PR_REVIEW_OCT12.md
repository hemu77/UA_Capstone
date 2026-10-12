# PR Review Response: Evidence Before More Spending

October 12, 2026. Branch `r1/social-netw`, PR #1. **Revise before main collection.**
Response to the [October 7 design review](PR_REVIEW_r1_social-netw.md), retained
unchanged on the review branch.
One bounded read-only reviewer used **GPT-6.1-Sol**, with medium reasoning,
to check the supplied PR review against committed source, receipts and tables.
The parent separately inspected the deployed interface. This is an engineering
cross-check, not human bilingual review or scientific approval.

## Confirmed Findings

| Finding | Evidence and consequence | Next action |
| --- | --- | --- |
| Global degeneracy | V5: 0/17 empty, including Luna 0/14. V6: 22/26 empty, including Luna 14/14. Same roster, but prompts and randomization changed. | Keep every valid empty outcome. Test the wording hypothesis with a separately versioned, budgeted experiment; causality is not yet established. |
| Portuguese grammar | All 20 V6 Portuguese receipts include `O cenário social é os Estados Unidos.` | Candidate replacement: `O contexto social é o dos Estados Unidos.` Review all country/language compositions, not just one string. Freeze a new version before any new generation; preserve V6 exactly. |
| Translation evidence mismatch | `TRANSLATION_REVIEW.md` fingerprints cover V5, not a complete semantic review of the exact V6 prompt surface. | Review exact-version instructions, country context, cardinality, retries and permitted outputs. Label automated reviews as automated. Parsing success is not semantic equivalence. |
| Contradictory handoff | README, plan and professor note still described a V5-only simulator while deployment defaults to V6. | Corrected current-status text; retained the dated October 2 report context. |
| Multiplicity and precision | The saved power CSV already includes n=8 thresholds 2.092758 and 2.559688 SD for families of 24 and 84 tests under its stated assumptions. | Use the actual proposed family sizes in planning, not an illustrative six-test figure. These are planning calculations, not achieved power guarantees. |

Evidence: [V6 results](CALIBRATION_V6_RESULTS.md),
[V5 reanalysis](../stats/public_v5_reanalysis/),
[candidate source](../revision_next.py),
[translation record](../TRANSLATION_REVIEW.md),
[protocol](../study_protocol_next.json).
The third-party `review_experiments/` folder is absent from this checkout.
We inspected the supplied review text and committed evidence, not an unavailable
external sandbox or its full nine-experiment reproduction.

## Statistical Recommendations: Not Automatic Approvals

- Narrowing confirmatory scope to RQ1/RQ4, with RQ2/RQ3 descriptive, is a proposal
  worth evaluating. Select estimands and meaningful raw-unit effects before new
  confirmatory data. Selecting outcomes after calibration requires a held-out study.
- Homophily belongs in the primary outcome definition if the confirmatory RQ
  claims homophily. Otherwise narrow that claim. Do not rank numeric age
  assortativity directly against categorical Coleman scores.
- More repetitions of the current mostly-empty global task do not repair it.
  The old global-32 allocation and roughly $101 forecast are not approvals or
  reliable prices for a changed design.
- Pooling four configurations does not automatically make 32 IID pairs or
  guarantee a +/-0.02 interval. Predeclare the across-model estimand, weighting,
  shared-seed blocking and heterogeneity handling. Models are fixed configurations,
  not a random representative sample of all LLMs.
- A valid paired design does not require positive observed calibration correlation.
  Do not choose paired versus unpaired inference after seeing which is significant.
- At n=8 the smallest exact two-sided sign-flip p-value is 2/256 = 0.0078125,
  above Holm's first 0.05/24 threshold. This is a resolution limit, not a bug.
  Conservative Hoeffding bounds can be valid yet uninformative: their checked
  radii are approximately 1.31024 at n=8 and 0.65512 at n=32 for the reviewed setup.
- One wording per language cannot separate language from that wording. Independent
  wording variants and English paraphrases are a proposed robustness study;
  machine review cannot be relabelled human validation.
- Keep valid `NONE` graphs distinct from transport, parsing and truncation failures.
  Replacement is not automatically unbiased: failure can depend on output length
  or content. Predeclare treatment, retain attempts and report sensitivity.
  Analysis conditional on nonempty graphs answers a different question.
- The saved 12.5% false-positive simulation applies to its specified scenario;
  the review's 11.4%, spread estimates and future-cost claims were not independently
  reproduced here. Neither proves every paired-t analysis fails.

## Gated Implementation Plan

- [x] Check the review's counts, Portuguese composition, version coverage and
  documentation conflicts against saved evidence with GPT-6.1-Sol.
- [x] Preserve frozen V5/V6 prompts, graphs, receipts, hashes and accounting.
- [x] Correct current handoff text and make the simulator's persona reset easier
  to find. Correct signed-difference and tie-count explanations without changing metrics.
- [ ] Agree on the primary RQs, raw-unit effect thresholds, multiplicity families,
  missingness policy and the role of model/seed blocks with the team/professor.
- [ ] Implement a new versioned prompt candidate and exact-version automated
  multilingual review, explicitly retaining the lack of human validation.
- [ ] Design a global-clause ablation with common randomization schedules.
  A 2 x 2 x 4 models x 4 repetitions proposal is 64 graphs, not an authorized run.
  Estimate tokens/retries from actual prompt lengths before requesting a finite cap;
  do not accept an unverified "under $1" estimate.
- [ ] After approved probes, inspect results and decide whether to revise again.
  Recalibrate only the accepted version; no automatic reuse of V6 for confirmation.
- [ ] Freeze outcomes, allocation, inference and stopping rules, then request
  an exact budget and main-collection authorization. No guarantee of acceptance.

## Simulator Interpretation

The live page already had rotation, Pan, pointer zoom, Play and filters when
inspected. The supplied link selected persona 4 and incident ties, so it was a
neighborhood view rather than all edges. Clear persona selection restores the
whole network; it is now beside the camera controls above the graph. In comparison
mode Play switches to the active run's recorded events; layers are not a shared
timeline. A dashed identity link joins the same persona across runs, not a new tie.

The link also chose RQ4 while varying methods at one language. The viewer correctly
flags this as an uncontrolled RQ4 comparison. Use **Language / RQ4** for matched
languages, **Method** for algorithm comparison. Matching does not establish causality.

No paid model calls, prompt changes, new graphs or statistical approvals were
made in this review response. See [deployment verification](SIMULATOR_DEPLOYMENT.md)
for the separate software checks and their limits.
