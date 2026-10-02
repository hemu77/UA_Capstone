# Professor and Team Review: Revised Calibration Complete

October 2, 2026. **Requested decision: review the findings before authorizing
main collection.** No main batch is running. The simulator was left unchanged.

Please read [the plain-language team brief](TEAM_KNOWLEDGE_BRIEF.md),
[the measured calibration results](CALIBRATION_V6_RESULTS.md), and
[the implementation/decision plan](../plan.md). The brief includes the earlier
capstone work, reviewer concerns, model rationale and remaining full scope.

## What We Completed

- [x] Revised singular/plural and global instructions, separate candidate-order
  randomization and exact prompt/schedule/response provenance.
- [x] A separately authorized 72-call English/Portuguese wording screen:
  revised 0/36 failures versus original 4/36. Cost $0.00852425.
- [x] The approved **104 full-50-person calibration graphs**, all four methods
  and models, English/Hindi/Japanese/Portuguese. GPT-6-Luna also covers India,
  Japan and Brazil in English. Other models have US-only calibration coverage.
- [x] All receipts, graph replay, metrics, PNG integrity and private accounting
  checked. **$7.854865725 of $10**, zero unresolved new requests. A local-file
  checkpoint issue was repaired without repurchasing an API response.
- [x] Public, ledger-free inspection and maintained notebook section, preserving
  old V5 evidence separately. Raw responses and corrections remain auditable.
- [x] Human bilingual review recorded as absent and waived by owner, not validated.

## Findings That Need Judgment

**Global: 22/26 graphs are empty.** The model returned the valid `NONE` response
permitted by the frozen prompt. We must decide whether unrestricted generation
with a large empty-output mass answers the intended question. We have not shown
that the `NONE` clause alone caused this. Buying more repetitions without deciding
the task is not an adequate response. Do not discard these valid outcomes.

**Quota methods:** density is constrained; local LCC is 1.0 in 24/26 graphs.
Clustering/modularity vary, but calibration ranges are not controlled treatment
effects. Iterative used about 71% of calibration cost.

**Compliance:** 12 initial failures in 11,536 decisions (0.104%), each corrected
once. There were zero failures among 243 US one-nomination decisions per language.
This supports operational compliance, not translation equivalence or population
validity. Graphs, not the thousands of decisions, are the replication units.

## Questions for Approval

1. Should global remain unrestricted, including empty graphs, or should a new,
   separately versioned wording comparison clarify the intended synthetic task?
2. Are country-framing and instruction-wording sensitivity on one fictional
   roster the accepted claim scope? No national friendship or empirical-realism
   claims follow from this design.
3. Is the purposeful single-provider model-configuration panel defensible, with
   recorded decoding differences and no architecture-only/general-LLM claim?
4. With human review unavailable, what independent wording/paraphrase robustness
   evidence is necessary? Candidate attributes remain English across conditions.
5. What raw-unit meaningful effects, precision targets, primary outcomes,
   missingness/correction policy and multiplicity families should be frozen?
6. What repetition allocation follows from those decisions? Eight or 32 is not
   a conference threshold; adaptive stages/reuse require a predeclared valid rule.
7. Are additional rosters, empirical benchmarks or other robustness studies
   necessary for the intended contribution before confirmation spending?

## Updated Cost and Time

For entirely fresh graphs using **unchanged prompts**, measured usage projects:

| Scenario | Additional usage | With 25% allowance | Serial API hours | With 50% time allowance |
| --- | ---: | ---: | ---: | ---: |
| 896 graphs: eight per cell | $102.16 | $127.70 | 23.54 | 35.31 |
| 1,568: global 32, other methods eight | $103.61 | $129.51 | 23.90 | 35.86 |

Neither scenario is approved. The low extra global cost reflects mostly empty
responses, not established research value. Prompt changes require reforecasting.
Non-US costs for three models are transferred from Luna country/US ratios;
all US language cells are directly measured but sparsely repeated. These are
conservative token estimates, not invoices or confidence bounds. Local analysis,
reviews, wording arms and downtime are not included. Current cumulative ledger
is $14.3367342; prior uncertain historical reservations remain counted.

Please record **revise first**, **approve a specified bounded next step**, or
**defer**, with reasons in [TEAM_APPROVAL.md](TEAM_APPROVAL.md). Main collection
requires an exact reviewed design and a separate finite budget authorization.
Passing software tests does not guarantee scientific validity or acceptance.

The [read-only reviewer package](../agents/README.md) is available to the team
for bounded evidence-backed engineering review. It cannot authorize spending,
provide human bilingual validation or replace the professor's scientific decision.
