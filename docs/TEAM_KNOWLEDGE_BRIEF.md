# Team Knowledge Brief: What We Built and What Needs a Decision

**October 2 update: revised calibration complete; main study still on hold.**
All 104 approved graphs passed replay/accounting checks at **$7.8549 of $10**.
Read [the new results](CALIBRATION_V6_RESULTS.md) and [plan](../plan.md).
The important finding is **22/26 global graphs are empty**. This is genuine
output under the frozen instructions, not a software failure to conceal.
We recommend resolving the intended global task before spending on main collection.
Human bilingual review remains absent, explicitly accepted by the owner.
Do not pool V5 or tuned calibration with fresh confirmation samples automatically.

## 1. The Project in Plain Language

We give a language model a list of fictional people and ask it to choose
friendships. Each person becomes a node; each friendship becomes an undirected
edge. We then measure which people connect and what the resulting network looks
like. We are studying **the model's generated choices**, not observing real human
friendships. An undirected tie does not establish mutual consent or who initiated it.

Two useful terms:

- **Homophily:** whether similar people connect more than expected from the
  composition of this roster. A large group will have many within-group ties
  even by chance, so a raw same-group percentage is not enough.
- **Topology:** the shape of the network, such as how dense it is, how much
  friends form triangles, and whether everyone belongs to one connected group.

A graph can be valid software output without being a realistic model of society.
Likewise, a beautiful visualization does not validate the research.

## 2. The Four Research Questions

| RQ | Question we can investigate | What it does not establish |
| --- | --- | --- |
| RQ1 | With instructions in English, does changing the stated country alter generated mixing and topology? | Actual national differences in human friendship behavior. |
| RQ2 | Which demographic attributes show stronger within-attribute mixing under the conditions? | Causal importance of an attribute, or a universal ranking of age against categorical scores. |
| RQ3 | Do four model configurations generate similar or different networks under matched settings? | Cross-vendor robustness or differences caused solely by model architecture. |
| RQ4 | With country fixed to US, do instruction-language changes alter generated networks? | Effects of participants' spoken language or fully translated persona descriptions. |

All four generation methods contribute to these questions. Method expansion is
not a fifth RQ. Comparisons across methods concern the complete generation
procedures, which differ in information, initialization and constraints.

## 3. What Was Done Earlier

The starting point was Stanford SNAP's
[LLM social-network repository](https://github.com/snap-stanford/llm-social-network).
The capstone added OpenAI-only support, experiment runners, results, notebook
analysis and project documentation.

| Earlier phase | What was implemented |
| --- | --- |
| Step 1 | OpenAI-key-only bring-up and a full 50-person sequential acceptance run. |
| Step 2 | Sequential culture study: 3 GPT-4.1-family models x 4 cultures x 2 seeds = 24 networks. |
| Step 3 | Global, local and iterative added across the same matrix = 72 networks. Together with Step 2, these addressed RQ1-RQ3. |
| Step 4 | 4 methods x 3 models x 4 languages x 2 seeds, with US culture fixed = 96 networks for RQ4. |

The earlier roster had US-specific party and race labels. Earlier instruction
languages were English, Spanish, Hindi and Japanese. Those results are history,
not interchangeable with the new Portuguese-based experiment.

The revision inventoried 192 historical study graphs. It retained 176 and
quarantined 16 containing 23 self-links. It did not silently repair invalid
graphs and pretend they were original valid observations. Earlier metric tables
and figures may use definitions that were subsequently corrected.

Next came 28 engineering-pilot graphs, used to exercise models, methods,
budgeting and visualization. They used historical personas and debugging
variants. They are not 28 extra clean repetitions for the current study.

Older reports and PDFs remain historical documents. See
[the archived capstone README](../CAPSTONE_README_ARCHIVE.md) and
[project history](../PROJECT_HISTORY.md); do not quote their completion claims
as current research validation.

## 4. Why the Reviewers Were Not Satisfied

The following summarizes the supplied review PDF as documented in the
[review-response matrix](../REVIEW_RESPONSE_MATRIX.md). We are not claiming that
the reviewers have approved our fixes or that every concern has disappeared.

| Concern | Plain-language meaning | Current response |
| --- | --- | --- |
| Homophily definition/aggregation | The calculation must match the mathematical claim and account for roster composition. | Corrected categorical Coleman calculation, per-group evidence, separate numeric age assortativity and recomputation from saved graphs. |
| Country and language mixed together | Changing two things at once prevents attributing a difference to either one. | Separate country framing from instruction language; preserve identical candidate JSON. |
| US personas treated as general populations | Saying "Japan" does not transform US-labelled fictional people into a representative Japanese population. | Fresh designed adult roster without US party/race labels; restrict claims to model sensitivity to country framing. |
| Too few repetitions | Two results can differ because of generation variability; a mean alone hides that. | Final allocation remains pending meaningful-effect/precision approval. Neither eight nor 32 repetitions is a universal validity threshold. |
| Weak reference comparisons | A generated pattern needs a meaningful reference, not just a visually different graph. | Added matched random, attribute-similarity and degree-preserving controls. They are synthetic references, not real-world validation. |
| Methods described inaccurately | Readers must know what the model actually saw and how decisions were made. | Documented local candidates, sequential initialization, iterative starting graph and add/drop rounds; recorded actual prompt actions. |
| Realism/benchmark inconsistencies | Dataset counts, provenance and network types must support claims of matching real society. | No new claim of empirical realism. Dataset reconciliation and appropriate real-network validation remain undone. |
| Limited robustness/novelty | More plots alone do not establish a mechanism, mitigation or contribution beyond prior work. | Model/configuration comparisons are supported; prompt, temperature, roster, open-weight and method ablations remain unrun. |

The diagnosis is not simply "we needed more graphs." Some issues were software
bugs; others were mismatches between evidence and claims. Spending more fixes
neither category automatically.

## 5. What Changed in the Implementation

- [x] Strict parsing validates a whole response before mutating a graph. Unknown
  IDs, self-links, duplicate choices and invalid additions/removals are rejected.
- [x] Global format repair preserves the original recoverable valid tie set;
  it cannot quietly replace a large graph with a short corrected fragment.
- [x] Retry instructions avoid illustrative IDs that could contaminate replies.
  An initially empty global reply has a declared bounded nonresponse policy.
- [x] Output limits accommodate full-roster replies: 8,192 completion tokens for
  global; 512 for per-person replies. Abnormal endings still stop collection.
- [x] Deterministic ordering stabilizes graph metrics after saving/reloading.
- [x] Receipts bind graph files, actual decisions, usage, source, roster and runtime.
  Replaying additions/removals must reproduce the saved final edge set.
- [x] A shared SQLite ledger reserves cost before a request. Cached successful
  replies prevent repayment on restart; lost responses retain their reservation.
- [x] Client/server request IDs and safe diagnostics support investigation without
  logging credentials. A tracing ID is not billing proof or an idempotency guarantee.
- [x] Reviewed recovery preserves original evidence and charges. A replacement
  needs a single-use authorization and a separate request identity.
- [x] Notebook analysis separates historical, fixture, pilot and fresh results.
  Undefined/missing evidence is not replaced with invented values.

These fixes were tested before the final calibration. The bounded reviewer found
and rechecked execution-provenance and replacement-authorization gaps. That is
independent engineering review, not human bilingual or conference peer review.

## 6. The Current Experiment

The fresh roster has 50 fictional adults aged 18-67, designed gender/religion
categories and left/center/right political orientation. It has no US party names
or race labels. Its deliberate proportions are not census estimates, and these
categories are not guaranteed to have identical meaning across countries.

Models: **GPT-4.1, GPT-5.6-Luna, GPT-6-Luna, GPT-6-Sol**. All are OpenAI model
configurations, not four independent vendors. Provider aliases do not guarantee
immutable weights. Supported decoding settings differ and are recorded.

| Method | What happens |
| --- | --- |
| Global | One response proposes the network's pairs of friends. |
| Local | Each persona selects from the other 49 candidates, without growing-network degree information. |
| Sequential | Personas act in order with current degree information after the first three local-style prompts. |
| Iterative | A local-generated network is revised through three per-person add/drop rounds. |

Country framing: US, India, Japan, Brazil. Instruction languages: English,
Hindi, Japanese, Brazilian Portuguese. Candidate fields/values stay English and
participants' spoken languages are not assigned.

We use **seven settings**, not a 4-country x 4-language full factorial:
four countries in English, plus US in the other three languages. US-English is
shared. Consequently, country-by-language interactions are not identified by
this design.

The former **4 x 4 x 7 x 8 = 896** proposal is now an illustrative allocation,
not the approved next stage. Giving global 32 and other methods eight would mean
1,568 graphs, also unapproved. Every graph has 50 people. Current calibration
seeds 21000/21001 control recorded randomization, not population size or guaranteed
provider reproducibility. Older filenames and seeds belong to preserved versions.

## 7. What Is Actually Complete Now

### Why This Model Panel Was Chosen

The selection is a **purposeful, budget-aware within-provider comparison**, not
a claim that four models represent all LLMs. The rationale is to test whether
observed patterns persist across different deployed configurations while using
one provider/authentication path and a common graph-output contract.

| Model | Intended role | Why retain it in this panel |
| --- | --- | --- |
| GPT-4.1 | Legacy reference configuration | It was present in the original capstone, giving a named reference alongside newer configurations. Fresh-roster results are still required; old-roster graphs cannot serve as matched new controls. |
| GPT-5.6-Luna | Additional lower-cost configuration | Adds a distinct configuration between the old reference and the GPT-6-labelled panel without paying standard-tier prices for every request. Its role is empirical comparison, not an assumed quality ranking. |
| GPT-6-Luna | Low-cost calibration coverage | The frozen price schedule made it the cheapest panel member, so it covered all seven settings and all four methods twice within the calibration budget. This bought broad parsing/cost coverage, not proof that other models behave identically. |
| GPT-6-Sol | Higher-priced comparison within the GPT-6-labelled pair | Lets us test whether Luna/Sol outputs differ under the same external task. A higher price or different suffix is not evidence of better social realism, known parameter count or a specific architecture. |

The old Nano/Mini configurations remain archived rather than being mixed into
the current four-model matrix. This avoids an expanding, partly populated panel.
No claim is made that their earlier results are invalid simply because they are
older, or that the newer configurations must be more robust.

This choice has limitations the team must accept: all four come from one
provider; model selection is not random; some decoding controls differ; aliases
may change; and non-US country behavior of the three non-Luna models has not
been directly calibrated. Their US multilingual coverage is now complete at one
repetition per method/language. Saved requested/resolved model IDs and settings provide
provenance, not immutable-weight guarantees. Hugging Face/open-weight models,
other vendors and GPT-6.1-Sol were discussed but **are not part of the frozen
four-model scope**. Adding them requires a revised design, access checks and budget.

### Entire Plan, From Handoff to Findings

| Stage | Scope and deliverable | Status |
| --- | --- | --- |
| Historical preservation | Keep original capstone, 28 engineering pilots and superseded calibration versions separate; retain reviewer-response history. | Complete for this handoff. |
| Corrected implementation | Shared parser/metrics, fixed roster, four instruction languages, seven settings, four methods, durable usage and replay evidence. | Implemented and tested. |
| Historical V5 calibration | 68 full-roster graphs, 204 original controls; later 2,788 separate reference draws for public reanalysis. | Preserved, not revised confirmation data. |
| Revised wording screen | 72 first replies, English/Portuguese, original/revised wording. | Complete; revised 0/36 failures, not proof of equivalence. |
| Revised calibration | 104 full-roster graphs across all model/method/language combinations; exact replay, compliance, accounting and forecasts. | Complete; $7.8549, 22/26 global graphs empty. |
| Professor/team review | Decide claim scope, model-panel rationale, translation standard, replication/precision, analysis plan and finite budget. | Awaiting approval; this is the current stop point. |
| Main collection | Freeze task/estimands, allocation, scientific decision and finite cap; collect fresh confirmation evidence under the original ledger. No automatic calibration reuse. | Not authorized or started. |
| Complete analysis | Verify all collected artifacts; per-network topology and group mixing; matched within-repetition country/language/model contrasts; honest missingness and correction reports. | Implemented analysis path, but complete-data results do not exist yet. |
| Controls | Edge-count-matched random graphs, degree-preserving rewires and demographic similarity references, kept separate from observations. Check stability/mixing before reference-tail claims. | V5 controls exist; revised/main controls remain to be specified and run. |
| RQ interpretation | Answer RQ1-RQ4 using observed effect patterns, uncertainty/precision decisions and limitations; assess whether results support the proposed contribution. | Pending final data and agreed analysis. |
| Reporting/team tooling | Update notebook, tables, publication figures, reproducibility package and optional private viewer exports. Do not invent conclusions or replay histories. | Existing tools available; final-study refresh pending. |
| Submission decision | Professor/team evaluate novelty, claim strength, ethics, benchmark needs and venue fit. | No acceptance guarantee; not implied by collection completion. |

Proposed method-specific primary outcomes are global density/degree-one share
and clustering/modularity for quota-based methods, subject to scientific approval.
The global empty-output mass now requires reconsideration before freezing them.
Density is constrained by quotas and local LCC is saturated in most runs, so
neither is a sensitive main endpoint for those methods. Demographic analysis retains
categorical group-level Coleman evidence and numeric age assortativity.
The implemented summaries preserve paired counts, means, sample standard
deviations and ranges; they do not currently produce confirmatory p-values or
independent-edge confidence intervals. The team must decide whether that
exploratory analysis is sufficient before approving main collection.

The seven-setting design does not cover every country-language combination.
Multiple independent rosters, empirical-human benchmarks, causal demographic
ablations, temperature/wording sweeps and cross-provider generalization are
**outside the current scope and cost estimate**. They are not silently treated
as resolved reviewer concerns.

### Historical V5 Calibration Evidence

- [x] 68/68 V5 calibration graphs, with adjacency, PNG and JSON receipts.
- [x] GPT-6-Luna covers all 7 settings x 4 methods x 2 repetitions: 56 graphs.
- [x] Other 3 models cover US-English x 4 methods x 1 repetition: 12 graphs.
- [x] All 68 full-roster graphs passed receipt, replay and metric verification.
- [x] 204 matched offline controls generated and stored separately.
- [x] All 88 Python tests; 896 offline fixture cells plus 12 control fixtures.
- [x] Notebook JSON/syntax checks and the fresh-study loading cell passed.
- [x] The V5 authorization window stayed below $5; the later revised calibration
  used a separate $10 approval, not this old allowance.

There were 7,749 received replies and 140 rejected parse attempts, or 1.81%.
Rejected replies were retained, charged and corrected under the bounded policy.
Two lost-response reservations remain conservatively counted with actual billing
unknown. A passing final graph is not a claim of perfect first-prompt compliance.

The old analysis says `PARTIAL` because only 68 of its former 896 target exist.
That does not describe revised calibration completeness. The new inspector says
`COMPLETE_CALIBRATION`: 104/104, 11,548 received requests, 12 first failures in
11,536 decisions, no unresolved new charges. Exactly-one US nomination prompts
had zero failures in 243 decisions per language; that is not translation accuracy.
The [new report](CALIBRATION_V6_RESULTS.md) includes method ranges and missingness.

## 8. What the Interactive Viewer Does

The private team viewer uses saved graph data, not edges reconstructed from a PNG.
It supports filters, layered comparisons, persona inspection and recorded replay
where event evidence exists. An animation cannot show unrecorded reasoning or
pretend all personas acted simultaneously when generation was sequential.

The unchanged default viewer exposes the 68 verified V5 calibration runs. Earlier
historical/pilot evidence is retained as a separate dataset because its persona
IDs refer to a different roster. Model/country/language/method comparisons use
actual matched records; the coverage grid distinguishes available results from
the eight repetitions planned per condition. Selection exports include the
underlying graph data and provenance, not just screenshots.

See the [interaction checks](VIEWER_INTERACTION_REVIEW.md) for observed flows,
tests and remaining device checks. Improving UI or figures does not resolve the
research decisions below. Source PNGs are unchanged inspection artifacts, not
finished paper figures. It does not yet display the new 104 graphs; simulation
work was explicitly deferred while the core calibration was completed.

## 9. What Remains Before Spending Again

- [ ] Approve the narrow claim: this is a controlled study of LLM-generated
  networks on one fictional roster, not national population behavior.
- [x] Owner accepted proceeding without human bilingual review. Disclose it as
  absent; the wording screen and calibration do not replace human validation.
- [ ] Resolve global empty-output behavior before proposing a main budget. If
  the task changes, use a new contract and bounded comparison, not cherry-picked reruns.
- [ ] Decide the required independent wording variants/English paraphrase
  robustness evidence. These experiments are not yet complete.
- [ ] Agree on primary outcomes, paired comparisons, missing-data/failed-run
  handling and multiple-comparison treatment before reading the full results.
- [ ] Justify precision/power, or explicitly designate the study exploratory.
  Eight repetitions are not a magic conference threshold.
- [ ] Decide whether multiple rosters, empirical benchmarks or robustness
  experiments are necessary for the intended claims. They are not in the quoted scope.
- [ ] Approve the protocol and a finite budget. Changed prompts or settings can
  invalidate calibration reuse and require a revised forecast.
- [ ] Collect the newly approved scope, verify it, perform the agreed analysis,
  and write findings only after observing the results. There is no current
  "remaining 828" commitment and no automatic calibration reuse.

## 10. Cost and Time

| Item | Conservative estimate/accounting |
| --- | ---: |
| Completed revised 104-graph calibration | **$7.854865725 of $10** |
| Entire historical + wording probe + calibration ledger | **$14.3367342** |
| Hypothetical 896 entirely fresh graphs, unchanged prompts | **$102.16**, or **$127.70** with 25% allowance |
| Hypothetical global 32 / other methods 8: 1,568 fresh graphs | **$103.61**, or **$129.51** with 25% allowance |
| 896 serial API time | **23.54 hours**, or **35.31 hours** with 50% allowance |
| 1,568 serial API time | **23.90 hours**, or **35.86 hours** with 50% allowance |

These are hypothetical additional costs, not quotes, approval, power evidence
or the old "$101 remaining" estimate. All US language cells are now measured;
non-US country costs for the other models still transfer Luna country/US ratios.
Runtime excludes future local processing, reviews and downtime. Global is cheap
because most replies are `NONE`; a revised task could cost more. Calibration
used 2.81 API hours over a 3.59-hour dispatch span, including pauses/recovery.
See [the calibration report](CALIBRATION_V6_RESULTS.md) for exact assumptions.

## 11. The Decision We Need From the Team

**Does the intended global task permit these mostly empty outcomes, or should
we revise and probe it first? Then, what claims, primary outcomes, precision
targets and fresh-sample allocation can the team defend before more spending?**

Do not approve merely because the software tests pass. Review
[TEAM_APPROVAL.md](TEAM_APPROVAL.md), record any required design changes, and then
approve both the scientific scope and a finite spending ceiling. Once those
decisions are approved and reflected in the matching execution review, we can
proceed with the remaining implementation/collection. GitHub publication itself
does not authorize paid generation.

## 12. Where to Look

Start at [the README](../README.md). Read [the current architecture](../ARCHITECTURE.md)
for file responsibilities, [the review matrix](../REVIEW_RESPONSE_MATRIX.md) for
feedback, [translation review](../TRANSLATION_REVIEW.md) for its actual scope,
and [the revised cost/results report](CALIBRATION_V6_RESULTS.md) for measured evidence.
Historical PDFs and earlier plans are not the current protocol.

Share [the professor/team review note](PROFESSOR_TEAM_REVIEW.md) with the brief.
The [reviewer agent package](../agents/README.md) is included for bounded,
read-only engineering review. Team members can use it; it does not replace the
professor's scientific judgment, bilingual reviewers or budget authorization.
