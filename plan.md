# Revised Research Plan: Fix Design Before Spending

Updated: October 2, 2026. **104 revised calibration graphs complete at
$7.854865725 of the separately approved $10. Main collection remains unauthorized.**
The earlier 72-call wording probe had its own $1 permission. Read the
[new calibration findings](docs/CALIBRATION_V6_RESULTS.md) before any spending decision.
See [the statistical/probe decision](docs/STATISTICAL_ANALYSIS_DECISION.md).
Human bilingual review is unavailable and explicitly waived by the owner; it
must remain disclosed as absent, not completed.

**Paid screen complete:** 72 first replies, $0.00852425 conservative charge,
0/36 revised failures versus 4/36 original failures. All original failures were
Portuguese one-choice cases. No replacements, uncertain charges or new networks.
The revised wording passed its engineering screen, then all 104 calibration
graphs passed replay/accounting. **22/26 global graphs are empty**; the task must
be reviewed before main collection. Statistical/sample-allocation decisions
remain open. The simulation was deliberately left unchanged.

## 1. Decision in Plain Language

Do not launch the old 896-network batch. The saved graphs are genuine, replayable
model outputs, but correct software does not make every research outcome useful.
The review identifies design weaknesses worth fixing before buying more data.

Preserve the 68 V5 calibration graphs as exploratory evidence, correct prompt
candidates and analysis offline, test a small paid probe only after approval,
then freeze a new study with a defensible sample allocation. **896 is no longer a
committed target.** Increasing global to 32 repetitions while retaining eight for
the other methods would produce **1,568**, not 896, graphs.

No honest reviewer can promise zero errors, zero wasted calls or conference
acceptance. The safeguard is small gated batches, immutable evidence, bounded
retries, a finite authorized budget, and stopping before repeating a defect.

## 2. Review Findings and Independent Checks

The supplied review text was inspected against local code and all 68 receipts.
Its `review_experiments/` scripts and bundle were not available here; this is an
independent check, not a claim to have rerun that missing bundle.

| Issue | Verified evidence | Consequence / correction |
| --- | --- | --- |
| Nomination quotas | Local/sequential/iterative initialization draw `floor(clamp(Exponential(scale=5), 1, 20))` choices per actor. | Constrains density, but does **not** fix final degrees or density: duplicate undirected nominations collapse into one tie. Study whom models choose under a quota, not spontaneous friendship quantity. |
| Saturated connectivity | All 17 local calibration graphs have LCC share 1.0. | Local connectivity cannot distinguish these saved conditions. Keep it as a diagnostic. |
| Global matching behavior | The two Hindi GPT-6-Luna graphs have 25 and 24 edges; 48/50 nodes have degree one in each. | Clarify whole-network output without forcing connectivity or an edge count. Not every non-English graph is a perfect matching; bimodality is not established by these few observations. |
| Portuguese retries | GPT-6-Luna US local: 25/100 decisions re-asked in Portuguese, 3/100 English, 4/100 Hindi, 0/100 Japanese. | Cardinality-one failures concentrate the problem. The plural/comma wording explanation is plausible, not a proven cause. |
| Candidate order | Global/local/sequential and iterative initialization used ID order. Iterative add/drop already shuffled candidates. | Separate display RNG removes persistent identity-position coupling. It does not prove absence of all order effects. |
| Public analysis | Original analysis opens the private ledger. | New `analyze_saved_study.py` rechecks graphs/receipts without a key or ledger. Spending reconciliation stays private. |
| Model configuration | GPT-4.1 uses temperature 0.8; Luna/Sol use reasoning off without the same temperature setting. | Compare configurations, not isolated architecture, generation, intelligence or vendor effects. |
| Review's language comparison | English pooled four countries; non-English used US. | Confounded as RQ4 evidence. New descriptive contrasts match US against US. No copied exploratory p-values. |
| Power table | At n=8, alpha=.05/288 and 80% power, exact two-sided noncentral-t MDES is 3.097 SDs of paired differences, not 2.11. | Correct the planning table. These distributional assumptions are not established by the calibration. |

The 50 people are one **designed fictional adult roster**, not 50 observations
from a 1,000-person population. Seeds are randomization labels. Neutral political
orientation avoids US party names but is not universally equivalent across
cultures. More repetitions do not establish population validity.

## 3. Completed Offline Work

Checked boxes mean implemented and checked offline, not human/model validation.

- [x] Preserve all seven frozen V5 source files; isolate the candidate in
  `revision_next.py` and `study_protocol_next.json`.
- [x] Add singular instructions for exactly one friend, without comma-separated
  plural wording, in English, Portuguese, Hindi and Japanese.
- [x] Revise Portuguese country articles, Hindi's unchanged-English-data wording,
  Japanese non-duplicate wording and plain-language global directionality.
- [x] Clarify global means the entire network, not one-to-one matching. Zero,
  one or multiple friends and disconnected groups are permitted. Explicit `NONE`
  represents no friendships; an empty response is not silently accepted.
- [x] Preserve the first recoverable valid global tie set during corrections.
  Carry required pairs in the retry; reject omissions or invented ties.
- [x] Separate actor, nomination-count and display-order RNG streams. Shuffle one
  display permutation per repetition, shared across conditions and filtered by
  eligibility. Record schedules; seeds do not guarantee API determinism.
- [x] Exercise the actual generation engine with 56 full-roster offline fixtures:
  four methods x seven settings x two seeds. Check replay and common schedules.
- [x] Export 144 instruction/retry cases and a 72-first-attempt wording manifest.
  Preparation exports are fixtures; separately saved paid results now complete
  the screen. Neither dataset is a set of new model-generated graphs.
- [x] Recompute all 68 reviewed saved graphs without private billing access;
  verify receipt, adjacency, PNG, protocol/source and runtime contracts.
- [x] Add degree-one share, isolation and degree concentration. Keep undefined
  quantities undefined, rather than inventing zero values.
- [x] Generate 2,788 offline reference graphs: per observed graph, 20 edge-count
  controls, 20 degree-preserving rewires and one deterministic similarity rule.
  All rewiring targets completed; this is **not proof of adequate mixing**.
- [x] Add decision-level compliance and exact paired-t power sensitivity tables.
- [x] Obtain one bounded read-only code review; no material issue reported.
  This is not human translation validation or scientific approval.
- [x] Record final regression/notebook checks in Section 8.

## 4. Research Scope and Outcomes

**Models:** `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol`, subject to current
access/price verification before payment. No invented capability ranking or
substitution. This is a purposeful single-provider configuration panel.

**Methods:** global, local, sequential, iterative. **Framings:** US, India, Japan,
Brazil. **Instructions:** English, Hindi, Japanese, Brazilian Portuguese, not
Spanish. The same 50 adult profiles and English JSON values stay fixed.

Seven settings: US/English, India/English, Japan/English, Brazil/English,
US/Hindi, US/Japanese, US/Portuguese. This does **not** identify country-language
interactions. A four-by-four factorial needs separate approval and costing.

| RQ | Comparison | Proposed evidence and limits |
| --- | --- | --- |
| RQ1: country framing | India/Japan/Brazil versus US, English fixed, within model/method. | Method-specific topology and attribute-specific mixing/reference differences. Not national behavior. |
| RQ2: demographic mixing | Gender, religion, political orientation; age separately. | Coleman indices and each attribute's own reference comparison; age assortativity. Descriptive, not causal dominance. Never directly rank age against categorical scores. |
| RQ3: configurations | Six model pairs within identical method/setting. | Matched schedules, distributions, graph overlap and topology. Includes decoding differences; no cross-provider claim. |
| RQ4: instruction wording/language | Hindi/Japanese/Portuguese versus English, US fixed. | Reviewed variants and English paraphrase control. Until robustness is checked, sensitivity to these wordings, not pure language causation. |

Proposed primary outcomes: **global density and degree-one share**; **clustering
and modularity** for the three quota-based methods. LCC, isolation, degree Gini,
homophily and age assortativity remain secondary/descriptive. RQ2 needs its own
predeclared family if confirmatory claims are desired. Do not choose a
matching-like threshold after seeing results.

`G(n,m)` controls hold total edges fixed; degree-preserving rewiring also holds
each person's degree fixed. Equal-weight demographic similarity is a deterministic
reference, not human behavior. Adjustment describes structure, not removal of all
bias or proof of cultural causality. Twenty draws are an offline starting point;
expand and check stability/mixing before strong reference-tail claims. Reference
percentiles are **not** confidence intervals for treatment effects.

## 5. Remaining Gates, in Execution Order

### A. Human Review and Wording Probe

- [x] Owner decision recorded: human bilingual review is unavailable and waived.
  This removes the operational requirement but does not supply validation.
  AI assistance must not be presented as native-speaker review.
- [ ] Obtain a second independently produced translation per non-English language
  and an English paraphrase. We have **not** produced/validated that independent
  set. Label wording variants in receipts and analyses.
- [x] Approve up to $1 for the initial probe: GPT-6-Luna, English/Portuguese, counts
  1/2/8, six actors, original/revised wording = 72 first attempts. Keep candidate
  data/order identical across arms; randomize and log call order before execution.
  Evaluate first replies, not only successful corrected answers.
- [x] Declare engineering screen before payment: zero revised-arm parse failures
  required for `NO_FAILURES_OBSERVED`; exact descriptive intervals reported.
  Six actors per stratum do not establish close equivalence. A ratio to an
  English rate of zero is not a sound gate; no main-study approval follows.
- [x] Complete the 72-call probe and verify all reply hashes against the ledger:
  72 unique received requests, zero replacements and no uncertain charges.
- [x] Log cardinality, eligible IDs, self/duplicate ties, truncation, retries,
  usage and cost by count/language/model/wording. Stop on systemic failures or
  uncertain paid outcomes rather than repeatedly paying for blind reruns.

Live evidence is recorded separately. `revision_next.py` stays offline-only;
`wording_probe.py` adds only the authorized first-response probe, not a full-study
execution path. Every future batch needs its own scope/budget approval.

### B. New Calibration and Robustness

- [x] Integrate the revised adapter into `calibration_v6.py`, with the existing
  ledger/caller, exact request journaling, cached replay and explicit finite-cap
  authorization. A bounded review and 128 Python tests passed before payment.
  The owner then authorized `extended104` and $10. See [runner instructions](docs/CALIBRATION_V6.md).
- [x] Prepare new source/prompt/roster/runtime/response-limit/decoding contract
  `5fb3715550db`; 104 full-roster offline fixtures and their saved artifacts pass.
  This engineering freeze is not paid authorization. Never relabel V5 graphs as
  revised runs or mix versions in analysis.
- [x] Generate the 68-run revised core after the wording screen passed, preserving
  V5 outputs separately. This core alone does not cover every model/language.
- [x] Add the separately approved **36 multilingual graphs**: three other models
  x three non-English languages x four methods. Combined total **104**, all verified.
- [ ] Run the translation-noise arm on global/local with Luna; inspect full
  distributions. Size, wording schedule and budget must be fixed in advance.
- [x] Reforecast from all 104 graphs; verify current access/prices. New hypothetical
  full-fresh-sample estimates are $102.16/896 or $103.61/1,568 before allowances.
- [ ] Resolve the global task: 22/26 valid `NONE` outcomes. Either accept the
  unrestricted-task estimand and its limitations, or prerecord a bounded,
  separately versioned wording comparison. Do not selectively regenerate empties.

Do not discard valid sparse/disconnected graphs because they look odd. Accept
on protocol compliance, not on producing a preferred result. Changed instructions
create a changed experiment and require fresh evidence.

### C. Statistical Plan and Allocation

- [ ] Define smallest scientifically meaningful differences in raw metric units
  and equivalence margins. They are currently **unset**, not invented.
- [ ] Review 32 global and 8 local repetitions per model/setting as a precision
  stage; choose sequential/iterative targets afterward using a predeclared rule.
  All four methods remain in scope. No silent expensive-method cut or fixed total.
- [ ] Freeze estimands, model-by-setting effects, uncertainty and failed-run policy.
  Network generations are repetitions, not nodes, edges, decisions, retries or
  controls. Block on a recorded common schedule only when justified. Check
  heterogeneity before pooling configurations.
- [ ] Proposed multiplicity: per method, RQ1 and RQ4 each have 4 models x
  3 baseline contrasts x 2 outcomes = **24** hypotheses; RQ3 has 6 model pairs x
  7 settings x 2 outcomes = **84**. Holm within each family is **not** study-wide
  control. Professor must accept these families or require broader adjustment.
- [ ] Validate the selected inferential procedure offline under correlated metrics,
  bounded/bimodal global outcomes, missingness and small samples. Mixed-model/
  t-test assumptions cannot be presumed valid from two calibration repetitions.
- [ ] Keep adaptive variance stages separate from confirmation, or approve a valid
  predeclared reuse/stopping procedure. Do not pool tuned calibration merely to
  save money. Null significance is not equivalence; margins and valid intervals
  or tests must support any bounded-null conclusion.
- [ ] Decide whether a second independent roster is necessary. More seeds do not
  answer roster generalization. Real-network validation is required for realism
  claims and is not supplied by synthetic controls.

Power sensitivity is executable in `stats/public_v5_reanalysis/power_sensitivity.csv`.
Eight pairs detect about 1.156 SDs of paired differences unadjusted, 1.650 with a
six-test Bonferroni bound; 32 pairs reduce these to 0.511 and 0.652. These are
normal-difference paired-t assumptions, **not measured power**, and not a guarantee
for the proposed 24/84-family design.

### D. Explicit Spending Decision

- [ ] Professor/team accept exact revised claims, outcomes, robustness and precision
  targets, acknowledging remaining external-validity limits.
- [ ] Author approves a finite cumulative dollar ceiling and specific batch scope.
- [ ] Execution owner verifies gates, access and prices; preserves original ledger,
  caches and reservations; rotates exposed credentials without publishing them.
- [ ] Run the smallest approved batch, inspect evidence, then resume only within
  that approval. The main batch remains blocked today.

## 6. Cost and Runtime: What We Can Honestly Say Now

Offline revisions/tests make no generation API calls; coding service usage is
separate. The completed wording screen cost $0.00852425 and the newly approved
104-graph calibration **$7.854865725 of $10**. Current cumulative ledger is
**$14.3367342**, including all historical charges/reservations. New requests are
fully reconciled; two historical abandoned reservations remain counted.

Current forecasts below assume entirely fresh samples and **unchanged prompts**:

| Scenario | Graphs | Additional estimate | With 25% cost allowance | Serial API hours / with 50% allowance |
| --- | ---: | ---: | ---: | ---: |
| Eight per model/method/setting | 896 | $102.16 | $127.70 | 23.54 / 35.31 |
| Global 32; all other methods eight | 1,568 | $103.61 | $129.51 | 23.90 / 35.86 |

Neither is approved or a guarantee. Mostly empty global responses make the
second scenario cheap, not automatically useful. Any revised wording needs a
new forecast. Non-US costs for three models transfer Luna country/US ratios;
all US language costs are directly but sparsely measured. Future local analysis,
reviews and downtime are additional. Actual calibration took 2.8148 summed API
hours over a 3.5852-hour dispatch span including pauses and local recovery.

### Historical Planning Estimates (Superseded)

The table below preserves the earlier V5-based discussion, not current quotes:

| Scenario | Graphs | Old-rate estimate | Status |
| --- | ---: | ---: | --- |
| Retained V5 calibration | 68 | $2.37807 | Already generated; excludes other historical/superseded charges. |
| Completed wording screen | 0 (72 replies) | $0.00852425 actual conservative charge | Narrow English/Portuguese compliance screen, not calibration networks. |
| Original eight-repeat design | 896 | $103.37 full / about $101 remaining | Superseded unchanged-protocol forecast. |
| Global 32 plus local 8, four models/seven settings | 1,120 | $3.64 + $14.46 = **$18.10** | Proposal, not permission or quote. |
| Global 32, other three methods 8 | 1,568 | **$106.10** | Illustration; excludes new preflight, contingencies, price changes. |

Review estimates of a sub-$1 probe, ~$2.4 replacement calibration and $5-10 wording
arm must be bounded and rechecked, not promised. Changed global instructions can
increase output tokens substantially. Multilingual calibration, variants, separate
confirmation or rosters add costs. The old $128 cumulative suggestion is neither
new authorization nor a promise this scope fits. Final scientific allocation is
still unset; the current unchanged-prompt estimates are above.

Old API-only runtime was about 28.44 remaining serial hours, 42.66 with the old
allowance. It cannot be transferred unchanged to different graph counts, prompts
or retry rates. Gate B's updated measured estimates are now delivered above.

## 7. Files, Reproduction and Team Handoff

| Location | Meaning |
| --- | --- |
| Original V5 sources, protocol, outputs/stats | Preserved historical contract/calibration, not the new protocol. |
| `revision_next.py`, `study_protocol_next.json` | Offline prompts, schedules, parser adapter and blocked protocol candidate. |
| `outputs/offline_revision_v6/` | Prompt catalog, preparation-only manifest, schedule and hashes. Never model observations. |
| `wording_probe.py`, `resume_wording_probe.py` | Frozen $1 probe and reviewed checkpoint-specific recovery; no main-study path. |
| `outputs/wording_probe_v6/` | Completed paid report, 72 raw response receipts, hashes and original authorization; no graphs. |
| `statistical_review.py`, `outputs/statistical_review_v6/` | Offline statistical safeguards and 8,000 synthetic null worlds; not scientific approval. |
| `calibration_v6.py`, `outputs/calibration_v6/<contract-hash>/` | Frozen revised runner, manifest, labelled fixtures and 104 separately approved paid artifact sets. |
| `calibration_v6_io.py`, `recover_calibration_v6_checkpoint.py` | Audited local-file retries and one checkpoint-specific recovery; no paid reply replacement. |
| `inspect_calibration_v6.py`, `docs/CALIBRATION_V6_RESULTS.md` | Public replay/metrics/compliance verification, forecasts and findings requiring a team decision. |
| `analyze_saved_study.py` | Public graph/receipt reanalysis; no key/private ledger needed. |
| `stats/public_v5_reanalysis/` | V5 metrics, descriptive contrasts, compliance, reference draws/summaries, power sensitivity and provenance report. |
| `analyze_networks.ipynb` | Historical sections retained; new review section loads isolated public analysis with hash checks. |
| `tests/test_offline_revision.py` | Contract, replay, RNG, retry, zero-edge, key-free analysis and power regressions. |

From the repository root with the pinned Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -B revision_next.py --prepare
.\.venv\Scripts\python.exe -B analyze_saved_study.py --control-repetitions 20
.\.venv\Scripts\python.exe -B inspect_calibration_v6.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
npm --prefix viewer test
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
```

Reanalysis requires the reviewed runtime and verifies source/report/receipt/artifact
contracts. Do not bypass that check for different dependencies. It does not need
the private billing DB or independently audit provider invoices. The viewer still
displays saved V5 graphs; no offline fixture is inserted into that dataset.

## 8. Verification Record

Completed on October 2, 2026, with the pinned Python 3.11.9 environment:

- [x] Latest revised-calibration suite: **142 tests passed in 162.368 seconds**.
  Reviewer-found stale-report and retained-notebook-state defects were reproduced
  with failing tests, fixed and rechecked. Frozen generation sources unchanged.
- [x] **104 real graphs** verified via exact request/parse/event replay and
  artifact/metric checks; 11,548 unique requests. Original-ledger reconciliation
  passed with zero unresolved new requests and total additional $7.854865725.
- [x] Latest notebook: **45 code cells handled**, with 32 archival bodies skipped;
  source unchanged by execution, revised receipt/CSV hashes checked, no API calls.

The earlier checkpoints below are retained as history, not the latest suite totals:

- [x] Original offline revision: **104 tests passed**, including roster-drift and
  unauthorized-preparation checks. Additional probe/statistical checks are
  recorded below; the 104 count is historical, not the final expanded suite.
- [x] Final expanded Python suite: **116 tests passed in 50.187 seconds**,
  including first-response accounting, no-repurchase recovery, missing-receipt
  rejection, hidden credential fallback, and statistical safety checks.
- [x] Subsequent calibration-runner integration: **128 Python tests passed in
  165.155 seconds**, including intercepted SDK requests across four models/four
  methods, receipt replay, zero-edge global results, correction preservation,
  cached execution, cap enforcement and rejection of abandoned-request replacement.
- [x] Calibration V6 preflight: **104 labelled full-roster fixtures**, zero paid
  requests. Re-read/replayed all adjacency files, checked all PNGs and JSON hashes,
  and compared every metrics CSV value to the graph measurements. Viewer tests
  remain **24/24 passing**. Historical ledger and frozen contracts are unchanged.
- [x] Bounded reviewer checked the paid path and required the exact 61/11
  pre-client receipt partition. Added it, tested deletion of a historical receipt,
  and obtained clearance before the remaining 11 paid requests.
- [x] Paid probe completed within its original cap; 72 request IDs, response
  contents, receipt hashes and summed charges reconciled. No uncertain attempts.
- [x] Eight thousand synthetic null worlds exposed assumptions behind small-sample
  paired-t inference. Conservative sensitivity is implemented, but meaningful
  effects, precision targets and final scientific approval remain unset.
- [x] 56 full-roster method/setting/seed fixtures replay correctly, included in
  the Python suite. Fixture graphs are not model observations.
- [x] 24 viewer tests; static viewer build passed. No new browser-interaction
  audit was performed for these offline research revisions.
- [x] Notebook: 43 code cells handled successfully, including 32 deliberately
  skipped archival bodies. Executed copy: `outputs/qa/analyze_networks.executed.ipynb`.
- [x] All 68 reviewed graphs, receipt/artifact hashes and recomputed metrics verified.
- [x] Public analysis produced 2,788 references and 2,784 descriptive contrast rows.
  Country contrasts hold English fixed; language contrasts hold US fixed.
- [x] All seven frozen source files and `study_protocol_896.json` are byte-identical
  to the existing published checkout. Original study outputs were not replaced.
- [x] Scoped whitespace check passed with intentional CRLF recognized. Do not
  normalize frozen files just to silence line-ending warnings in the dirty tree.
- [x] Bounded read-only reviewer found no material scientific-integrity or
  offline-spend issue in the selected new modules; Python compilation passed.
- [x] New protocol keeps `generation_authorized=false` and total target unset.

The test suite emits existing pandas/NumPy deprecation warnings; no runtime
failure occurred. Paid behavior, human translation, empirical realism and
conference acceptance cannot be certified by these software checks.
