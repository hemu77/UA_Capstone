# Revised Calibration Results and Main-Study Hold

October 2, 2026. **104/104 calibration graphs complete. Main collection remains
unauthorized. Engineering verification passed; scientific approval is separate.**

## What Was Collected

Contract: `5fb3715550db87bb86061bf9f3c1484ddc3203b6a1b989758e96fa8ff1772b3d`.
Evidence folder: `outputs/calibration_v6/5fb3715550db/`.

Every graph contains the same 50 designed fictional adults. Four methods are
global, local, sequential and iterative. Country framings are US, India, Japan
and Brazil. Instruction languages are English, Hindi, Japanese and Brazilian
Portuguese; persona JSON stays English. The seven settings are four countries
in English, plus US in the three non-English languages. This is not a full
country-language factorial and does not identify their interactions.

| Model configuration | Graphs | Coverage | Conservative usage |
| --- | ---: | --- | ---: |
| GPT-4.1 | 16 | US, all four languages/methods, one repetition | $3.412258 |
| GPT-5.6-Luna | 16 | Same | $0.352191 |
| GPT-6-Luna | 56 | All seven settings/four methods, two repetitions | $0.598252 |
| GPT-6-Sol | 16 | US, all four languages/methods, one repetition | $3.492165 |
| Total | **104** | **26 per method** | **$7.854865725** |

Seeds 21000/21001 label saved randomization schedules, not population sizes or
guaranteed provider determinism. Actor order, candidate display order and
nomination counts use separate streams. These revised graphs are separate from
the earlier 68 V5 graphs, 28 engineering pilots and original capstone outputs.
None is automatically reused as confirmation data.

## Principal Finding: Global Is Often Empty

**22 of 26 global networks have zero edges.** Raw received `NONE` responses
reconstruct those graphs. They are not missing files, parser replacements or
fixture data. The frozen prompt permits zero friendships; valid empty outputs
were retained rather than rerun to obtain a preferred network.

| Global configuration | Empty / total | Nonempty observations |
| --- | ---: | --- |
| GPT-4.1 | 4/4 | None |
| GPT-5.6-Luna | 3/4 | US-Portuguese: 118 edges |
| GPT-6-Luna | 14/14 | None across its seven settings |
| GPT-6-Sol | 1/4 | US-English: 84; US-Hindi: 70; US-Portuguese: 69 edges |

**Interpretation:** this wording often elicits no proposed ties. Calibration
cannot establish why. In particular, we have not isolated the effect of the
explicit `NONE` option from other prompt changes. These data do not show that
people in any country form no friendships. Increasing repetitions alone does
not fix an uninformative or misaligned task.

Before main collection, decide whether unrestricted generation with a large
zero-edge mass is the intended estimand. If not, compare a separately versioned
instruction clarification in a bounded probe. Do not retroactively force an edge
count, delete valid empty graphs, or mix revised prompts into this frozen dataset.
Any change requires a new contract and cost/precision review.

## What Can Move in the Other Methods

| Method | Edge range | Clustering range | Modularity range | LCC fraction range | Usage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 0-118 | 0.000-0.629 | 0.593-0.792 on 4 defined graphs | 0.02-1.00 | $0.033839 |
| Local | 183-207 | 0.404-0.596 | 0.368-0.562 | 0.96-1.00 | $1.056336 |
| Sequential | 185-228 | 0.444-0.652 | 0.163-0.568 | 0.96-1.00 | $1.184223 |
| Iterative | 185-216 | 0.512-0.773 | 0.293-0.657 | 0.38-0.98 | $5.580468 |

Ranges combine unequal exploratory conditions; they are not treatment effects,
significance tests or causal method comparisons. Requested nomination counts
still constrain local/sequential/iterative density. Local LCC is 1.0 in 24/26
graphs, so it remains a poor sensitive primary endpoint. Clustering/modularity
vary, but useful effect sizes and uncertainty targets still need agreement.
Iterative accounts for about 71% of this calibration's cost.

Each of the 22 empty graphs has undefined modularity, degree Gini, categorical
Coleman indices and age assortativity; the report also leaves its LCC path
quantities undefined. Undefined is not zero homophily. Density and clustering
are zero, and LCC fraction is 1/50. `inspection.json` records every missingness
count. Representative empty-global and iterative PNGs were visually inspected;
all 104 PNGs passed file/metadata/hash checks. These are inspection artifacts,
not polished publication figures or independent empirical validation.

## Compliance, Not Translation Accuracy

There were **11,536 initial decisions**, **12 first-response failures (0.104%)**,
and **12 correction calls**, making **11,548 unique received requests**. All
final responses passed the declared parser/replay checks. Corrections and their
costs remain recorded. Decisions within a graph are not independent repetitions.

For a country-matched descriptive language check, US-only counts are:

| Instruction language | Initial decisions | First failures |
| --- | ---: | ---: |
| English | 2,219 | 2 |
| Hindi | 2,217 | 3 |
| Japanese | 2,224 | 4 |
| Portuguese | 2,215 | 1 |

The other two failures occurred in Brazil-English. Across US exactly-one
nomination prompts, each language had 243 initial decisions and zero failures.
This subgroup includes local-style initialization in iterative/sequential runs;
it does not combine all iterative add/drop choices under a nomination label.
The CSV retains model, country, language, generation method, actual prompt
method and requested-count strata; pooling these is not an equivalence test.

This follows the separate 72-call English/Portuguese wording screen: revised
0/36 versus original 4/36 failures. Neither study supplies human bilingual
validation, a language-accuracy percentage, or evidence that error rates are
zero. The owner accepted the absence of human bilingual review.

## Accounting and Runtime

- Approved additional ceiling: **$10**; calibration usage: **$7.854865725**.
- Historical anchor: **$6.481868475**; cumulative ledger: **$14.3367342**.
- Original historical rows are unchanged. There are **zero unresolved new
  requests** and no new abandoned requests. Two historical abandoned
  reservations remain conservatively counted; their actual billing is unknown.
- Recorded usage: **8,659,035 input tokens**, **73,185 output tokens**.
- Summed recorded API time: **2.8148 hours**. First dispatch to last response:
  **3.5852 hours**, including local work, review pauses and recovery. Neither
  includes all subsequent analysis/documentation time.

Charges use the frozen price schedule with a 25% input-cost uplift and no cache
discount. They are conservative accounting, not a provider invoice. Model access
and prices were checked on October 2 against official model pages:
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[GPT-5.6-Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[GPT-6-Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[GPT-6-Sol](https://developers.openai.com/api/docs/models/gpt-6-sol).

### Recovery Without Repurchasing Replies

A Windows journal replacement failed after a reply was received and parsed.
An exact 45-request cached replay and parse-only hash comparison proved which
checkpoint was stale. One journal was reconciled, its original bytes preserved,
and every ledger row remained unchanged. No model reply was repurchased.

An audited I/O-only adapter then handled **16 local-file retry events** during
the successful resumed execution. These are not 16 API retries. It retries only
identical journal writes on `PermissionError`, at most five times; unexpected
errors stop. Its exact source snapshot and unchanged-source check are saved.
The frozen prompt/generation/probe sources did not change.

## Hypothetical Main Forecasts, Not Approval

These estimates are for **entirely fresh samples using unchanged prompts**;
the 104 calibration graphs are not subtracted. The old "$101 remaining / 828
remaining" quotation is superseded.

| Hypothetical allocation | Fresh graphs | Usage estimate | With 25% allowance | Serial API hours | With 50% time allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 repetitions for every model/method/setting | 896 | **$102.16** | **$127.70** | **23.54** | **35.31** |
| Global 32; other three methods 8 | 1,568 | **$103.61** | **$129.51** | **23.90** | **35.86** |

The second scenario is cheap because the present global replies are usually
`NONE`. That is not an argument to buy more of them. A changed global task may
increase output length, runtime and costs. These are planning allowances, not
confidence intervals, spending caps or guaranteed completion times.

US model/method/language costs are directly measured, with only one repetition
for the three non-Luna configurations. Their non-US English costs/time transfer
Luna's country-to-US ratios: 288/896 or 504/1,568 projected graphs use this proxy.
Provider changes, retries and prompt revisions can invalidate those assumptions.
API time excludes future local verification, analysis and downtime. No final
main-study allocation or finite budget has been approved.

## Team Decisions and Verification

- [x] 104 receipts replayed against the actual prompts, schedules, parser and
  event engine; adjacency/metrics and artifact hashes checked.
- [x] Same 50-person roster, no self-links, no request counted in two networks.
- [x] Original-ledger reconciliation passed; charged total equals all completed
  receipt charges. Public inspection uses neither an API key nor the private DB.
- [x] Separate source-backed notebook section and machine-readable tables added.
- [x] Full offline suite: **142 tests passed in 162.368 seconds**; existing
  pandas/NumPy deprecation warnings remain, with no test failures.
- [x] Notebook: **45 code cells handled**, including 32 deliberately skipped
  archival bodies, no API calls. Two reviewer-found stale-evidence paths were
  reproduced with failing tests and fixed before the final inspection.
- [x] One bounded read-only reviewer independently checked receipt hashes,
  CSVs, denominators, cost/runtime forecasts, empty-graph missingness and handoff
  claims; no remaining material inconsistencies reported. The reviewer did not
  rerun the private-ledger reconciliation, full suite or notebook; those checks
  were executed by the parent workflow. This is not scientific peer approval.
- [ ] Decide the global task/empty-graph interpretation before main collection.
- [ ] Approve raw-unit meaningful effects, uncertainty/precision, multiplicity
  families and repetition allocation; more repetitions alone are not validation.
- [ ] Decide whether wording variants, another roster or empirical benchmarks
  are required for the intended contribution; none is silently marked complete.
- [ ] Authorize an exact new scope and finite cap only after those decisions.

Read [the team brief](TEAM_KNOWLEDGE_BRIEF.md) and
[professor/team review request](PROFESSOR_TEAM_REVIEW.md). The simulator was
deliberately left unchanged and still shows the older V5 dataset.

Reproduce offline from the pinned Python 3.11.9 environment:

```powershell
.\.venv\Scripts\python.exe -B inspect_calibration_v6.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
```

The JSON report binds all receipt and CSV hashes. The notebook refuses stale
reports and does not expose a rejected report to downstream cells. Private
authorization/ledger files and credentials must not be published.
