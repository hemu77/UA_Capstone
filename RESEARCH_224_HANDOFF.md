# Fresh 224-Network Study: Astra Review Handoff

**Historical planning record, superseded by [RESEARCH_896_HANDOFF.md](RESEARCH_896_HANDOFF.md).**
The author selected eight repetitions (896 graphs) and removed the $50 maximum.
The active runner now reads `study_protocol_896.json`; commands, limits and
review instructions below describe the previous 224 proposal, not authorization.

This is the replacement study prepared on September 30, 2026. The previous
experiments remain available as historical evidence. No previous graph counts
toward the fresh 224. The local simulation is a team analysis tool.

The actual six-page rejection is mapped in [REVIEW_RESPONSE_MATRIX.md](REVIEW_RESPONSE_MATRIX.md).
Its two-repetition criticism remains unresolved by this reduced matrix. The
scope is **224 total**, not 224 per method; software readiness is not acceptance.

## Verification Completed

- [x] 60 offline unit/regression tests passed. Existing NumPy/pandas deprecation
  warnings remain; these runs had no failing tests.
- [x] All 224 planned cells exercised with labelled fixture replies, using the
  full 50-adult roster and the real parser/graph/metric path.
- [x] Twelve matched baseline fixtures checked across all four methods.
- [x] Notebook JSON validated and the fresh section executed: current source
  hashes matched, the verification table was 4 by 4, collected fresh runs were 0.
- [x] Existing viewer parity: 204 graphs, 28 replay traces and 28 presentation
  figures checked; 193 historical source hashes preserved.
- [x] Two read-only reviewer lanes inspected generation/spending and research/
  evaluation. Their concrete defects received regression coverage; remaining
  evidence requirements are listed in the response matrix rather than hidden.
- [x] Paid calls during this preparation: **0**. API spend: **$0**.

This verifies the checked software paths. Bilingual equivalence, fresh provider
responses, final cost and scientific generalizability still require review.

## Exact Study

Four models: GPT-4.1, GPT-5.6-Luna, GPT-6-Luna and GPT-6-Sol.
Four methods: global, local, sequential and iterative.
Four country framings: US, India, Japan and Brazil.
Four instruction languages: English, Hindi, Japanese and Brazilian Portuguese.

There are seven distinct settings: four countries in English, then US framing
with Hindi, Japanese and Portuguese. US-English is counted once. Four models
times four methods times seven settings times two repetitions equals **224**.
Each model has 56 runs. Every graph uses all 50 personas. Seeds 11000 and 11001
control local ordering and choice-count draws; they are not population sizes
and do not make the provider deterministic.

This design estimates main contrasts around US-English. It does not estimate
country-by-language interactions; that requires a larger crossed design.

## What Changed and Why

- [x] A new deterministic fictional adult roster replaces the old US roster
  for this study. Ages are 18-67. Gender counts are 20 men, 20 women and 10
  nonbinary people. Each of five religious categories has 10 people. Political
  orientation has 17 left-leaning, 16 centrist and 17 right-leaning people.
  These marginals are experimental choices, not census estimates.
- [x] Attributes are shuffled independently with seed 224. This avoids assigning
  all members of one category the same other attributes. Random correlations
  can still occur in one small roster; causal demographic claims remain unsupported.
- [x] US party names and US race categories are removed. Political orientation
  is still an operational scale whose meaning may differ across settings.
- [x] Complete fresh instructions cover all four languages and five prompt
  actions, including iterative add and drop. Spanish remains outside this study.
- [x] Candidate JSON has the same English column names, values and ID strings
  across instruction-language treatments. A header specifies the positional
  rows. This reduces repeated input tokens and avoids translating demographic
  categories while claiming only instructions changed.
- [x] Country framing never specifies participants' spoken language or changes
  their attributes. Unknown country codes and invalid languages raise errors.
- [x] Decimal/punctuation replies cannot silently become valid IDs. Blank global
  replies fail parsing. Self-links, unknown IDs, duplicates and invalid iterative
  actions are rejected before graph mutation.
- [x] Recorded event additions/removals reconstruct the final graph exactly.
  The engine retains each batch; the simulation cannot invent per-edge chronology.
- [x] Completed run receipts link request IDs to provider response IDs, resolved
  model names, token usage and parse decisions. Runtime/library versions are
  recorded and checked against the review receipt before execution.
- [x] Metrics reuse the corrected common analysis functions. Undefined group
  homophily stays null. Age assortativity is separate from categorical Coleman
  scores; their raw magnitudes are not a common importance scale.
- [x] Matched random-edge, feature-similarity and degree-preserving controls
  accept the actual fresh attributes. Swap completion is reported separately
  from convergence. Controls are synthetic references, not empirical validation.
- [x] Offline fixture outputs are labelled and stored separately from collected
  LLM results. They do not enter the web viewer or research result counts.
- [x] Preflight reports are invalidated before reruns and marked failed on error.
- [x] Source changes during preflight invalidate success. A crash-released OS
  lock prevents concurrent fresh writers; unique temporary files protect JSON.
- [x] Paid reservations share the original ledger and cumulative $50 ceiling.
  Uncertain transport failures stop the run without automatic paid resending.
  A missing/incompatible ledger or charges below the historical pilot total
  stop execution. Frozen source, protocol and roster identities protect resume.
- [x] Every completed fresh receipt is reconciled against ledger request identity,
  token usage and charges before more calls. Missing charges block execution.
  A fully completed selection resumes offline without constructing an API client.
- [x] Fresh offline analysis verifies actual receipts, exports group homophily,
  matched controls and individual paired RQ contrasts. Missing observations
  remain missing; fixtures and archive graphs never fill fresh-result tables.
- [x] Live execution checks a matching review receipt before creating an API
  client. Its default scope is one US-English run per method/model (16 cells)
  for actual calibration; those completed runs count toward the final 224.
  The review receipt bounds the number of cells and cumulative dollar ceiling.
  The initial template caps cumulative spending at $5, including earlier pilots;
  approving those 16 does not authorize the complete 224 or a $50 spend.

## What Each RQ Can Answer

1. **Country framing:** Does a country label change generated topology or
   homophily on this fixed fictional roster with English instructions?
   This does not establish national social behavior.
2. **Demographics:** Which attributes show within-category association above
   their own reference controls? Report group-specific evidence and age
   assortativity separately. Association does not establish causal dominance.
3. **Models:** How do four OpenAI configurations differ on matched methods,
   settings, roster and repetition? This panel cannot establish consistency
   across vendors or open-weight families.
4. **Language:** How do instruction-language treatments change outputs while
   US framing and English candidate data stay fixed? This does not test fully
   localized persona dossiers or the participants' spoken language.

Local generation asks each persona to choose among all other 49 people.
Sequential exposes candidate degrees after the first three local-style prompts.
Both use exponential choice-count draws with mean 5, clamped to 1-20.
Iterative starts from a local network and performs three add/drop rounds,
skipping an add when no eligible new friend exists. Its request upper bound is
350 per graph. Global generates the graph in one reply without matching the
other methods' degree constraints. Interpret method contrasts as differences
between these complete protocols.

## Review Still Required

- [ ] Astra reviews the actual code, protocol, roster and fixture evidence.
- [ ] Bilingual reviewers check instruction and retry equivalence, including
  Hindi, Japanese and Brazilian Portuguese. Automated checks cannot certify it.
- [ ] Confirm account access and supported settings for the four API models.
- [ ] Calibrate the fresh compact prompts and review completion/retry costs
  before the complete batch. Earlier prompt costs cannot guarantee this price.
- [ ] Freeze the protocol and record actual review decisions against its hashes.

Two repetitions of one roster support exploratory descriptions. Show both run
values and their ranges; do not manufacture narrow confidence intervals by
treating edges, nodes or duplicated baseline samples as independent experiments.
Additional rosters, replication, open-weight models, temperature/wording sweeps,
local candidate-size ablations, iterative initialization/round ablations and a
documented empirical benchmark remain outside the reduced 224-run budget.
This plan therefore does not close every concern in the rejection action plan.

## Evidence to Inspect

- `study_protocol_224.json`: the exact matrix, claims and pending reviews.
- `text-files/revision224_adults.json`: generated fictional roster.
- `revision224_prompts.py`: roster construction and all translated instructions.
- `revision224.py`: prepare, offline preflight, review checks and bounded execution.
- `outputs/revision224_preflight/prompt_catalog.json`: 80 review examples.
- `outputs/revision224_preflight/verification_summary.csv`: checks for 224 cells.
- `outputs/revision224_preflight/report.json`: current status and source hashes.
- `outputs/revision224_preflight/baseline_fixtures.json`: 12 baseline examples.

Fixture metrics are test outputs. They are not findings about any GPT model.
Collected networks will go to `outputs/revision224/`, with their adjacency,
PNG, exact events, metrics, homophily, artifact hashes and provenance receipt.

## Run Without API Spending

From the repository root in PowerShell:

```powershell
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --prepare
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --preflight
rtk proxy .\.venv\Scripts\python.exe -B -m unittest discover -s tests
# After fresh results exist; safe before collection too (reports NOT_COLLECTED):
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --analyze
```

The preflight uses test replies to exercise the real generation/parser path.
It reports zero paid calls and never constructs an API client.
Fresh analysis writes `stats/revision224/analysis_report.json`,
`network_metrics.csv`, `group_homophily.csv`, `matched_controls.csv`, and
`paired_contrasts.csv`. Each contrast retains both run IDs and a repetition;
`comparison_minus_reference` is a descriptive difference, not significance.
Baseline adjacency files live in `stats/revision224/baselines/`.

After reviews, the reviewer records completed decisions in
`outputs/revision224_preflight/review.json` using the template's exact hashes.
Do not set review flags merely to unblock execution. Paid execution also needs
`OPENAI_API_KEY` in the process environment:

```powershell
# First actual calibration cells, counted toward the 224:
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --execute --limit 16
# After reviewing actual cost, failures and outputs:
rtk proxy .\.venv\Scripts\python.exe -B revision224.py --execute --limit 224
```

The $50 ceiling stops requests before the next reservation exceeds the limit.
It can stop an incomplete experiment; it cannot guarantee all 224 will fit.
