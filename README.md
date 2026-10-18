# Cross-Cultural and Linguistic LLM Network Study

A research revision of [Stanford SNAP's LLM social-network code](https://github.com/snap-stanford/llm-social-network).
We study how language models generate friendships between fictional personas
under different country framings, instruction languages and generation methods.

## Current Review Gate (October 17)

Read the [pre-collection repair and team handoff](docs/PRECOLLECTION_REVIEW.md)
first. The new offline audit preserves all 327 public V8 files, replays all 104
observations, and reports failed, empty and endpoint-defined graphs separately.
No API calls or historical replacements were made in this repair.
The [final verification record](docs/PRECOLLECTION_VERIFICATION.md) records
**210 passing offline tests**, notebook checks and the exact remaining limits.

The [analysis candidate](precollection_spec.json) uses paired schedule blocks,
four fixed models and 24 primary comparisons. RQ4 is explicitly about **named
instruction contracts**, not a general language effect. A separate Hindi/global
parser candidate is offline-only, not silently adopted into V8.

The 32-schedule planning scenario is **3,584 fresh attempts**, approximately
**$409.77 / 105.22 serial API hours** at historical V8 usage/rates, before
allowances. It is not the old ~$101 study and **is not authorized**. Final
contract review, changed-prompt live checks, execution safeguards and current
prices remain launch gates. Tests cannot guarantee scientific validity.

```powershell
python -X utf8 -B precollection_audit.py
python -X utf8 -B -m unittest tests.test_precollection -q
```

## Preserved V8 Completion (October 13)

V8 calibration completed and verified. All **104 planned
observations** finished: **100 valid graphs and four retained global failures**.
Four valid global graphs are empty; empty graphs and invalid responses are not
interchangeable. Calibration usage plus both retained server-error reservations
is **$7.8912884 of $15**. Exact, separately approved replacements reused the
236- and 139-response prefixes without repurchasing them.

The [96-cell wording screen](docs/V8_EXECUTION_NOTES.md) produced 94 valid graphs
and two preserved failures at $5.9461527. The prospectively selected reference
completed 32/32 graphs. A separate four-cell Hindi-alternative probe cost
**$0.30618** and produced **one valid graph, three failures**. The clarification
did not reliably fix the format error and **is not adopted**. Neither frozen
stage is relabeled or replaced by that diagnostic.

Read the [plain-language handoff](docs/V8_TEAM_HANDOFF.md),
[final calibration/probe evidence](docs/V8_COMPLETION_RESULTS.md), and
[professor/team review note](docs/PROFESSOR_TEAM_REVIEW.md).
Fresh main-study planning estimates are **$102.44 for 896 attempts** or
**$104.49 for 1,568 attempts**, before contingencies. They are not guaranteed
valid-graph counts or authorized spending. Main collection remains blocked.

Checks passed: **191 offline regressions**, final recovery tests, and seven
focused publishing-checkout tests. Calibration private accounting and public
replay verified all observations with unchanged ledger rows and no API calls.
Costs are conservative local accounting, not reconciled invoices. V5/V6/V7
history stays intact; the simulator still shows observed V6 results. Human
bilingual validation is absent, and tests cannot certify research validity.

## Preserved V6 Completion (October 2)

**Historical status:** 104 revised calibration graphs complete at $7.8549 of the
approved $10. Main collection was NOT authorized. All saved graphs, receipts,
metrics and accounting passed verification. **22/26 global graphs are empty**,
so the global task needs scientific review before scaling. Completion is not
research-validity or conference-acceptance certification.

Read [the revised plan and checkboxes](plan.md): corrected offline prompt candidates,
independent candidate-order randomization, method-specific outcomes, public
key-free reanalysis, and remaining calibration/statistical approval gates.
The owner separately authorized a **$1, 72-first-response wording probe** and
then **104 calibration graphs under a $10 ceiling**. Human bilingual review was
unavailable and waived. That limitation remains disclosed, not marked validated.
**Probe complete:** 72 first responses cost $0.00852425 conservatively. Revised
wording had 0/36 failures; original wording had 4/36, all Portuguese one-choice
cases. No replacement calls or new networks. This passes a narrow engineering
screen, not final statistical approval. See [results and decision](docs/STATISTICAL_ANALYSIS_DECISION.md).

The revised [calibration runner](docs/CALIBRATION_V6.md) is implemented separately
in `calibration_v6.py`. It prepares a 68-run core and an optional 36-run multilingual
extension (104 total), now both complete. Offline fixtures are explicitly labelled
and excluded from observed-network datasets. See [measured results and new
forecasts](docs/CALIBRATION_V6_RESULTS.md). Old V5 graphs remain separate;
neither calibration is automatically reused for confirmation.

## Start Here

- [Latest repair and pre-collection review](docs/PRECOLLECTION_REVIEW.md):
  Raghav's findings, implemented safeguards, exact remaining gates and revised scenarios.
- [Current professor/team handoff](docs/V8_TEAM_HANDOFF.md): earlier versus now,
  debugging evidence, partial results, costs, and the exact remaining decisions.
- [Current V8 execution and evidence](docs/V8_EXECUTION_NOTES.md): paid wording
  outcomes, prospective recalibration, failure accounting and replay commands.
- [Hindi wording candidate](docs/HINDI_WORDING_CANDIDATE.md): the observed failure,
  isolated clarification, offline checks and outstanding live-validation limit.
- [Latest V7 execution and findings](docs/V7_EXECUTION_NOTES.md): completed paid
  diagnostics, verification-only recovery, preserved evidence and stop decisions.
- [Active V7 repair and budget plan](docs/REPAIR_V7_PLAN.md): what changes,
  which outputs need regeneration, staged cost assumptions and remaining gates.
- [V7 implementation and commands](docs/REPAIR_V7_RUNNER.md): safeguards,
  diagnostic workflows, historical-data preservation and remaining calibration gates.
- [V7 independent review evidence](docs/V7_REVIEW_OCT12.md): exact reviewed
  version, regressions, current pricing, billing-tier fix and limitations.
- [October 12 PR review response](docs/PR_REVIEW_OCT12.md): independently checked
  findings, unresolved scientific decisions and versioned next steps; dated review history.
- [Public simulator for team review](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html):
  104 revised V6 graphs, recorded replay, matched filters and original artifacts.
  [Deployment and verification notes](docs/SIMULATOR_DEPLOYMENT.md). No API calls;
  this is an exploratory review tool, not approval for main collection.
- [Team knowledge brief](docs/TEAM_KNOWLEDGE_BRIEF.md): plain-language history,
  reviewer concerns, improvements, current evidence and unfinished work.
- [Team approval checklist](docs/TEAM_APPROVAL.md): decisions before main collection.
- [Revised calibration results](docs/CALIBRATION_V6_RESULTS.md): 104-graph findings,
  actual accounting, verification, runtime and forecast assumptions.
- [Documentation map](docs/README.md): architecture, review response, translation,
  logging, viewer and historical records.
- [Professor/team review note](docs/PROFESSOR_TEAM_REVIEW.md): requested scientific
  decisions through calibration; the brief includes model rationale and full scope.
- [Reviewer agent package](agents/README.md): reusable read-only task prompt and
  canonical project skill; team members can use it for bounded engineering review.

## Research Questions

| RQ | Controlled question |
| --- | --- |
| RQ1 | With instruction language fixed to English, does country framing change homophily and topology? |
| RQ2 | Which demographic attributes show stronger mixing patterns under the conditions? |
| RQ3 | Do the four model configurations converge or diverge under matched settings? |
| RQ4 | With country framing fixed to US, how do the named English, Hindi, Japanese and Portuguese instruction contracts change the networks? |

All four generation methods contribute to these questions. A larger categorical
homophily score is not proof of causal demographic importance; numeric age
assortativity is not directly ranked against categorical Coleman scores.
Country framing is a prompt intervention, not a representative national sample.

## Design Under Revision

| Dimension | Current scope |
| --- | --- |
| Models | `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol` |
| Methods | Global, local, sequential, iterative |
| Country framing | US, India, Japan, Brazil |
| Instruction languages | English, Hindi, Japanese, Brazilian Portuguese |
| Personas | Same 50 designed fictional adults; no US party names or race labels |
| Settings | Four countries in English, plus US in the other three languages |
| Repetitions | Candidate: 32 fresh shared schedule blocks. Allocation and final collection contract not approved. |
| Total | Candidate: 3,584 attempts; no automatic calibration reuse. Historical 896/1,568 scenarios remain separately labelled. |

This is a seven-setting design, **not** the full country-by-language factorial.
Candidate attributes stay in English JSON. It tests instruction-language
sensitivity, not participants' spoken language or fully translated personas.
Seeds control local randomization; they are not population sizes or guarantees
of provider determinism. `revision224.py` retains its filename for compatibility
but reads the preserved V5 `study_protocol_896.json`. The current analysis candidate is
`precollection_spec.json`; `study_protocol_next.json` is preserved V6 preparation.
Neither `precollection_audit.py` nor `revision_next.py` can execute paid collection.
Country labels are prompt framings, not validated cultural constructs. RQ4
concerns these instruction wordings until wording robustness is tested.

## Historical V6 Progress (October 2)

This section records V6, not the current V8 completion or test status.

- [x] **104 revised calibration graphs**, each with 50 nodes, adjacency, PNG and receipt.
- [x] GPT-6-Luna: 7 settings x 4 methods x 2 repetitions = 56 graphs.
- [x] Other three models: US x 4 languages x 4 methods x 1 repetition = 48 graphs.
- [x] Exact prompt/schedule/parser/event replay, metrics, artifact hashes and private accounting.
- [x] **142 Python tests passed**; **104 revised offline fixture graphs** remain separately labelled.
- [x] Notebook executed: 45 code cells handled, including 32 explicitly skipped archival bodies.
- [x] Bounded checkpoint recovery and 16 local-file retry events; no API reply repurchased.
- [x] Historical V5 68 graphs, 204 original controls and 2,788 later reference draws preserved separately.
- [ ] Resolve mostly empty global outcomes and approve method-specific estimands.
- [ ] Team agreement on claim scope, wording robustness, analysis/precision and finite budget.
  The old **828 remaining** count is not a current target; no automatic calibration reuse.
- [ ] Final RQ analysis and a paper with claims supported by the completed evidence.

The historical V6 calibration contains **11,548 received requests**, including **12
first-response failures among 11,536 decisions (0.104%)**, each corrected once.
Costs include rejected responses. Undefined metrics in 22 empty graphs remain
undefined, not invented zeros. These clustered decision counts do not establish
translation equivalence. Eight repetitions are not proven adequate power.

## Historical V6 Costs (October 2)

These V6 measurements and forecasts are preserved history, not current V8 totals.

| Item | Conservative accounting/estimate |
| --- | ---: |
| Completed revised calibration | **$7.854865725 of $10** |
| Cumulative historical + probe + calibration ledger | **$14.3367342** |
| Hypothetical 896 fresh graphs, unchanged prompts | **$102.16 additional; $127.70 with 25% allowance** |
| Hypothetical global 32 / others 8: 1,568 fresh graphs | **$103.61 additional; $129.51 with 25% allowance** |
| Serial API time for 896 | **23.54 hours; 35.31 with 50% allowance** |
| Serial API time for 1,568 | **23.90 hours; 35.86 with 50% allowance** |

These are not invoices, approved budgets or guaranteed times. All US language
cells are measured; other models' non-US country costs transfer Luna ratios.
Global is unusually cheap because most replies are `NONE`; changed prompts need
new forecasts. Runtime excludes future local work and pauses. See [the complete
assumptions](docs/CALIBRATION_V6_RESULTS.md). The old ~$101 remaining forecast is
historical, not the current quote. Two historical uncertain billing reservations
remain counted; V6 had zero unresolved requests at its October 2 completion.
The current V8 HTTP-500 reservation is additional and remains unresolved.

**Team question:** does the intended paper accept this controlled, limited
model-behavior study, or require stronger population, translation, replication
or empirical-validity evidence before another paid batch?
Record that decision in [TEAM_APPROVAL.md](docs/TEAM_APPROVAL.md).
Only after approval and matching execution review can collection proceed.
Publishing this repository does not authorize paid generation.

## Earlier Work and Reviewer Response

The original capstone implemented OpenAI-only bring-up, then culture/model
comparisons across all four methods (Steps 2-3 / RQ1-RQ3), then fixed-US language
comparisons (Step 4 / RQ4). It used GPT-4.1 Nano/Mini/4.1, a US-labelled roster,
two repetitions and Spanish rather than Portuguese.

The revision corrected homophily and undirected comparisons, hardened parsing
and retries, clarified actual method behavior, separated country from instruction
language, added matched controls and replaced the roster for fresh work.
It does **not** claim that reviewers approved these changes or that all scientific
concerns are resolved. Read [the feedback explanation](docs/TEAM_KNOWLEDGE_BRIEF.md#4-why-the-reviewers-were-not-satisfied).

| Evidence collection | How to use it |
| --- | --- |
| Original 192 inventoried study graphs | Historical; 176 retained, 16 quarantined for self-links during revision. |
| 28 engineering-pilot graphs | Model/method/UI debugging on historical personas and labelled variants. |
| Superseded fresh calibration versions | Failure/debugging evidence, excluded from the current study. |
| V5 calibration: 68 graphs | Verified exploratory evidence on a superseded design; not relabelled or automatically pooled into revised confirmation. |
| Revised calibration: 104 graphs | Current exploratory evidence under `5fb3715550db`, not final RQ answers or automatic confirmation data. |
| Offline fixtures and synthetic controls | Software checks/reference rules, never additional LLM observations. |

Previous README and architecture content is preserved in
[project history](PROJECT_HISTORY.md),
[the original capstone README](CAPSTONE_README_ARCHIVE.md) and
[archived architecture](ARCHITECTURE_BEFORE_REVISION.md).
Historical PDFs and plots are not current validity claims.

## Repository Map

| Location | Purpose |
| --- | --- |
| `revision224.py`, `revision224_prompts.py`, `study_protocol_896.json` | Preserved V5 generation contract and execution gates. |
| `revision_next.py`, `study_protocol_next.json`, `plan.md` | Offline candidate and approval-gated next steps; no paid command. |
| `analyze_saved_study.py`, `stats/public_v5_reanalysis/` | Ledger-free reanalysis, repeated references, compliance and power sensitivity. |
| `generate_networks.py`, `constants_and_utils.py` | Shared generation, strict parsing and bounded corrections. |
| `paid_study.py` | Durable reservations, usage, diagnostics and cached requests. |
| `calibration_v6.py`, `calibration_v6_io.py` | Frozen revised calibration runner and audited local-file-only retry adapter. |
| `inspect_calibration_v6.py`, `outputs/calibration_v6/5fb3715550db/` | Public replay/verification, all 104 artifact sets, compliance tables and forecasts. |
| `analyze_networks.py`, `make_matched_baselines.py`, `inspect_calibration.py` | Metrics, controls and ledger-backed calibration inspection. |
| `analyze_networks.ipynb` | Maintained analysis; separates fresh and historical evidence. |
| `outputs/revision896_retry_v5/` | 68 historical V5 adjacency/PNG/receipt artifact sets. |
| `outputs/revision896_retry_v5_preflight/` | Manifest, exact prompts, checks, calibration reports and closed authorization. |
| `stats/revision896_retry_v5/` | Historical V5 metrics, homophily, controls, contrasts and timing. |
| `text-files/`, `plots/`, earlier `stats/` and `outputs/` | Preserved historical data and figures; follow each collection's provenance. |
| `viewer/`, `export_revised_viewer.py`, `export_calibration_viewer.py` | Three.js interface: default 104-graph V6 calibration, separate 68-graph V5 and historical/pilot views. |
| `tests/`, `docs/` | Offline regression checks and team documentation. |

The viewer reads saved graph data, not PNG pixels. Recorded formation replay is
available only where decisions were saved. It does not invent reasoning,
simultaneous agent actions or an "average" observed network. The deployed default
is the 104-graph V6 calibration. V5 and older links remain separate; their
rosters cannot be mixed by matching persona IDs. Coverage describes the selected
collection's allocation, not an approved main study or 896 completed runs.

## What Is Not Published

Credentials, private execution authorizations and ledger-linked request journals,
the original private `budget.sqlite`, build caches, diagnostic
logs, raw manuscript/review files and promotional/demo media are excluded.
Existing already-published historical reports are retained.

**A public clone can inspect artifacts and run tests, but cannot independently
reconcile provider requests or resume paid generation without the original
private ledger.** That fail-closed behavior is intentional. Do not create a new
empty ledger to bypass budget/provenance checks. API credentials should be
rotated after exposure and supplied locally, never committed or pasted into notebooks.

## How to Run

Use Python **3.11.9** exactly (pinned in `.python-version`); calibration used it
and the versions in `requirements.txt`. A different runtime, including another
3.11 patch release, can invalidate frozen execution checks. Install it from
[python.org](https://www.python.org/downloads/release/python-3119/), then run
from the repository root:

```powershell
# Creates .venv, installs the full transitive lock, checks versions.
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
# Same, then runs the offline suite: no key and no model calls.
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -RunTests
```

`requirements.lock.txt` pins every transitive dependency of `requirements.txt`
for Windows x64. The manual equivalent is `py -3.11 -m venv .venv`, then
`.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt` and
`.\.venv\Scripts\python.exe -B -m unittest discover -s tests`.

Prepare revised prompts and recompute public analysis without a key or ledger:

```powershell
.\.venv\Scripts\python.exe -B revision_next.py --prepare
.\.venv\Scripts\python.exe -B analyze_saved_study.py --control-repetitions 20
# Inspect the revised 104 graphs and execute the maintained notebook offline.
.\.venv\Scripts\python.exe -B inspect_calibration_v6.py
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
```

`analyze_saved_study.py` verifies all 68 V5 graphs before creating separate exploratory tables
in `stats/public_v5_reanalysis/`. Reference graphs are synthetic controls, not new
LLM observations. It does not need `budget.sqlite` or authorize any spending.

Inspect preserved historical reports without a key or ledger:

```powershell
.\.venv\Scripts\python.exe -m json.tool outputs/revision896_retry_v5_preflight/calibration_report.json
.\.venv\Scripts\python.exe -m json.tool stats/revision896_retry_v5/calibration_inspection.json
```

Open `analyze_networks.ipynb` in your notebook editor and select the fresh-study
section. Its data-loading cells are read-only; legacy sections are separately
labelled. To recompute the **full ledger-backed inspection**, the authorized
execution owner needs the original local ledger:

```powershell
.\.venv\Scripts\python.exe -B inspect_calibration.py
```

For the public simulator's local build, install Node.js, then:

```powershell
npm --prefix viewer ci
# Optional: recheck the frozen V6 receipts and rebuild the public derivative.
.\.venv\Scripts\python.exe -B export_revised_viewer.py
npm --prefix viewer test
npm --prefix viewer run build
npm --prefix viewer run verify:public
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory viewer/dist
```

Open `http://127.0.0.1:8765/layers.html` (the root URL opens the same workspace).
The default shows 104 revised V6 calibration graphs. The 68 V5 graphs and earlier
historical/pilot results remain separate datasets. Matched comparison
buttons vary one recorded factor; coverage cells load existing repetitions.
No viewer action calls a model. See [viewer checks and limitations](docs/VIEWER_INTERACTION_REVIEW.md).

**Main paid execution remains blocked.** After team approval, the execution
owner must retain the original ledger, configure a fresh `OPENAI_API_KEY` locally,
validate current prices/model access, and supply a truthful hash-matching main
review with a finite cumulative ceiling. Do not edit flags just to force it to run.
Do not launch the old 896-target command. V8 calibration is complete; the
separate Hindi candidate failed its reliability screen and is not adopted.
Scientific decisions and main-budget authorization remain in [plan.md](plan.md).
No automatic replacement or main-study generation is running.

### Current Offline Review Commands: No API Spending

```powershell
.\.venv\Scripts\python.exe -X utf8 -B precollection_audit.py
.\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_precollection tests.test_inspect_calibration_v6 -q
```

This canonical audit needs the checked-in artifacts and matching repository
environment, not an API key or private billing ledger. It runs the second V8
recovery adapter inside a temporary public-data copy. New tables and the figure
go to `outputs/precollection_review/`; historical calibration stays unchanged.
The notebook's October 17 section checks attempt state and all output bindings
through `precollection_audit.verified_report()` before displaying the results.
Do not run older in-place inspectors to refresh these new reports.
