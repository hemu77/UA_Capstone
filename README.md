# Cross-Cultural and Linguistic LLM Network Study

A research revision of [Stanford SNAP's LLM social-network code](https://github.com/snap-stanford/llm-social-network).
We study how language models generate friendships between fictional personas
under different country framings, instruction languages and generation methods.

**Status, October 2: 104 revised calibration graphs complete at $7.8549 of the
approved $10. Main collection is NOT authorized.** All saved graphs, receipts,
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
| RQ4 | With country framing fixed to US, does instruction language change the networks? |

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
| Repetitions | Proposed staged allocation: global 32, local 8; remaining targets and precision approval pending |
| Total | Not frozen. Global 32 plus eight for each other method would be 1,568; no automatic calibration reuse |

This is a seven-setting design, **not** the full country-by-language factorial.
Candidate attributes stay in English JSON. It tests instruction-language
sensitivity, not participants' spoken language or fully translated personas.
Seeds control local randomization; they are not population sizes or guarantees
of provider determinism. `revision224.py` retains its filename for compatibility
but reads the preserved V5 `study_protocol_896.json`. The new offline candidate is
`study_protocol_next.json`; `revision_next.py` cannot execute paid collection.
Country labels are prompt framings, not validated cultural constructs. RQ4
concerns these instruction wordings until wording robustness is tested.

## Verified Progress

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

The revised calibration contains **11,548 received requests**, including **12
first-response failures among 11,536 decisions (0.104%)**, each corrected once.
Costs include rejected responses. Undefined metrics in 22 empty graphs remain
undefined, not invented zeros. These clustered decision counts do not establish
translation equivalence. Eight repetitions are not proven adequate power.

## Measured Cost and Next Decision

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
remain counted; revised calibration has zero unresolved requests.

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
| `viewer/`, `export_calibration_viewer.py` | Three.js interface with a separate, verified 68-graph calibration dataset and preserved historical/pilot views. |
| `tests/`, `docs/` | Offline regression checks and team documentation. |

The viewer reads saved graph data, not PNG pixels. Recorded formation replay is
available only where decisions were saved. It does not invent reasoning,
simultaneous agent actions or an "average" observed network. V5 calibration
remains the default dataset; the new 104 graphs have not been exported because
simulation work was deferred. Old links still open the historical roster. These
rosters cannot be mixed by matching persona IDs. Coverage shows the historical V5
plan, not the revised allocation or a claim that 896 runs are complete.

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

Use Python **3.11**; calibration used 3.11.9 and the versions pinned in
`requirements.txt`. A different runtime can invalidate frozen execution checks.
Run these PowerShell commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Offline tests: no key and no model calls.
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

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

For the optional private viewer, install Node.js, then:

```powershell
npm --prefix viewer ci
# Optional: recheck receipts, hashes, replay and metrics, then rebuild the fresh export.
.\.venv\Scripts\python.exe -B export_calibration_viewer.py
npm --prefix viewer test
npm --prefix viewer run build
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory viewer/dist
```

Open `http://127.0.0.1:8765/layers.html`. The default shows the 68 historical V5 calibration
graphs. Use the dataset selector for historical/pilot results. Matched comparison
buttons vary one recorded factor; coverage cells load existing repetitions.
No viewer action calls a model. See [viewer checks and limitations](docs/VIEWER_INTERACTION_REVIEW.md).

**Main paid execution remains blocked.** After team approval, the execution
owner must retain the original ledger, configure a fresh `OPENAI_API_KEY` locally,
validate current prices/model access, and supply a truthful hash-matching main
review with a finite cumulative ceiling. Do not edit flags just to force it to run.
Do not launch the old 896-target command. Revised calibration is complete;
global-task, wording-robustness, estimand/allocation and main-budget decisions
remain pending as listed in [plan.md](plan.md). No further API spending is started.
