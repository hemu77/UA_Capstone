# Cross-Cultural and Linguistic LLM Network Study

A research revision of [Stanford SNAP's LLM social-network code](https://github.com/snap-stanford/llm-social-network).
We study how language models generate friendships between fictional personas
under different country framings, instruction languages and generation methods.

**Status: 68/68 calibration networks verified. Main collection is stopped pending
team research-design and budget approval. This is not a completed 896-run study
or a guarantee of research validity.**

## Start Here

- [Team knowledge brief](docs/TEAM_KNOWLEDGE_BRIEF.md): plain-language history,
  reviewer concerns, improvements, current evidence and unfinished work.
- [Team approval checklist](docs/TEAM_APPROVAL.md): the decision required before
  spending approximately **$101 more**.
- [Calibration results](CALIBRATION_RESULTS.md): actual accounting, verification,
  estimated runtime and assumptions.
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

## Current Design

| Dimension | Current scope |
| --- | --- |
| Models | `gpt-4.1`, `gpt-5.6-luna`, `gpt-6-luna`, `gpt-6-sol` |
| Methods | Global, local, sequential, iterative |
| Country framing | US, India, Japan, Brazil |
| Instruction languages | English, Hindi, Japanese, Brazilian Portuguese |
| Personas | Same 50 designed fictional adults; no US party names or race labels |
| Settings | Four countries in English, plus US in the other three languages |
| Repetitions | Eight planned per model/method/setting |
| Total | **4 x 4 x 7 x 8 = 896**, including reusable calibration runs |

This is a seven-setting design, **not** the full country-by-language factorial.
Candidate attributes stay in English JSON. It tests instruction-language
sensitivity, not participants' spoken language or fully translated personas.
Seeds control local randomization; they are not population sizes or guarantees
of provider determinism. `revision224.py` retains its filename for compatibility
but reads `study_protocol_896.json`.

## Verified Progress

- [x] **68 calibration graphs**, each with 50 nodes, an adjacency file, PNG and receipt.
- [x] GPT-6-Luna: 7 settings x 4 methods x 2 repetitions = 56 graphs.
- [x] Other three models: US-English x 4 methods x 1 repetition = 12 graphs.
- [x] Receipt/hash checks, exact event replay and recomputed topology/homophily.
- [x] **204 offline matched controls**, separate from model-generated graphs.
- [x] **88 Python tests**, **896 offline fixture checks** and 12 control fixtures.
- [x] Notebook JSON/syntax checks and fresh-study data-loading execution.
- [x] Reviewed cost logging, budget reservations, cached recovery and source provenance.
- [ ] Team agreement on claim scope, translation evidence and analysis/precision plan.
- [ ] Separate main-study approval and collection of the remaining **828**.
- [ ] Final RQ analysis and a paper with claims supported by the completed evidence.

The 7,749 received calibration replies included **140 rejected parse attempts
(1.81%)**. Their costs remain recorded; bounded corrections produced the verified
graphs. Tests passing does not mean every first model response was valid.
Human bilingual signoff is not complete. Eight repetitions are not proven power.

## Cost and Next Decision

| Item | Conservative accounting/estimate |
| --- | ---: |
| Additional calibration usage/reservations, including superseded attempts | $4.45452 of $5 |
| Current cumulative historical + calibration ledger | $6.47334 |
| Remaining 828, unchanged reviewed protocol | **About $101.00 additional** |
| Remaining with 20% planning allowance | **$121.20 additional** |
| Proposed cumulative ledger ceiling if approved | **$128 total** |
| Remaining serial API time | **28.44 hours**, or **42.66 hours** with allowance |

This is not a provider invoice or a fixed-price guarantee. Forecasts transfer
Luna treatment ratios to other models' US-English baselines; most other-model
multilingual conditions have not been calibrated. Runtime excludes local
analysis, rendering and pauses.

**Team question:** does the intended paper accept this controlled, limited
model-behavior study, or require stronger population, translation, replication
or empirical-validity evidence before another ~$101 is spent?
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
| Current V5 calibration: 68 graphs | Verified current-contract evidence; potential reuse within 896 if protocol remains unchanged. |
| Offline fixtures and synthetic controls | Software checks/reference rules, never additional LLM observations. |

Previous README and architecture content is preserved in
[project history](PROJECT_HISTORY.md),
[the original capstone README](CAPSTONE_README_ARCHIVE.md) and
[archived architecture](ARCHITECTURE_BEFORE_REVISION.md).
Historical PDFs and plots are not current validity claims.

## Repository Map

| Location | Purpose |
| --- | --- |
| `revision224.py`, `revision224_prompts.py`, `study_protocol_896.json` | Current experiment, fresh personas/prompts and execution gates. |
| `generate_networks.py`, `constants_and_utils.py` | Shared generation, strict parsing and bounded corrections. |
| `paid_study.py` | Durable reservations, usage, diagnostics and cached requests. |
| `analyze_networks.py`, `make_matched_baselines.py`, `inspect_calibration.py` | Metrics, controls and ledger-backed calibration inspection. |
| `analyze_networks.ipynb` | Maintained analysis; separates fresh and historical evidence. |
| `outputs/revision896_retry_v5/` | 68 current adjacency/PNG/receipt artifact sets. |
| `outputs/revision896_retry_v5_preflight/` | Manifest, exact prompts, checks, calibration reports and closed authorization. |
| `stats/revision896_retry_v5/` | Current metrics, homophily, controls, paired contrasts and timing. |
| `text-files/`, `plots/`, earlier `stats/` and `outputs/` | Preserved historical data and figures; follow each collection's provenance. |
| `viewer/` | Private Three.js analysis interface; currently exports earlier historical/pilot data. |
| `tests/`, `docs/` | Offline regression checks and team documentation. |

The viewer reads saved graph data, not PNG pixels. Recorded formation replay is
available only where decisions were saved. It does not invent reasoning,
simultaneous agent actions or an "average" observed network. Current fresh
calibration data should be inspected through their reports/notebook until
explicitly exported to the viewer.

## What Is Not Published

Credentials, the original private `budget.sqlite`, build caches, diagnostic
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

Inspect existing reports without a key or ledger:

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
npm --prefix viewer test
npm --prefix viewer run build
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory viewer/dist
```

Open `http://127.0.0.1:8765/layers.html`. This serves the included historical/pilot
export; it does not create model results or represent the 68 fresh runs automatically.

**Paid execution is intentionally blocked.** After team approval, the execution
owner must retain the original ledger, configure a fresh `OPENAI_API_KEY` locally,
validate current prices/model access, and supply a truthful hash-matching main
review with a finite cumulative ceiling. Do not edit flags just to force it to run.
Only then is the intended command:

```powershell
# NOT authorized now. This targets 896 total and reuses verified eligible receipts.
.\.venv\Scripts\python.exe -B revision224.py --execute --limit 896
```
