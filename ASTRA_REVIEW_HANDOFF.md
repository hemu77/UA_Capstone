# Astra Review Gate: Before Main Research Collection

This is a read-only scientific/software review, not an instruction to spend API
tokens or certify conference acceptance. Review the files and measured evidence;
do not assume that a checklist or a plausible PNG proves validity.

## Evidence Available

- [x] Historical inventory: 192 graphs, 176 retained, 16 quarantined for 23
  self-links. Original graph/roster hashes remain unchanged.
- [x] Shared metrics, strict parsing, bounded retries and a durable $5 pilot / $50
  cumulative reservation ledger. Failed paid replies are not refunded or sanitized.
- [x] Thirty-three offline Python checks and nine viewer checks at this revision gate.
  Rerun them rather than assuming this document remains current after edits.
- [x] Full-50 engineering pilot completed: 24/24 verified (20 original + four
  Sol US-English checks), $1.886188 conservative cost, 2,702 attempts and zero
  unresolved requests. Read
  `outputs/revision_budget_v1/pilot_summary.json` for the actual completed count,
  conservative cost, unresolved requests and missing parser annotations.
  Sixteen parser failures are annotated; 454 earlier replies lack annotations.
- [x] Real pilot event replay and provenance in the Three.js workbench. Historical
  networks have no invented chronology; manual/intermediate views hide final metrics.
- [x] Multi-select filters drive actual topology/mixing plots. Counts, missing
  combinations and prompt-variant boundaries are explicit; exported mixing
  values are recomputed from verified graphs. Sol covers only US-English.
- [x] 528 matched offline controls on historical graphs. These are not extra
  LLM runs, independent replications, real-network benchmarks or converged null samples.
- [x] Notebook execution without paid requests; archived unavailable experiments
  are skipped explicitly, not treated as successful scientific experiments.

## Review These Decisions

- [ ] **Population:** the engineering roster contains nine minors with political
  labels. Do not use it for confirmatory adult political-network inference.
  Define five new adult rosters from defensible documented sources; resampling
  the same 41 adults does not establish five independently grounded populations.
- [ ] **RQ1:** the reduced study estimates country-framing responses on
  US-structured synthetic populations. It cannot establish authentic Indian,
  Japanese or Brazilian social behavior. Verify a true no-country control and
  parallel context descriptions without stereotype-priming differences.
  The neutral setting now omits the country instruction; the supplied demographics
  still imply a US population, so this is a control for the explicit country label.
- [ ] **RQ2:** categorical Coleman scores and numeric age assortativity describe
  association, not causal importance. Attribute masking/intervention and
  correlated-demographic controls are absent from the reduced matrix. Narrow the
  claim unless those experiments are added and separately budgeted.
- [ ] **RQ3:** main scope compares two declared OpenAI configurations; the Sol
  extension adds four engineering checks, not a third confirmatory model.
  Luna/Sol use reasoning effort `none`; Mini uses temperature 0.8. Local seeds do
  not guarantee provider determinism. Freeze model identifiers and settings.
- [ ] **RQ4:** bilingual reviewers must check scenario equivalence, category
  meanings, candidate lists and retry instructions for English, Hindi, Japanese
  and Brazilian Portuguese. Instructions vary, not participants' spoken language.
- [ ] **Coverage:** live non-English pilot cells use Luna only. Mini translations,
  other country settings and the neutral setting still need the selected main
  study's bounded preflight. The viewer currently uses one historical roster;
  introducing five rosters requires a matching roster/attribute export contract.
- [ ] **Sampling:** 640 is a draft budget target, not a power calculation.
  Eight settings x two models x four methods x five rosters x two repetitions
  gives ten runs per condition, clustered within five rosters. Prespecify
  meaningful effect sizes, paired contrasts, uncertainty and multiplicity.
  The single-roster engineering pilot cannot estimate between-roster variance.
- [ ] **Methods:** candidate opportunities, selection counts and initialization
  differ. Compare within methods first; density-matched controls do not remove
  all whole-protocol confounding. Broad candidate/temperature/size panels are deferred.
- [ ] **Failure policy:** Mini global reversed duplicates and Hindi sequential
  wrong-count replies triggered prompt/retry fixes during debugging. Earlier
  variants and missing annotations remain visible. Freeze the policy before main
  collection; do not restart confirmatory failed cells until one happens to pass.
- [ ] **Metrics:** canonical node/edge ordering fixes serialization-dependent
  Louvain results. Ties for the largest component use canonical node order;
  their path statistics can depend on that declared tie choice. Primary density,
  clustering and largest-component shares were unchanged by this correction.
- [ ] **Benchmarks/manuscript:** reconcile the eight saved real-network entries
  with the submission's larger dataset claims and the reported method settings.
  Neither the viewer nor synthetic baselines repair manuscript/provenance gaps.

## Required Review Output

Report at most six material findings with exact file/line or data evidence,
consequence, confidence, minimal repair and a runnable check. Separate software
correctness, experiment completion and supported scientific claims. Do not call
the project complete or recommend resubmission while a required gate remains open.

Commands from the repository root (prefix with `rtk proxy` when using Codex):

```powershell
.\.venv\Scripts\python.exe -B paid_study.py --verify
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
npm --prefix viewer test
```

Main execution is intentionally blocked. No API key belongs in a review prompt,
notebook, browser, report or Git commit. Raw provider logs stay local and ignored.
