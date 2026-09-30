# Revision Validation Report

## Current Budgeted Revision Recheck

- [x] Paid pilot: **24/24 networks** (20 original + four Sol checks), exactly 50 roster nodes each, nonzero edges,
  no self-links, matching graph/PNG hashes and exact replay of recorded changes.
- [x] Cost receipt: **$1.886188** conservative total, **2,702 attempts**, zero
  unresolved requests. All failed replies remain charged. Sixteen parser failures
  are annotated; 454 earlier responses lack annotations, so a complete failure
  rate cannot be inferred.
- [x] **33 Python tests passed**. Red/green cases reproduced and repaired reversed
  duplicate guidance, wrong response counts, neutral framing and insertion-order
  dependence. Budget tests cover caps, persistent uncertainty, cached resume,
  model mismatch, artifact loss and offline verification without API dispatch.
- [x] **9 Node tests passed** for data/edge parity, layout, actual pilot replay,
  filter intersections, metric handling, SVG escaping and real Sol coverage.
- [x] Notebook: **41 code cells**, with **32 archived bodies skipped** and nine
  maintained/setup cells executed offline. Pilot metrics and source variants,
  usage summaries, comparison figure and 528 synthetic controls are included.
- [x] `pip check`: **No broken requirements found**. `git diff --check` passed.
  A targeted key-pattern scan found no credentials in the changed sources,
  notebook, executed notebook, viewer or maintained documentation.
- [x] Canonical scalar analysis regenerated 176 retained historical measurements
  and 528 controls. All 193 original source hashes remain unchanged. Earlier
  corrected measurements and manifest are archived as `*.insertion_order.*`.
- [x] Browser: actual pilot A/B graphs have 50 personas and 236/229 edges;
  replay starts at zero edges and reaches the saved graph. Intermediate views
  hide final metrics. Desktop and 390 x 844 mobile checks have no document overflow;
  the checked console returned no errors or warnings. Screenshot is in `outputs/qa/`.
- [x] The twelve-panel engineering comparison PNG was generated from verified
  adjacency lists and visually checked. It names model, method, seed, node/edge
  counts, political-label colors and the shared layout. It marks pilot limitations.
- [x] Independent read-only review found stale-score export and partial-model
  figure gaps. Both were repaired and regression-tested; the final bounded
  recheck found **no material findings**. Mixing scores are recomputed from
  verified adjacency lists; partial comparison figures label missing panels.
- [x] Notebook plotting dependency is pinned to `matplotlib-inline==0.1.7`,
  compatible with this repository's existing Matplotlib 3.6.2. Full execution
  exercises the new condition-selection plot cell, not just image-file loading.
- [x] Browser filter checks covered multiple models/methods/languages, historical
  country/seed intersections, empty selections, metric changes and URL restoration.
  Sol/Luna sequential panes displayed 50 nodes each and 204/217 edges. Mobile
  checks found no document overflow; wide plots scroll within their own regions.
- [x] Saved Sol/Luna topology and mixing SVGs contain eight actual US-English
  runs; both parse as XML. Their CSV includes run identities and source hashes.
- [ ] In-app browser download completion remains unverified: its download-event
  hook timed out. Rendering and generated export content were verified separately;
  this report does not claim a successful browser file-save test.
- [ ] Astra protocol review, adult rosters, bilingual review, main collection,
  statistical inference, benchmark reconciliation and publication remain pending.

The earlier report below records the initial offline phase. Its missing paid
runner/replay statements are superseded by the current recheck above. A passing
software check does not establish real-world validity or conference readiness.

## Archived Initial Offline Recheck

Scope: corrected historical export, shared metric/parser repairs, notebook and
static viewer. This is not a claim of zero undiscovered errors or scientific
validation. Existing dirty-worktree changes outside this scope were preserved.

## Observed Checks

- [x] Python regression suite: **19 tests passed**, including full 50-person mocked
  generation across **4 methods x 5 languages** and exact event replay. These are
  synthetic responses for software testing, not newly generated research networks.
- [x] Viewer Node suite: **4 tests passed**, including the complete exported data
  contract, orientation-invariant comparison, invalid-edge rejection and event replay.
- [x] Dependency check: `pip check` reported **No broken requirements found**.
- [x] Notebook: **37 code cells executed**, with **32 archival cell bodies
  explicitly skipped** by default; five maintained/setup cells ran successfully.
  Source JSON validates. The executed copy is under `outputs/qa/`.
- [x] Historical matrix: exact **192 expected run IDs** inventoried, not merely
  192 arbitrary files. **176** graphs retained, **16** excluded, **23** self-links.
- [x] All **193 source hashes** (192 graphs and one roster) unchanged. Viewer
  edges and scalar metrics independently recomputed from every retained source
  graph and compared in the Python regression suite.
- [x] All **176 retained graphs' historical PNGs decode**. This does not prove
  the old image labels or pixels match a particular computation. Graph validity
  and image readability are separate; a missing-PNG fixture proves that a valid
  graph is not falsely quarantined.
- [x] New `topology_comparison.png` and `network_comparison.png` regenerated from
  the corrected pipeline and visually inspected. Both name the model, country,
  method scope and seed aggregation. Identity coordinates are explicitly not
  social distance. The topology plot shows individual runs, not error bars/CIs.
- [x] Browser rendered both actual 50-node graphs and the 50-row accessible table.
  Default reference network has 190 edges; the tested Japanese comparison has
  191. These were also checked against exported data.
- [x] Browser persona selection displays attributes and neighbors. Sandbox adds
  one tie without changing research data, hides final metrics, rejects self-links,
  and resets after discard. Language filtering and 2D mode worked.
- [x] Responsive check at **390 x 844**: one-column panels, readable controls,
  rendered graph, and no horizontal document overflow. Viewport override reset.
- [x] Downloaded browser JSON verified on disk against the source export:
  **50 personas, 190 edges**, identical metrics. The browser's download-event
  wait timed out, but the actual downloaded file existed and passed parity checks.
- [x] Browser console inspection returned no errors/warnings during the checked
  desktop interaction sequence. No full assistive-technology audit is claimed.
- [x] Independent read-only reviewer rechecked the payment lock, PNG/graph-status
  separation and sandbox. **No material findings in that bounded recheck.**

## Explicit Non-Passes and Limits

- [ ] New paid experiments: **not run**. Shared dispatch is locked before client
  creation. No exposed key was stored or used; rotation is still the user's task.
- [ ] Cost-capped execution, actual usage and durable failed-request/run records:
  **not implemented**. A request-count estimate is not a dollar cap.
- [ ] Conference-ready inference: **not established**. Baselines, benchmark
  reconciliation, source-grounded adult rosters, human translation review,
  replication/power and prespecified uncertainty analyses remain pending.
- [ ] Historical attempt/refusal/retry rates and model snapshots: **unknown**.
  Stored successful graph files cannot reconstruct them.
- [ ] **19 undefined homophily scores** remain missing by definition; missing
  values are not replaced with zeros. Group-level CSVs preserve their context.
- [ ] The roster includes **nine minors**, including **four under five**, with
  political labels. Existing results are therefore not evidence for a defensible
  adult political-network population.
- [ ] New-run replay import, public GitHub Pages deployment and publication:
  **not completed**. The static viewer currently exports historical final graphs.
- [ ] Full repository-wide security audit, live API integration, human translation
  validation and all manuscript figures: **not covered by this bounded check**.

Known non-fatal warning: pandas 1.5 uses NumPy's deprecated `find_common_type` in
some grouping operations. Tests still pass. The notebook verification script
selects the Windows selector event loop to avoid the Jupyter/Tornado Proactor
warning observed during the initial manual execution.

## Reproduce

From the repository root, after installing `requirements.txt` and running
`npm --prefix viewer ci --ignore-scripts`:

```powershell
.\.venv\Scripts\python.exe -B export_research_viewer.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/verify_revision_notebook.py
.\.venv\Scripts\python.exe -m pip check
npm --prefix viewer test
npm --prefix viewer run build
```

Read the manifest, verification and quarantine tables together. Export completion
does not mean every input graph passed validation.
