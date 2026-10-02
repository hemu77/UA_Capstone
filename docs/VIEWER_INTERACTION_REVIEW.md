# Viewer Interaction Review

## Current Check: 2026-10-01

This pass upgrades the offline viewer, not the frozen study protocol. No model
was called, no original graph was regenerated, and no new spending was authorized.
It includes a focused parent review and live Codex-browser checks, not an
independent scientific review or a guarantee of error-free software.

### Data and Research Boundaries

- [x] Default dataset: all 68 fresh calibration graphs and their 50 designed
  adult personas. Four models, four methods, seven country/language settings.
- [x] Historical/pilot evidence remains a separate 204-graph dataset. The same
  numeric persona ID cannot be compared across different roster hashes.
- [x] Export verifies reviewed report/receipt hashes, source/runtime contract,
  adjacency/PNG hashes, exact event reconstruction and recomputed measurements.
- [x] Coverage shows observed repetitions against eight planned; unavailable
  conditions stay unavailable. Calibration is not presented as 896 completed runs.
- [x] Model, country, language and method shortcuts hold other recorded matching
  fields constant. The UI flags mismatches and model decoding limitations.
- [x] Demographic tables use the actual roster attributes. Age assortativity is
  separate from categorical Coleman homophily. Differences are descriptive, not
  accuracy scores, causal effects or significance tests.

### Automated Checks

- [x] `cd viewer; npm test`: 24 passing tests, including exact final replay of
  all 68 fresh exports and preservation of the historical dataset.
- [x] `python -B -m unittest discover -s tests -p test_calibration_viewer.py`:
  five passing tests, including altered receipts/artifacts/metrics and portable
  review-path matching. Use the frozen Python/dependency environment to re-export.
- [x] `cd viewer; npm run build`: passed. Content-versioned assets prevent a new
  page from silently using an older cached controller.
- [x] Scope is the viewer/export change; this does not claim the full Python
  study suite or the paid calibration was rerun in this pass.

### Observed Browser Flows

- [x] Four-country and four-language matched comparisons, persona 3 neighbor
  lists, selecting the same persona again to clear, and whole-network reset.
- [x] Persona journey stepping, complete-network playback/pause, iterative
  recorded history, and playback from the comparison view.
- [x] Apply, add, remove, undo, unavailable-language disabling and coverage-cell
  selection. A stale matched-comparison note now clears after changing layers.
- [x] 3D/top camera controls and expansion; wheel input zooms without page scroll
  when navigation is on. Turning it off lets wheel input scroll the page.
- [x] Desktop and 390 x 844 responsive layout; no document-width overflow in the
  checked mobile view. Wide research tables have their own scrolling container.
- [x] Legacy/fresh dataset switching, reload/share state, source links and numeric
  research readouts. No browser warnings/errors observed in these checked flows.
- [x] Selection export downloaded and parsed as JSON: the selected runs, personas,
  recorded events, measurements and provenance, not raw API prompts or billing logs.

### Remaining Gates

- [ ] Teammates should test physical trackpad pinch/inertia on their own devices.
- [ ] Broader cross-browser and screen-reader acceptance testing remains open.
- [ ] A final-study viewer refresh and publication-quality figures follow the
  approved collection/analysis, not invented data or a generic average graph.
- [ ] Scientific approval, human bilingual validation where required and the
  main-batch spending decision remain separate, unresolved gates.

The old UI-stage $50 proposal is superseded by the current
[team approval request](TEAM_APPROVAL.md). Viewer readiness does not authorize
API spending or resolve translation, population or statistical limitations.
