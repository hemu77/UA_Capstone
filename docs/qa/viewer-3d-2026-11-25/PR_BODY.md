# Layers viewer: all 890 valid graphs in 3D, one filter system, no repeated statements

Branch `claude/viewer-3d-all`. **Not deployed, not pushed.** No provider calls, no generation, no credentials.
Frozen data, sealed outputs, `viewer/archive/` and the pinned fixtures (`primary-viewer.json`,
`saved-networks.json`, `saved-plots.json`) are unchanged.

Study facts are exact and stated once, in the study ledger: 896 attempts, 890 valid graphs (12 empty), 6 failures,
6 supplemental graphs excluded.

## What changed

**All graphs in 3D**
- **Atlas** (default view): all 896 attempt slots as 112 condition towers of 8 seed planes. 890 slabs carry a graph
  (12 are visibly empty), the 6 failed attempts are hollow frames. 6 draw calls.
- **Stack**: 1 to 8 runs as lit planes with shared persona positions; a whole condition (8 seeds) fits; a 9th is
  refused with a message. Shared-tie ribbons, optional lift by degree, optional aggregate plane.
- **Replay**: one run in its recorded generation order (not elapsed time), checksum-verified as before.
- Every valid graph is two clicks away (tower, plane), one click from the result list, or one link (`?v=stack&l=<slot>`).

**Filters**
- One live filter system (model, method, country framing / language, seed, graph state, search) with counts,
  removable chips, Reset, and a separate "On stage" group, in a sticky bar. The bar turns green when a filter is set.
- Short readable links (`?v=stack&m=gpt-4.1&p=local&s=us-english&l=56.57&a=56&k=f50`); old `?view=` links still open.

**Integration work in this PR**
- Store: new validated `consensus` key (`null | {scope:'stack'|'filters', min}`), with tests. The 3D toggle, the
  Compare panel and the link (`k=f50`) all drive the same aggregate plane.
- `layers.js` no longer writes the address bar or paints the legend; node and tie colours come from the insight
  module; Atlas emphasis uses the filter module's `slotMask`; saved metrics are awaited before metric bars,
  homophily and export. One update that changes the active run and the view applies both.
- Removed: `#legacy-controls`, the old primary gallery markup and dialog, their CSS, the `plot-gallery.js` script tag.
  Tests now assert those ids stay deleted.
- Repetition removed: the population is stated only by the ledger (the empty filter bar reads "All 890 valid graphs ·
  no filters set"); "seed order, not time" appears once in the Atlas key; one legend title; one identity-thread
  explanation; the supplement panel no longer restates the primary count.
- Layout: Atlas setting labels sit in an aligned column left of the towers (country / language codes on phones);
  plane labels sit outside their plane and slide inside the canvas instead of disappearing; the stage key is one
  compact card; 3D toggles carry an on/off lamp; the study-design popover can no longer be clipped; the Compare panel
  no longer overflows its dock; replay fills the stage; phone stacks keep their corners in frame.
- Raw model keys replaced by display names in filters and chips ("GPT-4.1").

## Before / after

| | Before (`baseline/`, 1x) | After (`final/`, 2x) |
|---|---|---|
| First screen, desktop | `docs/qa/viewer-3d-2026-11-25/baseline/desktop-top.png` | `docs/qa/viewer-3d-2026-11-25/final/desktop-atlas.png` |
| Filters | `docs/qa/viewer-3d-2026-11-25/baseline/desktop-filters.png` | `docs/qa/viewer-3d-2026-11-25/final/desktop-atlas-filtered.png` |
| Layers | `docs/qa/viewer-3d-2026-11-25/baseline/desktop-top.png` (6 at most) | `docs/qa/viewer-3d-2026-11-25/final/desktop-stack-8.png` |
| Replay with persona | `docs/qa/viewer-3d-2026-11-25/baseline/desktop-stage-replay.png` | `docs/qa/viewer-3d-2026-11-25/final/desktop-replay-persona.png` |
| Attribute lens | none | `docs/qa/viewer-3d-2026-11-25/final/desktop-attribute-lens.png` |
| Overlap and aggregate plane | none | `docs/qa/viewer-3d-2026-11-25/final/desktop-overlap-consensus.png` |
| Browsing all graphs | `docs/qa/viewer-3d-2026-11-25/baseline/desktop-gallery.png`, `docs/qa/viewer-3d-2026-11-25/baseline/desktop-gallery-dialog.png` | the Atlas and result list above |
| Phone | `docs/qa/viewer-3d-2026-11-25/baseline/phone-top.png` | `docs/qa/viewer-3d-2026-11-25/final/phone-atlas.png`, `docs/qa/viewer-3d-2026-11-25/final/phone-stack.png` |

Final screenshots are 256-colour palette PNGs (2.4 MB in total) made by `final-shots.cjs` and `compress-shots.py`.

## Tests run (2026-11-25, headless Chromium with SwiftShader)

| Command | Result |
|---|---|
| `npm run build` | pass (also proves `public/data/graphs/` equals a fresh generation) |
| `npm test` | 107 of 107 pass |
| `npm run verify:public` | PASS: 896 attempts, 890 graphs, 12 empty, 6 failures, 890 verified event logs, 6 supplements separate |
| `python3 scripts/verify_viewer_archive.py` | not run to completion here: the script shells out to `rtk`, which is not installed in this environment. `git status` shows no change under `viewer/archive/` or to the pinned fixtures |
| `browser-check.mjs` | pass |
| `primary-browser-check.mjs` | pass |
| `polish-browser-check.mjs` | pass |
| `plot-gallery-browser-check.mjs` | pass |
| `navigation-check.mjs` | pass |
| `layers-3d-browser-check.mjs` | pass |
| `layers-filters-browser-check.mjs` | pass |
| `layers-shell-browser-check.mjs` | pass |
| `layers-all-graphs-browser-check.mjs` (new) | pass: each of the 890 run ids appears exactly once in the result list; all 890 open in Stack and Replay with a rendered frame; 28 slots open by link; a failed slot is refused |
| `check-preview.cjs` | pass: `file://`, network blocked, 0 requests, 0 console errors |

No assertion on counts, failures, checksums or supplement isolation was dropped. Where the filter bar stopped
restating the population, the same checks now assert the exact numbers on the ledger and assert that the bar does
not repeat them. The identity-thread assertion moved from the key to the in-scene label.

Draw calls measured: Atlas 6 (budget 12); Stack 6 to 9 for 1 to 8 planes including persona thread and aggregate plane (budget 6 per plane); Replay 6.
Zero console errors or warnings from page code at 1440x900, 1024x768 and 390x844.

## Self-contained preview

`docs/qa/viewer-3d-2026-11-25/preview/layers-preview.html`, 3.33 MB, built by `build-preview.mjs` from `viewer/dist`.
All scripts (including three.js), styles, the index, ties, metrics, supplements, both study figures and all 890
recorded histories are inlined; integrity checks still run. Nothing is "not available" in it.

## Remaining limitations

- Performance was measured only under software rendering. The first Atlas frame takes 2.6 to 3.0 s in the cold-start
  check (target 2.5 s) and about 0.6 to 0.8 s after navigation start on a warm browser. Frame rate and first frame on
  real integrated graphics have not been measured.
- The last one-line fix (the Compare panel adopting an aggregate setting that arrives by link) was followed by the unit
  tests, the two checks that touch that module and the preview check, not by a full re-run of all nine browser checks.
- Phones show country / language codes on the Atlas rows and no seed or method labels; full names are in the filter
  panel and the tower card. On touch, a tap on a tower opens it directly, so the hover card is desktop only.
- With eight planes on a phone the plane labels sit over the left edge of the planes.
- On a 1024 px wide window the stage key can cover the nearest corner of a replay plane.
- The preview needs a secure context for its checksums (`file://`, `https://` or localhost) and a browser with
  `DecompressionStream`. It was tested in Chromium only.
- `shot.cjs` is the baseline capture script and targets the old page; use `final-shots.cjs` for the new one.
- Dark theme only. No visual regression baseline is automated; screenshots were reviewed by eye.
