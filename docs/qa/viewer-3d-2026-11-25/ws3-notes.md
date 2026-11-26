# Workstream 3 (filters): notes for the integrator

## What was built

There is now one filter system for the 890 valid primary graphs. It is live (no Apply step) and it talks to the rest
of the page only through the store.

- **Pure model** (`layers-filters.js`, DOM-free and unit-tested):
  - `applyFilters`: values within a facet are OR, facets are AND, and results are always valid primary graphs.
    `isPrimaryRun` rejects supplements (`supp*` ids) and failures.
  - `facetCounts` gives each value's count under the other active facets. `failuresInScope` counts failed attempts in
    scope; failures are never counted once a graph-state filter is set.
  - `activeChips`, `scopeSummary`, `sortRuns` (one row per run_id), `slotMask` (896 slots for the atlas), and
    `normalizeFilters`, `toggleValue`, `withoutChip`.
  - `search` matches whole words across model, method, country name, language, seed, setting key and run_id.
    All words must match.
- **URL**:
  - `encodeState` and `stateQuery` write the B.4 grammar: `v m p s b g q l a n c x f`, plus `i` for isolated
    `groups`. Commas and dots stay literal and defaults are left out.
  - `decodeState` reads the short grammar and legacy `?view=<json>` links. Legacy single-select filters become facet
    values; culture and language map to settings.
  - Archived links throw today's exact message. Unknown or failed slots, an active run missing from `l`, more than
    8 runs, and unknown values all throw a `LinkError` with a readable message. The page then shows a dismissible
    notice and keeps the default view, never a substitute.
  - `syncUrl(app)` is the debounced address-bar writer. It is off until the integrator passes `writeUrl:true`
    (requests-ws3.md item 1).
- **`mountFilters(app,{chips,root,results,writeUrl})`**:
  - `#active-filters`:
    - A summary line that matches the result count exactly.
    - One removable chip per *set* value only. Chips are filled accent; there are no chips for unset facets.
    - `Reset filters`, shown only when something is set.
    - A separate **On stage** group: the view (Atlas / Stack n of 8 / Replay run k of n), one chip per stacked run
      with its layer colour (click to inspect, × to remove), and a single note when stacked runs fall outside the
      filters.
  - `#filter-root`:
    - Search box.
    - Toggle buttons for Model, Method, Country framing / language (the 7 real settings) and Seed, each with
      `aria-pressed` and a live count. Zero-count values stay visible and disabled.
    - Graph: Any / With ties / Empty.
    - Per-facet "n of N" and Clear. Arrow keys move within a facet, so each facet is one tab stop.
  - `#run-results`:
    - Sort (design order, ties, density).
    - "Stack these N" for N of 8 or fewer. Above 8, guidance to narrow to one condition (8 seeds).
    - A virtualised list (about 20 DOM rows for 890) with one row per run_id. Each row has the run label from the
      shared `runLabel`, ties, density and an Empty badge.
    - Row actions: **Replay** and **Add to stack** / **Remove**. A full stack refuses with a spoken message.
    - Roving tabindex, with Up/Down/PageUp/PageDown/Home/End between rows and Left/Right between actions.
    - Row hover and focus write `focusSlot`.
- **`layers-ui.js`**: the old chip painter is removed. Keys `1`–`8` (`MAX_STACK_LAYERS`) replay that stacked run
  through the store (`replayRun`). `/` focuses the filter search.
- **Style** (`layers-filters.css`): tokens only, all selectors scoped. Set and unset values differ in fill, border,
  weight and the check mark. Focus is visible. Targets are 44 px on phone or coarse pointer. The phone bar is two
  rows: summary + Reset, then one horizontal rail. The result list fills WS4's dock and falls back to `70vh`
  elsewhere (`--flt-list-h` overrides).

## Files touched

New: `viewer/layers-filters-browser-check.mjs`. Rewritten: `viewer/layers-filters.js`, `viewer/layers-filters.css`,
`viewer/layers-filters.test.js`, `viewer/browser-check.mjs`, `viewer/polish-browser-check.mjs`. Edited:
`viewer/layers-ui.js`. Docs: this file and `requests-ws3.md`. No other file was changed.

## Verification (private build on :8793, Chromium, deviceScaleFactor 1)

- `node --test layers-filters.test.js`: 16/16 pass.
  - Exact 890 / 12 empty / 878 with ties / 6 failures, and failures split by model and cell.
  - Supplements and failures rejected; no run twice.
  - Facet counts equal `applyFilters` for 20 seeded random combinations; summary count equals list length.
  - Search, chips, slot mask and sorts.
  - URL round trip, every error message, legacy and archived links, `replayRun`, `syncUrl`.
- `npm test`: 104/105. The one failure is in `graph-data.test.js` (requests-ws3.md item 6).
- `layers-filters-browser-check.mjs` (new), `browser-check.mjs` and `polish-browser-check.mjs` all PASS against the
  integrated build of 2026-11-25 evening, with zero page errors and warnings.
  - Kept from the old checks: every assertion on counts (896/890/12/6), failures, supplement isolation, archived-link
    rejection, export, routing aliases and phone overflow.
  - New: the DOM counts for 20 combinations, the 890-row walk, the < 100 ms filter-change budget, sticky bar on both
    sizes, phone targets ≥ 44 px, and the short, failed-slot and legacy links.
- Screenshots at 1440x900 and 390x844 were reviewed (bar, facet panel, results tab, phone sheet). No PNG was left
  in `viewer/`. `browser-check.mjs` now writes its phone screenshot to the OS temp dir unless `VIEWER_SCREENSHOT` is set.

## For the integrator

1. Apply `requests-ws3.md` items 1–4 (URL writer hand-off, subscriber fix, atlas mask, `/` in the help dialog), and decide item 5.
2. The checks open WS4's tabs and docks (`#tab-filters`, `#tab-results`, `#tab-persona`, `#tab-stack`) when the controls
   are hidden. If the shell renames those ids, update `reveal()` in the three checks.
3. Browser checks must import the page's versioned module URL (`layers.js?v=<hash>`). A bare `import('./layers.js')`
   starts a second app. The checks use `window.__layersModule()` for this.
