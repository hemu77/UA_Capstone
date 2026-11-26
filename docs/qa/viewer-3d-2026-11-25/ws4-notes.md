# Workstream 4 (shell) notes, 2026-11-25

## What I built

- **Layout (`layers.html`, `layers.css`).** Sticky header, then the sticky `#active-filters` bar, then a workspace that fills
  the rest of the screen: a left dock (study ledger always on top; tabs **Filters | Results**), a stage column with the
  canvas filling it edge to edge, and a right dock (tabs **Stack | Persona | Compare**). The analysis cards sit below the
  fold: trajectory and decision ledger, research questions, tie comparison, study coverage table, findings and
  supplements. The footer is a real `contentinfo`.
  - 1150 px and wider: docks sit beside the stage, not over it. The left dock starts open. The right dock starts as a 48 px
    rail of vertical tabs below 1600 px and starts open at 1600 px and wider. Toggles are saved per viewer in
    `localStorage` and the code works without it.
  - 700 to 1149 px: one tab strip with all six panels (Study, Filters, Results, Stack, Persona, Compare). The chosen panel
    opens as a drawer over the left of the stage.
  - Below 700 px: the same tab strip is fixed at the bottom and opens a bottom sheet. The sheet never covers the sticky
    filter bar, so live counts stay visible while filtering. It closes with the Close button, Escape, a swipe down on its
    head, or by tapping the active tab.
- **Stage chrome.** Everything sits in one CSS grid cell over the canvas: the heading top-left, the view switch
  (Atlas / Stack / Replay) with drag mode and the persona Center / Clear buttons top-right, the key (replay note, legend,
  tie key, status) bottom-left, and the camera controls bottom-right. On phone the view switch, transport, persona tools
  and key become rows around the canvas. The transport row only appears when `body[data-view=replay]`. Center / Clear
  persona are hidden while they are disabled.
- **Design tokens.** All type comes from `--text-*` and all spacing from `--space-*`. New tokens: `--focus`, `--overlay`,
  `--glass`, `--scrim`, `--shadow*`, `--radius-lg`, `--accent-line`, `--warn-line`, `--control-h` (32 px; 44 px below
  700 px or on coarse pointers, and `--chip-h` follows), `--dock-left-w`, `--dock-right-w`, `--rail-w`, `--header-h`.
  `--sticky-top`, `--chrome-h` and `--tabbar-h` are measured at runtime. The theme is dark only, as the SPEC states.
- **`layers-shell.js`.** `mountShell(app) -> {destroy()}`. It is idempotent: `layers.html` mounts it from a one-line
  inline module before the data loads, so the shell also works in the load-error state, and the `mountShell(app)` call in
  `layers.js` returns the same handle. It covers:
  - tabs with roving tabindex, arrow keys, Home and End
  - dock toggles and the drawer / sheet
  - focus moves into a panel when it opens and returns to its tab when it closes
  - Escape closes the panel before `layers-ui.js` clears the persona
  - links to a panel (header "Study ledger", skip links) open that panel
  - the header menu below 900 px
  - focus returns to the opener when a native dialog closes
  - the toast for `#action-status`
  - `body[data-view]` and `data-shell` / `data-sheet`
  - a `shell:layout` event on `document` whenever panels change
  - pure helpers: `layoutMode`, `defaultDocks`, `parsePrefs`, `panelDock`, `PANELS`, `BREAKPOINTS`
- **One population statement.** The static page no longer states 896 or 890 anywhere.
  - `#dataset-summary` is now a fallback inside the ledger panel, shown only while `#study-ledger` is empty. It also
    carries load errors.
  - `#coverage-summary`, `#stage-state` and the normal-state `#evidence-notice` are hidden. `#evidence-notice`, Retry and
    the reset link show only when `body[data-state=error]`, in a warning band. The error message is centred on the stage.
- **Clean-up.** Removed the dead `#color` options and the `selected` on RQ3, dropped the 01-06 section numbering, and
  removed the duplicated camera and replay copy. Help now lists keys `1`-`8` and "Esc closes a panel, then clears the
  persona". Supplements card: titled "Supplemental graphs, excluded from the primary study", with `#plot-cohort`
  defaulting to `supplement`. Added an inline SVG favicon, because the `/favicon.ico` 404 was a console error.
- **Legacy.** `#legacy-controls[hidden]` holds `#filter-form` (with `#model #method #culture #language #seed #apply
  #reset-filters #filter-preview`), `#candidate #add #candidate-detail`, `#open-filters`, `#filter-chips` and the old
  `.replay-banner`. Every id from the baseline page still exists exactly once.

## Files touched

`viewer/layers.html`, `viewer/layers.css`, `viewer/layers-shell.js`, `viewer/current-study.test.js` (7 tests: frozen ids,
legacy block, mount roots inside `.stage`, no static population count, one Filters entry, no dead options, breakpoint
helpers, preference parsing) and the new `viewer/layers-shell-browser-check.mjs`.

## How I verified

I ran a private build on port 8794 using SwiftShader Chromium at 1440x900, 1024x768 and 390x844 (deviceScaleFactor 1),
plus 1920x1080 and 1200x800, and looked at every screenshot.

`node layers-shell-browser-check.mjs` passes. It checks:
- there is no horizontal overflow
- the 3D canvas covers 62.7% of the first screen on desktop, 72% on tablet and about 57% on phone
- there are no duplicate ids and every frozen id is present
- the legacy controls are hidden
- the landmarks are present (banner, main, one h1, contentinfo, one live region)
- there is one "Filters" entry
- shell controls on phone are at least 44x44 px, including the Stack panel and the transport
- the ledger is visible without scrolling on desktop and one tap away elsewhere
- dock open / close moves focus correctly
- the sheet opens, closes with Escape, and focus returns
- the transport shows only in replay
- the header menu opens, and Escape closes it
- the help dialog returns focus to its opener
- the toast shows
- a 45-step Tab pass always has a visible focus ring
- there are zero console errors

`node --test current-study.test.js` passes (7/7).

## What the integrator must do

1. **Browser checks of other streams wait on panels that are now hidden by default.** At 1440 wide the right dock is a
   rail, so `#layers`, `#person` and `#insight-root` are `display:none` until their tab opens. To fix this, call
   `page.locator('#tab-stack').click()` (or `#tab-persona`, `#tab-compare`, `#tab-results`) first, or wait with
   `{state:'attached'}`. Affected checks: `primary-browser-check:14` and `navigation-check:11` (both wait on a visible
   `#layers`), and `plot-gallery-browser-check:24` (clicks inside `#insight-root`). `navigation-check:12-16` also asserts
   "#play precedes .laboratory" and uses `#open-filters` / `.run-drawer`. Both are now legacy: the transport lives inside
   the stage.
2. **Repeated statements I cannot fix from the shell.**
   - The empty-state line in the WS3 bar ("All 890 valid graphs · 896 attempts · 6 failures · 12 empty") restates the
     ledger. SPEC B.4 asks for it, but it conflicts with B.5 and D. I suggest "All 890 valid graphs" plus a link to the
     ledger.
   - WS2 states "seed order, not time" three times in Atlas: the axis label, the `#stage-overlay` caption and the
     `.canvas-legend`.
   - `#legend-heading` ("Node colors", written by `layers.js`) sits above the WS5 legend's own "Age band" label. One of
     them should go.
3. **WS2:**
   - Atlas axis labels are clipped at the left edge of the canvas at 1440 and 1920.
   - The `#stage-overlay` legend covers about a third of the phone canvas. Collapse it below 700 px.
   - `#layer-count` still says `/6` in Stack.
   - `mountShell(app)` in `layers.js` is now optional, because `layers.html` mounts the shell. Keep one call.
   - The canvas sits at `inset:0` under the chrome: frame content with roughly 110 px of clear space top and bottom, or
     listen to `shell:layout`.
4. **WS3:** the chip row in `#active-filters` is wider than a phone. The bar box now scrolls horizontally, but the row
   should wrap or scroll itself. `#run-results` is a flex child with a definite height and `overflow:auto`; virtualise
   against it or against an inner element at `height:100%`. Panels are `display:none` while closed, so measure with a
   `ResizeObserver`.
5. Delete `#legacy-controls`, the `.ledger-fallback` paragraph (once WS5's ledger is always present) and the
   `body[data-state=error] .panel-study` rule last. Then remove `#coverage-summary` and `#stage-state` if nothing writes
   them any more.
6. Add `layers-shell-browser-check` to the section D loop (it is already listed there).
