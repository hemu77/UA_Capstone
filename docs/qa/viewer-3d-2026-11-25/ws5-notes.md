# Workstream 5 notes: insight (attribute lens, comparison, consensus, ledger, provenance, supplements)

## What was built

`viewer/layers-insight.js`
- Pure, DOM-free (used by WS2 and node:test):
  - `nodeEncoding(dataset,state)`: `colorOf(id,layerIndex)`, `groupOf(id)`, `dimmed(id)` (legend isolation),
    `legend:[{key,label,color,count}]`. `colorBy:'layer'` uses `LAYER_COLORS`; an attribute colours every plane the same.
  - `edgeEncoding(dataset,state)`: `classOf(edge)` same/mixed/null, `colorOf(edge)` (mirrors `--same`/`--mixed`),
    `dimmed(edge)` (other class under a lens; neither end isolated).
  - `consensus(runs)` -> `[{edge,count,total}]` sorted by count desc then numeric edge; `consensusStats(list,total,min)`.
  - `overlapMatrix(runs)` -> `{ids, matrix[i][j]={shared,union,jaccard|null}}`.
  - `tieMixing(personas,edgeLists,key)`: same-group tie share and the roster-pair reference (share of all possible
    persona pairs that are same-group).
  - `ledgerFacts(dataset)`: 896/890/12/6/6/50 recomputed from `slots` and asserted against `COUNTS`; throws on mismatch.
  - Palettes: `ATTRIBUTES` (gender, age band, religion, political orientation), `AGE_BANDS` (18–24, 25–34, …, 65+,
    viridis segment), Okabe–Ito categorical hues. Tests prove >= 3:1 vs `--panel`, deuteranopia dE76 >= 15 between
    every pair (Machado 2009 simulation), and evenly increasing lightness for age.
- `mountInsight(app,{root,ledger,legend})`:
  - `#study-ledger`: the single population line ("896 attempts · 890 valid graphs (12 empty) · 6 failures · 6
    supplemental graphs excluded") plus a "Study design" popover (design, outcome, roster, supplements, how to read
    the view, limitations, verification). Becomes a 2-column grid in narrow hosts (container query).
  - `#legend`: clickable group chips (`aria-pressed`) writing `groups`; "Show all"; run swatches in layer mode.
  - `#insight-root`: Attribute lens (colour + tie-lens toggles, per-layer same-group share with reference tick and
    the saved Coleman / age assortativity read via `loadMetrics`, pooled share over the filtered graphs); Layer
    overlap (Jaccard matrix, shared-by-all / one-layer-only / distinct); Consensus ties (stacked or filtered scope,
    share threshold, top-10 ties with persona buttons writing `persona`, labelled "Aggregate of N graphs, not a
    run"); Provenance of the active run (label stated once, run id, slot, source link, all hashes, replay wording).
  - Writes only `colorBy`, `lens`, `groups`, `persona` (and `consensus` once that key exists), source `layers-insight`.
- `formation-charts.js`: the one run-label formatter: `runLabel` ("GPT-4.1 · Local · United States / English ·
  seed 44000"), `runParts`, `layerLabel` (differing fields only, same format), `modelName`, `methodName`,
  `countryName`, `languageName`, `settingName`, `seedName`, `plural`. Charts use it; plural fixes in charts.
- `research-plots.js`: row labels via the formatter; demographics are the recorded attributes only.
- `plot-gallery.js`: primary gallery retired. `validatePlotCatalog`/`filterPlots` kept (build pins the fixture).
  New `validateSupplements`, `supplementContext` (relation to the primary slot, e.g. "Same condition as failed
  primary slot 214. It does not fill that slot"), `mountSupplementPanel` (loads `loadSupplements()` only when
  opened; six graphs; separate canvas watermarked "Supplement · not primary"; tie-list fallback without WebGL).
- `network-stage.js`: supplement-only renderer: lit instanced personas sized by degree, one tie buffer, fog,
  damping (off under reduced motion), render on demand, instance picking.
- `layers-insight.css`: scoped to `#study-ledger`, `#insight-root`, `#legend`, `.ins-`; tokens only.

## Files touched
`layers-insight.js`, `layers-insight.css`, `layers-insight.test.js`, `formation-charts.js`, `research-plots.js`,
`plot-gallery.js`, `plot-gallery.test.js`, `network-stage.js`, `plot-gallery-browser-check.mjs`;
`docs/qa/viewer-3d-2026-11-25/requests-ws5.md`, this file.

## Verification
- `node --test layers-insight.test.js plot-gallery.test.js`: 19/19 pass (brute-force consensus check, exact
  roster references 425/1225 etc., Jaccard cross-checked with `researchEvidence`, 156,730 pooled ties, ledger
  mismatch refusal, supplement leak/foreign-tie rejection, palette contrast and CVD separation).
- `npm test`: 90 pass, 1 fail = `graph-data.test.js` layer-label test (WS1 file; exact fix in requests-ws5.md).
- Private build on :8795. `plot-gallery-browser-check.mjs` passes when `layers.js` no longer paints `#legend`
  (simulated in the private copy only; see request WS2.1). Harness screenshots at 1440x900 and 390x844 reviewed.
  Supplements request only after the panel opens; no monolith JSON requested.

## Integrator to-do
Apply `requests-ws5.md` (WS2 legend removal and encodings are the important ones), add the `consensus` key,
then re-run `plot-gallery-browser-check.mjs`.
