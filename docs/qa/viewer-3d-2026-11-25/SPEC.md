# Layers viewer 3D upgrade: audit, target design and work split

Date: 2026-11-25. Branch: `claude/viewer-3d-all`. Scope: `viewer/layers.html` and its modules.
Audience: five engineers working in parallel in one working tree, then one integrator.

Owner's request: "want these kind of 3d layers for all, no repetitions, active filters, make it more 3D
so we can show this to simulation companies; make sure it's not sloppy, upgrade it to max."

## 0. Rules for everyone

1. No paid or provider API calls, no new LLM generation, no credentials.
2. Write only inside `viewer/` and `docs/qa/viewer-3d-2026-11-25/`. Never touch `viewer/archive/`
   (`scripts/verify_viewer_archive.py` pins its exact file set), `outputs/`, `text-files/`, or any hash constant
   of a frozen file. Build scripts may READ `outputs/` and `text-files/`.
3. Study facts stay exact and visible: **896 primary attempts, 890 valid graphs (12 of them empty), 6 failures.**
   The **6 supplemental graphs stay out of the primary set**: never in the atlas, stack, filters, counts or comparisons.
4. Honest wording: replay is the recorded **generation order**, not elapsed time, human motion or simultaneous agents.
   Positions are a layout, not geography or social distance. A plane is a saved run, not a time slice or a country.
   Aggregates and overlaps are descriptive, not significance tests. Country framing is a prompt condition.
5. Edit only the files your workstream owns (section C.9). If you need a change in someone else's file, write it in
   `docs/qa/viewer-3d-2026-11-25/requests-<your stream>.md`; the integrator applies it.
6. No deploy, no push. No new npm dependencies. Three.js additions are limited to what is already vendored
   (`three`, `OrbitControls`, `LineSegments2`, `LineSegmentsGeometry`, `LineMaterial`). No post-processing passes.
7. Do not delete a DOM id listed in C.1 "frozen ids". Workstream 4 may move such elements, not remove them.

## A. Audit (behaviour as found on 2026-11-25; line numbers refer to the tree after the section E scaffolding)

Build, unit tests (44), `verify:public` and all five browser checks pass on the baseline. No console errors and no
page errors on desktop or phone; no duplicate DOM ids. Baseline screenshots: `baseline/*.png`.

### A.1 Which graphs can be shown, and how many are missing

- **Data coverage is complete: 0 of 890 are missing.** `viewer/public/data/primary-viewer.json` holds 890 runs with
  edges, 6 failures and the 50-person roster. Cross-check against
  `outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv`: 896 rows, 890 `valid_graph`, 12 `empty_graph`,
  6 failures; identical run_id set; all 890 `.adj` files hash to `adj_sha256` and equal `source_sha256`; edge counts
  match. Totals: 156,730 edges, at most 279 per graph.
- **The layered 3D view is the bottleneck, not the data.** `layerComparison` (`graph-data.js`) rejects more than 6
  layers; `#apply` is disabled when a filter matches more than 6 (`layers.js:174`). A condition has 8 seeds, so even
  one condition cannot be shown whole (the coverage cell click falls back to "Choose a seed", `layers.js:393`).
  At most 6 of 890 (0.7%) are visible as layers at once, reached through five dropdowns.
- The only way to browse all graphs is section "06 / Saved network explorer" (`layers.html:90-103`,
  `plot-gallery.js`): text-only cards, 8 per page, 112 pages, one graph at a time in a modal with a different,
  flatter renderer (`network-stage.js`: unlit `MeshBasicMaterial`, one hue, 1px lines, no layers). It mixes in
  the 6 supplements through a cohort dropdown.
- Payload: the page downloads `primary-viewer.json` 3.28 MB (`cache:'no-store'`, so never cached); opening the
  explorer downloads it **again** plus `saved-networks.json` 2.09 MB and `saved-plots.json` 0.56 MB. 5.9 MB of JSON
  for 313 KB of actual edge information.

### A.2 Repetition and duplication

Data
- Every primary edge list ships twice (`primary-viewer.json` and `saved-networks.json`; `verify-public.mjs` asserts
  they are deep-equal). Run metadata ships three times (also `saved-plots.json`).

Controls
- "Filters & saved runs" appears twice: drawer summary (`layers.html:31`) and `#open-filters` (`layers.html:47`).
- Two unrelated filter systems: five selects for layers (`layers.html:33-37`) and cohort/method/search for the
  explorer (`layers.html:94-98`). Neither affects the other.
- Two 3D renderers with different looks and controls (`layers.js` scene, `network-stage.js`), two persona selects
  (`#person`, `#network-person`), two zoom/reset button sets.
- `#candidate` lists every match as an `<option>`: 890 options with no filter (`layers.js:168`).

Content
- The population statement is written six times with different wording: nav "896 networks in 3D"
  (`layers.html:8`, misleading: 890 are graphs), `#dataset-summary` (`layers.js:404`), `#evidence-notice`,
  coverage summary "890 / 896 saved", section 06 heading "896 networks. Four methods." plus paragraph
  (`layers.html:91-92`), `#plot-status`, footer.
- Camera help is written four times (navigation bar paragraph, controls hint, canvas aria-label, help dialog).
- Replay honesty text is written five times (banner, `#playback-scope`, `#replay-note`, card hint, help dialog).
- Each selected run's full label is repeated in up to eight places on one screen: layer list, stage title, plane
  badge, metric bars, signed-difference sentence, "Inputs:" sentence, exact-values table, source gallery. The source
  gallery repeats the same two links and the same sentence once per run (`layers.js:355-361`).

### A.3 Filters and active-filter state

- Filters are a pending "query", not what is displayed. Layers change only on Apply; chips
  (`layers-ui.js:86-108`) show the query, so after removing a layer, using a Match button, or opening a shared link the
  chips can describe something other than the stage. The chip bar itself says "Filter query — Apply to replace layers".
- Chips are rendered for unset filters too ("METHOD All methods"), so active and inactive look alike; no per-chip
  remove; Reset lives inside the collapsed drawer.
- Single-select only. Country and language are two selects although the design has exactly 7 settings; impossible
  combinations must be disabled option by option (`layers.js:176-180`).
- No counts per option, no empty/non-empty filter, no text search, no way to act on more than 6 matches.
- URL state is one JSON blob in `?view=` (about 960 characters for the default view), not readable or editable.
- On phone the filter controls sit about 2,900 px down the page, below the stage, both rails and the legend.
- Grammar: "1 saved runs match" (`layers.js:173`), "1 recorded events", "1 recorded updates".

### A.4 What makes the 3D weak

- **Flat by construction.** `point()` (`layers.js:273`) drops the layout's third coordinate; every node sits exactly
  on its plane. Replay mode is a single flat disc seen obliquely.
- **No lighting or materials.** Nodes are `MeshBasicMaterial` spheres (`layers.js:305`); planes are a 6%-opacity quad
  with a 1px outline (`layers.js:289-290`). No lights, fog, shadows, depth fade or thickness: depth is carried by
  perspective alone, and stacked layers read as one tangle (see `baseline/desktop-top.png`).
- **Edges** are uniform grey screen-space lines with `depthWrite:false`; far and near ties look identical; nothing
  shows which ties are shared between adjacent planes except a filter dropdown.
- **Camera.** Presets jump with no transition; no damping; "Top" is edge-on to nothing useful; no framing of the
  active layer; fit ignores the rails on wide screens until a resize.
- **Labels.** DOM badges are projected every frame with `offsetWidth/offsetHeight` reads inside the loop
  (`layers.js:196-203`), forcing layout per label per frame; plane badges pile up at one corner and overlap nodes.
- **Stage size.** At 1440x900 the canvas is 868x369 px, 25% of the screen; rails and chrome take the rest.
- **Performance model.** `draw()` (`layers.js:259`) disposes and rebuilds every geometry, material and DOM list on any
  input, including each tick of the opacity slider and each replay step. Each node is its own mesh, geometry and
  material: about 55 draw calls per layer, about 330 at six layers. This cannot scale to an 890-graph view.
- Node colour for "age" is six near-identical blue-green steps; node size carries no information.

### A.5 Sloppiness

- Static `#color` options "political affiliation" and "race/ethnicity" (`layers.html:61`) do not exist in the data;
  JS replaces them at start. `#research-question` marks RQ3 `selected` in HTML while JS sets "method".
- Raw keys leak into the UI: "us / english" in metric bars vs "United States" elsewhere; method and language in
  lower case; "seed 44000" vs "Rep 1 / seed 44000".
- "Apply 1 matching run" wraps onto three lines in the primary button (`baseline/desktop-filters.png`).
- Touch targets below 44 px: layer remove "×" 24x22, zoom "-" 23x30, 13 px checkboxes, 6 px timeline track.
- Phone: header nav wraps into two ragged rows; page is 8,475 px tall; the study population line is below the fold;
  the legend and "Final primary study" bar land between unrelated blocks.
- Two transport rows and one navigation row stack above the stage, so the 3D view starts at y=448 of 900.
- Section numbering 01-06 mixes analysis cards with a second explorer; "Persona inspector" heading says
  "Follow a connection history" even in comparison mode.
- `browser-check` scripts default to `channel:'msedge'`; they need `BROWSER_CHANNEL=chromium` here.

## B. Target design

### B.1 One explorer, three views of the same 896 slots

The design is a 4 x 4 x 7 x 8 lattice. Use it as the 3D structure instead of hiding it in dropdowns.

| View | What is on stage | How you get there |
|---|---|---|
| **Atlas** | All 896 slots: 112 condition towers (16 model x method columns by 7 setting rows), each tower 8 stacked planes (seeds 44000-44007, bottom to top). 890 planes carry a graph; 12 of those are visibly empty (nodes, no ties); 6 failed slots are hollow hatched frames with no nodes. | Default view. `#atlas-mode`. |
| **Stack** | 1 to 8 chosen runs as large lit planes with shared persona positions. A whole condition (8 seeds) fits. | Click a tower, "Stack these" on a filter result of 8 or fewer, Match buttons, coverage cell. |
| **Replay** | One run, recorded update order. | Click a plane or a result row; existing transport. |

Every valid graph is reachable in two clicks (tower, plane) or one result-row click, and by URL. The second
explorer (section 06) and its renderer are retired for primary graphs. Supplements get one small, separately
labelled panel ("6 supplemental graphs, excluded from the primary study") that never shares a stage with primary data.

### B.2 Data: compact, generated, on demand

Generated by `viewer/build-primary-graphs.mjs` from the frozen CSV, the `.adj` files and the roster; committed under
`viewer/public/data/graphs/`; verified and copied by `build.mjs`. Deduplicated by `run_id` (one record per slot).

| File | Content | Size (raw, estimated) | Loaded |
|---|---|---|---|
| `index.json` | design axes, roster, atlas layout, 896 slot records, hashes | about 300 KB | at start |
| `edges.bin` | all 890 edge lists, 2 bytes per tie | 313,460 bytes exactly | at start (needed by the atlas) |
| `metrics.json` | 16 saved metrics per run, columnar | about 250 KB | when a metric, lens or comparison is first shown |
| `supplements.json` | the 6 supplemental graphs, labelled | under 10 KB | only when the supplement panel opens |
| `../primary-events/<run_id>.json.gz` | recorded updates (unchanged) | 3.6 MB total, about 4 KB each | per run, on replay |

First paint needs about 0.6 MB instead of 3.3 MB; nothing is fetched twice. `dist/` stops shipping
`primary-viewer.json`, `saved-networks.json` and `saved-plots.json`; they stay in `public/data/` unchanged as
cross-check fixtures (their pinned hashes must not change). One 313 KB binary beats 890 per-graph files or chunks:
the atlas needs every edge list anyway and one request is cheaper than any chunking scheme at this size.
Exact schemas: C.3.

### B.3 3D visual language

- **Planes are glass slabs**: thin lit box with a bright rim in the run hue, faint inner grid, soft contact shadow on
  the plane below. Failed slots: dashed hollow frame in `--failed`, no fill. Empty graphs: slab plus 50 unlinked nodes,
  rim in `--empty`.
- **Nodes are lit spheres** (`MeshStandardMaterial`, instanced), radius scaled by degree within a fixed range,
  lifted off the plane on a short stem by an optional, labelled encoding ("Lift by degree"; off means flat on plane).
- **Ties** are in-plane lines with depth fade. Ties present in two adjacent planes get a thin vertical ribbon between
  the planes ("shared with next layer"); the selected persona's identity thread stays a gold dashed line and is always
  labelled "same persona, not a tie".
- **Depth cues**: hemisphere plus one key light, exponential fog matched to the background, depth-sorted
  translucency, ground grid with horizon fade, size attenuation with a legibility floor, damped orbit, eased camera
  moves (500 ms; instant under `prefers-reduced-motion`), slow optional auto-orbit.
- **Atlas**: axis labels in 3D (models and methods along X, settings along Z, seeds along Y); towers outside the
  active filter drop to 12% opacity and lose ties; hover highlights a tower and shows a card with condition,
  "n of 8 valid", empties and failures; click flies the camera in and opens the Stack.
- **Labels**: one label layer; positions cached and only reprojected when the camera or data changes; collision
  culling by priority; no layout reads in the frame loop.
- **Rendering budget**: Atlas at most 12 draw calls (points for 44,500 nodes, one line buffer for 156,730 ties, one
  instanced mesh for 896 slabs, labels in DOM). Stack at most 6 draw calls per layer. Render on demand; idle GPU cost
  is zero. Geometry is built once per view and updated in place (attributes, instance colours, draw ranges) on filter,
  persona, opacity and replay changes.

### B.4 Filter model

- State: `filters = {model[], method[], setting[], seed[], graph:'any'|'nonempty'|'empty', search}`. Values within a
  facet are OR, facets are AND. `setting` is one of the 7 real settings (for example `us-hindi`), so impossible
  combinations cannot be expressed.
- Filters apply **live**: Atlas emphasis, result list and counts update immediately. There is no pending query.
- The Stack selection (`layers`) is separate and explicit, shown as its own chip group, so "what is filtered" and
  "what is stacked" can never be confused.
- **Active-filter bar** (`#active-filters`), always visible, sticky under the header on all sizes:
  `Showing 128 of 890 valid graphs · 2 empty · 1 failed attempt in scope (not rendered)` then one removable chip per
  active value (`Model: gpt-4.1 ×`), then `Reset filters`. With nothing active it reads
  `All 890 valid graphs · 896 attempts · 6 failures · 12 empty` and shows no chips.
- Facet panel (`#filter-root`): toggle buttons per value with live counts under the other active facets; zero-count
  values stay visible and disabled.
- Result list (`#run-results`): one row per run_id (no duplicates, virtualised, sortable by design order, ties,
  density); row actions "Replay" and "Add to stack"; header action "Stack these N" when N is 8 or fewer, otherwise
  "Stack one condition" guidance. Replaces `#candidate`, `#add`, `#apply`.
- URL state, short and readable; legacy `?view=<json>` links still open:
  `?v=stack&m=gpt-4.1&p=local,global&s=us-english&b=44000&g=nonempty&q=&l=0.8.16.24&a=8&n=12&c=religion&x=same`
  where `l` and `a` are slot numbers (0-895), `n` the persona, `c` colour attribute, `x` lens, `f` replay frame.
  Unknown or failed slots in a link produce a clear message and the default view, never a silent substitute.

### B.5 Layered comparison and persona / attribute analysis

- Stack of up to 8: per-layer card (label once, ties, density, clustering), pairwise edge-overlap matrix
  (Jaccard, descriptive), shared-tie ribbons, tie filter all / shared by all / not shared by all.
- **Consensus layer** (optional extra plane, clearly labelled "Aggregate of N filtered graphs, not a run"): tie
  opacity and width by the share of the N in-scope graphs that contain the tie. Works for any N up to 890 from
  `edges.bin`.
- **Attribute lens**: colour by gender, age band, religion or political orientation; click legend entries to isolate
  groups; lens "same-group ties" or "mixed-group ties" recolours ties; per-layer readout of the saved Coleman
  index or age assortativity from `metrics.json` (never recomputed in the browser).
- Persona: select by click, list or keys; neighbours per layer; identity thread; follow in replay (existing).
- **Study ledger** (`#study-ledger`): the single place that states 896 / 890 / 12 / 6, the 4 x 4 x 7 x 8 design, the
  50-person roster, and "6 supplemental graphs excluded". Other places link to it instead of restating it.

### B.6 Quality bar

- Desktop 1440x900: stage at least 60% of the viewport area in default layout; nothing overlaps; no wrapped buttons.
- Phone 390x844: no horizontal scroll; stage at least 45% of the first screen; filter bar and study ledger reachable
  without scrolling past the stage (sticky bar plus bottom sheet); touch targets at least 44x44 px.
- Typography from tokens only (`--text-*`), spacing from `--space-*`; one label formatter for runs everywhere
  ("GPT-4.1 · Local · United States / English · seed 44000").
- Keyboard: every action reachable; visible focus; `aria-pressed` on toggles; one live region; reduced-motion honoured.
- Performance: Atlas first frame under 2.5 s on a cold local load; orbit at 50 fps or better on integrated graphics at
  1440x900 (SwiftShader in CI is exempt from fps, not from draw-call counts); filter change under 100 ms.
- Zero console errors or warnings from page code. WebGL-unavailable fallback keeps counts, lists and tables usable.

## C. Work split

### C.1 DOM contract (`layers.html`, owned by workstream 4)

Mount points already scaffolded. The owning module builds all inner DOM; workstream 4 places and styles the box only.

| id | Filled by | Purpose |
|---|---|---|
| `#study-ledger` | WS5 `mountInsight` | population and design facts |
| `#active-filters` | WS3 `mountFilters` | always-visible chips, counts, reset |
| `#filter-root` | WS3 | facet panel |
| `#run-results` | WS3 | deduplicated result list |
| `#insight-root` | WS5 | lens controls, overlap matrix, consensus controls |
| `#legend` | WS5 | attribute legend (clickable) |
| `#stage-overlay` | WS2 | atlas axis labels, hover card, HUD (inside `.stage`) |
| `#viewport`, `#plane-labels`, `#node-tooltip` | WS2 | canvas, labels, tooltip |
| `#atlas-mode`, `#compare-mode`, `#formation-mode` | WS2 wires | view switch: atlas, stack, replay (`#atlas-mode` is `hidden` until WS2 enables it) |

Frozen ids (read by `layers.js`, `layers-ui.js`, `plot-gallery.js` or browser checks; may be moved, restyled or put in
`<div id="legacy-controls" hidden>`, never deleted during the parallel phase): every id currently in `layers.html`.
The integrator deletes `#legacy-controls` last. Expected to end up there: `#filter-form` with `#model #method
#culture #language #seed #apply #reset-filters #filter-preview`, `#candidate #add #candidate-detail`,
`#open-filters`, `#filter-chips`, and the primary part of `#saved-plots`.

### C.2 Shared store (`viewer/layers-app.js`, frozen, integrator only)

```js
import {createStore, INITIAL_STATE, EMPTY_FILTERS, LAYER_COLORS, MAX_STACK_LAYERS, VIEWS, GRAPH_STATES, LENSES} from './layers-app.js';
// layers.js exports the single instance:  export const app = createStore();
app.ready                      // Promise<Dataset>, resolves after layers.js validated the data
app.data                       // Dataset (C.3) or null before ready
app.getState()                 // frozen state object
app.update(patch, source)      // -> Set of changed keys; unknown key throws; no-op emits nothing
app.subscribe((state, changed /* Set<string> */, source) => {})   // -> unsubscribe
app.announce(text)             // the one live region
```

State keys and writers:

| key | type | written by |
|---|---|---|
| `filters` | `{model:string[],method:string[],setting:string[],seed:number[],graph,search}` | WS3 (and WS2 coverage cell / tower click) |
| `view` | `'atlas' \| 'stack' \| 'replay'` | WS2 |
| `layers` | run_id[], 1..8, unique | WS2, WS3 ("Stack these", "Add to stack") |
| `active` | run_id in `layers` | WS2, WS3 ("Replay") |
| `frame` | integer | WS2 |
| `persona` | id or `''` | WS2, WS5 |
| `colorBy` | `'layer'` or attribute key | WS5 |
| `lens` | `'none' \| 'same' \| 'mixed'` | WS5 |
| `groups` | string[] | WS5 |
| `focusSlot` | 0..895 or null | WS2, WS3 (row hover) |

Pass your module name as `source` and ignore your own emissions. The scaffold in `layers.js` already mirrors the
legacy variables into the store (`saveView`) and applies external `layers`, `active`, `view`, `persona`, `colorBy`
updates, so WS3 and WS5 can develop against real behaviour from the first hour.

### C.3 Data contract (workstream 1)

`viewer/layers-data-all.js`:

```js
export const COUNTS  // {attempts:896, valid:890, empty:12, failures:6, supplements_excluded:6, personas:50, cells:112, repetitions:8}
export function settingKey(culture, language)            // 'us-hindi'
export function slotOf(design, {model,method,culture,language,seed})   // 0..895, throws outside the design
export async function loadPrimaryDataset({fetchImpl, base})            // -> Dataset; throws on any mismatch
export async function loadMetrics(dataset, {fetchImpl, base})          // fills run.metrics for all runs, once   (to add)
export async function loadSupplements({fetchImpl, base})               // -> labelled supplement graphs          (to add)
export function decodeEdges(bytes, offsetEdges, countEdges)            // -> [[a,b],...] ids as strings          (to add)
```

`Dataset` is a superset of today's object (must still pass `validateData`): `schema_version, dataset, roster_id,
personas[{id, attributes, position}], runs[], failures[], models, methods, settings, planned_networks,
planned_repetitions, analysis_version, evidence_notice, calibration_note, limitations, verification` plus
`design {models, methods, settings, seeds}`, `counts`, `slots[896] {slot, status:'valid'|'empty'|'failed', run_id,
failure?}`, `layout` (atlas x,y per persona). Each run keeps all current fields (`run_id, model, method, culture,
language, seed, study, status, roster_id, prompt_variant, source, source_sha256, record_sha256, protocol_sha256,
homophily, analysis_version, edges, events:[], events_url, events_sha256, replay_status`) plus `slot`, `setting`.
`run.metrics` must exist at load with at least `density` and `avg_clustering_coef` (carried in `index.json`); the
remaining 14 arrive with `loadMetrics`. Edge ids are strings; each edge is `[a,b]` with numeric `a < b`, sorted.

`public/data/graphs/index.json` (example, abbreviated):

```json
{
  "schema_version": 2,
  "dataset": "primary",
  "source": {"attempts_csv": "outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv", "attempts_csv_sha256": "<64 hex>",
             "roster": "text-files/revision224_adults.json", "roster_file_sha256": "<64 hex>"},
  "roster_id": "26744d2c8fb08406aef3ccba968799473bf7b47456d38baadd7595a5882e90ac",
  "counts": {"attempts": 896, "valid": 890, "empty": 12, "failures": 6, "supplements_excluded": 6, "personas": 50},
  "design": {"models": ["gpt-4.1","gpt-5.6-luna","gpt-6-luna","gpt-6-sol"],
             "methods": ["global","local","sequential","iterative"],
             "settings": [["us","english"],["india","english"],["japan","english"],["brazil","english"],["us","hindi"],["us","japanese"],["us","portuguese"]],
             "seeds": [44000,44001,44002,44003,44004,44005,44006,44007]},
  "personas": [{"id": "0", "gender": "Nonbinary", "age": 20, "religion": "Muslim", "political orientation": "Centrist"}],
  "layout": {"basis": "force layout on tie frequency across all 890 valid graphs; algorithmic, not geographic", "xy": [75.0, 0.0, 61.2, -40.1]},
  "edges": {"file": "edges.bin", "sha256": "<64 hex>", "bytes": 313460, "encoding": "uint8 pairs a<b, runs in slot order"},
  "metrics": {"file": "metrics.json", "sha256": "<64 hex>"},
  "columns": ["slot","status","edges","density","avg_clustering_coef","adj_sha256","record_sha256","events_sha256","failure"],
  "attempts": [
    [0, "valid", 46, 0.0375510204, 0.0533333333, "<64 hex>", "<64 hex>", "<64 hex>", null],
    [300, "failed", 0, null, null, null, "<64 hex or null>", null, "invalid_global_response"]
  ],
  "shared": {"prompt_variant": "frozen060d_reference", "protocol_sha256": "060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f",
             "run_id_template": "final_060dcd3a86d5_main_{method}_{model}_{culture}_{language}_s{seed}",
             "source_template": "outputs/final_study/060dcd3a86d5/main/runs/{run_id}.adj"},
  "notes": {"evidence_notice": "...", "calibration_note": "...", "limitations": "...", "verification": "..."}
}
```

`attempts` has exactly 896 rows in slot order, `slot = ((model*4 + method)*7 + setting)*8 + seed_index`; the edge
offset of a run is the sum of `edges` of all earlier rows. `metrics.json`: `{"schema_version":2,"columns":[16 names],
"rows":[[...16 numbers or null] x 890 in slot order of valid runs]}`, full CSV precision. `supplements.json`:
`{"schema_version":1,"note":"Excluded from the primary study","graphs":[{run_id,cohort,model,method,culture,language,
seed,edges:[[a,b]],source,source_sha256}] x 6}`.

Integrity checks the generator and tests must enforce: 896 rows, 890/12/6 split, unique run_ids, every `.adj` hashes
to `adj_sha256`, parsed edges equal `primary-viewer.json` edges for all 890, no id outside 0-49, no self or duplicate
tie, metrics equal the CSV, no supplement run_id present, roster equals `personas` of `primary-viewer.json`
(note: `roster_id` is a canonical roster hash from the CSV column `roster_sha256`, not the file hash of the JSON),
output is byte-deterministic (running the generator twice changes nothing). The browser loader verifies
`edges.bin` against `edges.sha256` with `crypto.subtle` before use and throws the existing message
`Data request failed: <status>. No substitute data is loaded.` on HTTP errors.

Also WS1: raise `MAX_LAYERS` in `graph-data.js` from 6 to 8 (constant already scaffolded) and update the affected
unit tests; update `build.mjs` (verify + copy `graphs/`, stop copying the three monolith files, keep hash checks of
the fixtures) and `verify-public.mjs` accordingly.

### C.4 3D engine contract (workstream 2)

WS2 owns `layers.js` and may reshape it freely as long as the store contract, frozen ids and these imports hold.
Suggested module split:

```js
// layers-3d-scene.js
export function createStage3D(host, {overlay, labels, reducedMotion}) // -> {scene, camera, controls, requestRender(), flyTo(target, opts), pose(name), resize(), stats(), dispose()}
// layers-3d-atlas.js
export function atlasLayout(design)                     // pure: slot -> {x,y,z}, tower boxes, axis label anchors
export function createAtlas(stage, dataset)             // -> {setMatch(Uint8Array(896)), setEncoding(enc), setFocus(slot|null), pick(ray) -> {slot}|null, dispose()}
// layers-3d-stack.js
export function stackGeometry(runs, layout, options)    // pure: typed arrays for nodes, ties, shared ribbons (unit-testable)
export function createStack(stage, dataset)             // -> {setRuns(runs, activeId), setFrame(edges, event), setPersona(id), setEncoding(node, edge), setTies('all'|'shared'|'different'), setConsensus(list|null), pick(ray) -> {run_id, person}|null, dispose()}
// layers-3d-labels.js
export function createLabels(host)                      // -> {set(items[{id,text,position,priority,className}]), project(camera, size), dispose()}
```

Consumes from other streams (call, never copy): `nodeEncoding`, `edgeEncoding`, `consensus` from
`layers-insight.js`; `applyFilters` from `layers-filters.js` to build the 896-slot match mask; `LAYER_COLORS`,
`MAX_STACK_LAYERS` from `layers-app.js`; `MAX_LAYERS` from `graph-data.js`. Expose for tests and checks:
`canvas.dataset.view`, `canvas.dataset.drawCalls`, `canvas.dataset.slots` (896), `canvas.dataset.layers`.
Remove from `layers.js` once WS3 lands: `readFilters`, `candidates`, `updateAdd`, the `#filter-form`, `#add`,
`#candidate`, `#reset-filters`, `#open-filters` handlers and the per-key filter restore in `start()`; guard every
legacy `$()` lookup you still need with a null check so the integrator can delete `#legacy-controls`.
Coverage table stays as the accessible, non-WebGL equivalent of the Atlas; a cell click sets `filters` and
`view:'stack'` with that condition's valid runs.

### C.5 Filters contract (workstream 3)

```js
export function applyFilters(runs, filters)        // pure; valid graphs only
export function facetCounts(dataset, filters)      // {model:{v:n}, method:{}, setting:{}, seed:{}, graph:{nonempty:n, empty:n}}; plus failuresInScope(dataset, filters) -> number
export function activeChips(dataset, filters)      // [{facet, value, label}]
export function encodeState(state, dataset)        // -> URLSearchParams (B.4 grammar)
export function decodeState(params, dataset)       // -> partial state; accepts legacy ?view=<json>; throws on archived datasets with today's message
export function mountFilters(app, {chips, root, results})   // -> {destroy()}
```

`search` matches whole words against model, method, country name, language, seed and run_id. WS3 owns URL writing
(`history.replaceState`) after integration; until then keep writing nothing and only read. WS3 also removes the chip
painter and widens the `1`-`6` shortcut to `1`-`8` in `layers-ui.js`.

### C.6 Shell contract (workstream 4)

`mountShell(app) -> {destroy()}` owns panel open/close state, the phone bottom sheet, sticky offsets
(`--sticky-top` set from the measured header), focus return, and `body[data-view]` mirroring `state.view`.
Layout target: header (brand, study ledger link, share, shortcuts), sticky active-filter bar, then a full-bleed stage
with a left dock (filters, results) and right dock (inspector, insight) that collapse to tabs below 1150 px and to a
bottom sheet below 700 px; transport appears only in replay; analysis cards below the fold. One population statement
(the ledger). One "Filters" entry point. Remove dead static options (A.5). Keep `<title>`, the brand `aria-label`,
the visible study name, the `NETWORKS.md` link and `id="saved-plots"` (`current-study.test.js`).

### C.7 Insight contract (workstream 5)

```js
export function nodeEncoding(dataset, state)   // {colorOf(personId, layerIndex) -> '#rrggbb', dimmed(personId) -> boolean, legend:[{key,label,color,count}]}
export function edgeEncoding(dataset, state)   // {classOf([a,b]) -> 'same'|'mixed'|null}
export function consensus(runs)                // [{edge:[a,b], count, total}] sorted by count desc, then edge
export function overlapMatrix(runs)            // {ids, matrix[i][j] = {shared, union, jaccard|null}}
export function mountInsight(app, {root, ledger, legend})   // -> {destroy()}
```

Colours: categorical palettes with at least 3:1 contrast against `--panel`, distinguishable under deuteranopia; age
uses 6 ordered bands with a perceptually even ramp. Also WS5: one run-label formatter in `formation-charts.js` used
everywhere (replace raw `us / english`), singular/plural fixes, and reduce `plot-gallery.js` + `network-stage.js` to
the supplement-only panel reading `loadSupplements()`.

### C.8 CSS contract

Tokens live in `layers.css :root` (WS4 owns values): existing `--bg --panel --raised --hover --line --line-strong
--text --muted --faint --accent --accent-strong --accent-ink --accent-wash --warn --warn-wash --add --remove --actor
--radius --gutter --sans --mono`, plus scaffolded `--layer-1..8`, `--space-1..6`, `--text-xs..xl`, `--touch`,
`--chip-h`, `--z-stage-ui --z-sticky --z-sheet --z-dialog`, `--failed --empty --same --mixed`.
Other stylesheets use tokens only (no raw hex except canvas-matched data colours) and scope every selector:
`layers-3d.css` under `#stage-overlay`, `#plane-labels`, `#node-tooltip` or class prefix `.g3-`;
`layers-filters.css` under `#active-filters`, `#filter-root`, `#run-results` or `.flt-`;
`layers-insight.css` under `#study-ledger`, `#insight-root`, `#legend` or `.ins-`.
No stylesheet other than `layers.css` sets position, size or margin of a mount root itself.

### C.9 Five workstreams and file ownership (disjoint)

| WS | Scope | New files | Existing files (sole editor) |
|---|---|---|---|
| **1 Data** | generator, compact loader, integrity, build | `build-primary-graphs.mjs`, `layers-data-all.js`, `layers-data-all.test.js`, `public/data/graphs/*` | `build.mjs`, `verify-public.mjs`, `package.json`, `graph-data.js`, `graph-data.test.js`, `primary-data.test.js`, `network-catalog.js`, `network-catalog.test.js`, `plot-catalog-build.mjs`, `verified-events.js`, `verified-events.test.js` |
| **2 3D engine** | scene, atlas, stack, labels, camera, performance, replay wiring | `layers-3d-scene.js`, `layers-3d-atlas.js`, `layers-3d-stack.js`, `layers-3d-labels.js`, `layers-3d.css`, `layers-3d.test.js`, `layers-3d-browser-check.mjs` | `layers.js`, `navigation.js`, `navigation.test.js`, `navigation-check.mjs`, `primary-browser-check.mjs` |
| **3 Filters** | filter model, chips, facets, results, URL, shortcuts | `layers-filters.js`, `layers-filters.css`, `layers-filters.test.js`, `layers-filters-browser-check.mjs` | `layers-ui.js`, `browser-check.mjs`, `polish-browser-check.mjs` |
| **4 Shell** | layout, tokens, responsive, accessibility | `layers-shell.js`, `layers-shell-browser-check.mjs` | `layers.html`, `layers.css`, `current-study.test.js` |
| **5 Insight** | attribute lens, comparison, consensus, ledger, provenance, supplements panel | `layers-insight.js`, `layers-insight.css`, `layers-insight.test.js` | `formation-charts.js`, `research-plots.js`, `plot-gallery.js`, `plot-gallery.test.js`, `network-stage.js`, `plot-gallery-browser-check.mjs` |
| Integrator | merge order, legacy removal, QA | `docs/qa/viewer-3d-2026-11-25/*` | `layers-app.js`, `layers-app.test.js` (frozen for everyone else) |

`package.json` and `build.mjs` already list every new module, stylesheet and unit-test file above, so no other stream
needs to touch them. New browser-check files are run by path, not through `package.json`.

Briefs

- **WS1**: implement `build-primary-graphs.mjs` (read CSV, `.adj`, roster; write the four files deterministically;
  compute the atlas layout once with `comparisonLayout` weighted by tie frequency). Swap `loadPrimaryDataset` to the
  compact files with SHA verification. Add `loadMetrics`, `loadSupplements`, `decodeEdges`. Set `MAX_LAYERS = 8`.
  Tests: every check in C.3, round trip against `primary-viewer.json` for all 890, generator determinism.
- **WS2**: build the stage, Atlas and Stack per B.3 and C.4 with in-place updates and the draw-call budget; eased
  camera, damping, picking for towers, planes and nodes; label layer without layout thrash; make the store the single
  source of truth in `layers.js`; keep replay semantics and verification exactly as they are; WebGL fallback.
- **WS3**: pure filter functions with facet counts; sticky active-filter bar; facet panel; virtualised result list
  with "Replay", "Add to stack", "Stack these N"; URL grammar with legacy-link support; update shortcuts and the two
  owned browser checks to the new controls.
- **WS4**: rebuild the page shell around a dominant stage per C.6 and B.6; token-driven type and spacing; phone
  bottom sheet; 44 px targets; focus order and landmarks; keep all frozen ids (move legacy ones into
  `#legacy-controls[hidden]`); screenshot check at 1440x900 and 390x844 asserting no horizontal overflow and the
  stage-area thresholds.
- **WS5**: encodings, lens, legend isolate, overlap matrix, consensus list; study ledger with the exact counts read
  from `app.data.counts` and asserted against `COUNTS`; provenance drawer for the active run (label stated once);
  supplement-only panel; single run-label formatter; honest captions on every aggregate.

Integration order: WS1, WS4, WS2, WS3, WS5; then apply `requests-*.md`, delete `#legacy-controls` and dead code,
regenerate screenshots, run section D.

## D. Acceptance checklist (integrator)

Commands (from `viewer/`):

```bash
npm ci
node build-primary-graphs.mjs && git diff --exit-code public/data/graphs     # generator is deterministic
npm run build
npm test                         # all unit tests, including the 5 new files
npm run verify:public
(python3 -m http.server 8781 --bind 127.0.0.1 --directory dist &)
export NODE_PATH=$(npm root -g) PLAYWRIGHT_MODULE=$(npm root -g)/playwright BROWSER_CHANNEL=chromium VIEWER_URL=http://127.0.0.1:8781
for t in browser-check primary-browser-check polish-browser-check plot-gallery-browser-check navigation-check \
         layers-3d-browser-check layers-filters-browser-check layers-shell-browser-check; do node $t.mjs || echo "FAIL $t"; done
SHOT_DIR=after node ../docs/qa/viewer-3d-2026-11-25/shot.cjs               # then read after/*.png
cd .. && python3 scripts/verify_viewer_archive.py && git status --short     # only viewer/ and docs/qa/viewer-3d-2026-11-25/
```

Data and honesty
- [ ] `index.json` has 896 attempts: 890 valid, 12 of them empty, 6 failed; no supplement run_id anywhere in `graphs/`
      except `supplements.json`.
- [ ] All 890 edge lists decoded from `edges.bin` equal the `.adj` files and `primary-viewer.json`.
- [ ] `public/data/primary-viewer.json`, `saved-networks.json`, `saved-plots.json`, `primary-events/*` are byte-identical
      to `main`; nothing outside `viewer/` and the QA folder changed.
- [ ] The page states 896 / 890 / 12 / 6 exactly once as the canonical ledger, visible without scrolling on desktop and
      within one tap on phone; "6 supplemental graphs excluded" is stated there.
- [ ] Wording review: no "simulation", "live", "time" or "movement" claims for replay; every aggregate says descriptive.
- [ ] Network panel shows no request to any host other than the page origin; no `primary-viewer.json`,
      `saved-networks.json` or `saved-plots.json` request; no file fetched twice.

All 890 in 3D
- [ ] Atlas shows 112 towers x 8 slots; `canvas.dataset.slots === "896"`; 6 failed slots are visibly hollow and not
      pickable as graphs; 12 empty graphs render with nodes and no ties.
- [ ] A scripted walk opens every one of the 890 valid slots in Stack and in Replay by URL without error.
- [ ] Any condition's 8 seeds stack together; a 9th layer is refused with a clear message.

Filters
- [ ] Active-filter bar is visible at every scroll position on both sizes; shows chips only for narrowed facets; each
      chip removes itself; Reset clears all; count line always matches the result list length.
- [ ] Facet counts are correct for 20 random filter combinations (compare with `applyFilters`).
- [ ] URL round trip: reload reproduces filters, view, stack, active run, persona, colour, lens, frame. A legacy
      `?view=` link from the deployed site opens the same runs.
- [ ] Result list never shows a run twice; no control or sentence appears twice on a screen (A.2 list re-checked).

3D quality
- [ ] Lighting, fog, slab planes, degree-scaled lit nodes, shared-tie ribbons and eased camera present; depth is
      readable in a still screenshot.
- [ ] Draw calls: Atlas at most 12, Stack at most 6 per layer; no geometry rebuild on opacity, persona or frame change.
- [ ] No layout reads in the frame loop (performance trace shows no forced reflow during orbit).
- [ ] Reduced motion: no camera animation, no auto-orbit. WebGL off: lists, counts and tables still work.

Shell
- [ ] 1440x900: stage at least 60% of the viewport; 390x844: no horizontal scroll, stage at least 45% of the first
      screen, all interactive targets at least 44x44 px (or 24x24 with 44 px spacing for inline links).
- [ ] Keyboard-only pass through header, filters, results, stage, inspector; focus always visible; dialogs trap and
      return focus; one live region announces filter and stack changes.
- [ ] Zero console errors and warnings from page code on both sizes.

## E. Scaffolding already in the tree (2026-11-25)

Verified after scaffolding: `npm run build`, `npm test` (51 pass), `npm run verify:public`, all five browser checks,
and a store round trip in the page (external `app.update({layers, active, persona, colorBy})` re-renders six layers).

- New: `layers-app.js` (+test, complete), `layers-data-all.js` (+test; working shim over `primary-viewer.json` that
  already adds `design`, `counts`, `slots[896]`, `run.slot`, `run.setting`), stubs `build-primary-graphs.mjs`,
  `layers-3d-{scene,atlas,stack,labels}.js`, `layers-filters.js`, `layers-insight.js`, `layers-shell.js`, three scoped
  stylesheets, three placeholder unit tests.
- `build.mjs`: new modules and stylesheets in the copy list. `package.json`: new unit tests in `npm test`.
- `graph-data.js`: `export const MAX_LAYERS = 6` used by `layerComparison`.
- `layers.html`: stylesheet links, mount points `#study-ledger #active-filters #filter-root #run-results
  #insight-root #stage-overlay`, hidden `#atlas-mode`.
- `layers.css`: new tokens in `:root`.
- `layers.js`: `export const app`, `loadPrimaryDataset()` instead of the inline fetch, store mirror in `saveView`,
  inbound subscriber and the three `mount*` calls at the end of `start()`.

## F. Risks

1. **`layers.js` is one dense 490-line file owned by WS2 but read by everyone's features.** Mitigation: all other
   streams go through the store and their own mount roots; WS2 lands the store-as-source-of-truth refactor first,
   visuals second.
2. **Frozen-id discipline.** If WS4 deletes instead of moving a legacy element, `layers.js` and four browser checks
   crash at load. Mitigation: `#legacy-controls[hidden]`, integrator deletes last.
3. **Existing browser checks encode the old UX** (six-layer limit, Apply button, gallery of 896). Their owners must
   rewrite them to the new behaviour, not delete assertions about counts, failures, checksums or supplement isolation.
4. **Raising the limit to 8** touches unit tests and strings ("one and six", "6-layer limit", `/6` counter, keys 1-6)
   across WS1, WS2, WS3; all read `MAX_LAYERS` / `MAX_STACK_LAYERS` rather than literals.
5. **Atlas performance on integrated GPUs and phones**: 44,500 nodes and 156,730 ties are fine as two buffers but not
   as meshes; headless SwiftShader is slow and will under-report fps. Budget by draw calls and measure on real hardware.
6. **Honesty drift**: a more cinematic view invites "simulation" language, and the consensus plane can be mistaken for
   a run. Captions are part of acceptance. The towers' Y axis is seed order, not time.
7. **Supplements leaking in** through the old gallery code path or `saved-networks.json` (896 graphs). The compact
   primary files must contain exactly 890 graphs; tests assert it.
8. **Pinned hashes**: `SAVED_NETWORKS_SHA256` and the catalog hash in `build.mjs` guard fixtures; do not regenerate
   those fixtures. Root-level Python exporters write into `viewer/public/data/`; do not run them.
9. **`metrics.json` laziness**: `layers.js` reads `run.metrics.density` synchronously; `index.json` therefore carries
   density and clustering, and anything else must await `loadMetrics`.
10. **Gzip assumption**: sizes above are raw; the static host may not compress `.bin`. 313 KB raw is acceptable either way.
