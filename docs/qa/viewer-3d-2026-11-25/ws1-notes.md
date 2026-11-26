# Workstream 1 (Data): notes for the integrator

Date: 2026-11-25. Scope: SPEC sections B.2 and C.3, brief "WS1" in C.9.

## What was built

- `viewer/build-primary-graphs.mjs`: the generator. It reads the frozen attempts CSV, the 890 `.adj` files, the roster,
  the supplemental CSV and `.adj` files, and the recorded event logs. It cross-checks against the pinned fixtures
  `primary-viewer.json` and `saved-networks.json`, then writes `viewer/public/data/graphs/`. It reads only those
  sources and writes only to that folder. Every check throws; nothing is repaired. Running it again produces the
  same bytes (tested). `npm run build:graphs` runs it. It also exports `generateGraphFiles()` and its parsers for
  `build.mjs`, `verify-public.mjs` and the tests.
  - Checks: 896 rows; 890 valid, 12 of them empty, 6 failed; unique run ids that follow the run id template; no
    supplement id in the primary set; every `.adj` hashes to `adj_sha256`; parsed ties equal the CSV counts and
    the `primary-viewer.json` edges for all 890; ids are 0-49; no self ties or duplicate ties; every event log hash
    matches; the canonical roster hash (Python `json.dumps(sort_keys=True)`) equals the CSV `roster_sha256` and
    `roster_id`; the roster equals the fixture's personas; the six supplements equal `saved-networks.json`.
  - Atlas layout: computed once with `comparisonLayout` in planar mode, with springs weighted by
    `count / max count` across all 890 graphs. The basis text says the layout is algorithmic, not geographic.
- `viewer/public/data/graphs/` (committed output):

  | File | Bytes | Contents |
  |---|---|---|
  | `index.json` | 244,414 | the C.3 schema, plus `counts.cells`/`counts.repetitions`, `supplements {file, sha256}`, the supplements CSV hash, and `shared` run fields (`study`, `analysis_version`, `replay_status`). Rows are one per line, so diffs stay reviewable. |
  | `edges.bin` | 313,460 | one uint8 pair per tie (156,730 ties) |
  | `metrics.json` | 186,635 | 16 columns x 890 rows, with the exact CSV values |
  | `supplements.json` | 11,346 | 6 labelled graphs |

  Start-up loads `index.json` and `edges.bin`, 557,874 bytes. Before this change it loaded 3.28 MB.
- `viewer/layers-data-all.js`, which has the C.3 contract:
  - `loadPrimaryDataset` loads the compact files. It checks `edges.bin` with `crypto.subtle` before using it,
    rebuilds every run from its slot (the run id comes from the template), and passes `validateData`. On an HTTP
    error it throws the existing message `Data request failed: <status>. No substitute data is loaded.` Any other
    mismatch throws `Primary data failed its integrity check: ... No substitute data is loaded.`
  - `loadMetrics(dataset)` fills all 16 metrics and `run.homophily`, checks the SHA-256 against `index.json`, and
    fetches only once per dataset (concurrent calls share one promise; a failed request can be retried).
  - `loadSupplements()` checks that the file has the exclusion label, exactly 6 graphs, no primary run id, and only
    canonical ties.
  - `decodeEdges(bytes, offsetEdges, countEdges)`: offsets and counts are in ties, not bytes. It rejects ties that
    are not pairs a<b in ascending order.
- `graph-data.js`:
  - `MAX_LAYERS = 8`. The message now reads `Choose between 1 and 8 layers.` (generated from the constant).
  - `comparisonLayout` takes an optional `{weights, planar}`. Callers that pass no options get exactly the old
    result (tested).
- `build.mjs` and `verify-public.mjs`:
  - `build.mjs` first checks that the committed `graphs/` files are byte-identical to a fresh generation, then
    copies them. It gzips the event logs using the hashes from the compact index. It keeps the fixture hash checks
    (saved-plots pin and PNG hashes, `SAVED_NETWORKS_SHA256`, catalogue validation). It no longer deploys
    `primary-viewer.json`, `saved-networks.json` or `saved-plots.json`.
  - `verify-public.mjs` loads `dist/` through the real loader. It asserts that the monolith files are absent, that
    start-up requests only the index and `edges.bin`, and that all 890 replays rebuild the `edges.bin` graph. It
    also checks supplement isolation, credential and legacy paths.

## Files touched

New: `build-primary-graphs.mjs` (the scaffold was replaced), `layers-data-all.js` (the shim was replaced),
`layers-data-all.test.js`, `public/data/graphs/*`.
Edited: `graph-data.js`, `graph-data.test.js`, `primary-data.test.js`, `build.mjs`, `verify-public.mjs`,
`package.json` (`build:graphs` script).
Unchanged, but checked: `network-catalog.js` and its test, `plot-catalog-build.mjs`, `verified-events.js` and its
test. No fixture or file outside `viewer/` changed; `git status public/` shows only the new `graphs/`.

## Verification

- `node --test layers-data-all.test.js primary-data.test.js graph-data.test.js verified-events.test.js
  network-catalog.test.js`: 49 of 50 pass. The one failure is explained below. `layers-data-all.test.js` (12 tests)
  covers every C.3 check: a round trip for all 890 graphs against the `.adj` files and the fixture, metrics equal to
  the CSV, the roster hash, generator determinism against the committed bytes, a stack of 8 seeds that refuses a 9th,
  and rejection of tampered or HTTP-failed inputs.
- `npm run build` and `npm run verify:public` pass.
- Browser check on a private copy served on port 8791 at 1440x900 and 390x844. The page loads `index.json` and
  `edges.bin`, then one `.json.gz` replay and `metrics.json` (loaded on demand by WS5). It makes no other data
  requests and has no page errors. The only console entries are a `favicon.ico` 404 from the Python server and
  SwiftShader driver warnings. A cold `loadPrimaryDataset` in Chromium takes about 130 ms.

## The integrator must know

1. **`graph-data.test.js` test 3 ("layer labels identify ...") fails.** It asserts the output of WS5's
   `formation-charts.js` `layerLabel`, which WS5 changed (`2. Portuguese`, `2. seed 11001`). This is intended by
   A.5. Once WS5's formatter is final, update the three expected strings in `graph-data.test.js` lines 22-27. WS5
   cannot edit that file.
2. **The edge order changed.**
   - Compact runs list ties as numeric `a < b`, sorted numerically, as C.3 requires.
   - `primary-viewer.json` used string order (`["10","30"]` before `["2","5"]`).
   - The two contain the same tie sets. Anything that compares edge arrays by position must use `edgeKey`.
3. **Metrics and homophily load lazily.**
   - At load, `run.metrics` holds only `density` and `avg_clustering_coef`, and `run.homophily` is `{}`.
   - Code that reads any other metric must `await loadMetrics(app.data)` first: `layers.js` metric bars, and RQ2
     `homophilyMatrix`. WS5 already calls it.
   - Values are the exact CSV values. They differ from the fixture's 15-digit rounding in the last digits.
4. **Dataset additions beyond C.3:**
   - `layout = {basis, xy, points: Map(id -> [x, y])}`.
   - `edgeData = {bytes: Uint8Array, offsets: Uint32Array(897)}`, indexed by slot. Pass these to `decodeEdges` for
     consensus over any N.
   - `files` (base and hashes for the lazy files) and `source` (provenance hashes).
   - Each failure also carries `slot`, `setting` and `record_sha256`.
   - `schema_version` stays `1`. This is the in-memory shape that `validateData` accepts; `index.json` itself is
     schema 2.
   - `personas[].position` is the legacy roster circle that `validateData` requires (equal to the fixture within
     1e-9).
5. `events_url` is `${base}primary-events/<run_id>.json.gz`, which matches what `dist` previously wrote.
6. Integrity checks use Web Crypto, so the page must be served over https or from localhost. On another origin it
   throws a clear message. Replay verification already had the same requirement.
7. Old shared code that fetched `saved-plots.json` or `saved-networks.json` now gets a 404 from `dist`. WS5's
   supplement panel uses `loadSupplements()`, so check that no other caller remains.
8. `verify:public` takes about 20 s because it replays all 890 histories. It can fail if another engineer rebuilds
   `dist/` during the run; run it again on a quiet tree.

No contract change to `layers-app.js` is needed.
