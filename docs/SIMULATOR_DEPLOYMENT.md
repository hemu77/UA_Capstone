# Current Demo And Deployment

[Open Network Observatory](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html)
or [explore all 896 networks in 3D](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html#saved-plots).

## Scope

The public Three.js workspace uses only the final primary study:
896 original attempts, 890 valid graphs (12 empty), six failures retained in the
repository audit. Replay loads saved, hash-verified graph updates, not live agents.

The separate final-graph browser contains 890 primary networks and six explicitly
labeled supplements, grouped by Global, Local, Sequential and Iterative (224 per
method in this descriptive browser, not the primary cohort). Rotate, pan, zoom
and select a persona to inspect its actual saved neighbors. One reusable canvas
renders one selected graph; graphs are never merged into a fictitious network.
Supplements never enter primary replay, comparisons or statistics. Twelve primary
and one supplemental graph are valid empty graphs and retain all 50 nodes.
Original PNG and adjacency links remain in GitHub; PNG tiles are no longer deployed.

Calibration, engineering pilots and their old interface live in
[`viewer/archive/`](../viewer/archive/README.md). They are excluded from the
production build. Old entry paths (`index.html`, `study.html`, `archive.html`)
open the current lab; old dataset queries stop with an explicit reset link.
No historical selection is silently relabeled as primary.

## Build And Verify

From `viewer/`:

```powershell
npm ci
npm test
npm run build
npm run verify:public
python -m http.server 8781 --bind 127.0.0.1 --directory dist
```

The build pins the descriptive catalog, final-edge projection and all 896 original
PNG hashes (repository files only). Regenerate the final-edge export offline with
`python scripts/export_saved_networks.py`; it verifies every adjacency hash and
primary graph parity against the unchanged primary export.
All 890 recorded histories are compressed for transport, then decompressed,
size-limited and checked against their original SHA256 before replay.
The build verifier checks exact history bytes, final graph parity, primary /
supplement separation, absence of archive payloads and credential patterns.

Browser scripts use installed Playwright with Edge; set `PLAYWRIGHT_MODULE`
only when Playwright is outside this checkout, and `VIEWER_URL` to the server.

```powershell
npm run test:browser
npm run test:navigation
node primary-browser-check.mjs
node polish-browser-check.mjs
npm run test:gallery
```

The tests cover current-only routing, legacy link rejection, empty graphs,
selection/download, actual pointer navigation, delayed/invalid replay,
keyboard/dialog behavior, mobile layout, 3D browser pagination/neighbor inspection, cohort isolation, checksum refusal
and WebGL fallback. The final-graph browser does not synthesize replay events.
Headless mobile viewports are not physical-device or Safari validation.

## Publication Boundary

Publish only verified `viewer/dist/` to the existing Site. Do not publish
`viewer/archive/`, raw responses, ledgers, credentials or operational scripts.
Never call the collection or supplemental runners for a UI build.

The prior version 6 release and its exact deployment record are preserved in
[the dated handoff snapshot](history/20261114/SIMULATOR_DEPLOYMENT-before-cleanup.md).
## Verified Publication: November 14, 2026

Site version **7** published at **2:11 a.m. America/Phoenix**.
GitHub implementation commit: `4318066db60848cb8129a1c6f8337de3077cdb23`.
Site source commit: `e902e0e6ac828535cae0f4155428039d99d8065f`.

The deployment archive contains 918 files / 10,721,280 bytes. Static assets are
10,006,074 bytes, down from 261,353,250 bytes in version 6; this is package size,
not a measured loading-speed claim. Original PNGs stay in GitHub.

The live public browser passed the four-method / 896-network check, exact graph
catalog checksum, actual rotate/zoom, neighbor inspection, primary-state
isolation, supplement labeling, empty graphs, mobile viewport, keyboard close,
retry, checksum refusal and WebGL fallback. Local checks additionally verified
all 890 replay histories, 435 byte-preserved archived files and all 69 frozen
source pins. See [the QA record](qa/VIEWER_CLEANUP_20261114.md).
