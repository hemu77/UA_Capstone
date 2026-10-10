# Simulator Team Handoff

## Public Review Link

[Open Network Observatory](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html)

Revision branch: `r1/social-netw`, [pull request #1](https://github.com/hemu77/UA_Capstone/pull/1).
The root URL also opens the revised workspace; `archive.html` retains the older
two-network/condition-analysis interface. Publication does not merge the branch
or authorize model generation.

## Evidence Included

- [x] Default: 104 V6 calibration graphs from contract `5fb3715550db`.
- [x] The same 50 designed fictional adults, four models and four methods.
- [x] English country framing: US, India, Japan, Brazil. US-language comparison:
  English, Hindi, Japanese, Brazilian Portuguese. Seven settings, not a full factorial.
- [x] Coverage uses actual calibration allocations: Luna two repetitions in all
  settings; other models one repetition per method/language at US framing.
  Conditions not allocated are labelled **Not in scope**.
- [x] All 22 empty global graphs are retained. A `NONE` reply leaves 50 nodes
  and zero ties; it does not imply a missing file. Undefined homophily stays NA.
- [x] V5's 68 graphs and earlier historical/pilot evidence remain selectable
  separately. IDs shared by different rosters are not evidence of shared people.
- [x] Exported graphs, source PNGs and adjacency files remain byte-for-byte
  linked to inspected receipts. The interactive layout is not a PNG reconstruction.

## Review Workflows

1. Open **Filters & saved runs**, choose a model/method/country/language/repetition,
   then **Apply filters**. This replaces the displayed layers, up to six; larger
   selections are rejected rather than silently truncated. **Add selected layer**,
   **Undo remove** and **Restore** preserve explicit user control.
2. Use **Match active run by** for RQ1 country, RQ3 model or RQ4 language contrasts.
   RQ2 displays recorded demographic homophily. Matching checks recorded fields,
   not causal validity or statistical significance.
3. Replay a saved run or select a persona and follow their relevant recorded
   events. Clear selection returns to whole-network scope. Global results have
   one batch, not an invented sequence of individual decisions.
4. Inspect exact descriptors, shared ties and source hashes; download a selection
   or its original PNG/adjacency artifacts. NA overlap means no union of ties
   exists to divide by, not a fabricated zero-similarity score.

### Camera and Persona Controls

Play/pause, timeline and Filters are above the graph in both modes, not buried
below it. **Rotate 360 degrees** is the default drag tool; choose **Pan** to move
freely across the view. Shift-drag swaps the tool. Scroll or trackpad pinch zooms
toward the pointer without the previous far-away zoom limit. Shift-scroll pans;
arrow keys pan a focused canvas and Shift-arrows rotate it. Touchscreens support
one-finger rotation/panning and two-finger pan/pinch. **3D** resets the view.

Clicking a rendered node shows its direct neighbors in each displayed run. Gold
dashed lines join the same selected persona across layers; these are identity
guides, never added friendships. The selected identity stays labelled even with
other labels off. **Center selected persona** recenters the camera without
changing data. Clicking the same node again or **Clear persona selection**
returns to the whole graph. Replay uses the selected event's actual neighbors.

## Verification

- [x] Offline export rechecks contract, complete inspection, receipt/artifact
  hashes, prompt/parser/event replay and recomputed metrics before publication.
- [x] 29 JavaScript tests pass, including exact final replay of all 104 V6 and
  all 68 V5 graphs, empty/null handling, collection isolation and old URL routing.
- [x] Eight focused Python tests pass: three V6 export tests and five V5 regressions.
- [x] Built-output verification checks 208 V6 artifact hashes, recorded replay,
  prohibited private paths and credential patterns.
- [x] Headless Edge/Playwright desktop and 390-pixel mobile flows pass: default
  data, root/archive navigation, Portuguese comparisons, persona journey/clear,
  remove/undo, empty global replay, artifact links and JSON download.
- [x] The same flow passes against the deployed public URL in a fresh anonymous
  browser context, with no HTTP/page errors. Hosted clean-URL redirects are supported.
- [x] Mobile playback controls no longer stick over and obscure the graph.
- [x] Actual pointer-event tests verify Rotate/Pan, wheel zoom, Shift-wheel,
  keyboard panning, click/toggle and exact per-layer saved neighbor lists.
  Browser-native two-finger touch gestures work in comparison and replay.
- [x] One bounded read-only reviewer found no material/blocking findings. The
  suggested root/archive navigation regression was added and passed.

The Browser plugin was unavailable; checks used installed Playwright. Automated
mouse/mobile-viewport checks do not certify every physical touchpad/browser.
These checks establish tested software behavior, not error-free research or
conference acceptance. No study prompts, frozen contracts, source graphs or
accounting records were changed. No model API calls were made.

## Reproduce and Maintain

From the project root, using the pinned Python environment:

```powershell
.\.venv\Scripts\python.exe -B export_revised_viewer.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_revised_viewer.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_calibration_viewer.py
npm --prefix viewer ci
npm --prefix viewer test
npm --prefix viewer run build
npm --prefix viewer run verify:public
.\.venv\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1 --directory viewer/dist
```

In a second terminal, run `npm --prefix viewer run test:browser` and
`npm --prefix viewer run test:navigation` with Playwright
installed. `PLAYWRIGHT_MODULE` optionally gives its installed module path;
`BROWSER_CHANNEL` defaults to `msedge`. `VIEWER_URL` defaults to
`http://127.0.0.1:8766`. No packages are installed automatically by the test.

Static hosting project: `appgprj_6ac225ce79208191ae15aaca77d8a299`.
Deployment version: 2; source bundle commit `ea4848b58b6c258134e2befb92476183819aa7ff`.
The hosting service confirmed publication succeeded on October 4, 2026.
The deployment contains only the verified `viewer/dist` derivative, with Three.js
vendored locally. It has no model endpoint, connector, database or runtime secret.
API replies, request journals, billing ledgers and authorization files are not
included. A public link is intentionally accessible to anyone, not team-only.
Future deployments must use the same hosting project, rebuild, verify, publish,
then test the hosted URL; never point the static server at the repository root.
On this Windows host, the Sites packaging helper needs Git Bash ahead of the
Windows WSL stub in the process PATH, plus `TAR_OPTIONS=--force-local` so GNU tar
treats `C:/...` as a local archive path. These are packaging settings only;
credentials must still be passed through the helper's hidden stdin, never files.

## Still Pending

- [ ] Team/professor decision on the predominantly empty global outcomes.
- [ ] Approved estimands, precision/repetition allocation and finite main budget.
- [ ] Main collection and supported final research findings.

The viewer does not remove these gates. See [calibration results](CALIBRATION_V6_RESULTS.md)
and [team approval checklist](TEAM_APPROVAL.md).
