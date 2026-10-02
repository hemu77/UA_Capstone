# Formation studio: recorded history, not invented animation

## Current Calibration Workspace

`layers.html` defaults to the 68 verified fresh calibration runs, not the old
engineering pilots. Use **Evidence dataset** to open the historical archive;
these rosters stay separate even though both use IDs 0 through 49.

- **Study coverage** shows available repetitions out of eight planned per
  condition. Click a nonempty cell to inspect its recorded repetitions. Empty
  cells do not synthesize or substitute a graph.
- **Match active run by** selects model (RQ3), country (RQ1), instruction
  language (RQ4), or method contrasts while retaining the other matching fields.
  RQ2 tables use the actual saved demographic measurements.
- **Export selection** downloads the selected final graphs, personas, event
  deltas, measurements and provenance. It is not an averaged synthetic graph
  and does not export raw prompts, replies or provider accounting.
- **Expand** enlarges the stage without removing playback or inspection tools.
  Neighbor buttons inspect the same person across the selected runs.

The offline refresh command is `python -B export_calibration_viewer.py`, followed
by `npm run build` in `viewer`. Re-exporting requires the frozen environment and
unchanged reviewed artifacts; serving the checked-in export needs neither an API
key nor a paid model call. Original PNGs remain byte-for-byte unchanged; browser
positions are a separate display layout over the same verified adjacency data.

Coverage is **68 / 896**, not a completed main study. GPT-6-Luna supplies two
repetitions across all seven settings; the other three models supply one
US-English repetition per method. The older pilot counts below describe only
the separately selectable historical dataset.

## Use the workspace

Current interaction checks and remaining team gates are recorded in
[the interaction handoff](docs/VIEWER_INTERACTION_REVIEW.md). Click a selected
node again, click empty space, or use Clear persona selection to restore the
whole graph. Trackpad navigation defaults on in replay and comparison; disable
it to scroll the page over the canvas. Replay active run starts at event zero.
Connection comparisons now show exact per-run presence and counts first, with
the matrix available only as an optional view.

1. Open `http://127.0.0.1:8765/layers.html` after `npm run build` in `viewer`.
2. Select a saved run in the left rail. **Whole network** shows that one run
   and clears persona-only filtering; **Compare runs in 3D** shows final networks.
3. Press **Play formation**. Use Pause, Back, Next, Start, Final, speed, or the
   scrubber. Event zero is the predefined roster with no ties. Displayed event
   one corresponds to the first logged decision (raw logs use zero-based steps).
4. Select a persona by node or selector, then **Follow persona**. Press **Play
   journey** in the same timeline. This includes the persona's own recorded
   decisions, even no-ops, and other actors' changes to that person's ties.
5. The event actor is explicitly distinguished from the selected persona. The
   graph is undirected; the recorded actor is not an edge direction.
6. Manual scrubbing or choosing a ledger event switches back to all-event scope.
   Back/Next in persona mode visits only relevant events, while reconstruction
   still incorporates every preceding recorded event. The journey can finish
   before the run's final step when no later event involves that persona.
7. **Choose experiments** retains explicit Apply, Add, Undo remove and Restore.

Playback never requests a new model response. Pause is automatic when the page
is hidden. Share URLs retain run, selected persona, mode, frame and filters;
they do not resume playback automatically.

## What the display means

- Bright green edges: added in the displayed event.
- Dashed coral edges: removed in that event, drawn as temporary visual markers;
  they are excluded from current counts and neighbor lists.
- Dim edges: already present. In persona focus, unrelated edges are hidden,
  while the current network edge count still refers to the full graph.
- Attribute colors remain unchanged when a persona is selected or acting.
  An outline marks the actor; selection increases the node size. The legend
  stays outside the scrolling controls and uses one mapping across runs.
- The event header and inspector report actions and attempt counts. No model
  reasoning, motives or causal explanations are invented.
- The frozen algorithmic layout uses saved final edges for visual stability.
  Spatial positions do not provide evidence about when or why a tie formed.

The trajectory chart shows exact edge counts, or the selected persona's degree,
after each recorded event. It deliberately displays the complete recorded trace,
including later steps; the cursor marks the currently displayed snapshot.

Final density, clustering and other graph descriptors appear in a separate
comparison chart, with definitions and source values. They are not evaluation
accuracy scores. Higher density or clustering is not automatically better.

## Available evidence

- Local and sequential pilot runs: 50 logged decisions each.
- Iterative pilot runs: 350 logged decisions, including initialization and updates.
- Global pilot runs: one logged batch. No internal per-person sequence exists.
- Historical runs without event logs: static final networks only. Playback is
  disabled rather than manufacturing a history from an adjacency list.

All 28 pilot event logs are checked against their saved final edge sets. The
test fixtures cover removals, no-ops, incoming incident changes, unknown IDs,
missing histories, and inconsistent final graphs.

## Review and references

Two read-only reviewers covered distinct scopes: UI hierarchy/interaction and
scientific explanation/user understanding. Their findings led to larger text,
one Play/Pause control, explicit actor relevance, an event-change header, corrected
manual-scrub scope and neutral wording for recorded attempt counts.

The interaction reference was the timeline plus contextual inspection pattern
documented by [Gephi](https://docs.gephi.org/desktop/User_Manual/Import_Dynamic_Data/)
and [Linkurious](https://doc.linkurious.com/user-manual/latest/timeline/).
Their time-based workflows informed the design; our logs contain generation
order, not observed real-world timestamps. No affiliation or product equivalence
is claimed.

The earlier [usability review](VIEWER_USABILITY_REVIEW.md) remains a record of the
previous layer interface. This document describes its replay-led replacement.
Software checks and reviewer feedback do not validate real-human simulation.
Adult-roster design, translation review, uncertainty and Astra's scientific
review remain separate gates before additional confirmatory collection.

## September 30 evidence and interaction correction

The monochrome charcoal workspace replaces the unrelated white/blue surfaces.
Colors encode attributes and changes, not decoration. Nodes have a screen-size
minimum; edges use Three.js screen-space widths. Hover reveals the same persona
attributes available through the keyboard selector. Full hashes live in the
provenance disclosure, not run names. Plane spacing changes only display geometry,
is disabled outside multi-run comparison, and no longer resets camera orientation.

Several new edges in one event are not several successive decisions. Playback
fades the complete recorded batch together. Reduced-motion settings disable the
fade; counts always report the exact recorded state. Pause holds that recorded
state, not an interpolated scientific measurement. Local/sequential generation
is not a simultaneous-agent simulation. Global has only one recorded batch.

The RQ selector shows comparison requirements and flags other varying fields,
unknown source versions and source-version differences. RQ2 displays categorical
Coleman scores separately from age assortativity; RQ3 reports exact edge Jaccard
overlap. None of these displays supplies a causal explanation or accuracy score.

The pilot runner saves `.adj` and `.png` from the same in-memory graph. The viewer
exports adjacency edges, not pixels from the PNG. The original image is linked in
Source provenance via a hash-checked copy. Layouts differ intentionally. Historical
PNG/graph correspondence is unknown, so no verified historical PNG claim is made.

Run `rtk proxy .\.venv\Scripts\python.exe -B verify_viewer_sources.py` after export
and build. This checks all 204 browser graphs, recomputed topology, 28 pilot
receipts/replays/PNG copies and 193 unchanged historical sources. It does not
independently reconcile responses against the private API ledger.

See [the handoff gate](PILOT_HANDOFF.md) before spending on another study.

## Persona selection, source panels and manuscript exports

- A plane label such as **Run 3: gpt-4.1-mini** identifies the third experiment,
  not persona 3. Persona labels explicitly say **Persona**. A recorded actor is
  labeled **actor at event N**, even while paused; nobody is continuously acting.
- Clicking a node or its ID label, or selecting a persona in the dropdown, now
  isolates their direct ties immediately. In 3D comparison this happens in every
  selected run. Each run lists that persona's neighbors; summary cards compare
  the neighbor intersection and union, not unrelated whole-network statistics.
- Selecting Whole network restores everyone. Automatic labels show IDs while
  suppressing overlaps; hovered nodes and the selector identify every persona.
  The label selector can hide them when the graph is crowded. An isolated persona
  remains visible and is reported as having no neighbors.
- Scroll moves the page. Explicit +/- buttons zoom; mouse drag orbits. Touch
  scrolling remains available with the interaction toggle off. Enable wheel
  zoom and touch rotation for direct manipulation; right-drag pans. Rotation is
  unrestricted except at the poles; use 3D to reset an unreadable camera angle.
- The neutral summary uses each selected final run once to calculate per-edge
  frequency. It is not a newly generated average network or a probability model.
  Selecting a persona restricts this summary to their direct ties too.
  It now uses a symmetric persona-by-persona matrix, not overlapping lines.
  Darker cells indicate more selected runs containing that tie; mirrored cells
  represent the same undirected edge. Node labels default to hover-only.
- The figure gallery covers all selected pilot runs, while the detailed receipt
  is explicitly labeled as the active run. Missing historical presentation panels
  are stated, not fabricated. Missing optional gallery data cannot disable replay.
- `render_pilot_figures.py` produces clean 400-dpi PNGs and vector SVGs for all
  28 pilots under `outputs/presentation`. Only persona IDs appear inside the
  panels: no branding, headers, metrics or explanatory paragraphs. Suggested
  captions are separate `.caption.md` files; provenance stays in the manifest.
  These are paper-layout assets, not a claim of scientific acceptance.

Build after rendering: `rtk proxy .\.venv\Scripts\python.exe -B render_pilot_figures.py`,
then `rtk npm --prefix viewer run build`, then the offline source verifier.
