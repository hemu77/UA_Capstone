# Viewer polish 2: 3D stage, sections below the stage, Replay rework

Not deployed. Nothing was pushed or committed by the integrator; the GitHub Pages site still serves the previous build.
No provider or LLM calls, no credentials. Frozen data, pinned fixtures and `viewer/archive/` are untouched.
Study facts unchanged: 896 attempts, 890 valid graphs (12 empty), 6 failures; the six supplements stay out of every primary view.

## What changed

### Replay (reworked)

What the owner saw ("there is only one event for replay button"):

- The default stack is Global, Local, Sequential, Iterative with the **Global run active**. Every Global record is a
  single response, so pressing Replay showed "Event 0 / 1": one step, then nothing.
- Every other route into Replay (plane label, stacked-run list, result-row Replay, keys 1-8) opened on the **final
  frame**, so a 50-event run looked finished and Next was disabled.
- The transport was a row of same-weight text buttons; what the current event did was only in the closed Persona dock.

Recorded events per method, checked against all 890 logs: Global 1 (219 runs), Local 50 (224), Sequential 50 (224),
Iterative 334-350 (223; 50 Local choices, then recorded add / drop events).

Now:

- Transport: jump to start, step back, Play / Pause (primary), step forward, jump to end, speed (0.5x, 1x, 2x, 5x), and
  a scrubber with one tick per recorded event (up = ties added, down = removed, dashed line = phase change; events
  not involving a selected persona are dimmed). Space, arrows, Home and End still work.
- A card beside the stage shows "Event k of N", the acting persona, the recorded phase, ties added and removed, the
  tie count and the ties themselves.
- Replay always opens at event 0 with no ties drawn (also from links without a position); "Play again" at the end.
- From a mixed stack, Replay picks the active run if it has steps, otherwise the first stacked run that does, and
  says so ("Replaying run 2 (Local): run 1 (Global) was returned in one response and has no steps.").
- A Global run shows "Global generation returned the whole network in one response — there is no step-by-step
  record to replay." in place of the transport, with buttons to replay the Local, Sequential or Iterative run of the
  same model, setting and seed (only those saved as valid graphs) and "View ties by persona". The saved graph stays
  on stage. No order is invented and nothing is interpolated.
- On a phone the transport sits above the canvas, so Play is on the first screen.
- New module `viewer/layers-replay.js` (pure helpers), `viewer/layers-replay.test.js`, `viewer/replay-browser-check.mjs`.

### 3D stage

- Engineer's rework of Atlas, Stack, labels and scene integrated as delivered.
- Zoomed-in Atlas towers: tie level-of-detail starts earlier and the glass top faces are clearer, so ties read
  instead of dissolving into haze.
- Aggregate plane: thin lines only, strength rising steeply with the share of graphs containing the tie (was a
  heavy block of wide gold lines).

### Sections below the stage

- Engineer's rework integrated as delivered.
- Finding 03 now reads: "With uniform 50-node samples of the eleven observed records, Global is closest on
  efficiency in 9 records; with snowball samples of the same records it is closest in 0."
- A one-response run no longer plots a trajectory with fractional event ticks; it says all ties arrived in one response.
- Phone: metric-axis labels no longer collide.

### Share view

The header button no longer sticks on the fallback text. It tries the clipboard API, then a hidden-textarea copy;
shows "Link copied" or "Not copied" (the hint "Copy the URL from your address bar" goes to the live region / toast);
and always returns to "Share view" after 2 s. Repeated clicks restart the timer; the button width is fixed.

## Before / after

| | Before | After |
|---|---|---|
| Replay from the default stack | `docs/qa/viewer-polish-2/replay/before-replay-default-global.png` | `docs/qa/viewer-polish-2/final/replay-mid-playback.png` |
| Replay opened from a plane | `docs/qa/viewer-polish-2/replay/before-replay-opens-finished.png` | `docs/qa/viewer-polish-2/replay/after-replay-persona-selected.png` |
| Replay of a Global run | `docs/qa/viewer-polish-2/3d/before-desktop-replay.png` | `docs/qa/viewer-polish-2/final/replay-global-run.png` |
| Atlas | `docs/qa/viewer-polish-2/3d/before-desktop-atlas.png` | `docs/qa/viewer-polish-2/final/atlas.png`, `docs/qa/viewer-polish-2/final/atlas-filtered.png` |
| 8-plane stack | `docs/qa/viewer-polish-2/3d/before-desktop-stack8.png` | `docs/qa/viewer-polish-2/final/stack-8-planes.png` |
| Sections below the stage | `docs/qa/viewer-polish-2/sections/before-desktop-full.png` | `docs/qa/viewer-polish-2/final/sections-below-stage.png` |
| Phone | `docs/qa/viewer-polish-2/3d/before-phone-atlas.png` | `docs/qa/viewer-polish-2/final/phone-atlas.png`, `docs/qa/viewer-polish-2/final/phone-replay.png` |

README media (captured from the built viewer, 960 x 600, 12.5 fps): `docs/media/atlas-to-stack.gif` (2.6 MB, 10.2 s),
`docs/media/replay.gif` (2.3 MB, 9.9 s). The README itself is not edited in this change; it does not reference them yet.

## Tests (all on the final tree)

| Check | Result |
|---|---|
| `npm run build` | pass |
| `npm test` | 125 of 125 pass (7 new in `layers-replay.test.js`) |
| `npm run verify:public` | PASS (896 / 890 / 12 / 6, 890 verified event logs, 6 supplements separate) |
| `layers-3d-browser-check.mjs` | pass |
| `layers-all-graphs-browser-check.mjs` | pass (all 890 graphs open in Stack and Replay) |
| `layers-filters-browser-check.mjs` | pass |
| `layers-shell-browser-check.mjs` | pass |
| `plot-gallery-browser-check.mjs` | pass |
| `polish-browser-check.mjs` | pass |
| `primary-browser-check.mjs` | pass |
| `replay-browser-check.mjs` (new) | pass |
| `browser-check.mjs` | pass |
| `navigation-check.mjs` | pass |

Draw calls (budget: Atlas 12, Stack 6 per layer): Atlas 4 (7 zoomed in with a tower hovered), 8-plane Stack 5 in
total, Replay 6, Stack with aggregate plane 5. Zero console errors or warnings at 1440x900, 1024x768 and 390x844.

Checks changed to follow the new behaviour, with no count or failure assertion removed: `browser-check.mjs` (a Global
condition now asserts the one-response state, then steps its Local counterpart; the empty-graph case asserts the
statement plus "valid run has no ties"), `primary-browser-check.mjs` (waits for the verification-failure note of the
run Replay now picks), `layers-filters.test.js` (replayRun also sets the replay position in the same single update).
The `layers-3d.test.js` failure reported against in-progress atlas work did not reproduce on the integrated tree.

## Remaining limitations

- At 1024x768 the Replay plane is small: the legend card and the event card take much of the canvas height.
- In a zoomed Atlas tower the lower planes' ties are still veiled by the planes above; dense towers remain busy.
- A bare `?v=replay` link whose active run is Global shows the one-response state; it does not switch runs.
- Using a Global run's alternative when the stack already holds 8 runs replaces the stack with that one run (announced).
- The scrubber ticks of Iterative runs (about 340 events) are about 2 px apart on a phone.
- Final screenshots are 256-colour PNGs to stay under the size budget; faint banding in dark gradients is an artefact
  of that, not of the viewer.
- Checks ran in headless Chromium with software WebGL only; no real GPU, Safari or Firefox pass was made.
