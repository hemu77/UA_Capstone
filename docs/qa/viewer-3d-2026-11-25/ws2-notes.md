# Workstream 2 (3D engine): notes for the integrator

Date: 2026-11-25. Scope: SPEC.md B.1, B.3, C.4 and the WS2 row of C.9. Screenshots: `ws2-shots/` (1440x900 and 390x844,
deviceScaleFactor 1, SwiftShader).

## What was built

**Store as the single source of truth (`layers.js`, landed first).** Every control writes the store through
`commit()`; one subscriber derives everything on screen from the store, whoever wrote it (`view`, `layers`, `active`,
`frame`, `persona`, `colorBy`, `lens`, `groups`, `filters`, `focusSlot`). Only presentation-local controls stay outside
the store (tie opacity, plane spacing, tie filter, label density, "only this persona's ties", persona journey, playback
timer). External writes are validated: an oversized or unknown stack is refused with a message and the previous stack is
kept; an `active` run outside the stack joins it (or replaces it when the stack is full). Replay loading,
checksum verification, caching and wording are unchanged.

**Stage (`layers-3d-scene.js`).** Renders on demand only (no frame is drawn while nothing moves; checked). Hemisphere +
warm key + cool rim light, `FogExp2`, ground grid with horizon fade and a soft footprint shadow, damped orbit, eased
camera moves (500 ms ease-in-out cubic, instant under `prefers-reduced-motion`; auto-orbit disabled there), per-view
field of view, framing that fits a sphere or a box, and a "safe area": the shell's heading and key are measured on
resize (never per frame) and the camera is framed in the unobstructed part of the canvas via a view offset. Shared
glass-slab shader (rim in the run hue, fading inner grid, lit sides), per-slot state texture.

**Atlas (`layers-3d-atlas.js`).** All 896 slots: 112 towers (16 model x method columns, model groups separated; 7
setting rows; 8 seed planes per tower, bottom to top). 890 lit glass slabs (12 valid-empty slabs in the `--empty` hue with
nodes and no ties), 6 failed slots as dashed, hatched, hollow frames with no nodes. 44,500 nodes as one point buffer
(lit sphere impostors, size by degree with a legibility floor, colour from the WS5 encoding through a 64-texel palette),
156,730 ties as one additive line buffer with depth fade. Filter emphasis (out-of-scope towers at 12% and without ties),
hover focus, stacked-run marker, persona highlight across all 890 planes and the same/mixed lens are texture or attribute
writes only. Axis labels in 3D (models, methods, settings, seeds) with the caption "Seed order ↑ not time". Hover shows a
tower card ("n of 8 valid", empties, failures, the hovered seed's ties or failure reason) and sets `focusSlot`; click
flies into the tower and opens its Stack (filters set to that condition). Atlas build: about 26-42 ms.

**Stack and Replay (`layers-3d-stack.js`).** 1..8 runs (+1 optional consensus plane) as large lit glass planes with
shared persona positions; instanced `MeshStandardMaterial` spheres sized by degree; contact shadows and outline rings
(selected persona, recorded actor) in one sprite pass; in-plane ties (fat lines, additive, tinted by plane hue);
translucent ribbons between adjacent planes for ties present in both ("Ribbon: tie shared with next plane"); gold
dashed identity thread labelled "Persona n: same persona, not a tie"; optional "Lift by degree" (lollipop stems; off =
flat on plane), labelled in the key. Replay is the same engine with one plane: new ties of the current recorded update
fade in, removed ties are dashed coral, the actor is ringed. Plane labels are buttons: tower -> plane label = Replay
of that run (two clicks). Consensus plane (HUD toggle, Stack only): `consensus(applyFilters(runs, filters))` from WS5,
labelled "Aggregate of N filtered graphs · not a run", tie brightness and width by share. All buffers are allocated once
and rewritten in place.

**Labels (`layers-3d-labels.js`).** One pooled label layer in `#plane-labels`; text is measured once per `set()`,
`project()` only writes `transform` and `hidden` (zero layout reads inside animation frames; checked); priority-based
collision culling; labels whose box would be clipped are hidden.

**HUD (`#stage-overlay`).** View caption and key (Atlas: graph / valid-no-ties / failed / in your stack), toggles
"Lift by degree", "Shared-tie ribbons", "Consensus plane" (`aria-pressed`, 44 px). Collapses to one row below 700 px.

**Also done in `layers.js` on request of WS3/WS5:** URL writing handed to `layers-filters.js` (`mountFilters(...,
{writeUrl:true})`; this file only reads links: short grammar via `decodeState`, legacy `?view=` handled here so
stale-link errors keep today's message); Atlas emphasis uses WS3's `slotMask`; legend/palette fallback removed (WS5 owns
`#legend`); tie colour/dimming from WS5 `edgeEncoding` (`colorOf`, `dimmed`); population text no longer written to
`#dataset-summary`/`#evidence-notice` (only load errors); per-run source gallery removed; run text through
`runParts`/`modelName`/`methodName`/`settingName`/`plural`. Legacy filter-drawer code removed (`readFilters`,
`candidates`, `updateAdd`, `#filter-form`/`#add`/`#candidate`/`#reset-filters`/`#open-filters` handlers, per-key
restore); every remaining legacy lookup is null-guarded.

Exposed for checks on the canvas: `data-view`, `data-draw-calls`, `data-slots` (896), `data-layers`, `data-planes`,
`data-geometries`, `data-frames`, `data-camera`, `data-atlas` (`{"slabs":890,"failed":6,"nodes":44500,"ties":156730}`);
`export function drawables()` from `layers.js` lists the objects submitted to the GPU.

## Draw calls (measured, SwiftShader, 1440x900)

| View | Draw calls | Budget |
|---|---|---|
| Atlas (896 slots) | 6 (7 while hovering a tower) | 12 |
| Stack, 4 planes | 6-9 | 24 |
| Stack, 8 planes | 6-9 | 48 |
| Replay (1 plane) | 4-6 | 6 |

Geometry count is unchanged by filter, persona, colour, opacity, tie filter, encoding toggles and replay frames
(checked). Store update latency in page: filter change 14-31 ms, persona 8 ms, view switch 10-17 ms, frame 2 ms.
First rendered frame on a cold local load: about 3.0-3.8 s under headless SwiftShader (data ready at about 0.25 s; the
remainder is software shader compilation), to be confirmed on real hardware against the 2.5 s target.

## Files touched

`viewer/layers.js` (rewritten), `viewer/layers-3d-scene.js`, `viewer/layers-3d-atlas.js`, `viewer/layers-3d-stack.js`,
`viewer/layers-3d-labels.js`, `viewer/layers-3d.css`, `viewer/layers-3d.test.js`, new `viewer/layers-3d-browser-check.mjs`,
`viewer/navigation-check.mjs` and `viewer/primary-browser-check.mjs` (rewritten for the shell; same assertions on
replay verification, failures, counts; the navigation check no longer writes into `outputs/`). `navigation.js` unchanged.
Docs: this file, `requests-ws2.md`, `ws2-shots/*.png`.

## How it was verified

- `node --test layers-3d.test.js navigation.test.js`: 10/10 (atlas layout and slot formula, picking, persona
  positions, stack geometry/tie filters/isolation/ribbons/lift, identity thread, geometry dashes, label culling,
  framing and easing). Full `npm test`: 104/105; the one failure is `graph-data.test.js` test 10 (WS1 file, label
  capitalisation from WS5; already in `requests-ws3.md` / `requests-ws5.md`).
- Private build on port 8792, all 8 browser checks PASS: `primary-browser-check`, `navigation-check`,
  `layers-3d-browser-check` (Atlas budget/idle GPU/in-place updates, tower hover + click, 8-seed stack and refused 9th,
  plane label -> Replay, no layout reads in frames, coverage cell = tower, a store-driven walk opening all 890 valid
  slots in Stack and Replay without error, reduced motion, phone), plus `browser-check`, `polish-browser-check`,
  `plot-gallery-browser-check`, `layers-filters-browser-check`, `layers-shell-browser-check`. No console errors or
  warnings from page code (only SwiftShader "GPU stall due to ReadPixels" driver messages caused by screenshots).
- Screenshots read at every iteration; final set in `ws2-shots/`.

## What the integrator must do

1. Merge in SPEC order; `layers.js` already mounts the other modules and hands the URL to WS3 (`writeUrl:true`).
2. Delete `#legacy-controls` and its contents: `layers.js` no longer reads `#filter-form`, `#model`, `#method`,
   `#culture`, `#language`, `#seed`, `#apply`, `#reset-filters`, `#filter-preview`, `#candidate`, `#add`,
   `#candidate-detail`, `#open-filters`, `#filter-chips`.
3. Decide the open items in `requests-ws2.md` (consensus state key, chrome class names used for the camera safe area,
   `#stage-overlay` placement, `graph-data.test.js` test 10).
4. Check first-frame time and orbit fps on real integrated graphics (SwiftShader is exempt from fps only).
