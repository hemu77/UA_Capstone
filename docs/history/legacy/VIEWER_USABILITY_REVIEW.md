# Layer viewer: critical review and revision

Reviewed 2026-09-29. Scope: the local saved-network viewer, not the validity of
the complete scientific study. No affiliation with Simile is claimed.

## Judgment before the fixes

The 3D view was visually stronger than its interaction design. As a first-time
research user, I could not reliably answer "what did my filter change?" or
"how do I recover that removed layer?". That is a core workflow failure.

- [x] **Filters had no commit action.** They narrowed a run picker but did not
  update the canvas. Added Apply matching runs, with an exact preview count.
- [x] **Add existed but was difficult to discover.** A crowded floating panel
  hid the action, and scene updates reset the selected candidate. Moved filters
  to a separate bar, kept layer management visible, and preserved the candidate
  through persona/color/display changes.
- [x] **Removal lacked recovery.** Added Undo remove, independent of current
  filters, plus restore-starting-comparison and ordinary re-add from saved runs.
  Removing never deletes research artifacts. At least one layer remains.
- [x] **The figure was a dense network without an immediate interpretation.**
  Added on-canvas layer labels, selected-person degree counts, shared-tie counts,
  intersection/union overlap and a list of varying conditions.
- [x] **Trust limitations were too far from the result.** Show synthetic-population
  status, unestablished human-behavior accuracy, and differing source variants or
  historical language confounding alongside the canvas.
- [x] **Too many matches could encourage arbitrary selection.** Apply rejects
  zero or more than six matches without replacing the current comparison.
  Individual addition remains available within the six-layer limit. No hidden
  truncation, averaging across incompatible conditions or new API generation.

## What Simile's public work contributes

I inspected its public homepage visually, public research descriptions and the
linked research lineage. I did not access or evaluate its private application.
The useful presentation pattern is an explicit question and action, with population
construction and validation explained close to the outputs. Its homepage uses
a restrained hierarchy and one primary action rather than a screen of equally
weighted controls. [Simile homepage](https://www.simile.com/).

Its research agenda calls for better human grounding and calibrated uncertainty.
That supports showing our provenance and limitations; it does not license adding
an invented confidence percentage. [Simulation research agenda](https://www.simile.com/blog/simulation-next-frontier).

The associated self-report-grounded agent study evaluates agents built from
interviews and surveys against held-out responses, including a demographics-only
comparison. Our synthetic-persona networks have not undergone equivalent human
validation. Better graphics do not bridge that evidence gap.
[Park et al., version 3](https://arxiv.org/abs/2411.10109v3).

## Remaining limits

- [ ] Human-population representativeness, external behavioral validation and
  uncertainty remain research tasks, not UI features.
- [ ] Dense 3D networks still have occlusion. Use selected-person ties, alternate
  cameras and the exact table/2D topology chart when judging measurements.
- [ ] More than six simultaneous layers is intentionally unsupported. Narrow
  the comparison or use the archive's multi-condition plots.
- [ ] Undo is local to this page session; share links restore displayed runs and
  filter choices, not the undo history or camera orientation.

## Verification

The Node tests cover exact filter intersections, invalid/oversized selections,
recoverability of removed saved runs, roster compatibility and undirected overlap.
Browser checks exercise draft versus applied filters, removal under changed
filters, undo, manual re-add, candidate stability after persona changes, empty
and oversized results, URL restoration, and mobile layout. These are software
checks; they do not establish the scientific validity of the displayed findings.
