# Re-review of 7887da0 ("Qualify reachability and mixing claims and publish the primary network explorer")
AI review, not peer review. No paid calls; only files in this directory were added. Script: `recheck_7887da0.py` (run from repo root, stdlib only).

## Verdict
The commit resolves the prior high-severity findings. Remaining issues are medium/low and about framing and figure polish.

## Reproduced
- Target-informed ER efficiency W1 for all 11 records recomputed from the published 4,928 simulated values and the held-out draws: max abs diff 5.6e-17. Equal-record means 0.119224 / 0.365267 / 0.153019; gap closed 86.3% (my independent estimate was ~85%).
- Cramer's V religion x politics = 0.3156 (chi2 9.96, n=50). Politics first on 90/112 (Coleman) and 85/112 (excess share); all switches are Global (22 and 27 religion-first).
- Clustering margins table matches (WS vs BA 0.0037 at FB/40; ER vs Global 0.0171 at school/25).
- Explorer data: 896 records (890 graphs, 12 empty, 6 failures), no duplicate edges, edge codes decode with a<b, edge counts match `network_metrics.csv` density for every graph, every linked `.adj` file exists, no absolute paths, keys or supplement strings. JS uses textContent/encodeURIComponent (no innerHTML).
- Frozen paths unchanged since f339989; `tests/test_deep_review.py` 3/3 pass (in a scratch Python 3.9 env, NOT the pinned 3.11.9). Not run: `check_deep_review.py`, notebook execution, node viewer build/browser check, the full test suite.

## Findings
**Medium**
1. *The 86% summary hides a density-dependent sign pattern.* Per record (`efficiency_records.csv`): target ER beats or ties Global on every sparse roster (density < 0.06: FB01, FB02, FB07, FB09, School) but Global beats target ER on 5 of the 6 denser Facebook rosters (FB03, FB04, FB06, FB08, FB10; FB05 tie). So density explains the sparse records; on dense ones fragmentation of 50-node draws matters and Global's sparse, fragmented graphs fit better. Report per-density-band results next to the pooled 86%, and say "density alone is not sufficient" with this evidence, not only as a caveat.
2. *Fig 08 efficiency panel still reads as an LLM win.* The only qualifier is a figure-level footnote; panel c has no ER-target marker. Add the target-ER reference to panel c or an in-panel note.
3. *Fig 15 polish.* Legend is a grey sentence ("Black: observed / Blue: Global / Dashed: target ER") with no colour keys; x-range 0-1 although efficiency <= ~0.6; overplotted markers at FB02/FB05; ER curve in panel C is clearly right of observed school draws, so "closer than Global" (0.121 vs 0.138) is a small margin on one record.
**Low**
4. Fig 14: panel A shows only the Coleman score though excess share is used in B; sequential/iterative show outlier dots (e.g. Coleman 0.10, 0.17) unexplained; Cramer's V is descriptive of one fixed roster (chi2 p ~0.27 at 8 df if treated inferentially), so "co-vary" should be read as a design fact, not evidence.
5. Brief/viewer say ER "closed most of the gap" and "86%"; both carry the not-causal caveat, but the school ratio is 105% (>100%), so pooled percentages are not a bounded measure.
6. The viewer build copies fig 15 at build time; `viewer/public/data/study-figures/` is absent from the checkout, so the page's image only exists after `build.mjs`.

## Figure status
08 KEEP+add ER ref | 11 KEEP | 14 KEEP (minor) | 15 REVISE (legend, axis, overplot). Others unchanged from REVIEW_V3_DEEP.md.

## Words for the professor
The reachability advantage of the whole-network LLM mostly shrinks when a random graph gets the real density, but not uniformly: it holds on sparse records and reverses on denser Facebook records. Clustering has no stable winner. Politics dominates demographic mixing in person-level methods on one fixed roster/prompt order.
