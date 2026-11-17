# Deep review of HEAD f339989 (v3 figures, matched-size sensitivity, RQ2 fig 14, professor brief)
AI review, not peer review. No paid calls; nothing outside `outputs/figure_review/` was modified. Supersedes the figure verdicts in REVIEW.md where they conflict.

## Verified
- Frozen/sealed paths (`final_study`, `primary_analysis`, `heldout_evaluation`, `analysis_ready`, supplements, v2 figures, `analysis_heldout`, `analysis_primary`, `study_amendments`): zero diff vs 398ce7a.
- v3 KEEP figures 02/04/10: byte-identical to v2 in png/svg/pdf. Notebooks 01/02/03/04 embed 3/7/6/13 PNGs, every hash matches v3; 05 embeds fig 14 only.
- Sensitivity table numbers in docs match `sensitivity_source_summary.csv`; ER connected in 447/448 graphs.
- Earlier findings addressed: LCC moved to appendix (13), spreads added (08, 11), Hindi 3/8 flag added (01), 09/12 rebuilt, axes fixed.
NOT done: `scripts/review_publication.py check`, pytest (77 tests claimed), raw-data rebuild, SVG/PDF inspection beyond identity checks, v3 figures 03/05/06/13 not viewed.

## New findings
**High (claim-level)**
1. *"Closest on efficiency = LLM" is largely a density artifact.* Baselines were fixed at dev density 0.189; targets are 0.03-0.14. Target-informed oracle (`density_matched_oracle.py`: ER with p = each record's mean target density, 56 graphs): mean efficiency W1 = 0.154 vs Global LLM 0.119 vs fixed WS 0.365. A density-matched random graph closes ~85% of the WS-to-Global gap; it beats or is within 0.015 of Global on 6/11 records (FB01, FB02, FB05, FB07, FB09, School2013). Professor brief ("best match for how easily people can reach each other was an LLM") must say: Global wins mainly by being sparse. Oracle uses target info, so it is a diagnostic, not a competitor; do not retune baselines and call them held-out.
2. *Winners with negligible margins are reported as winners.* Facebook clustering at n=40: WS 0.1352 vs BA 0.1389 (gap 0.0037); n=50 WS/BA gap 0.0196; school n=25: ER 0.0456 vs Global 0.0627. Tables name one "closest" without margins; the ranking flips across n, so "closest classical model" should read as "classical models are competitive; ordering unstable".
**Medium**
3. Fig 14 (RQ2): Coleman score is not obviously comparable across attributes with different group counts/shares; dots are means over 28 combinations with no spread (panel A) and panel B counts ties-broken-by-unrounded means (global 22/28 vs 6/28 hides margin). Attribute salience/order in the persona prompt and roster correlation (religion vs politics) are unverified confounds. Exploratory only; add per-combination spread and ranges.
4. Fig 08 v3: Facebook "range across 10 records" and the single School diamond share axes but different meaning; ranges are heterogeneity and overlap, so ranking in the efficiency/clustering panels is not supported beyond Global-vs-fixed-baselines. Efficiency panel needs a density-matched reference (finding 1).
5. Fig 09 v3: separate LLM vs baseline panels are clean, but efficiency (where headline claim lives) was dropped from the main figure; if the efficiency headline stays, show it.
6. Fig 11 v3: ranges across 32 overlapping draws look like uncertainty; FB09 crosses 0; keep the "not a CI" label in the figure, not only the doc.
**Low**
7. Fig 12 v3 is dense (19 markers x 11 records) but legible; ER teal vs Sequential green are close hues across panels. Fig 07 v3 omits per-graph dots (fine for context).
8. Docs say proposed artwork "was not available in the repositories": my R-figures were local/untracked, which is accurate.

## Figure status after v3
01 KEEP (flag added) | 02 KEEP | 03,05,06,13 not re-viewed | 04 KEEP | 07 KEEP | 08 REVISE (add density-matched reference or caveat efficiency) | 09 KEEP + add efficiency to appendix | 10 KEEP | 11 KEEP (label ranges in-figure) | 12 KEEP | 14 REVISE (spread, scope caveat).
