# Critical review of figures and held-out evaluation (AI review, not human peer review)

Repo state reviewed: `398ce7a` (origin/main). No paid calls; frozen data, v1/v2 figures and seals untouched
(only this directory is new). Proposed replacements: `R08`, `R09`, `R12`, `R_appendix_lcc`
(rebuild: `python -I -B build_review_figures.py <repo_root>`; needs numpy+matplotlib; reads sealed CSVs only).

## Reproduced independently (pure Python from saved CSVs)
- Condition-balanced W1 (1/7 per condition spread over valid graphs, vs 32 target draws): all 1,056 LLM rows in
  `per_cohort_distances.csv` match; max abs diff 8.9e-16. Failure-affected groups (global gpt-5.6-luna 3 failures,
  global gpt-6-luna 1, global gpt-6-sol 1, iterative gpt-4.1 1; 5 invalid_global_response + 1 local_corrections_exhausted) reproduce.
- 896 LLM run_ids, none from supplement paths; 12 valid empty LLM graphs use LCC=0.02, Gini=0 (same convention as targets; no empty target draws).
- Baselines: 3 x 448 = 1,344 graphs (8 banks x 56). Dev fit: ER p=0.18894, BA m=5, WS k=10 p=0.2 (`development_fit.json`;
  `benchmark.py:63-82` fits only the 8-record dev panel, Tailor waves averaged, 7 equal community weights). No held-out input in fit.
- Hindi iterative: mean -0.04528, raw p 0.00099, Holm 0.0238, simultaneous CI [-0.0850, -0.0056]; 3/8 leave-one-out refits pass (fig 02 filled markers).
- Notebooks 01/02/03/04 embed 3/7/5/12 PNGs, every one SHA-256-identical to `publication_figures/20261105_v2/`; zero error outputs, no temp paths.
NOT done: `scripts/check_publication_notebooks.py` (Windows/frozen-kernel script), raw SNAP/School downloads, 24-test rerun, SVG/PDF files (only PNG viewed), the five papers themselves (relied on the reference notes).

## Findings by severity
**Medium**
1. *Largest-component panels (08 f, l) carry no information.* ER/BA/WS are connected in 99.8-100% of graphs, observed draws always <1 (mean 0.563), so all three baselines get the same distance (1 - target mean); Local/Sequential nearly identical too. Evidence: `candidate_metrics.csv`, fig 08 panels f/l. -> remove from main figure (`R_appendix_lcc`).
2. *Source means imply precision the cohorts do not have.* Fig 08 shows one dot per source; per-cohort Facebook W1 ranges overlap heavily (R08 bars: e.g. clustering WS ~0.01-0.27, Global ~0.08-0.42). Baseline "bank ranges" in fig 08 are invisible (tiny) while the dominant variation (cohort heterogeneity) is not shown. 22/45 cohort pairs share people.
3. *Sampling dependence.* Observed clustering falls sharply under 50-node draws (fig 10b: draw means 0.20-0.57 vs full rosters 0.41-0.68), so the transfer ranking is conditional on n=50 uniform sampling. A naive bootstrap over the 32 draws (understates uncertainty: draws overlap) gives clustering mean-W1 intervals that never reorder WS (FB, 200/200) but split BA/WS for School (195/5). Treat winners as descriptive.
4. *Fig 01 omits the fragility of the one "significant" contrast.* Iterative/US-Hindi interval excludes 0 with Holm 0.0238 but 3/8 leave-one-block-out refits pass; fig 01 shows no flag and fig 02 is a separate figure.
5. *Fig 09 overpacked/redundant.* LLM and baseline curves are identical in both rows (only the black curve changes); density panels waste 0.35-1.0; ER/BA/WS vertical lines are indistinguishable greys; efficiency repeats the story; School observed curve is 32 overlapping draws of one 133-node roster (steps look like precision).
**Low / design limits**
6. Pooling all 7 instruction conditions (incl. Hindi, Japanese, Brazil framing) against US-school/Facebook targets mixes estimands; baselines were fit to dense (0.189) dev graphs, so density/efficiency losses are expected (disclosed in docs).
7. Fig 11: G(n,m) density residual is identically 0 and degree-swap Gini residual 0 by construction; only means shown (no spread over the 32 draws). Doc range "0.1516-0.4259" does not say it is the degree-swap clustering residual.
8. Unsupported wording to avoid: "closest procedure", leaderboard, equivalence, realism; 352 draws are not 352 samples; two studies are not representative.

## Figure verdicts (v2)
| Fig | Verdict | Reason / remedy |
|---|---|---|
| 01 primary contrasts | REVISE | Add Holm-significant marker + "3/8 LOO" note on iterative/Hindi; mixed density/clustering axes are labeled but panel a x-range differs; keep block dots. |
| 02 robustness | KEEP | Clear; give block seeds on axis, drop empty y tick on panel b. |
| 03 model profiles | REVISE | 0-1 axis compresses 0.45-0.75 differences; use zoomed shared range, add valid-n for the 6 cells with n<8. |
| 04 coverage | KEEP | Good use of hatching and bounds-not-CI labelling. |
| 05 resource use | REVISE | Global markers clipped at the zero axis and overlapping; use log or dot per method with offset; fine otherwise. |
| 06 G(n,m) context | REVISE | x range -0.7..0.7 leaves half empty; use 0..0.65 or symmetric but tighter. |
| 07 legacy comparison | REVISE | Box + whisker + jitter + diamonds is heavy; Global whiskers run to 0; drop boxes, keep dots + medians; label sizes 31-70 vs 50. |
| 08 heldout distances | REPLACE | -> `R08` (5 informative metrics, both sources in one panel, cohort ranges, no LCC). |
| 09 heldout distributions | REPLACE | -> `R09` (clustering + zoomed density, direct 2x2, one legend; efficiency/LCC to appendix). |
| 10 sampling effect | KEEP | Directly shows the sampling effect; clip of FB09 range at 0 should be annotated. |
| 11 conditional controls | REVISE | Add spread over the 32 target draws; say both controls are target-informed; keep to clustering/transitivity. |
| 12 cohort sensitivity | REVISE | -> `R12` puts LLM and baselines on one axis so comparison needs no panel hopping; vertical stagger is still busy (could use small multiples). |

## Paper vs appendix
Paper: 01 (revised), 04, 10, R08, one of R09/12. Appendix: 02, 03, 05, 06, 07, 11, R12/R_appendix_lcc.

## Words for the professor
No procedure is closest across metrics or sources; classical dev baselines stay competitive (WS on Facebook clustering, BA on School). Observed clustering exceeds degree-swap controls in all 11 records, yet these are two public studies, 50-node samples, possibly in pretraining, and not independent. The only primary effect that survives the planned correction (iterative/Hindi clustering) is fragile under leave-one-block-out.
Remaining validation gaps: pretraining-unseen data, matched-person link prediction, mixing of swap nulls, population-level uncertainty.
