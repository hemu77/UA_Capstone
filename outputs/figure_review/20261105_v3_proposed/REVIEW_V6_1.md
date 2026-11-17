# Deep review of HEAD 340fc6a (v6/v6.1 19-figure release, Copenhagen extension, notebook 06, 3D viewer data)
AI review, not peer review. No paid calls; nothing outside `outputs/figure_review/` modified. Scripts: `recheck_v6_*.py` (run from repo root; hindi/spearman need scipy, mixing/copenhagen are stdlib).

## Reproduced independently
- Sealed paths (final_study, primary_analysis, heldout_evaluation, analysis_ready, supplements, v2-v4 figure sets, analysis_heldout/primary, study_amendments): zero diff vs 7887da0. v6 vs v6.1: only SVGs (glyph outlines), fig 16, fig 17 and manifest differ; 17 PNG/PDF/CSV sets identical. Notebooks 01-06 embed 3/7/7/19/1/4 = 41 PNGs, every hash = v6.1, zero error outputs.
- Efficiency vs density: Spearman(W1_eff, W1_dens) over 84 record x procedure pairs = 0.866 (reported 0.87); LCC 0.755; Facebook-only 0.82.
- Hindi iterative: mean -0.04528, t raw p 0.000993, Holm 0.0238, simultaneous CI [-0.0850,-0.0056]; Shapiro-Wilk p=0.0126; exact sign-flip p = 2/256 = 0.0078 -> x24 = 0.1875, i.e. the smallest value that test can return (v6 doc is right; docs before v6 framed 0.1875 as evidence, and I repeated that figure without noticing the floor - superseded).
- Fig 16 correction: Religion, Sequential minus Local: mean -0.001162, 12 negative / 16 positive cells (v6.1 says 12/28).
- Copenhagen: all 24 LLM procedure x metric W1s at n=50 recomputed from candidate_metrics + extension draws match to 1e-9.
- RQ2: from the 878 non-empty adjacency files + `text-files/revision224_adults.json`, excess same-group tie share for all 112 cells x 3 attributes matches `attribute_cells.csv` to 1e-16; politics first 85/112 (Global 1/28, others 28/28 each), religion 27 (all Global). Roster chance shares 0.320/0.184/0.347 (politics/religion/gender).
- Viewer data: 896 saved graphs = 890 primary + 6 supplements (4+2) each labeled by source folder; no absolute paths or key-like strings.
Not done: frozen-kernel `check_figure_release_v6.py`, notebook execution, 410-test suite, viewer build/browser checks, raw Copenhagen/SNAP rebuild, SVG/PDF content beyond identity checks.

## Findings
**Medium**
1. *Fig 02/doc framing of Hindi "fragility" overstates it.* After omitting any one block, the raw t p stays 0.00006-0.0036 and all eight block differences are negative; adjusted p crosses .05 only because n drops 8->7 (t df 6) under an x24-type multiplier. "3/8 survive" measures closeness to a Bonferroni-scale cut, not instability of direction or size. Conversely the rejection itself rests on t-test normality at n=8 (Shapiro p=0.013; block 44000 at -0.099): no distribution-free test (sign-flip, Wilcoxon) can reach the Holm threshold with 8 blocks. Report both: strong raw/directional consistency, but no nonparametric corroboration is possible in this design. Add a raw-p column to fig 02b.
2. *Copenhagen n=50 targets are near-empty graphs.* Density 0.020 (~25 edges), ~47% isolates, clustering 0.08, efficiency 0.05 (extension tables); LLM graphs have ~75-210 edges. So "closest" mostly measures edge count/fragmentation: target-density ER wins density, clustering, efficiency, LCC and loses transitivity, and fixed ER beats Global on clustering only because its clustering (~0.19, tracking its density) happens to sit closer to the target's 0.08 than Global's (~0.26); both are far above it. Treat fig 19/18 as a sampling-regime stress test, drop "closest/closer" ranking language, and if realism is the goal pre-specify a connected sampler (snowball/ego ball) rather than uniform induced draws.
3. *Density dependence is now documented but not yet drawn for clustering.* Fig 12a shows Global's clustering distance rising with record density (0.09 at FB09 to 0.42 at FB06) like fig 15; state this in the caption and avoid ordering claims by record.
**Low**
4. Viewer default: layers.html cohort selector defaults to "All 896 / descriptive display only", so the six selectively completed supplements are in the default gallery (labeled). "896" now means both attempts and saved graphs. Make Primary-only the default and name the two counts differently.
5. Fig 10: y-range clipped at 0-2 / 0 with no marker (density bars exceed 2.0, clustering ranges floor at 0); Copenhagen square overplots FB at ~6%.
6. Fig 18: legend row touches the first caption line; fig 15b: labels 01/07/School collide; fig 02a: x tick labels crowd; fig 19f repeats the uninformative largest-component panel (explained in 13).
7. Spearman 0.87 pools 12 non-independent records x 7 procedures; descriptive only (already described as such).

## Figure verdicts (v6.1)
KEEP: 01, 03, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14, 15, 16, 17. REVISE: 02 (raw p, ticks), 10 (clipping, overplot), 18 (legend/caption collision), 19 (LCC panel, stress-test framing). Main text suggested by repo: 01, 02, 04, 08, 15, 14, 18 - I agree, with 02 revised.

## What to tell the professor
The one adjusted instruction effect (iterative Hindi, -0.045 clustering) is consistent in all eight blocks and in every leave-one-out refit (raw p<0.004), but with 8 blocks it depends on a t approximation and cannot be confirmed non-parametrically. Efficiency "advantage" for Global is largely density (rho~0.87); clustering has no stable winner; politics leads RQ2 in person-level methods on one roster/prompt order. Copenhagen shows how uniform 50-person draws wreck large sparse networks, not which generator is realistic.
