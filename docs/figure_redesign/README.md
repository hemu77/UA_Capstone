# Figure release v6: methods review and redesign of all 19 figures

**Review follow-up:** current notebooks use [v6.2](../../outputs/publication_figures/20261118_v6_2/).
[Codex review and fixes](CODEX_REVIEW.md) correct Figure 16, caption scope, verification
coverage and notebook integration. This document records the original v6 review;
the original commit and release remain unchanged.

**November 15, 2026.** Reviewed from commit `b24e20d`. Offline only: no provider calls, no new
generation, no credentials, no ledger queries. Primary population unchanged: 896 attempts, 890 valid
graphs including 12 empty, six failures. The six supplementary graphs are excluded from every table
and figure here.

Release: [`outputs/publication_figures/20261115_v6/`](../../outputs/publication_figures/20261115_v6/)
(19 figures as PNG, SVG and PDF, one `*_plotted.csv` per figure, `manifest.json`).
Renderer: [`analysis_figures/v6/`](../../analysis_figures/v6/). Checker:
`scripts/check_figure_release_v6.py`. Handoff: [HANDOFF.md](HANDOFF.md).

**This release changes presentation only.** No estimate, weight, endpoint, hypothesis family or
baseline setting changed, and no computational bug was found in the published tables. Earlier figure
releases (v2 to v5, extension v2) and every sealed table are byte-identical to `b24e20d`.

## 1. Findings, most serious first

What was re-derived here from saved graphs or raw public files is marked **reproduced**; everything
else is a reading of the existing tables.

1. **High: the efficiency "advantage" of Global over the fixed baselines is mostly a density
   mismatch.** Across the 84 record-by-procedure pairs at 50 nodes, efficiency W1 and density W1 have
   Spearman correlation 0.87 (largest-component W1: 0.75). The fixed baselines were calibrated on eight
   small, nearly complete development networks (density target 0.189), while 50-node draws from the
   large held-out records have mean density 0.02 to 0.14. Global graphs happen to be sparse (mean
   density 0.061; person-level procedures 0.17). The comparison therefore mixes a sampling-frame
   mismatch with generator quality. Figure 15 shows the consequence: an ER graph given the record's
   density is closer than Global on four of the five sparsest records. Do not headline "Global is more
   realistic on reachability".
2. **High: the sign-flip result was described as evidence of fragility, but it is the test's
   floor.** With eight blocks an exact two-sided sign-flip test cannot return a p-value below
   2/256 = 0.0078. The first Holm threshold for 24 contrasts is 0.05/24 = 0.0021. That test cannot
   reject any contrast in this family; its adjusted p of 0.1875 is the smallest value it can produce
   (**reproduced**). The brief's sentence "an exploratory sign-flip test gave adjusted p = 0.1875" is
   corrected. The real fragility evidence is separate: 3 of 8 leave-one-block-out refits stay below
   .05 (adjusted p from 0.0015 to 0.0825), and the eight block differences are not close to normal
   (Shapiro-Wilk p = 0.013, one block at -0.099), which matters for a t interval on n = 8.
3. **Medium: intervals and p-values in figure 01 come from two different procedures.** The interval
   is a Bonferroni simultaneous t interval at 0.05/24; the p-value is Holm step-down over the same 24
   (the untested contrast is carried as p = 1). They use the same threshold for the smallest p but can differ at later ranks. The old figure did
   not say so and printed no numbers. All 24 means, intervals and Holm p-values were **reproduced**
   from `primary_attempts.csv` to within 3e-15. Seven contrasts have unadjusted p < .05, all negative
   for clustering or involving a non-English instruction; only one survives correction. That pattern
   is worth reporting as unadjusted description, never as seven findings.
4. **Medium: "politics first" is robust in this panel but cannot be read as importance.**
   **Reproduced** from the 890 saved adjacency files: all three scores match the published tables to
   1e-16, pairing before averaging is correct, all six procedure pairs and the 28-cell equal weighting
   match, 878/12/6 defined/empty/failed. Additional reviewer diagnostic (not in any figure): among
   tie pairs of *different* religion, politics excess still exceeds religion excess among
   different-politics pairs in 28/28 cells for each person-level procedure and 0/28 for Global, but this conditioning is not an independent intervention and does not rule out
   confounding by the single fixed roster. What remains
   unseparated: one roster, three near-equal politics groups (17/17/16) against five religion groups
   of 10, and one fixed prompt field order. Chang et al. used a shuffled-demographics control for this;
   that needs new generation and is out of scope.
5. **Medium: the repository's "Coleman" is not the classic Coleman index.** It is a size-weighted
   normalized same-group share; against the classic piecewise index the maximum difference is 0.166
   (correlation 0.999; the politics-first counts are identical). v6 labels it "Coleman-type index".
   The roster also has an `age` field that RQ2 does not analyse; say so when describing RQ2.
6. **Medium: held-out "records" are far from independent.** Mean pairwise node overlap of 50-node
   draws is 76% for FB06 (N=66) and 85% for FB10 (N=59), 6% for Copenhagen. Ranges over 32 draws are
   sampling variation of one record. v6 never draws them as intervals without saying so.
7. **Low: Copenhagen provenance holds.** **Reproduced** from the cached public files: figshare
   version 1, MIT in deposit metadata, published MD5 and recorded SHA-256 match; 6,429 rows, 11
   self-links, no duplicate undirected pairs, 800 endpoints, 6,418 edges, one component. 787 gender
   rows, 747 intersect endpoints, 40 gender-only IDs (not treated as isolates); 53 endpoints have no
   gender row. Every node has degree at least 1 by construction, so "zero isolates" describes the
   release, and expected edges in a 50-node draw are about 25. It supports a structural,
   endpoint-conditioned comparison only.
8. **Low: the whole extension rebuilds.** `scripts/run_research_extension.py` was run into a
   disposable directory: all 15 CSV tables match the published v2 tables numerically (maximum absolute
   difference 2e-16); files differ in bytes only by line endings. That run re-checks the historical
   50-node metrics and W1 distances to 1e-12. The disposable output was deleted.
9. **Low: no procedure is closest on clustering.** Closest procedure by record at 50 nodes: BA 3,
   WS 3, ER 2, Iterative 1, Local 1, Sequential 1; the split changes at 25 and 40 nodes. Figure 12 now
   shows the crossing lines and the gap to the runner-up instead of a five-panel marker cloud.

Not established by any of this: equivalence of the non-significant contrasts, human validity,
population representativeness, or publication readiness. This is an AI-assisted review, not peer
review.

## 2. All 19 figures: verdict and redesign

Before = the version each notebook displayed at `b24e20d`. After = `20261115_v6`. Denominators are
printed in the figure or its caption.

| # | Question | Estimand and denominator | Source table | Problem with the old figure | Verdict | v6 redesign |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Do instruction settings shift the planned endpoint? | Mean paired block difference, 24 contrasts, up to 8 blocks | `primary_contrasts.csv`, `paired_schedule_blocks.csv` | Four cramped panels with different x-ranges; no numbers; interval/p procedures unlabelled | Replace | One forest plot on a shared axis with blocks, mean, simultaneous CI and Holm p columns |
| 02 | How fragile is the one adjusted finding? | Block differences; refit Holm p; attainable p-values | `leave_one_seed_out.csv`, `sensitivity.csv` | Showed refit means, which barely move; said nothing about test resolution | Replace | Three panels: blocks, refit adjusted p against .05, sign-flip resolution against the Holm threshold |
| 03 | Do the four models move the same way? | Model-specific mean difference on common blocks | `model_contrasts_descriptive.csv` | Plotted levels, so differences had to be read by eye | Replace | Same forest layout as 01 with four model markers, mean, same-sign count and range |
| 04 | What was attempted and what did failures cost? | 896 attempts; 3 affected contrasts | `primary_attempts.csv`, `primary_contrasts.csv` | Four near-full bars; 6 failures and 12 empty graphs invisible | Replace | One square per attempt (878/12/6) plus identification bounds |
| 05 | How unequal was computation? | Requests, cost, seconds per attempt, 56 attempts per mark | `resources_by_method_model.csv` | Linear totals compress Global to zero | Replace | Per-attempt log axes with ratio to Global |
| 06 | Is clustering above same-edge-count chance? | Observed minus mean of 8 G(n,m) graphs, per valid graph | `synthetic_reference_per_run.csv` | Jitter clouds; no quantiles; four panels | Redraw | Tick per graph with median, IQR and 5-95% span, n per row |
| 07 | Where do generated graphs sit beside the 8 archived networks? | Density, clustering, efficiency | `empirical_reference_metrics.csv`, primary | Shared 0-1 axis wasted space; "coefficient" axis label | Redraw | Distribution strips with the archived range as a band and numbered networks |
| 08 | How far is each generator from held-out records? | Record-level W1, 11 records, 5 metrics | `transfer_record_summary.csv` (n=50) | "Not estimated" rows; target-informed ER mixed into a transfer panel | Replace | Ten record ticks, Facebook mean, school diamond; diagnostic moved to 15; explicit "not comparable across panels" |
| 09 | What distributions produce those distances? | Metric per 50-node graph | `heldout_sample_metrics.csv`, `candidate_metrics.csv` | Eight overlapping ECDFs with duplicated references | Replace | Aligned distribution strips for observed draws and all seven generators |
| 10 | What does a 50-node draw keep? | Draw value / full-record value, 12 records, 32 draws | `full_network_metrics.csv`, `observed_sample_metrics.csv` | Two metrics, no link to sampling fraction | Replace | Retention ratio against share of the record sampled, three metrics |
| 11 | Is closure above target-informed controls? | Draw minus mean of 8 controls, 32 draws | `conditional_control_metrics.csv` | Ranges drawn like intervals; unordered | Redraw | Every draw as a tick, means, records ordered by density |
| 12 | Is the clustering ordering stable across records? | Record-level clustering W1 | `transfer_record_summary.csv` | Five panels of 220 markers; ordering unreadable | Replace | Profile lines by record plus gap between closest and runner-up |
| 13 | Why is the largest-component metric uninformative? | LCC share and its W1 | same as 08, 09 | Oversized single panel; identical baseline rows unexplained | Replace | Distribution panel showing baselines are 100% connected, beside the distances |
| 14 | Which attributes show same-group ties, and how firmly does politics lead? | Excess share per cell; politics minus best competitor, 28 cells | `mixing_scores.csv`, `mixing_rank_sensitivity.csv` | Two near-duplicate score panels; rank claim not shown | Replace | Excess by attribute, then the politics margin under three scores with k/28 |
| 15 | Does Global's efficiency advantage survive a density-informed reference? | Record-level efficiency W1 | `efficiency_by_density.csv` | ECDF panels repeated figure 09; density dependence only in text | Replace | Records ordered by density; difference against density |
| 16 | How do procedures differ on the same cases? | Paired difference, 28 cells, 206-224 pairs | `mixing_paired_cells.csv` | Min-max spans looked like intervals; headline title | Redraw | All 28 cell differences as ticks, mean, same-sign count, pair counts |
| 17 | Who is tied to whom? | Mean oriented edge fraction / roster-chance fraction | `mixing_matrices.csv`, roster | Raw fractions on one scale: darkness tracked group size and attribute, inviting cross-attribute reading | Replace | Ratio to roster chance, diverging scale centered on chance, group sizes shown |
| 18 | How do full and induced structure differ by size? | Full value; mean and range at 25/40/50 | extension sampling tables | Headline title; isolates not shown | Redraw | Four metrics including isolate share; full record as a distinct open diamond |
| 19 | Copenhagen at 50 nodes | W1, six metrics, 4 models or 8 banks | `transfer_record_summary.csv`, `target_er_summary.csv` | Diagnostic row barely separated; ranges only | Redraw | Model and bank ticks; diagnostic in a shaded tier with its own marker |

Display rescaling in 10 and 17 (a ratio to the full-record value; a ratio to roster chance) divides a
published mean by a constant. Both numerator and denominator are in the `*_plotted.csv` files.

**Proposed main text:** 01, 02, 04, 08, 15, 14, 18. **Appendix:** 03, 05, 06, 07, 09, 10, 11, 12,
13, 16, 17, 19. Inconvenient results stay in the main set (02, 04, 15).

## 3. Reference figures actually inspected

Figure pages were rendered from these files in this review and looked at; nothing is copied.

| Source read | Figure | Principle adapted | Not adopted |
| --- | --- | --- | --- |
| Chang et al., author version [arXiv:2408.16629v2](https://arxiv.org/abs/2408.16629v2), PDF pp. 5, 6, 8. The ICWSM page itself shows only the abstract. | Figs. 3, 5: every network as a faint point with a dark summary and a chance line. Fig. 6: group-pair observed/expected ratios on a shared diverging scale | 06, 07, 09, 14, 16 show every graph or cell; 17 uses ratio to chance | Their mean and standard error; their KS statistic |
| Argyle et al., publisher PDF (Political Analysis, doi 10.1017/pan.2023.2), p. 11 | Fig. 4: two sources joined per item in aligned small multiples | 15a joins Global and the informed ER per record | Their matched-survey validation claim |
| Papachristou and Yuan, author accepted version [arXiv:2402.10659v7](https://arxiv.org/abs/2402.10659v7), PDF p. 4 | Fig. 1: dashed null lines, consistent panels across model, temperature, environment | Zero and chance reference lines; diagnostic tier drawn differently from competitors | Bar charts with error bars |
| Piao et al., author version [arXiv:2501.05171v3](https://arxiv.org/abs/2501.05171v3), PDF p. 18. The Nature page was not retrievable. | Fig. 2: panel letters, one role per color | One color per procedure and one marker per model across all 19 | Time series and stacked areas |
| Helwe et al., [ACL proceedings PDF](https://aclanthology.org/2025.findings-acl.883.pdf), p. 16 | Appendix Fig. 1: repeated grid with identical axes | Shared axes across procedure panels (01, 03, 16) | Political-compass geometry |

Publisher and author versions are not asserted to be identical.

## 4. Verification run for this release

See [HANDOFF.md](HANDOFF.md) for commands and results, including the checks that are superseded.

## 5. Repository hazard found while verifying

The historical full-suite run could rewrite the frozen roster while preparing a test
fixture. The follow-up isolates the roster and every preparation write in a temporary
root. Do not repair failed checks by overwriting frozen evidence; investigate the
writer. See [review corrections](CODEX_REVIEW.md) for current verification evidence.
