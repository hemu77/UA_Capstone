# Prompt For Claude: Methods And Actual Figure Review

You are an independent computational-social-science, statistics and scientific
visualization reviewer. Audit this repository critically; do not rubber-stamp
the work because tests pass or because AI agents reviewed it. No paid provider
calls, credential/ledger access, graph regeneration, or changes to frozen data.

Start with `ANALYSIS_START_HERE.md`, `docs/HELDOUT_REAL_WORLD_EVALUATION.md`,
`docs/PRIMARY_ANALYSIS_FINDINGS.md`, `docs/MANUSCRIPT_FIGURE_CAPTIONS.md`,
`docs/VISUAL_REFERENCE_NOTES.md`, and executed notebooks 01, 03 and 04.
Read both evaluation manifests. The six selectively completed supplemental
graphs must remain outside every primary test and held-out benchmark.

**Inspect actual plots, not just source code or captions.** Open the rendered
PNG and SVG/PDF figures in `outputs/manuscript_figures/20261105_v1/` and
`outputs/heldout_evaluation/20261105_v1/figures/`. Read the five supplied papers
and inspect their figures as visual references, using the exact versions and
pages in the visual-reference notes. Do not copy source graphics, adopt their
uncertainty assumptions, or force static graph results into political-compass
or dynamic-intervention plots.

Audit the following independently:

1. **Held-out meaning and leakage.** Verify that ER/BA/WS settings use only the
   old development panel, with Tailor waves averaged before seven equal
   community weights. New public data are excluded from fitting. Challenge any
   implication of preregistration, pretraining-unseen evaluation or matched
   human link prediction. Review the original protocol plus timed clarification.
2. **Data semantics and sampling.** Check hash-pinned SNAP alter rosters,
   retained isolates, excluded egos and edge deduplication. Check the school
   positive-reporting-nominator roster, unknown nonresponses and three removed
   outside-roster nominations. Uniform 50-node induced draws overlap; 22/45
   Facebook cohort pairs share nodes. Do not count 352 draws as independent
   communities or treat two studies as a representative human sample.
3. **Weights and failures.** Reproduce condition-balanced W1 from the saved
   graph metrics, including failure-affected groups. Retain 896 attempts,
   890 valid graphs and six primary failures. Verify empty graph conventions,
   exact graph identifiers and no supplementary leakage. A distance conditional
   on successful graphs is not an unconditional generation-quality score.
4. **Baselines and controls.** Verify eight 56-graph banks per development
   baseline and the fixed WS grid. Explain why development-domain density can
   disadvantage classical transfer. Target-informed G(n,m)/degree swaps are
   conditional diagnostics, never transfer competitors. Check exact degree
   invariants, finite-chain mixing limits and all-or-missing residual coverage.
   Monte Carlo bank ranges and node-draw ranges are not population CIs.
5. **Statistical claims.** Retain the locked 24-test family, eight paired
   schedule blocks, four fixed models, minimum six complete blocks, full Holm
   adjustment and family-simultaneous intervals. Verify the Hindi result and
   its 3/8 leave-one-seed-out fragility. No equivalence, national-cultural,
   causal, polarization or universal-best-model claims should follow.
6. **Plot quality.** For EACH figure, return **KEEP / REVISE / REPLACE / REMOVE**.
   Assess the scientific question, plot type, denominator, units, axis ranges,
   uncertainty meaning, missing data, legend, ordering, color accessibility,
   text size at manuscript width, panel alignment and clipping. Is a dense
   heatmap actually useful, or would small-multiple dot/interval/distribution
   plots make the evidence clearer? Are point estimates wrongly made to look
   precise? Are repeated disclaimers, decorative titles or overpacked legends
   still obscuring the data? Identify the exact panel and concrete remedy.

Return findings first, ordered by severity with file/line or figure/panel
references and reproducible evidence. Separate numeric defects, design limits,
unsupported claims and stylistic improvements. Independently recompute at least
one weighted W1, the baseline settings and the main Hindi/robustness numbers.
State precisely which artifacts you could not inspect or rerun.

Then propose a short, prioritized figure-replacement plan. If implementation is
authorized, write corrected/replacement figures in a NEW versioned directory,
preserve original inputs/figures, update notebook/captions, and show rendered
before/after examples. Run focused tests and execute the notebook; verify all
old seals remain unchanged. Do not change tests or thresholds to obtain a more
favorable finding, and do not turn exploratory results into confirmatory ones.

Finish with simple-word findings we can responsibly tell the professor,
remaining real-world validation gaps, and which plots should enter the paper
versus appendix. Do not guarantee a venue outcome or describe this AI review
as independent human peer review.
