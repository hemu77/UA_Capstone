# Prompt For Claude: Methods And Actual Figure Review

You are an independent computational-social-science, statistics and scientific
visualization reviewer. Audit this repository critically; do not rubber-stamp
the work because tests pass or because AI agents reviewed it. No paid provider
calls, credential/ledger access, graph regeneration, or changes to frozen data.

Start with the [November 11 response](DEEP_REVIEW_RESPONSE.md). Recalculate the 88 target-density ER bank W1s and average by record, not from pooled ECDFs. Separate Facebook and school; verify the two clustering margins and the 90-to-85 change under the alternative mixing score. Look for wording that turns a descriptive gap into a causal explanation or an attribute ranking into independent importance.

**First deliverable: audit the [professor's brief](PROFESSOR_RESEARCH_BRIEF.md).**
Return a six-row checklist: comparative plots, effect sizes/significance, failure
analysis, limitations, strongest finding, and proposed contribution. Mark each
supported, needs revision, or missing; cite the exact table row or figure panel.
For the four headline comparisons, independently find the minimum within each
candidate family in the 50-node, original-weighting rows (not the US-English-only
rows). Reconcile the three failure counts to 5/6/7 complete blocks and distinguish
an untestable contrast from a nonsignificant one. Challenge the proposed novelty
against Chang et al. and the other four supplied papers: what is inherited,
what is demonstrated here, and what would require a new study? Recommend one
headline in plain English, one claim to omit, and the most useful next experiment.
Review only; do not edit files or launch any study run.

Start with `ANALYSIS_START_HERE.md`, `docs/HELDOUT_REAL_WORLD_EVALUATION.md`,
`docs/PRIMARY_ANALYSIS_FINDINGS.md`, `docs/FIGURE_REVIEW_RESPONSE_V3.md`, `docs/PUBLICATION_FIGURES_V2.md`,
`analysis/README.md`, and executed notebooks 01 through 05.
Read both evaluation manifests. The six selectively completed supplemental
graphs must remain outside every primary test and held-out benchmark.

**Inspect actual plots, not just source code or captions.** Open the rendered
15 figures: v4 overrides 08, 11, 14 and adds 15; the rest remain in v3. Inspect their PNG and vector versions.
The manuscript-v1 and heldout-v1 figures are archived comparisons, not current
displays. Read the five supplied papers
and inspect their figures as visual references, using the exact versions and
pages in the revision-2 reference notes. Do not copy source graphics, adopt their
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
   text size at manuscript width, panel alignment and clipping. The replacement
   uses dot/interval panels and ECDFs: are there too many panels/curves or
   overly small marks? Verify all 24 primary contrasts, individual block dots,
   all 16 model/method groups and three baselines. Ensure the descriptive
   Facebook mixture ECDF does not replace mean cohort-wise W1. Are point estimates wrongly made to look
   precise? Are repeated disclaimers, decorative titles or overpacked legends
   still obscuring the data? Identify the exact panel and concrete remedy.
7. **Review-response evidence.** Recompute the within-record averaging and ranges in 08;
   contrast these with bank ranges and population uncertainty. Verify the 32 paired
   residuals in 11, and ER 447/448 versus BA/WS 448/448 connected in the LCC appendix.
   Assess nested 25/40/50-node sensitivity on targets AND candidates, reconstructed
   50-node anchors, a single thinning per parent, US-English-only weights, and the
   absence of independent population replicates. Do not infer native small-network
   generation quality or treat either dataset as geographic validation.
8. **Actual delivery.** Run `scripts/check_deep_review.py` with the
   registered frozen kernel. Verify the notebooks embed exactly 3, 7, 7, 15 and 1 figures with current captions. Check gallery links, the 3 unchanged figure
   copies, the untouched v2 release, and the new v3 seal.

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
