# Publication Figures, Revision 2

The main notebooks 01, 02, 03 and 04 now display this set directly. Revision 1
and all original statistics remain intact for provenance, not as the recommended
presentation. These are programmatic plots of verified results, not generated
illustrations. No provider calls, new hypothesis tests, or supplemental graphs.

## Reference Decisions

All five supplied links were revisited. Publisher access was blocked for the
ICWSM, Nature and PNAS pages; identified author PDFs were used for the actual
figure inspection, not assumed to be identical to every publisher revision.

| Actual figure inspected | Concrete change in our figures |
| --- | --- |
| [Chang et al., Fig. 3, PDF p. 5](https://arxiv.org/pdf/2408.16629v2) | Small metric panels with visible observations; comparison is no longer a wall of colored cells. Our intervals remain our own 24-family intervals, not their standard errors. |
| [Piao et al., Fig. 2, author PDF p. 18](https://arxiv.org/pdf/2501.05171v3) | Consistent panel lettering, compact composition and one visual role per color. No invented time series or intervention diagram. |
| [Papachristou and Yuan, Fig. 1, author PDF p. 4](https://arxiv.org/pdf/2402.10659v7) | Explicit reference lines and separate panels for different metrics. Observed distributions are shown, not replaced by a composite score. |
| [Helwe et al., Appendix E Fig. 1, PDF p. 16](https://aclanthology.org/2025.findings-acl.883.pdf) | Aligned repeated panels with consistent scales for the same quantity. No political-compass geometry for network endpoints. |
| [Argyle et al., Fig. 4, PDF p. 11](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/035D7C8A55B237942FB6DBAD7CAA4E49/S1047198723000025a.pdf/out_of_one_many_using_language_models_to_simulate_human_samples.pdf) | Directly aligned source comparisons and restrained marks; no implied matched-person validation. |

The old notebooks were a delivery problem as well as a design problem: notebook
02 regenerated the original dashboard-like figures even after notebook 04 was
added. Revision 2 replaces those displays in place. The old renderer and its
sealed outputs are not edited. No source artwork is copied or republished.

## 01_primary_contrasts
**Instruction contrasts.** All 24 planned contrasts are retained. Colored points
are complete-block means, horizontal lines are the archived 95% simultaneous
Bonferroni t intervals for the 24-contrast family, and faint dots are individual
complete paired blocks. Each block averages the same four fixed models before
subtracting same-seed US-English. The top three rows change country framing;
the bottom three change language while retaining US framing. Most rows have
eight blocks; the three exceptions are labeled. Global Japanese has five and
therefore no interval or primary test. Global uses density; the other panels
use clustering and share an axis range. Zero denotes no difference, not
equivalence. Assumptions and missingness limits remain unchanged.

## 02_primary_robustness
**Sensitivity of iterative US-Hindi.** Left: all eight paired block differences.
Right: means after omitting the indicated block; blocks 1-8 correspond to seeds
44000-44007. The dashed line is the full-data mean, -0.045279. Filled refit
markers pass the full-family Holm threshold, open markers do not (3/8 pass).
No refit intervals are shown. Overlapping leave-one-out fits are not independent
replications, and 3/8 is not a probability of replication. The separate exact
sign-flip diagnostic has adjusted p=0.1875. This selected sensitivity view is
not a new confirmatory analysis or a statement about Hindi speakers.

## 03_model_profiles
**Fixed-model endpoint means.** Colors identify methods and shapes identify
the four fixed model configurations consistently. Each point averages valid,
defined endpoints in one eight-attempt cell. The US-English baseline is above
the separator. No confidence intervals or model rankings are implied. Global
uses density, other methods clustering; their quantities are not interchangeable.
Defined counts below eight are global Japanese for the three non-GPT-4.1
configurations (7 each), global Portuguese/gpt-5.6-luna (6), and iterative
Japanese/gpt-4.1 (7). Exact counts remain in descriptive_metrics.csv.

## 04_coverage_missingness
**Failures and missingness.** Left: all 896 original attempts, with six failures
hatched rather than erased. All 12 valid empty graphs remain valid outcomes.
Right: all three contrasts with fewer than eight complete blocks. Brown spans
allow unavailable endpoints anywhere in [0,1]; these are identification bounds,
NOT confidence intervals. Dots are complete-block means. The other 21 contrasts
have complete data and collapsed missingness bounds, not zero uncertainty.
Global bounds are density differences and iterative bounds clustering differences.

## 05_resource_use
**Compute is unequal.** Every point is one fixed model/method group (56 attempts),
including failed attempts. Received-cost estimates exclude unallocated retained
reservations and are not invoices. Serial provider hours are not wall runtime.
The original resource totals are unchanged; no efficiency winner is inferred.

## 06_gnm_context
**Generated graphs versus matched random-edge references.** Faint dots show
observed clustering minus the mean of eight same-node, same-edge G(n,m) draws
for each valid, defined primary graph. Larger shapes show model/method means.
All seven instructions are pooled without new reweighting, as in the archived
primary figure. Zero is equality to that synthetic mean, not a realism threshold.
No confidence intervals or new tests. This is not the held-out transfer ranking.

## 07_legacy_comparison
**Original development-panel context.** Primary distributions are unweighted
valid, defined observations pooled over fixed models and instructions. Dots
are graphs; boxes are medians/IQRs with 1.5-IQR whiskers, not confidence intervals.
All observations, including outliers and valid empty graphs, remain visible.
Diamonds are the eight individually named legacy records. Sizes (31-70 versus
50), tie semantics and dependence differ; Tailor waves share workers. This is
development context, not an independent held-out test. Archived primary metric
columns are used without substituting new recomputed values.

## 08_heldout_distances
**Transfer distances without a heatmap.** Every one of the 16 fixed LLM groups
and three classical baselines appears for all six metrics and both sources.
Within each LLM group, the seven instructions retain equal weight over valid
outcomes. Failures remain in the attempted denominator, never zero-filled.
Each source value averages its cohort distances equally. Lower Wasserstein-1
distance is closer on that metric only. Axes start at zero and are shared
between the two sources within each metric, not forced to the same maximum
across different metrics. Baseline diamonds are eight-bank means; tiny lines
show bank minima/maxima (simulation spread, NOT CIs). No aggregate winner.

## 09_heldout_distributions
**What the distance summaries hide.** Empirical cumulative distributions for
density, clustering and efficiency: the height is the fraction at or below
the horizontal value. Black curves are equal-cohort mixtures of observed
50-node draws; colors are equal mixtures of the four fixed models, each
retaining equal instruction weights over valid graphs. Gray curves pool eight
equal-sized banks per development-fitted baseline. All valid empty graphs stay
in the curves; failures do not become zeros. The Facebook mixture curve is
descriptive: distance to this mixture is NOT substituted for the benchmark's
mean of cohort-wise distances. These are not confidence bands or new tests.

## 10_sampling_effect
**Sampling changes the reference.** Open circles show the full observable
roster; diamonds show the average over 32 induced 50-node draws. Light spans
are minima/maxima of those draws, NOT CIs. Lines join full and sampled means
for the same record. The observable school roster excludes unrecoverable
nonresponders; Facebook cohorts and within-cohort samples overlap.

## 11_conditional_controls
**Closure beyond simple graph constraints.** Points are mean observed-minus-
control coefficients. Gray squares condition on edge count; blue circles
condition on degrees through finite swaps. Every target needs all eight controls
and each cohort all 32 targets. These target-informed controls are separate
from transfer baselines. Positive residuals describe topology, not significance,
uniform mixing, or proof that LLMs model human mechanisms.

## 12_cohort_sensitivity
**Show individual source records, not only source averages.** Left: clustering
distances averaged across four fixed models within each LLM procedure. Right:
baseline distances averaged across eight banks. All eleven source records are
retained, including the school record below the separator. Shapes and colors
identify series; no lines imply a temporal sequence. Facebook records can
share people and ties. These means are descriptive, with no p-values or CIs.

## Reproduce And Review

Render editable SVG/PDF and PNG with `scripts/run_publication_figures.py`.
The renderer reads verified CSVs only, preserves all old seals and 69 frozen
sources, and writes to `outputs/publication_figures/20261105_v2/`.
`scripts/check_publication_notebooks.py` executes notebooks 01-04, verifies
the exact replacement PNGs are embedded, and checks all protected bytes before
and after. The original independent statistics and W1 checks remain in 01/03.
No data download, credentials, ledger query, or paid generation is required.

The new visual encodings do not establish national representativeness, human
decision fidelity, pretraining-unseen evaluation, equivalence, or publication
readiness. Original 890 primary graphs and six selected supplements stay separate.
