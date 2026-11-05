# Primary Study: Findings And Limits

## What Was Analyzed

The primary study recorded all **896 planned attempts**: **890 valid graphs**,
including **12 empty graphs**, and **six failures**. Empty graphs are outcomes,
not missing attempts. The six subsequently selected supplemental graphs are
stored separately and enter **none** of this analysis's hypothesis tests.

This is one fictional 50-person roster, four fixed model configurations,
four generation methods and seven instruction conditions, with eight schedule
blocks. A block pairs the same seed's target and US-English graphs across all
four models. The independent inferential unit is the paired block, not each
person, edge, model row or generated response. Independence and t-distribution
assumptions remain assumptions, not facts established by a passing software test.

## Main Finding In Simple Words

Of the **24 locked comparisons**, one passes the original Holm-adjusted t-test
threshold: iterative generation under the Hindi instruction has mean clustering
**0.04528 lower** than under US-English. The adjusted p-value is **0.02383**;
the 95% familywise simultaneous interval is **[-0.08500, -0.00556]**.
Clustering measures how often a person's friends are also friends with each
other. This concerns these particular instructions and model versions, not
Hindi speakers, India or a national cultural trait.

The direction is negative in all eight paired blocks, but the strength of the
significance claim is fragile. Removing one block at a time retains adjusted
significance in only **three of eight** reruns. An exploratory exact sign-flip
check has raw p = **0.0078125** and 24-family adjusted p = **0.1875**.
This diagnostic requires sign symmetry; it is not a randomized causal test.
With eight blocks its p-values are coarse, so lack of adjusted significance
does not settle whether an effect exists.

The other comparisons do not show adjusted evidence in the locked analysis.
That does **not** show that methods, languages or conditions are equivalent.
No equivalence margin or equivalence test was prespecified. Twenty-one
comparisons have all eight blocks, two have six/seven, and global Japanese is
not tested because only five complete blocks remain. Missingness bounds allow
unavailable bounded endpoints anywhere in [0,1]; they are **not confidence
intervals**. Model-specific summaries also use common panel-complete seeds so
different availability is not hidden by per-model averages.

## Evaluation Beyond P-Values

All **890 primary adjacency files** are hash-checked, roster/loop-checked, and
their defined density and clustering values independently recomputed. All
24 primary test calculations are independently cross-checked against the
frozen analysis. Five locally recomputed floating values differ from the
archived runtime by at most 1.11e-16; agreement is numeric to 1e-12, not
byte-for-byte identity. The original outputs and 69 pinned source files are
preserved.

Eight deterministic **G(n,m)** draws per valid graph supply **7,120 synthetic
references** with equal node/edge counts. They ask whether clustering and
efficiency differ from uniform edge placement at the observed density. They
are not people, a fitted social model, an additional primary test, or proof of
realism. Their SD is Monte Carlo spread, not a confidence interval; eight draws
do not establish convergence. Matching density is automatic, not a success.

The additional **real-world structural evaluation** uses the eight existing
friendship networks documented in [Chang et al., Appendix C.3](https://arxiv.org/html/2408.16629v2#A3.SS3).
Exact layer/positive-edge selectors reproduce the published node/edge sizes.
The provenance table records exact local file hashes, CRLF-to-LF-only
cross-checkout hashes, selectors and tie definitions. Three legacy local text
files differ from Git only in newlines; no reference file is rewritten.
Comparisons show density, clustering and efficiency distributions and ECDF
distances, **without KS p-values, fitted winners or realism scores**.

For orientation, these are unweighted means across defined primary graphs,
pooling the four fixed models and seven instruction conditions. The reference
row is an unweighted mean across the eight named records, not a population
estimate. Different denominators and dependence prevent a naive pooled test.

| Group | Defined graphs/records | Density | Clustering | Global efficiency |
| --- | ---: | ---: | ---: | ---: |
| Global | 219 | 0.061 | 0.247 | 0.235 |
| Local | 224 | 0.169 | 0.522 | 0.509 |
| Sequential | 224 | 0.171 | 0.517 | 0.519 |
| Iterative | 223 | 0.173 | 0.653 | 0.411 |
| Eight real-network records | 8 | 0.197 | 0.454 | 0.513 |

In simple words: global graphs are much sparser than this reference-panel mean;
iterative graphs cluster more strongly, without equally high efficiency.
Local/sequential efficiency means are near the reference-panel mean, but that
is **not an equivalence result** or proof that their graphs model real people.
The per-model table and full distributions are essential; means alone hide
variation and no single method matches every aspect by definition.

These are small legacy convenience references, not population-matched ground
truth. Sizes range from 31 to 70 versus our 50 nodes. Two Tailor waves share
workers; Moreno positive ties union repeated ratings. Undirected projection
changes the meaning of directed nominations. The archived Hi-tech projection
matches the published size but does not independently verify all upstream
reciprocity processing. The panel cannot validate national behavior, language
fidelity or political homophily. Structural resemblance is not human behavioral
validation. Treat these comparisons as post-collection diagnostics only.

## Compute And Budget

Primary received-request accounting is **$101.91528510**; including retained
reservations, the historical conservative primary total is **$101.98695215**.
These are not invoices. All nine extra supplemental attempts bring the
historical cumulative conservative total to **$102.634921875**, still below
the original $120 cap; supplemental attempts remain outside primary tests.

Resource tables include failed primary attempts and report tokens, parse errors,
received-cost totals and serial API time. Retained reservations are included
only in the overall total because public evidence cannot defensibly allocate
them among methods. **90,984.973 serial API seconds are not wall-clock time.**
The iterative method uses many more requests; these are not equal-compute
comparisons and cannot establish an efficiency-adjusted method winner.

## Stronger Real-World Evaluation Next

1. **Held-out friendship prediction.** Obtain appropriately consented,
   de-identified networks with compatible friendship definitions and matched
   attributes. Freeze a separate evaluation plan before inspecting held-out
   edges; split by community/person cohort, not randomly by dependent dyads.
   Compare edge prediction/calibration and structural metrics to simple
   density-, degree- and attribute-aware baselines trained only on training data.
2. **Multiple independent rosters and communities.** Test variation across
   genuinely new persona samples and real settings. More draws on this one
   roster do not establish population generalization. Plan power around
   communities/rosters and meaningful effects, not the number of edges.
3. **Language and construct validity.** Use bilingual human review and
   measurement-equivalence checks before claiming language effects. Separate
   country framing, language, demographics and translation wording in a new
   factorial design; do not reinterpret confounded conditions retrospectively.
4. **Behavioral validation.** Compare simulated choices or survey responses
   against held-out human observations with consent, retest reliability and
   subgroup calibration. Polarization or diffusion requires separate temporal
   human validation; a static graph does not establish those dynamics.
5. **Fair baselines and transparent failures.** Add preregistered equal-budget
   comparisons and report all failures, empty graphs and robustness analyses.
   Keep post-hoc diagnostics distinct from the locked 24-test family.

These are recommended future studies, **not completed experiments**. They may
require permissions, ethics review and new data; no such data collection is
authorized or claimed here. The five professor-supplied papers inform framing
and design choices, not a target result or a promised publication outcome.

## Reproduction And Reading Order

- [Setup and separated catalogs](../ANALYSIS_START_HERE.md).
- [Executed primary statistics notebook](../analysis/01_primary_statistical_analysis.ipynb).
- [Executed comparison-figure notebook](../analysis/02_primary_comparison_plots.ipynb).
- [Hash manifest](../outputs/primary_analysis/20261105/evaluation_manifest.json),
  [primary contrasts](../outputs/primary_analysis/20261105/primary_contrasts.csv),
  [sensitivity](../outputs/primary_analysis/20261105/sensitivity.csv),
  [real-network provenance](../outputs/primary_analysis/20261105/empirical_reference_metrics.csv).
- [Five-paper review](PRIMARY_LITERATURE_REVIEW.md) and
  [Claude review prompt](CLAUDE_PRIMARY_ANALYSIS_REVIEW_PROMPT.md).
- [Offline checks and review scope](PRIMARY_ANALYSIS_VERIFICATION.md).

Run `scripts/run_primary_evaluation.py` with the pinned Python environment and
hash-verified release files installed. No API key, paid call or SQLite query is
used. Completed artifact hashes are checked by both notebooks; plot notebook
rerenders go to a temporary directory and cannot rewrite sealed analysis files.
