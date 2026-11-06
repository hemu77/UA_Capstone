# Figure review: decisions and results

**What changed:** five metric panels now compare the two observed sources directly. The blue span is the range across ten Facebook records, not a measure of statistical precision. The school diamond is one incomplete survey, not a population estimate. The locked primary tests, original 50-node benchmark and all v2 exports remain unchanged. Six selected supplements remain excluded. This responds to the supplied AI review; it is not independent human peer review. The proposed artwork was not available in the checked-out repositories or remote branches; these replacements were rebuilt from the verified data.

## The sampling result

We asked a narrower question than "which generator is best?": **does the clustering comparison depend on how many people we observe?** For each original 50-node target, valid generated graph and fixed-parameter baseline graph, use one deterministic random ordering, retain the first 25, 40 or all 50 people, and keep every tie between retained people. Isolates stay in. The same 50-node parent is used at all sizes. No model was asked to generate another graph and no baseline was refitted. The raw public inputs and all original adjacency files are hash-checked; all reconstructed 50-node clustering, efficiency and largest-component values and corresponding bank-level distances must match the archived results to absolute tolerance 1e-12.

| People retained | Closest on Facebook clustering (W1) | Closest on school clustering (W1) |
| ---: | --- | --- |
| 50, original comparison | WS, 0.1259 | BA, 0.0416 |
| 40, induced | WS, 0.1352 | BA, 0.0449 |
| 25, induced | BA, 0.1249 | ER, 0.0456 |

Lower W1 means a closer distribution on **one** metric. These are comparisons of seven procedures (four fixed-model distances averaged per LLM procedure; eight banks per classical procedure). Classical generators remain competitive, but the identity of the closest one changes. Global has the smallest efficiency distance on both sources at all three sizes here. Efficiency depends on disconnected pairs as well as on path lengths; it is not human communication quality. On Facebook the closest largest-component procedure changes from iterative at 50 to global at 40 and 25. No composite score or new significance test.

**Interpretation boundary:** these are induced subnetworks, not models prompted for 25 or 40 people. One fixed thinning per original graph at each size is a reproducible diagnostic, not an average over all possible subsets. The 32 target draws share nodes; 22/45 Facebook record pairs also overlap. We did not use the reviewer's 200 resamples to estimate confidence or a chance of winning. Smaller-size control residuals were not evaluated. The original 50-node benchmark remains the main comparison.

**Instruction pooling:** the original seven-condition weights are unchanged. A separate 50-node US-English-only diagnostic keeps identical human targets and fixed baseline banks. The closest procedure on clustering remains WS for Facebook and BA for school; restricting the prompt does not match people, geography, or language. The [school friendship survey](https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) is from Marseille, France. The [SNAP dataset description](https://snap.stanford.edu/data/ego-Facebook.html) does not establish a representative US sampling frame.

## Figure disposition

| Figure | Decision and reading guide |
| --- | --- |
| 01 | Revised. The original 24 contrasts stay visible; the iterative/Hindi panel explicitly flags that only 3/8 block-omission refits pass Holm. Full mean difference -0.04528, Holm p=0.02383, simultaneous interval [-0.08500,-0.00556]. This is about these instructions, not Hindi speakers. |
| 02, 04, 10 | Keep. All three export formats are byte-identical to v2. Respectively: omission refits; failures and identification bounds; full-roster versus sampled metrics. |
| 03 | Revised. Point-plot limits follow observed group means with padding. The three clustering panels share one scale; global density has its own. These are not zero-based bar lengths. |
| 05, 06 | Revised. Zero-valued resource markers have margin. Clustering residual limits include observed negatives and zero without reserving an unused symmetric half-axis. |
| 07 | Revised. Median and interquartile span replace jitter, whiskers and boxes; individual named development records stay visible. The span is not a CI. |
| 08 | Replaced. Five metrics, both sources on the same metric axis. Average four model W1s or eight bank W1s **inside each record first**. Facebook points and ranges then summarize ten records equally. School has no between-record range. |
| 09 | Replaced. Two metrics; one panel per candidate family. Both observed distributions are repeated as references; generated curves are not repeated across sources. Density is zoomed without losing observations. Separate baseline colors and strokes. Distances between these pooled curves do not reproduce figure 08. |
| 11 | Revised. Each residual subtracts the mean of eight matched controls from the same target. Show mean and range across all 32 target residuals. At 50 nodes, mean **degree-swap clustering residuals** are positive in all eleven records: 0.1516 to 0.4259 across Facebook records and 0.2771 for school. Not every draw is positive. G(n,m) density and degree-swap Gini residuals are zero by construction, not empirical successes. |
| 12 | Replaced. All 19 candidates shown per record in five aligned panels on one clustering scale. No invented ordering of records or population error bars. |
| 13 | Moved to appendix. Largest-component fraction is not discarded. ER was connected in 447/448 graphs; BA and WS in 448/448 each. Their W1 scores are nearly tied, not guaranteed identical. |

**Manuscript selection:** lead with 01 (primary contrasts), 04 (coverage), 10 (sampling), 08 (real datasets versus baselines), and 09 (distribution shapes). Keep 02, 03, 05, 06, 07, 11, 12 and 13 as supporting figures. Retain the sensitivity table whenever discussing which procedure was closest.

[Figures and seal](../outputs/publication_figures/20261105_v3/) | [All source-level sensitivity scores](../outputs/publication_figures/20261105_v3/sensitivity_source_summary.csv) | [Bank/record detail](../outputs/publication_figures/20261105_v3/sensitivity_bank_distances.csv) | [Original five-paper design references](PUBLICATION_FIGURES_V2.md)

## Offline checks

Verified for this release: the v2 checker passed before migration (3/7/5/12 embedded PNGs). The v3 checker passed after migration (3/7/6/13 exact images and captions); notebook 01 reran the frozen 24-contrast family and notebook 03 independently checked all 2,640 original W1 values. All 77 focused preparation/analysis/publication tests passed. All 13 SVGs parse, all 13 PDFs have one page, and the three KEEP figures are identical to v2 in each format. Staged checks found 69 unchanged frozen source pins, 1,842 manifest-bound public study blobs, 896 primary attempts, 890 valid primary graphs and zero supplemental benchmark rows.

Use the registered `ua-capstone-final060d` kernel. `scripts/review_publication.py check` executes notebooks 01-04 and verifies the ordered embedded PNGs and captions (3/7/6/13). It needs only the published release. To reproduce the entire revision, run `scripts/review_publication.py build --cache <hash-pinned-public-cache> --dest <unused-directory>` in the pinned environment. The build refuses an existing destination. Run `scripts/check_publication_notebooks.py` on the archived v2 checkout, not on v3 notebooks. No credentials, ledger, provider calls, or population bootstrap are needed.
