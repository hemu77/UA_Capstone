# Evaluation-Held-Out Public Friendship Networks

This strengthens the real-world **structural** evaluation without new paid
generation. It does not validate named people, national behavior, political
attitudes or human decisions. The original primary dataset remains **896
attempts / 890 valid graphs / six failures**. All six selected supplements are
excluded. The original 24-test analysis and its sealed outputs are unchanged.

## What Is Held Out

The two new public sources are held out from **this benchmark's parameter
fitting**, not necessarily from LLM pretraining. All LLM graphs were already
generated before this benchmark. This is post-collection evaluation, not a
retroactively preregistered primary hypothesis test.

The versioned [protocol](../analysis_heldout/protocol.json) was written before
downloading the graph files. Published overview metadata were already visible.
A separate [review clarification](../analysis_heldout/clarifications.json)
fixed empty-graph conventions, weighting after failures, and conditional-control
aggregation after downloads but before graph parsing, fitting or benchmark
outcomes. Both versions are preserved and SHA-256 bound in the results.

## Public Sources And Attribution

**SNAP Facebook:** [official dataset](https://snap.stanford.edu/data/ego-Facebook.html)
and [file descriptions](https://snap.stanford.edu/data/readme-Ego.txt), from
McAuley and Leskovec, *Learning to Discover Social Circles in Ego Networks*,
NIPS 2012. The ten alter graphs exclude the ego. We use only roster IDs and
friendship edges, never profile attributes. Keep feature-roster isolates and
collapse repeated edge orientations. Observed rosters range from 59 to 1,045
nodes. Twenty-two of 45 cohort pairs share nodes, with up to 44 shared nodes;
these are not ten independent populations. The official download is public;
we do not infer an additional redistribution license or republish its raw IDs
and features. Source-member filenames are retained as provenance identifiers.

**SocioPatterns, Marseille, France:** [official friendship-survey dataset](https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/),
Mastrandrea, Fournet and Barrat, [PLoS ONE 10(9):e0136497](https://doi.org/10.1371/journal.pone.0136497),
2015. Source data carry **CC BY-NC-SA 3.0** terms. Only aggregate benchmark
results are published here; retain attribution and source terms when reusing
the data. We use reported friendship, **not** sensor proximity or contact
diaries. The observable positive-reporting-nominator roster has 133 nodes;
restricting nominations to it and taking an undirected OR projection gives
403 edges. Three of 668 nomination rows point outside that roster. The edge
list cannot recover zero-nominating/nonresponding students; this is not a
complete school friendship census, nor a representative sample.

## Design And Baselines

For each of the 11 observed source records, take **32 deterministic uniform
50-node induced draws**, retaining isolates, empty and disconnected outcomes.
The 352 draws are overlapping samples from two studies, **not 352 independent
human populations**. This controls node count while making the remaining
sampling/coverage and domain differences visible.

Fit classical generators only to the old eight-network development panel.
Average the two Tailor waves first, then weight seven source communities equally.
The fitted settings are ER `p=0.1889356947`, BA `m=5`, WS `k=10, p=0.2`.
WS rewiring is chosen from the fixed eight-value grid using 64 development-only
draws per value. New public outcomes never enter parameter fitting.

Each generator supplies eight Monte Carlo banks of 56 graphs: **1,344 baseline
graphs total**. Compare them with the 16 fixed method/model groups from the
original primary study. Each LLM group originally has 56 attempts across seven
instruction conditions. Give each condition weight 1/7, distributed over its
valid graphs, to avoid changing the condition mixture after failures. Retain
failure denominators: distances condition on valid outcomes, not universal
generation success. No failed graph is treated as a zero-valued graph.

Report **Wasserstein-1 distance separately** for density, clustering,
efficiency, transitivity, degree Gini and largest-component fraction. All are
bounded graph-level metrics. Average cohort distances equally within each
source; keep the two sources separate. Baseline bank ranges measure simulation
sensitivity, not confidence intervals. No new p-values or optimized composite
winner are reported. Empty 50-node graphs have zero Gini by the declared new
benchmark convention and largest-component fraction 1/50; primary metrics are
not rewritten.

Separately, each held-out draw has eight target-matched **G(n,m)** and eight
**degree-preserving swap** controls: **5,632 conditional draws**, zero failures
in this run. They use held-out edge/degree information and **never enter the
transfer ranking**. Finite swaps do not prove uniform mixing. A residual needs
all eight controls and a cohort mean needs all 32 targets; missing controls
would remain missing rather than overweighting easier graphs.

## Findings In Simple Words

There is **no universally closest procedure**. Results change with metric and
source. Classical baselines remain competitive, which prevents a story built
only around LLM-versus-LLM comparisons.

For illustration, mean clustering-distribution distances are below. LLM rows
average the four fixed models; baseline rows average eight banks. Lower means
closer on clustering only. These are descriptive distances, **not significance
tests, equivalence tests, or a model-quality leaderboard**.

| Procedure | Facebook records | School observed roster |
| --- | ---: | ---: |
| Global | 0.1847 | 0.1067 |
| Local | 0.1716 | 0.2045 |
| Sequential | 0.1678 | 0.1991 |
| Iterative | 0.2687 | 0.3347 |
| Development ER | 0.2094 | 0.1298 |
| Development BA | 0.1454 | 0.0416 |
| Development WS | 0.1259 | 0.0813 |

WS is closer on this metric than each four-model LLM procedure average for
Facebook; BA is closer for the school roster. Other metrics tell a different
story: global graphs better match sparse density/efficiency here, while some
local/sequential or iterative graphs better match transitivity or degree
inequality. Dense legacy development graphs also disadvantage those frozen
baseline settings on sparse new sources. Do not retune them on these results
and continue calling the same data held out.

Uniform node sampling can materially lower observed clustering; the sampling
figure shows full-roster values alongside the 50-node distributions. Observed
clustering remains above the finite degree-swap controls in every source record:
mean degree-swap clustering residuals span **0.1516-0.4259** for Facebook and are **0.2771** for the
school roster. This is evidence about these observed topologies, not proof
that the LLMs match human mechanisms or that these effects generalize.

## Reading The Figures

1. **Transfer distances:** revision 3 puts both sources on the same axis for each
   of five metrics. Facebook spans are ranges of ten record-specific procedure
   values, not CIs or bank ranges; school is one record. Largest-component
   fraction remains available in appendix figure 13. Each column is a
   separate estimand; do not average columns into an invented realism score.
2. **Sampling shift:** open circles are the complete observed rosters; diamonds
   are 50-node-draw means. Lines show the minimum-to-maximum node-draw range,
   **not a CI**. Sampling effects and source coverage must remain visible.
3. **Conditional structure:** mean and range of the 32 paired observed-minus-control
   values for clustering and transitivity. Positive values mean excess closure relative to that control,
   not a p-value or human-behavior validation. Both controls are target informed.
4. **Cohort sensitivity:** do not hide differences among source records behind
   a source mean. Facebook records can share people and edges.
5. **Distributions:** ECDFs compare both observed sources with the LLMs or classical
   baselines for density and clustering. The Facebook mixture is descriptive, not a replacement for
   cohort-wise W1. No population uncertainty bands are inferred.

See the [executed notebook](../analysis/03_heldout_public_evaluation.ipynb),
[sealed results](../outputs/heldout_evaluation/20261105_v1/evaluation_manifest.json),
[all distances](../outputs/heldout_evaluation/20261105_v1/per_cohort_distances.csv)
and the [review response and matched-size findings](FIGURE_REVIEW_RESPONSE_V3.md).
The [verification receipt](HELDOUT_VERIFICATION.md) distinguishes completed
checks from remaining scientific limits.
Both earlier figure sets are archived unchanged. Notebooks 01-04 now display
the [revision-3 set](../outputs/publication_figures/20261105_v3/) directly.
The original 50-node results do not imply size-invariant rankings: on matched
25-node induced subgraphs, the closest clustering baselines change on both sources.

## Reproduce Without Paid Calls

Use the pinned Python 3.11.9 environment and registered notebook kernel:

```powershell
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\fetch_heldout_public.py --cache data_external\heldout_20261105
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\run_heldout_evaluation.py --cache data_external\heldout_20261105
rtk proxy .\.venv\Scripts\python.exe -I -B scripts\review_publication.py check
```

The fetcher accepts only the two pinned public downloads, bounds sizes,
rejects off-host/insecure redirects and refuses differing existing bytes.
The archive parser never extracts files to disk. Raw source caches are ignored
by Git; no API key, provider call or SQLite access is needed. Notebook-only
reproduction and independent checks of all 2,640 distances use published tables
without downloads. The original primary seal, its artifact/code hashes and 69
frozen source pins are checked before/after the full benchmark.

**Still not established:** pretraining-unseen evaluation, matched-person link
prediction, human decision fidelity, national representativeness, translation
equivalence, uniformly mixed swap nulls, or population-level significance.
