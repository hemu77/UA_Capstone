# Robustness checks v3 (exploratory, offline)

Release: `outputs/research_extension/20261125_v3/`. Rebuild into a new directory with
`python scripts/run_robustness_checks.py --output outputs/research_extension/<new>` (about one minute).

Boundary: primary collection only (896 attempts, 890 valid graphs including 12 empty, 6 failures). No supplemental
graph, no new generation, no paid or provider call, no ledger query. Not preregistered. Earlier releases are unchanged;
the script refuses to write into an existing directory and checks the protected hashes.

## 1. Density-informed ER on all six statistics (11 test records, uniform 50-node draws)

Mean W1 of an ER generator given each record's own density, against the best transfer candidate:

| Statistic | Target-density ER | Best transfer candidate | Records where target ER is closer |
|---|---|---|---|
| Global efficiency | 0.153 | Global 0.119 | 4 / 11 |
| Largest-component fraction | 0.342 | Iterative 0.339 | 2 / 11 |
| Degree Gini | 0.196 | Iterative 0.165 | 5 / 11 |
| Mean clustering | 0.309 | WS 0.122 | 0 / 11 |
| Transitivity | 0.426 | Local 0.109 | 0 / 11 |

Density carries most of the agreement on reachability and none of the agreement on closure. The efficiency column
reproduces the released `efficiency_records.csv` to 1e-9 (asserted in code).

## 2. Uniform versus snowball sampling of the same records

Snowball: breadth-first from a seeded random node to 50 nodes, restart if a component is exhausted; 32 draws per record.
Mean draw statistics over records: uniform density 0.083, clustering 0.383, efficiency 0.197; snowball 0.320, 0.651,
0.594. Closest candidate on efficiency: uniform Global 9, Iterative 2; snowball ER 8, Iterative 3, Global 0.
Closest on clustering under snowball: Iterative 9 of 11. Figure: `20_sampling_scheme.{pdf,png,svg}`.

## 3. Mixing conditional on the other two attributes

500 permutations per nonempty graph of one attribute's labels within strata of the other two. Conditional excess =
observed same-group edge share minus permutation mean; 28 model-setting cells per procedure.

| Procedure | Politics | Religion | Gender | First in 28 cells |
|---|---|---|---|---|
| Global | 0.251 | 0.496 | 0.167 | religion 28 |
| Local | 0.310 | 0.204 | 0.083 | politics 28 |
| Sequential | 0.280 | 0.202 | 0.071 | politics 28 |
| Iterative | 0.385 | 0.110 | 0.052 | politics 28 |

## What these checks cannot remove

One roster, fixed prompt field order, no shuffled-attribute generation control, procedures not compute-matched, eight
schedule blocks. Each needs new generation.

## Paper

`paper/icml2026/` holds the manuscript source that reports these results (figures are read from
`outputs/publication_figures/20261118_v6_2/` and this release).

---

# Tie-level checks v4 (exploratory, offline)

Release: `outputs/research_extension/20261125_v4/`. Rebuild with
`python scripts/run_tie_level_checks.py --output outputs/research_extension/<new>` (under ten seconds). Same boundary as v3.

## 4. Joint dyadic model (878 nonempty graphs, 1,225 pairs each)

Ridge-penalised logistic regression (C = 10) of tie presence on shared politics, shared religion, shared gender and
age gap. Mean log-odds; 95% block-bootstrap interval over eight schedule blocks for politics minus religion.

| Procedure | Politics | Religion | Gender | Politics minus religion | Cells with politics largest (of 28) |
|---|---|---|---|---|---|
| Global | 2.16 | 3.79 | 1.59 | -1.63 [-1.88, -1.35] | 0 |
| Local | 2.67 | 2.24 | 0.82 | 0.43 [0.29, 0.58] | 22 |
| Sequential | 2.34 | 2.09 | 0.70 | 0.25 [0.15, 0.35] | 20 |
| Iterative | 3.51 | 1.48 | 0.58 | 2.03 [1.82, 2.28] | 28 |

The sign of the gap agrees in 8 of 8 blocks for every procedure. Note the qualification: under the joint model politics
leads in 22 and 20 cells for Local and Sequential, not 28. Eight blocks make the bootstrap intervals optimistic.

## 5. Tie overlap across settings

Jaccard overlap of edge sets between runs on different blocks. Within a setting: 0.32 to 0.41. Reference versus another
setting, as a fraction of the within-setting value: Global 1.01, Local 0.99, Iterative 0.99, Sequential 0.97
(Sequential under Hindi 0.91, Japanese 0.94).

## 6. Detectable differences for the planned contrasts

80% power, two-sided paired t at 0.05/24, observed block variability. Median detectable clustering difference: Local
0.029, Sequential 0.045, Iterative 0.051 (6% to 9% of the reference mean). Global density: 0.031 against a reference
mean of 0.059, so the Global nulls are not informative about moderate effects.
