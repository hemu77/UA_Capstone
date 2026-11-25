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
