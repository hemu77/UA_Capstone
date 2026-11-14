# Network Observatory

How do language models generate friendship networks when prompts change country
framing, instruction language, model and generation method? This project extends
[Stanford SNAP's LLM social-network code](https://github.com/snap-stanford/llm-social-network)
using a fixed roster of 50 fictional adults.

**[Open the interactive 3D demo](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html)**
| **[Explore all 896 networks in 3D](https://ua-network-observatory.handm01042024.chatgpt.site/layers.html#saved-plots)**
| **[Read the research brief](docs/PROFESSOR_RESEARCH_BRIEF.md)**

The demo replays saved model decisions. It does not run agents, make paid calls,
or simulate real people. Historical calibration and pilot viewers are archived,
not offered as current study results.

## What Is Included

| Collection | Attempts | Valid graphs | Use |
| --- | ---: | ---: | --- |
| Original primary study | 896 | 890, including 12 empty | Primary statistics and 3D replay |
| Separate supplemental attempts | 9 | 6 selected successes | Separate final-graph 3D browser only |

The **896 selectable networks are 890 primary + 6 supplements**, not 896 primary successes.
All **6 primary failures and 3 supplemental failures** remain recorded. A valid
empty graph is not a failed attempt. Supplements never replace primary observations.

## Start Here

- [Professor's brief](docs/PROFESSOR_RESEARCH_BRIEF.md): research questions, main findings, effect sizes, significance and limitations.
- [Analysis guide](ANALYSIS_START_HERE.md) and [notebook gallery](analysis/README.md): readiness, primary statistics, comparison figures, held-out evaluation and demographic mixing.
- [Findings in plain language](docs/PRIMARY_ANALYSIS_FINDINGS.md) and [held-out evaluation](docs/HELDOUT_REAL_WORLD_EVALUATION.md).
- [All graph files and original PNGs](NETWORKS.md), with a [separate supplemental index](docs/network_catalog/supplemental.md).
- [Research file map](docs/research/README.md): primary data, supplements, failures and historical evidence.

## Repository Layout

| Location | Purpose |
| --- | --- |
| `viewer/` | Current Three.js lab, 896-network browser, public exports and checks |
| `analysis/`, `analysis_primary/`, `analysis_heldout/`, `analysis_figures/` | Notebooks and reproducible analysis code |
| `outputs/final_study/060dcd3a86d5/` | Original primary graphs and collection evidence |
| `outputs/supplement_20261105_v1/`, `outputs/supplement_completion_20261105_v1/` | Separately labeled extra attempts |
| `outputs/analysis_ready/060dcd3a86d5/` | Sealed input catalogs, cohort mapping and failure audit |
| [`viewer/archive/`](viewer/archive/README.md) | Historical viewer source, calibration exports and pilot figures; not deployed |
| [`docs/history/`](docs/history/README.md) | Superseded operational notes and earlier deployment documentation |

Frozen evidence stays at its original paths so hashes, manifests and notebooks
continue to resolve. Large response records are in the
[verified release](https://github.com/hemu77/UA_Capstone/releases/tag/main-study-20261105);
all 896 valid graph adjacency files and original PNGs are in this repository.

## Findings Without Overclaiming

Classical baselines remain competitive; rankings depend on metric and network size.
Density explains much of the apparent reachability advantage. The corrected Hindi
contrast is fragile under leave-one-block-out checks. These are model outputs from
one fixed roster, not representative human samples or evidence of cultural causation.
See the brief for estimates, uncertainty, failure consequences and caveats.

## Run The Demo Locally

```powershell
cd viewer
npm ci
npm test
npm run build
npm run verify:public
python -m http.server 8781 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8781`. The build uses local saved files; it does not call models.
See [deployment and browser checks](docs/SIMULATOR_DEPLOYMENT.md) and the
[analysis reproduction guide](ANALYSIS_START_HERE.md) for the pinned Python setup.
