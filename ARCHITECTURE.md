# Repository Architecture

This document explains the project as a beginner-friendly pipeline.

## Core Idea

The repo simulates a social network using LLMs.

It does that in four stages:

1. define the people in the network
2. ask a model who should become friends with whom
3. save the resulting graph and visual outputs
4. measure homophily and graph structure

## Main Files And Responsibilities

[generate_personas.py](generate_personas.py)
Creates or enriches personas. This is the "who exists?" layer.

[generate_networks.py](generate_networks.py)
Main generation engine. This is the "who becomes friends?" layer.

[constants_and_utils.py](constants_and_utils.py)
Shared infrastructure. It handles paths, API calls, retries, saving graphs, and drawing PNGs.

[analyze_networks.py](analyze_networks.py)
Main metrics layer. This is the "what kind of network came out?" layer.

[plotting.py](plotting.py)
Visualization helpers for graphs and analysis tables.

[run_cultural_study.py](run_cultural_study.py)
Step 2 experiment orchestrator. It runs the full culture/model/seed matrix and writes aggregate outputs.

[study_runner_utils.py](study_runner_utils.py)
Shared experiment helper layer. This keeps the Step 2, Step 3, and Step 4 runners on the same generation, aggregation, and verification path.

[run_method_study.py](run_method_study.py)
Step 3 experiment orchestrator. It compares `global`, `local`, and `iterative` under the same culture/model/seed setup used for Step 2.

[run_language_study.py](run_language_study.py)
Step 4 experiment orchestrator. It keeps culture fixed and varies the prompt language across English, Spanish, Hindi, and Japanese.

[analyze_networks.ipynb](analyze_networks.ipynb)
Exploratory notebook. This is where results are compared, plotted, and interpreted interactively.

[network_datasets.py](network_datasets.py)
Loads or converts real/reference datasets for comparison.

[bias.py](bias.py)
Bias-oriented text analysis helpers.

## Data Flow

The normal flow is:

1. Personas live in `text-files/*.json`
2. Generation creates graph files in `text-files/*.adj`
3. Graph images are written to `plots/*.png`
4. Analysis writes per-condition CSVs into `stats/<condition>/`
5. The Step 2 runner writes study-level summary files into `stats/cultural_study/`
6. The Step 3 runner writes study-level summary files into `stats/method_study/`
7. The Step 4 runner writes study-level summary files into `stats/language_study/`

## Step 1 Changes

Step 1 was about compatibility and getting the repo runnable with your setup.

Changes made:
- OpenAI-only authentication now works with `OPENAI_API_KEY`
- a Llama key is only required if a non-OpenAI model is actually requested
- default model path was updated to `gpt-4.1-mini`
- saving plots/graphs now creates directories automatically
- a plotting bug exposed during verification was fixed

## Step 2 Changes

Step 2 added the first controlled cultural-context experiment.

Changes made:
- `generate_networks.py` now accepts `--culture_context`
- the prompt can now say the network is set in `us`, `india`, `japan`, or `brazil`
- language remains fixed to English
- `run_cultural_study.py` automates the full study matrix
- aggregate outputs summarize demographic dominance and inter-model divergence

In the final capstone framing, Step 2 is not the whole story for the first three research questions. It is the sequential baseline that later gets combined with Step 3.

## Step 3 Changes

Step 3 expanded the project beyond sequential generation.

Changes made:
- `run_method_study.py` automates the full method-comparison matrix
- `global`, `local`, and `iterative` now have the same study-level aggregation and verification path as Step 2
- aggregate outputs summarize method effects, demographic dominance, and model divergence

This is what turns RQ1, RQ2, and RQ3 into four-method results instead of sequential-only results.

## Step 4 Changes

Step 4 added prompt-language variation while keeping culture fixed.

Changes made:
- `generate_networks.py` now accepts `--prompt_language`
- prompt instructions and persona labels can be rendered in English, Spanish, Hindi, or Japanese
- `run_language_study.py` automates the fixed-culture language study
- aggregate outputs summarize cross-language shifts in homophily and topology

In the final project write-up, this becomes Research Question 4.

## Full Experiment Summary

The final capstone is one connected experiment, not four unrelated scripts.

The structure is:

1. Step 1 makes the original repo runnable with OpenAI only.
2. Step 2 runs the first cultural study using the `sequential` method.
3. Step 3 adds the other three methods, `global`, `local`, and `iterative`, so RQ1 to RQ3 are supported by all four methods together.
4. Step 4 keeps culture fixed and changes prompt language, which becomes RQ4.

Shared experimental settings:
- one fixed 50-person roster from `text-files/us_50_gpt4o_w_interests.json`
- three GPT models: `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`
- two seeds per condition

Cultural contexts used:
- `us`
- `india`
- `japan`
- `brazil`

Prompt languages used:
- `english`
- `spanish`
- `hindi`
- `japanese`

Method coverage:
- `sequential`
- `global`
- `local`
- `iterative`

Verification status after the final refresh:
- Step 2 cultural study: `24/24` conditions passed
- Step 3 method study: `72/72` conditions passed
- Step 4 language study: `96/96` conditions passed

## Report Format

If you are writing the final capstone report or proposal follow-up, this is the cleanest structure:

1. Introduction
   Explain that the project studies how LLM-generated social networks change under controlled variations in culture, method, model, and prompt language.

2. Research Questions
   Present the four RQs exactly:
   - RQ1: cultural context effects with language held constant
   - RQ2: dominant demographic dimensions in tie formation
   - RQ3: consistency or divergence across LLM models
   - RQ4: prompt-language effects with culture held constant

3. Experimental Design
   State:
   - same 50 personas reused across conditions
   - same three GPT models across studies
   - Step 2 plus Step 3 together answer RQ1 to RQ3 across all four methods
   - Step 4 answers RQ4 by fixing culture to `us` and varying prompt language

4. Methods
   Describe the four generation methods:
   - `sequential`: people choose friends one at a time as the network grows
   - `global`: the model proposes friendship pairs for the whole network at once
   - `local`: one focal person chooses from the candidate list without the full sequential buildup
   - `iterative`: the network is revised through add/drop style friendship updates

5. Metrics
   Explain the two main families:
   - homophily metrics such as `same_ratio`
   - topology metrics such as density, clustering, modularity, and `prop_nodes_lcc`

6. Results
   Present one subsection per RQ using the brief findings below.

7. Verification and Reliability
   Include the pass counts from the final verification refresh and note that the study-level outputs were rebuilt from saved artifacts without re-running expensive generation.

8. Limitations
   State clearly that:
   - LLM outputs are stochastic
   - prompt wording still matters
   - these findings are empirical results for this setup, not universal social laws

9. Conclusion
   Summarize that culture, method, model, and prompt language all influence generated network structure, and that model choice is not interchangeable.

## Brief RQ Results

### RQ1 Brief Result
When language was fixed to English and culture varied across `us`, `india`, `japan`, and `brazil`, both homophily and topology changed. The strongest culture-driven homophily shift appeared in `political affiliation`, and the topology metric with the widest spread was `prop_nodes_lcc`.

### RQ2 Brief Result
The dominant demographic dimension was usually `political affiliation`, especially in `sequential`, `local`, and `iterative`. The main exception was `global`, where `age` most often emerged as the strongest homophily dimension.

### RQ3 Brief Result
The three GPT models were not interchangeable. Across cultural, method, and language studies, `gpt-4.1` and `gpt-4.1-mini` were consistently the closest pair, while `gpt-4.1-nano` was the most divergent relative to `gpt-4.1`.

### RQ4 Brief Result
When culture was fixed to `us` and prompt language varied across English, Spanish, Hindi, and Japanese, the network still changed. The largest language-driven homophily shift appeared in `religion`, and the topology metric with the widest spread was again `prop_nodes_lcc`.

## What To Read First

If you are new, read files in this order:

1. [ARCHITECTURE.md](ARCHITECTURE.md)
2. [generate_networks.py](generate_networks.py)
3. [constants_and_utils.py](constants_and_utils.py)
4. [analyze_networks.py](analyze_networks.py)
5. [run_cultural_study.py](run_cultural_study.py)
6. [analyze_networks.ipynb](analyze_networks.ipynb)

## Outputs To Care About

For one generation run:
- `text-files/<condition>_<seed>.adj`
- `plots/<condition>_<seed>.png`
- `stats/<condition>/cost_stats_*.csv`
- `stats/<condition>/homophily.csv`
- `stats/<condition>/network_metrics.csv`

For the Step 2 study:
- [condition_summary.csv](stats/cultural_study/condition_summary.csv)
- [demographic_dominance.csv](stats/cultural_study/demographic_dominance.csv)
- [model_divergence.csv](stats/cultural_study/model_divergence.csv)
- [research_answers.md](stats/cultural_study/research_answers.md)

For the Step 3 study:
- [condition_summary.csv](stats/method_study/condition_summary.csv)
- [method_summary.csv](stats/method_study/method_summary.csv)
- [research_answers.md](stats/method_study/research_answers.md)

For the Step 4 study:
- [condition_summary.csv](stats/language_study/condition_summary.csv)
- [language_summary.csv](stats/language_study/language_summary.csv)
- [research_answers.md](stats/language_study/research_answers.md)

## Results & Plots

### Plot Naming Convention

Every file in `plots/` follows this pattern:

```
{method}_{model}[_n{n}]_culture_{culture}[_lang_{language}]_{seed}.png
```

| Segment | Values | Meaning |
|---|---|---|
| `method` | `global`, `sequential`, `local`, `iterative` | How friendships were generated |
| `model` | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | OpenAI model used |
| `_n{n}` | `_n5` | Neighbourhood size (absent for global) |
| `culture` | `us`, `india`, `japan`, `brazil` | Cultural context injected into prompt |
| `lang` | `english`, `spanish`, `hindi`, `japanese` | Prompt language (Step 4 only; absent = English) |
| `seed` | `0`, `1` | Random seed index |

**Examples**
- `global_gpt-4.1-mini_culture_us_0.png` — global method, mini model, US culture, seed 0
- `sequential_gpt-4.1_n5_culture_india_1.png` — sequential, full model, India, seed 1
- `local_gpt-4.1-nano_n5_culture_us_lang_hindi_0.png` — local, nano model, US culture, Hindi prompt, seed 0

### What Each Plot Shows

Each PNG is a force-directed graph of the generated social network.

- **Nodes** — individual personas (50 per graph), labelled by index
- **Edges** — friendship ties proposed by the model
- Spring layout: tightly connected subgroups pull together into visible clumps

### Visual Patterns by Method

#### global

The model receives all 50 personas at once and proposes friendship pairs for the whole network in a single call.

| Model | Visual result |
|---|---|
| `gpt-4.1` | One or two medium-sized connected components in a loose elongated chain; 1–3 isolated nodes; low density (~0.056) |
| `gpt-4.1-mini` | Highly fragmented forest — 4 to 6 separate small trees of 3–8 nodes each with many isolated nodes scattered across the canvas |
| `gpt-4.1-nano` | Hub-and-spoke star: one dense central core with long radial arms extending outward; ~15–20 isolated nodes around the periphery |

The global method consistently produces the sparsest graphs (mean density 0.056 vs 0.18 for local/iterative). The nano model collapses into a centralized star rather than a distributed network, suggesting it defaults to assigning one or two highly popular nodes.

#### sequential

Each persona is added one at a time to a growing network. The model chooses friends from the existing pool as each new node arrives.

Visual result across all cultures and models: **two dense horizontal clumps bridged by a thin chain**. The first clump forms from early arrivals; the second clump forms from later arrivals; a few bridge edges connect them. Almost no isolated nodes. The bipartite-clump structure is consistent across all four cultures and all three models, making sequential the most structurally predictable method.

#### local

A focal persona is asked to choose friends from a neighbourhood of `n=5` candidates, without access to the full growing network.

Visual result: **two tight, well-separated dense cliques**. Each clique occupies a distinct canvas quadrant with minimal cross-cluster edges. The separation is the cleanest of all four methods. The nano model is the exception — it occasionally collapses all 50 nodes into a single massive highly connected ball (single giant component, very high within-cluster density), suggesting the smaller model over-connects when working locally.

#### iterative

The network is built through add/drop revision passes — existing friendship decisions can be reconsidered and updated.

Visual result: **two tight clumps in opposite corners of the canvas plus a notable fringe of ~10–15 scattered isolated nodes**. The two clusters themselves are as dense as local, but iterative uniquely leaves more nodes completely unconnected compared to sequential or local. Despite this, measured density (mean 0.182) is comparable to local because the two active clusters are very dense internally.

### Visual Patterns by Culture (global and sequential, gpt-4.1)

| Culture | global gpt-4.1 | sequential gpt-4.1 |
|---|---|---|
| **US** | Elongated chain + satellite cluster bottom-right + 1–2 isolates | Two dense clumps (top-left and bottom-right) joined by a diagonal bridge chain |
| **India** | Most fragmented: 3 separate components of different sizes + isolated nodes; no dominant cluster | Two near-identical dense bundles; cleanest separation of all four cultures |
| **Japan** | One dominant dense cluster upper-center + small satellite bottom-left + scattered isolates | Dense tight cluster upper-left + loose chain lower-right + isolated pair |
| **Brazil** | Compact single component with most nodes; 1 isolated node far left | Two parallel horizontal bundles connected by bridge edges |

**Key cultural observation**: India consistently produces the most fragmented global-method graphs — three separate components with no single dominant cluster. Japan tends to produce one highly dense core with isolated outliers. Brazil and US produce more connected single-component graphs under the global method.

### Visual Patterns by Prompt Language (global gpt-4.1-mini, culture = US)

| Language | Visual structure |
|---|---|
| **English** | Large tree-like spanning component covering most of the canvas; a few satellite clusters; fewest isolates — most connected output |
| **Spanish** | Three components: one large dense cluster (upper center), one medium chain (lower left), one small group; moderately fragmented |
| **Japanese** | Multiple small trees (4–6 nodes each) spread evenly across canvas; many isolated nodes; very fragmented, no dominant component |
| **Hindi** | Highest fragmentation: 5+ separate small trees (3–8 nodes each) spread across four quadrants; most isolated nodes of any language condition |

**Key language observation**: English prompts produce the most connected global-method networks. Hindi and Japanese prompts produce the most fragmented outputs, with the network fragmenting into many small disconnected trees. Spanish sits between the two extremes. This pattern is consistent with the measured density ranking: Spanish (0.164) > English (0.147) ≈ Japanese (0.146) > Hindi (0.145), though Hindi and Japanese are visually far more fragmented than density alone suggests because the few edges that exist cluster within small trees rather than bridging across the network.

### Plot Coverage Summary

| Study | Method(s) | Cultures | Languages | Models | Seeds | Total PNGs |
|---|---|---|---|---|---|---|
| Step 2 (cultural) | sequential | us, india, japan, brazil | english only | 3 | 2 | 24 |
| Step 3 (method) | global, local, iterative | us, india, japan, brazil | english only | 3 | 2 | 72 |
| Step 4 (language) | global, sequential, local, iterative | us only | english, spanish, hindi, japanese | 3 | 2 | 96 |
| **Total** | | | | | | **192** |

### Numeric Summary of Key Metrics

These values come from the validated study CSV files (`stats/*/method_summary.csv`, `stats/*/language_summary.csv`).

**Density by method (Step 3):**

| Method | Mean density | Std |
|---|---|---|
| iterative | 0.182 | 0.010 |
| local | 0.179 | 0.010 |
| sequential | — (Step 2 baseline) | — |
| global | 0.056 | 0.017 |

**Homophily (same_ratio) by method — top demographic per method:**

| Method | Top demographic | Mean same_ratio |
|---|---|---|
| global | age | 1.83 |
| iterative | political affiliation | 1.75 |
| local | political affiliation | 1.74 |
| sequential | political affiliation | ~2.0 (Step 2 baseline) |

A same_ratio above 1.0 means the model forms more same-group ties than chance. All methods show homophily (>1.0) on most demographics. Below-1.0 cases are concentrated in the `gpt-4.1-nano` + `global` combination, where the star topology reduces meaningful demographic clustering.

**Density by prompt language (Step 4):**

| Language | Mean density | Std |
|---|---|---|
| Spanish | 0.164 | 0.036 |
| English | 0.147 | 0.058 |
| Japanese | 0.146 | 0.061 |
| Hindi | 0.145 | 0.064 |

Spanish-prompt networks are measurably denser and visually more connected. Hindi and Japanese show the highest variance — the nano model in those languages sometimes produces near-empty graphs (heterophily on religion down to 0.79).

## Practical Reading Advice

Do not try to understand every utility or plot helper first.

Instead, ask:
- where does the input persona file come from?
- where is the model prompt built?
- where is the model response parsed?
- where is the graph saved?
- where are the metrics computed?
- where are the final summary tables written?

Those six questions cover most of the repo's logic.
