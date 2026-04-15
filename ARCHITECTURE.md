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
| `method` | `global`, `sequential`, `local`, `iterative` | How friendships were proposed (Step 3) |
| `model` | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | OpenAI model used |
| `_n{n}` | `_n5` | Neighbourhood size (local/iterative/sequential only) |
| `culture` | `us`, `india`, `japan`, `brazil` | Cultural context injected into the prompt |
| `lang` | `english`, `spanish`, `hindi`, `japanese` | Prompt language (Step 4 only; absent = English) |
| `seed` | `0`, `1` | Random seed index (two replicas per condition) |

Examples:
- `global_gpt-4.1-mini_culture_us_0.png` — global method, mini model, US culture, seed 0
- `iterative_gpt-4.1_n5_culture_india_1.png` — iterative method, full model, India culture, seed 1
- `local_gpt-4.1-nano_n5_culture_us_lang_hindi_0.png` — local method, nano model, US culture, Hindi prompt, seed 0

### What Each Plot Shows

Each PNG is a force-directed graph visualization of the generated social network:

- **Nodes** — individual personas (labelled by index)
- **Edges** — friendship connections proposed by the model
- Layout uses spring positioning; tightly connected clusters appear as dense clumps

### Visual Patterns by Method

| Method | Typical Structure | Why |
|---|---|---|
| `global` | Sparse, tree-like; many isolated or lightly connected nodes | The model decides all friendships in one shot — it tends to under-connect |
| `sequential` | Denser; a large main component with a few small satellites | Each persona is added one at a time; connections accumulate |
| `local` | Two or more tight dense clusters with near-isolation between them | Only nearby personas are considered; clique-like subgraphs form |
| `iterative` | Dense main component, fewer isolates than global | Repeated revision passes fill in missing links |

### Plot Coverage

The `plots/` directory contains **192 PNGs** spanning:

- **4 methods**: global, sequential, local, iterative
- **3 models**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- **4 cultures** (Step 2/3): us, india, japan, brazil
- **4 languages** (Step 4, culture fixed to `us`): english, spanish, hindi, japanese
- **2 seeds** per condition

Step 2 plots (cultural study, global method only):
- `global_{model}_culture_{culture}_{seed}.png`

Step 3 plots (method comparison, all four methods):
- `{method}_{model}_n5_culture_{culture}_{seed}.png`

Step 4 plots (language study, all methods, culture=us):
- `{method}_{model}_n5_culture_us_lang_{language}_{seed}.png`

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
