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

[generate_personas.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\generate_personas.py)
Creates or enriches personas. This is the "who exists?" layer.

[generate_networks.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\generate_networks.py)
Main generation engine. This is the "who becomes friends?" layer.

[constants_and_utils.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\constants_and_utils.py)
Shared infrastructure. It handles paths, API calls, retries, saving graphs, and drawing PNGs.

[analyze_networks.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\analyze_networks.py)
Main metrics layer. This is the "what kind of network came out?" layer.

[plotting.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\plotting.py)
Visualization helpers for graphs and analysis tables.

[run_cultural_study.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\run_cultural_study.py)
Step 2 experiment orchestrator. It runs the full culture/model/seed matrix and writes aggregate outputs.

[study_runner_utils.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\study_runner_utils.py)
Shared experiment helper layer. This keeps the Step 2, Step 3, and Step 4 runners on the same generation, aggregation, and verification path.

[run_method_study.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\run_method_study.py)
Step 3 experiment orchestrator. It compares `global`, `local`, and `iterative` under the same culture/model/seed setup used for Step 2.

[run_language_study.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\run_language_study.py)
Step 4 experiment orchestrator. It keeps culture fixed and varies the prompt language across English, Spanish, Hindi, and Japanese.

[analyze_networks.ipynb](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\analyze_networks.ipynb)
Exploratory notebook. This is where results are compared, plotted, and interpreted interactively.

[network_datasets.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\network_datasets.py)
Loads or converts real/reference datasets for comparison.

[bias.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\bias.py)
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

1. [ARCHITECTURE.md](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\ARCHITECTURE.md)
2. [generate_networks.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\generate_networks.py)
3. [constants_and_utils.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\constants_and_utils.py)
4. [analyze_networks.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\analyze_networks.py)
5. [run_cultural_study.py](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\run_cultural_study.py)
6. [analyze_networks.ipynb](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\analyze_networks.ipynb)

## Outputs To Care About

For one generation run:
- `text-files/<condition>_<seed>.adj`
- `plots/<condition>_<seed>.png`
- `stats/<condition>/cost_stats_*.csv`
- `stats/<condition>/homophily.csv`
- `stats/<condition>/network_metrics.csv`

For the Step 2 study:
- [condition_summary.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\cultural_study\condition_summary.csv)
- [demographic_dominance.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\cultural_study\demographic_dominance.csv)
- [model_divergence.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\cultural_study\model_divergence.csv)
- [research_answers.md](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\cultural_study\research_answers.md)

For the Step 3 study:
- [condition_summary.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\method_study\condition_summary.csv)
- [method_summary.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\method_study\method_summary.csv)
- [research_answers.md](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\method_study\research_answers.md)

For the Step 4 study:
- [condition_summary.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\language_study\condition_summary.csv)
- [language_summary.csv](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\language_study\language_summary.csv)
- [research_answers.md](C:\Users\Hemu\OneDrive\Desktop\D.s\UA_Captsone_SocN\stats\language_study\research_answers.md)

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
