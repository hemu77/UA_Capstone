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
