# Generating social networks with LLMs
This repo contains code and results for the paper ["LLMs generate structurally realistic social networks but overestimate political homophily"](https://arxiv.org/abs/2408.16629), by Serina Chang*, Alicja Chaszczewicz*, Emma Wang, Maya Josifovska, Emma Pierson, and Jure Leskovec (ICWSM 2025).

## UA Capstone framing
This repository has been adapted into a four-question capstone project.

The final project is organized like this:
- `Step 1`: make the repo run with only `OPENAI_API_KEY`
- `RQ1-RQ3`: use all four generation methods, `sequential`, `global`, `local`, and `iterative`, to study culture, dominant demographics, and model consistency
- `RQ4`: keep culture fixed and vary prompt language

The important framing change is this:
- Step 2 gave the first `sequential` evidence for RQ1-RQ3
- Step 3 added the other three methods so RQ1-RQ3 are supported by all four methods together
- Step 4 introduced prompt-language variation and became `RQ4`

## Experimental setup
The final project uses one fixed 50-person persona roster:
- persona file: `text-files/us_50_gpt4o_w_interests.json`
- the same personas are reused across studies so the changed variable is the prompt condition, not the underlying people

LLM models used in the final capstone studies:
- `gpt-4.1-nano`
- `gpt-4.1-mini`
- `gpt-4.1`

Generation methods used across the project:
- `sequential`
- `global`
- `local`
- `iterative`

Cultures used in the cultural-context experiments:
- `us`
- `india`
- `japan`
- `brazil`

Prompt languages used in the fixed-culture language experiment:
- `english`
- `spanish`
- `hindi`
- `japanese`

Study structure:
- Step 2: `sequential` only, 4 cultures, 3 models, 2 seeds
- Step 3: `global`, `local`, `iterative`, 4 cultures, 3 models, 2 seeds
- Step 4: all 4 methods, culture fixed to `us`, 4 prompt languages, 3 models, 2 seeds

Why this matters:
- RQ1, RQ2, and RQ3 are not based on one single prompting strategy
- Step 2 and Step 3 together give coverage across all four methods
- RQ4 isolates prompt language by holding culture constant

## Final verification status
- Step 2 cultural study artifacts were generated and analyzed
- Step 3 method-study verification passed `72/72`
- Step 4 language-study verification passed `96/96`
- the notebook and documentation now reflect the final four-RQ structure

## Final research questions and answers

### RQ1
When language is held constant, does varying cultural context alter homophily patterns and network topology?

How this was tested:
- prompt language was held fixed to `english`
- cultures were varied across `us`, `india`, `japan`, and `brazil`
- the same 50 personas were reused
- models were varied across `gpt-4.1-nano`, `gpt-4.1-mini`, and `gpt-4.1`
- evidence comes from Step 2 plus the method expansion added in Step 3

Answer:
- Yes. Holding prompt language fixed to English while varying culture still changed both homophily and graph structure.
- In the sequential cultural study, the largest culture-driven homophily shift appeared in `political affiliation`.
- The topology metric with the widest spread across cultures was `prop_nodes_lcc`.
- After adding `global`, `local`, and `iterative`, the broader project still showed meaningful method-sensitive structural differences, which means the culture question cannot be reduced to just one prompting method.
- In plain terms: changing only the cultural frame changed who clustered with whom and how connected the network became, even when the language stayed the same.

### RQ2
Which demographic dimensions dominate tie formation under varying linguistic and cultural conditions?

How this was tested:
- for the culture studies, language was held at `english` while culture varied
- for the language study, culture was held at `us` while prompt language varied
- dominance was measured by looking at which demographic had the highest `same_ratio` within each condition
- all four methods were considered by combining the Step 2 sequential baseline with the Step 3 method expansion

Answer:
- Across the English-language cultural study, `political affiliation` most often dominated tie formation.
- After extending the project to all four methods, the answer became more nuanced:
- `global` most often elevated `age`
- `local`, `sequential`, and `iterative` most often elevated `political affiliation`
- In the fixed-culture language study, `political affiliation` still appeared most often as the strongest homophily dimension, which suggests it remains the most stable dominant factor across many conditions.
- In plain terms: the strongest tie-formation signal was usually political similarity, but the `global` method behaved differently often enough that age became the leading factor there.

### RQ3
Do different LLM models produce consistent or divergent patterns under identical conditions?

How this was tested:
- the same personas, methods, seeds, and study conditions were matched across models
- graph disagreement was measured with pairwise edge distance
- lower edge distance means two models produced more similar networks under the same setup

Answer:
- They diverge in a repeatable way rather than behaving as interchangeable substitutes.
- In the cultural study, `gpt-4.1` and `gpt-4.1-mini` were the most similar pair, while `gpt-4.1` and `gpt-4.1-nano` were the most different.
- In the method study, the same pattern held again: `gpt-4.1 vs gpt-4.1-mini` was the closest pair (`0.074` average edge distance), while `gpt-4.1 vs gpt-4.1-nano` was the farthest (`0.119`).
- In the language study, the same ranking held for a third time: `gpt-4.1 vs gpt-4.1-mini` was the closest pair (`0.081`), while `gpt-4.1 vs gpt-4.1-nano` was the farthest (`0.126`).
- In plain terms: the models are not interchangeable. The nano model behaved like a meaningfully different network generator, while `gpt-4.1` and `gpt-4.1-mini` stayed the closest pair throughout the project.

### RQ4
When culture is held constant, does changing the prompt language alter homophily patterns and network topology?

How this was tested:
- culture was fixed to `us`
- prompt language was varied across `english`, `spanish`, `hindi`, and `japanese`
- the same personas were reused
- all four methods and all three GPT models were included

Answer:
- Yes. Keeping culture fixed to `us` while changing prompt language still shifted both homophily and topology.
- The largest language-driven homophily shift appeared in `religion`.
- The topology metric with the widest cross-language spread was `prop_nodes_lcc`.
- The closest language pair was `hindi vs japanese`, while the farthest was `japanese vs spanish`.
- In plain terms: even when the people and the culture stayed fixed, changing the language of the instructions still changed the network that came out.

## Method-level takeaways
- `iterative` produced the highest average density (`0.182`)
- `global` produced the lowest average density (`0.056`)
- the most similar method pair was `global vs local`
- the most different method pair was `iterative vs local`, although the gap was small

## Where the final answers live
For the exact generated summaries and tables, see:
- `stats/cultural_study/research_answers.md`
- `stats/cultural_study/condition_summary.csv`
- `stats/method_study/research_answers.md`
- `stats/method_study/method_summary.csv`
- `stats/method_study/verification_summary.csv`
- `stats/language_study/research_answers.md`
- `stats/language_study/language_summary.csv`
- `stats/language_study/verification_summary.csv`

## Prerequisites 
To run OpenAI models, set `OPENAI_API_KEY` in your environment. As a backward-compatible fallback, you can still put an OpenAI key on the first line of `api-key.txt`.

To run Llama, Gemma, or other open-source models, set `LLAMA_API_KEY` or add it as the optional second line of `api-key.txt`.

We used Python 3.10 in our experiments, see package requirements in `requirements.txt`.

## Generate personas
To sample 50 personas and save it to a file called us_50.json, run the following command.
This does *not* include names nor interests.

```python generate_personas.py 50 --save_name us_50```

If you would like to generate names and/or interests (based on demographics):

```python generate_personas.py 50  --save_name us_50 --include_names --include_interests```

With names and interests, the resulting filename will be `us_50_w_names_w_interests.json`. You can also specify which LLM to use with `--model`. In our experiments, we use the 50 personas saved under `text-files/us_50_gpt4o_w_interests.json`.

`generate_personas.py` also has functions for analyzing the personas and interests, such as `get_interest_embeddings()` and `parse_reason()`.


## Generate networks
To generate networks, run something like the following command.

```python generate_networks.py global --model gpt-4.1-mini --num_networks 30```

This will generate 30 networks using the Global method, using GPT-4.1 Mini. The networks will be saved as adjacency lists as `global_gpt-4.1-mini_SEED.adj`, for SEED from 0 to 29, under `PATH_TO_TEXT_FILES` (defined in `constants_and_utils.py`). The visualized network is also saved under `PATH_TO_SAVED_PLOTS` (defined in `plotting.py`) and the summary of the costs (number of tokens, number of tries, time duration) is saved as `cost_stats_s0-29.csv` under `PATH_TO_STATS_FILES/global_gpt-4.1-mini` (defined in `constants_and_utils.py`).

You can vary which LLM to use with `--model` and how many networks are generated with `--num_networks`. Other important arguments include `--persona_fn` (which file to get personas from) and `--include_interests` (whether to include interests, which need to be included in the persona file if so). See `parse_args()` in `generate_networks.py` for a full list of arguments.

To try other prompting methods, replace `global` with `local`, `sequential`, or `iterative`. These methods also come with the added option of `--include_reason`, where the model is prompted to generate a short reason for each friend it selects. If `--include_reason` is included, the networks will be saved as `METHOD_MODEL_w_reason_SEED.adj` and the reasons will be saved as `METHOD_MODEL_w_reason_SEED_reasons.json` (e.g., see `sequential_gpt-3.5-turbo_w_reason_0_reasons.json`) under `PATH_TO_TEXT_FILES`.

To run the Step 2 cultural study matrix described in this project extension, use:

```python run_cultural_study.py```

This runs the `sequential` cultural-context matrix and writes aggregate outputs under `stats/cultural_study`. In the final capstone framing, these outputs are the first piece of evidence for `RQ1-RQ3`.

To run the Step 3 method-expansion study, use:

```python run_method_study.py```

This runs the missing three methods, `global`, `local`, and `iterative`, across the same culture/model/seed matrix and writes aggregate outputs under `stats/method_study`. Together with Step 2, this is what makes `RQ1-RQ3` a four-method result rather than a sequential-only result.

To run the Step 4 fixed-culture language study, use:

```python run_language_study.py```

This keeps culture fixed to `us`, varies prompt language across English, Spanish, Hindi, and Japanese, and writes aggregate outputs under `stats/language_study`.

To analyze the generated networks, see `analyze_networks.py` and `plotting.py`.

## Our results
See `analyze_networks.ipynb` for our figures and tables. You can also find our generated networks and generated personas (with interests) in `text-files` and the summary statistics in `stats`.

## How to run the full project
If you want to reproduce the project from the command line, run the sections below in order.

Step 1: verify the repo works with OpenAI only
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python generate_networks.py sequential --persona_fn us_50_gpt4o_w_interests.json --model gpt-4.1-mini --num_networks 1 --start_seed 0 --mean_choices 5
python analyze_networks.py --persona_fn us_50_gpt4o_w_interests.json --network_fn sequential_gpt-4.1-mini_n5 --num_networks 1
```

RQ1-RQ3 part A: sequential cultural study
```powershell
python run_cultural_study.py
```

RQ1-RQ3 part B: add the other three methods
```powershell
python run_method_study.py
```

RQ4: fixed-culture language study
```powershell
python run_language_study.py
```

Notebook review
```powershell
jupyter notebook analyze_networks.ipynb
```

The notebook now has dedicated UA Capstone sections for:
- `RQ1-RQ3`, which combine the sequential cultural study with the added method study
- `RQ4`, which covers prompt-language variation with culture held constant
