# Generating social networks with LLMs
This repo contains code and results for the paper ["LLMs generate structurally realistic social networks but overestimate political homophily"](https://arxiv.org/abs/2408.16629), by Serina Chang*, Alicja Chaszczewicz*, Emma Wang, Maya Josifovska, Emma Pierson, and Jure Leskovec (ICWSM 2025).

## Project status
This repository has been fully adapted for the `UA_Capstone` project.

Completed project scope:
- Step 1: OpenAI-only compatibility using `OPENAI_API_KEY`
- Step 2: English-language cultural-context study across `us`, `india`, `japan`, and `brazil`
- Step 3: full method expansion for `global`, `local`, and `iterative`
- Step 4: fixed-culture prompt-language study across English, Spanish, Hindi, and Japanese
- final notebook and documentation updates for all four steps

Verification status:
- Step 3: `72/72` generated method-study graphs passed artifact and sanity checks
- Step 4: `96/96` generated language-study graphs passed artifact and sanity checks

## Final project findings

### Research question 1
When language is held constant, does varying cultural context alter homophily patterns and network topology?

Answer from Step 2:
- Yes. Cultural framing changed both homophily and topology.
- The largest culture-driven homophily shift appeared in `political affiliation`.
- The topology metric with the widest spread across cultures was `prop_nodes_lcc`.

### Research question 2
Which demographic dimensions dominate tie formation under varying linguistic and cultural conditions?

Answer across the final project:
- In the English-language cultural-context study, `political affiliation` most often dominated tie formation.
- In the Step 3 method study, `global` most often elevated `age`, while `local` and `iterative` most often elevated `political affiliation`.
- In the Step 4 fixed-culture language study, `political affiliation` was still the most frequent top-ranked homophily dimension, appearing in `29` conditions.

### Research question 3
Do different LLM models produce consistent or divergent patterns under identical conditions?

Answer across all phases:
- They are not fully consistent.
- In Step 2, `gpt-4.1` and `gpt-4.1-mini` were the most similar pair, while `gpt-4.1` and `gpt-4.1-nano` were the most different.
- In Step 3, the same pattern held: `gpt-4.1` vs `gpt-4.1-mini` was the closest pair (`0.074` average edge distance), and `gpt-4.1` vs `gpt-4.1-nano` was the farthest (`0.119`).
- In Step 4, the same pattern held again: `gpt-4.1` vs `gpt-4.1-mini` was the closest pair (`0.081`), and `gpt-4.1` vs `gpt-4.1-nano` was the farthest (`0.126`).

### Additional final findings

Step 3 method study:
- `iterative` produced the highest average density (`0.182`).
- `global` produced the lowest average density (`0.056`).
- The most similar method pair was `global vs local` and the most different pair was `iterative vs local`, although the gap between them was small.

Step 4 fixed-culture language study:
- With culture fixed to `us`, the largest language-driven homophily shift appeared on `religion`.
- The topology metric with the widest cross-language spread was `prop_nodes_lcc`.
- The closest language pair was `hindi vs japanese`; the farthest was `japanese vs spanish`.

For the exact study summaries, see:
- `stats/cultural_study/condition_summary.csv`
- `stats/cultural_study/research_answers.md`
- `stats/method_study/condition_summary.csv`
- `stats/method_study/method_summary.csv`
- `stats/method_study/research_answers.md`
- `stats/language_study/condition_summary.csv`
- `stats/language_study/language_summary.csv`
- `stats/language_study/research_answers.md`

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

This runs the full set of English-language cultural-context conditions across the configured GPT models and writes aggregate outputs under `stats/cultural_study`.

To run the Step 3 method-expansion study, use:

```python run_method_study.py```

This runs the missing three methods (`global`, `local`, `iterative`) across the same culture/model/seed matrix and writes aggregate outputs under `stats/method_study`.

To run the Step 4 fixed-culture language study, use:

```python run_language_study.py```

This keeps culture fixed to `us`, varies prompt language across English, Spanish, Hindi, and Japanese, and writes aggregate outputs under `stats/language_study`.

To analyze the generated networks, see `analyze_networks.py` and `plotting.py`.

## Our results
See `analyze_networks.ipynb` for our figures and tables. You can also find our generated networks and generated personas (with interests) in `text-files` and the summary statistics in `stats`.
