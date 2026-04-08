# Generating social networks with LLMs
This repo contains code and results for the paper ["LLMs generate structurally realistic social networks but overestimate political homophily"](https://arxiv.org/abs/2408.16629), by Serina Chang*, Alicja Chaszczewicz*, Emma Wang, Maya Josifovska, Emma Pierson, and Jure Leskovec (ICWSM 2025).

## Project status
This repository is currently being adapted for the `UA_Capstone` project and is still a work in progress.

Current completed work:
- OpenAI-only compatibility using `OPENAI_API_KEY`
- support for newer GPT models such as `gpt-4.1-mini`
- a Step 2 cultural-context study runner for `us`, `india`, `japan`, and `brazil`
- aggregate outputs written under `stats/cultural_study`

Planned / ongoing work:
- more notebook-level explanations and documentation
- more robust verification and repeatability checks
- further experiment extensions beyond the current Step 2 setup

## Current Step 2 findings
These are the current answers from the implemented Step 2 study. They should be treated as current project findings, not final polished claims.

### Research question 1
When language is held constant, does varying cultural context alter homophily patterns and network topology?

Current answer:
- Yes. In the current study, cultural framing changed both homophily and topology.
- The largest culture-driven homophily shift appeared in `political affiliation`.
- The topology metric with the widest spread across conditions was `prop_nodes_lcc`, meaning connectedness changed across cultural contexts.

### Research question 2
Which demographic dimensions dominate tie formation under varying linguistic and cultural conditions?

Current answer:
- In the current English-language cultural-context study, `political affiliation` most often dominated tie formation.
- It was the top-ranked same-group homophily dimension in most conditions, especially for `gpt-4.1` and `gpt-4.1-mini`.
- `gpt-4.1-nano` was less consistent and in some conditions elevated `age` or `race/ethnicity` instead.

### Research question 3
Do different LLM models produce consistent or divergent patterns under identical conditions?

Current answer:
- They are not fully consistent.
- `gpt-4.1` and `gpt-4.1-mini` produced the most similar network structures.
- `gpt-4.1` and `gpt-4.1-nano` showed the largest divergence in edge structure under the same culture and seed.

For the exact study summaries, see:
- `stats/cultural_study/condition_summary.csv`
- `stats/cultural_study/demographic_dominance.csv`
- `stats/cultural_study/model_divergence.csv`
- `stats/cultural_study/research_answers.md`

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

To analyze the generated networks, see `analyze_networks.py` and `plotting.py`.

## Our results
See `analyze_networks.ipynb` for our figures and tables. You can also find our generated networks and generated personas (with interests) in `text-files` and the summary statistics in `stats`.
