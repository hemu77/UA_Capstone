"""
Rebuild capstone study summaries from existing saved artifacts.

This script does not call any LLM APIs. It only reads graphs and per-condition
CSV files that already exist in the repo, then rebuilds the study-level
summary and verification files for:
- Step 2 cultural study
- Step 3 method study
- Step 4 language study

Use this when a summary file is missing or when you want to re-verify the
saved outputs without spending tokens on generation again.
"""

import os

import pandas as pd

from analyze_networks import load_list_of_graphs
from constants_and_utils import PATH_TO_STATS_FILES
from run_cultural_study import build_research_summary as build_cultural_research_summary
from run_language_study import build_research_summary as build_language_research_summary
from run_method_study import STEP3_METHODS, build_research_summary as build_method_research_summary
from study_runner_utils import (
    DEFAULT_CULTURES,
    DEFAULT_LANGUAGES,
    DEFAULT_METHODS,
    DEFAULT_MODELS,
    build_condition_summaries,
    build_dominance_df,
    build_focus_summary,
    build_model_divergence,
    build_pairwise_graph_divergence,
    load_personas,
    make_save_prefix,
    save_common_outputs,
    verify_condition_outputs,
)


PERSONA_FN = 'us_50_gpt4o_w_interests.json'
NUM_SEEDS = 2
START_SEED = 0
MEAN_CHOICES = 5
NUM_ITER = 3


def build_records_for_study(study_name):
    records = []
    if study_name == 'cultural_study':
        for culture in DEFAULT_CULTURES:
            for model in DEFAULT_MODELS:
                save_prefix = make_save_prefix(
                    method='sequential',
                    persona_fn=PERSONA_FN,
                    model=model,
                    num_seeds=NUM_SEEDS,
                    start_seed=START_SEED,
                    mean_choices=MEAN_CHOICES,
                    temp=0.8,
                    culture_context=culture,
                    prompt_language=None,
                    num_iter=NUM_ITER,
                )
                records.append({
                    'save_prefix': save_prefix,
                    'method': 'sequential',
                    'culture': culture,
                    'model': model,
                    'prompt_language': 'english',
                })
    elif study_name == 'method_study':
        for method in STEP3_METHODS:
            for culture in DEFAULT_CULTURES:
                for model in DEFAULT_MODELS:
                    save_prefix = make_save_prefix(
                        method=method,
                        persona_fn=PERSONA_FN,
                        model=model,
                        num_seeds=NUM_SEEDS,
                        start_seed=START_SEED,
                        mean_choices=MEAN_CHOICES,
                        temp=0.8,
                        culture_context=culture,
                        prompt_language=None,
                        num_iter=NUM_ITER,
                    )
                    records.append({
                        'save_prefix': save_prefix,
                        'method': method,
                        'culture': culture,
                        'model': model,
                        'prompt_language': 'english',
                    })
    elif study_name == 'language_study':
        for method in DEFAULT_METHODS:
            for prompt_language in DEFAULT_LANGUAGES:
                for model in DEFAULT_MODELS:
                    save_prefix = make_save_prefix(
                        method=method,
                        persona_fn=PERSONA_FN,
                        model=model,
                        num_seeds=NUM_SEEDS,
                        start_seed=START_SEED,
                        mean_choices=MEAN_CHOICES,
                        temp=0.8,
                        culture_context='us',
                        prompt_language=prompt_language,
                        num_iter=NUM_ITER,
                    )
                    records.append({
                        'save_prefix': save_prefix,
                        'method': method,
                        'culture': 'us',
                        'model': model,
                        'prompt_language': prompt_language,
                    })
    else:
        raise ValueError(f'Unknown study: {study_name}')
    return records


def load_graphs_by_condition(records):
    graphs_by_condition = {}
    for record in records:
        graphs_by_condition[record['save_prefix']] = load_list_of_graphs(
            record['save_prefix'],
            START_SEED,
            START_SEED + NUM_SEEDS,
            directed=False,
        )
    return graphs_by_condition


def rebuild_cultural_study(personas):
    records = build_records_for_study('cultural_study')
    graphs_by_condition = load_graphs_by_condition(records)
    verification_rows = []
    for record in records:
        verification_rows.extend(
            verify_condition_outputs(record['save_prefix'], START_SEED, NUM_SEEDS, expected_nodes=len(personas))
        )

    condition_summaries, homophily_summary, _ = build_condition_summaries(records)
    dominance_df = build_dominance_df(homophily_summary)
    divergence_df = build_model_divergence(records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])
    output_dir = os.path.join(PATH_TO_STATS_FILES, 'cultural_study')
    save_common_outputs(output_dir, condition_summaries, dominance_df, divergence_df, verification_rows)
    build_focus_summary(condition_summaries, ['culture']).to_csv(os.path.join(output_dir, 'culture_summary.csv'), index=False)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(build_cultural_research_summary(condition_summaries, dominance_df, divergence_df))
    return pd.DataFrame(verification_rows)


def rebuild_method_study(personas):
    records = build_records_for_study('method_study')
    graphs_by_condition = load_graphs_by_condition(records)
    verification_rows = []
    for record in records:
        verification_rows.extend(
            verify_condition_outputs(record['save_prefix'], START_SEED, NUM_SEEDS, expected_nodes=len(personas))
        )

    condition_summaries, homophily_summary, _ = build_condition_summaries(records)
    dominance_df = build_dominance_df(homophily_summary)
    model_divergence_df = build_model_divergence(records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])
    method_divergence_df = build_pairwise_graph_divergence(
        records,
        graphs_by_condition,
        group_keys=['culture', 'model', 'prompt_language'],
        compare_key='method',
        pair_label='method_pair',
    )
    output_dir = os.path.join(PATH_TO_STATS_FILES, 'method_study')
    save_common_outputs(output_dir, condition_summaries, dominance_df, model_divergence_df, verification_rows)
    build_focus_summary(condition_summaries, ['method']).to_csv(os.path.join(output_dir, 'method_summary.csv'), index=False)
    method_divergence_df.to_csv(os.path.join(output_dir, 'method_divergence.csv'), index=False)
    verification_df = pd.DataFrame(verification_rows)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(
            build_method_research_summary(
                condition_summaries,
                dominance_df,
                model_divergence_df,
                method_divergence_df,
                verification_df,
            )
        )
    return verification_df


def rebuild_language_study(personas):
    records = build_records_for_study('language_study')
    graphs_by_condition = load_graphs_by_condition(records)
    verification_rows = []
    for record in records:
        verification_rows.extend(
            verify_condition_outputs(record['save_prefix'], START_SEED, NUM_SEEDS, expected_nodes=len(personas))
        )

    condition_summaries, homophily_summary, _ = build_condition_summaries(records)
    dominance_df = build_dominance_df(homophily_summary)
    model_divergence_df = build_model_divergence(records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])
    language_divergence_df = build_pairwise_graph_divergence(
        records,
        graphs_by_condition,
        group_keys=['method', 'culture', 'model'],
        compare_key='prompt_language',
        pair_label='language_pair',
    )
    output_dir = os.path.join(PATH_TO_STATS_FILES, 'language_study')
    save_common_outputs(output_dir, condition_summaries, dominance_df, model_divergence_df, verification_rows)
    build_focus_summary(condition_summaries, ['prompt_language']).to_csv(os.path.join(output_dir, 'language_summary.csv'), index=False)
    language_divergence_df.to_csv(os.path.join(output_dir, 'language_divergence.csv'), index=False)
    verification_df = pd.DataFrame(verification_rows)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(
            build_language_research_summary(
                condition_summaries,
                dominance_df,
                model_divergence_df,
                language_divergence_df,
                verification_df,
            )
        )
    return verification_df


def summarize_verification(study_name, verification_df):
    checks = ['graph_exists', 'png_ok', 'homophily_ok', 'network_metrics_ok', 'node_count_ok', 'edge_count_ok']
    passed = int(verification_df[checks].all(axis=1).sum())
    total = len(verification_df)
    return f'{study_name}: {passed}/{total} passed'


def main():
    personas = load_personas(PERSONA_FN)
    cultural_df = rebuild_cultural_study(personas)
    method_df = rebuild_method_study(personas)
    language_df = rebuild_language_study(personas)

    summary_lines = [
        '# Capstone Verification Refresh',
        '',
        summarize_verification('cultural_study', cultural_df),
        summarize_verification('method_study', method_df),
        summarize_verification('language_study', language_df),
        '',
        'These files were rebuilt from existing graphs and per-condition stats.',
        'No model generation was re-run for this refresh.',
    ]
    with open(os.path.join(PATH_TO_STATS_FILES, 'capstone_verification_refresh.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')

    for line in summary_lines[2:5]:
        print(line)


if __name__ == '__main__':
    main()
