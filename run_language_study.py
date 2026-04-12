"""
Step 4 study runner.

This file is the dedicated Research Question 4 runner.

Unlike Step 2 and Step 3, which together answer RQ1 to RQ3, this study keeps
culture fixed and changes the surface language of the prompt itself across all
methods and models.
"""

import argparse
import os

import pandas as pd

from constants_and_utils import DEFAULT_TEMPERATURE, PATH_TO_STATS_FILES
from study_runner_utils import (
    DEFAULT_LANGUAGES,
    DEFAULT_METHODS,
    DEFAULT_MODELS,
    analyze_condition,
    build_condition_summaries,
    build_dominance_df,
    build_focus_summary,
    build_model_divergence,
    build_pairwise_graph_divergence,
    load_personas,
    run_generation,
    save_common_outputs,
    verify_condition_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the Step 4 fixed-culture language study.')
    parser.add_argument('--persona_fn', type=str, default='us_50_gpt4o_w_interests.json')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--prompt_languages', nargs='+', default=DEFAULT_LANGUAGES)
    parser.add_argument('--culture', type=str, default='us')
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--mean_choices', type=int, default=5)
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--output_subdir', type=str, default='language_study')
    return parser.parse_args()


def build_research_summary(condition_summaries, dominance_df, model_divergence_df, language_divergence_df, verification_df):
    lines = ['# Step 4 Language Study Summary', '']
    lines.append('## Verification')
    passed = int(verification_df[['graph_exists', 'png_ok', 'homophily_ok', 'network_metrics_ok', 'node_count_ok', 'edge_count_ok']].all(axis=1).sum())
    total = len(verification_df)
    lines.append(f'- {passed} of {total} generated language-study graphs passed the artifact and sanity checks.')
    lines.append('')

    lines.append('## Language effects with culture fixed')
    q1_df = condition_summaries[
        (condition_summaries['table'] == 'homophily')
        & (condition_summaries['metric_name'] == 'same_ratio')
    ].copy()
    if len(q1_df) > 0:
        q1_ranked = q1_df.groupby('demo', as_index=False)['_metric_value'].agg(['min', 'max', 'mean']).reset_index()
        q1_ranked['range'] = q1_ranked['max'] - q1_ranked['min']
        q1_ranked = q1_ranked.sort_values('range', ascending=False)
        top = q1_ranked.iloc[0]
        lines.append(f'- The largest language-driven homophily shift appears on `{top["demo"]}` with range {top["range"]:.3f}.')
    topology_df = condition_summaries[
        (condition_summaries['table'] == 'network')
        & (condition_summaries['metric_name'].isin(['density', 'avg_clustering_coef', 'modularity', 'prop_nodes_lcc']))
    ].copy()
    if len(topology_df) > 0:
        topology_ranked = topology_df.groupby('metric_name', as_index=False)['_metric_value'].agg(['min', 'max', 'mean']).reset_index()
        topology_ranked['range'] = topology_ranked['max'] - topology_ranked['min']
        topology_ranked = topology_ranked.sort_values('range', ascending=False)
        top_metric = topology_ranked.iloc[0]
        lines.append(f'- The topology metric with the widest cross-language spread is `{top_metric["metric_name"]}` with range {top_metric["range"]:.3f}.')
    if len(language_divergence_df) > 0:
        language_pairs = language_divergence_df.groupby('language_pair', as_index=False)['edge_distance'].mean().sort_values('edge_distance')
        lines.append(f'- The closest language pair was `{language_pairs.iloc[0]["language_pair"]}` ({language_pairs.iloc[0]["edge_distance"]:.3f}).')
        lines.append(f'- The farthest language pair was `{language_pairs.iloc[-1]["language_pair"]}` ({language_pairs.iloc[-1]["edge_distance"]:.3f}).')
    lines.append('')

    lines.append('## Dominant demographics')
    dominance_counts = dominance_df.groupby('top_demo').size().sort_values(ascending=False)
    if len(dominance_counts) > 0:
        top_demo = dominance_counts.index[0]
        lines.append(f'- `{top_demo}` was the most frequent top-ranked homophily dimension across Step 4 conditions ({int(dominance_counts.iloc[0])} conditions).')
    lines.append('')

    lines.append('## Model consistency')
    if len(model_divergence_df) > 0:
        ranked = model_divergence_df.groupby('model_pair', as_index=False)['edge_distance'].mean().sort_values('edge_distance')
        lines.append(f'- The closest model pair was `{ranked.iloc[0]["model_pair"]}` ({ranked.iloc[0]["edge_distance"]:.3f}).')
        lines.append(f'- The farthest model pair was `{ranked.iloc[-1]["model_pair"]}` ({ranked.iloc[-1]["edge_distance"]:.3f}).')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    personas = load_personas(args.persona_fn)

    condition_records = []
    graphs_by_condition = {}
    verification_rows = []

    # This loop spans every method because RQ4 is a full-project question, not
    # a sequential-only follow-up.
    for method in args.methods:
        for prompt_language in args.prompt_languages:
            for model in args.models:
                print(f'=== Running method={method} language={prompt_language} model={model} culture={args.culture} ===')
                save_prefix = run_generation(
                    method=method,
                    persona_fn=args.persona_fn,
                    model=model,
                    num_seeds=args.num_seeds,
                    start_seed=args.start_seed,
                    mean_choices=args.mean_choices,
                    temp=args.temp,
                    culture_context=args.culture,
                    prompt_language=prompt_language,
                    num_iter=args.num_iter,
                )
                graphs_by_condition[save_prefix] = analyze_condition(personas, save_prefix, args.start_seed, args.num_seeds)
                condition_records.append({
                    'save_prefix': save_prefix,
                    'method': method,
                    'culture': args.culture,
                    'model': model,
                    'prompt_language': prompt_language,
                })
                verification_rows.extend(verify_condition_outputs(save_prefix, args.start_seed, args.num_seeds, expected_nodes=len(personas)))

    condition_summaries, homophily_summary, _ = build_condition_summaries(condition_records)
    dominance_df = build_dominance_df(homophily_summary)
    model_divergence_df = build_model_divergence(condition_records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])
    language_divergence_df = build_pairwise_graph_divergence(
        condition_records,
        graphs_by_condition,
        group_keys=['method', 'culture', 'model'],
        compare_key='prompt_language',
        pair_label='language_pair',
    )

    output_dir = os.path.join(PATH_TO_STATS_FILES, args.output_subdir)
    save_common_outputs(output_dir, condition_summaries, dominance_df, model_divergence_df, verification_rows)
    build_focus_summary(condition_summaries, ['prompt_language']).to_csv(os.path.join(output_dir, 'language_summary.csv'), index=False)
    language_divergence_df.to_csv(os.path.join(output_dir, 'language_divergence.csv'), index=False)
    verification_df = pd.DataFrame(verification_rows)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(build_research_summary(condition_summaries, dominance_df, model_divergence_df, language_divergence_df, verification_df))


if __name__ == '__main__':
    main()
