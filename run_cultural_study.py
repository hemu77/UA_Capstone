"""
Step 2 study runner.

This file automates the cultural-context experiment:
- keep personas fixed
- keep the prompt language fixed to English unless explicitly overridden
- vary culture and model
- write aggregate summaries and verification outputs
"""

import argparse
import os

from study_runner_utils import (
    DEFAULT_CULTURES,
    DEFAULT_MODELS,
    analyze_condition,
    build_condition_summaries,
    build_dominance_df,
    build_focus_summary,
    build_model_divergence,
    load_personas,
    run_generation,
    save_common_outputs,
    verify_condition_outputs,
)
from constants_and_utils import DEFAULT_TEMPERATURE, PATH_TO_STATS_FILES


def parse_args():
    parser = argparse.ArgumentParser(description='Run the full Step 2 cultural study matrix.')
    parser.add_argument('--persona_fn', type=str, default='us_50_gpt4o_w_interests.json')
    parser.add_argument('--method', type=str, default='sequential', choices=['global', 'local', 'sequential', 'iterative'])
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--cultures', nargs='+', default=DEFAULT_CULTURES)
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--mean_choices', type=int, default=5)
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--prompt_language', type=str, default=None)
    parser.add_argument('--output_subdir', type=str, default='cultural_study')
    parser.add_argument('--num_iter', type=int, default=3)
    return parser.parse_args()


def build_research_summary(condition_summaries, dominance_df, divergence_df):
    lines = ['# Cultural Study Summary', '']

    q1_df = condition_summaries[
        (condition_summaries['table'] == 'homophily')
        & (condition_summaries['metric_name'] == 'same_ratio')
    ].copy()
    q1_ranked = q1_df.groupby('demo', as_index=False)['_metric_value'].agg(['min', 'max', 'mean']).reset_index()
    q1_ranked['range'] = q1_ranked['max'] - q1_ranked['min']
    q1_ranked = q1_ranked.sort_values('range', ascending=False)
    lines.append('## Q1: Cultural context effects')
    if len(q1_ranked) > 0:
        top = q1_ranked.iloc[0]
        lines.append(
            f'- The largest culture-driven homophily shift appears on `{top["demo"]}` with mean same-ratio range {top["range"]:.3f} across conditions.'
        )
    topology_df = condition_summaries[
        (condition_summaries['table'] == 'network')
        & (condition_summaries['metric_name'].isin(['density', 'avg_clustering_coef', 'modularity', 'prop_nodes_lcc']))
    ].copy()
    topology_ranked = topology_df.groupby('metric_name', as_index=False)['_metric_value'].agg(['min', 'max', 'mean']).reset_index()
    topology_ranked['range'] = topology_ranked['max'] - topology_ranked['min']
    topology_ranked = topology_ranked.sort_values('range', ascending=False)
    if len(topology_ranked) > 0:
        top_metric = topology_ranked.iloc[0]
        lines.append(
            f'- The topology metric with the widest cross-culture spread is `{top_metric["metric_name"]}` with range {top_metric["range"]:.3f}.'
        )
    lines.append('')

    lines.append('## Q2: Dominant demographics')
    dominance_counts = dominance_df.groupby('top_demo').size().sort_values(ascending=False)
    if len(dominance_counts) > 0:
        top_demo = dominance_counts.index[0]
        lines.append(f'- `{top_demo}` is the most frequent top-ranked homophily dimension across conditions ({int(dominance_counts.iloc[0])} conditions).')
    lines.append('')

    lines.append('## Q3: Model consistency')
    if len(divergence_df) > 0:
        top_pair = divergence_df.groupby('model_pair', as_index=False)['edge_distance'].mean().sort_values('edge_distance')
        most_similar = top_pair.iloc[0]
        most_different = top_pair.iloc[-1]
        lines.append(
            f'- The most consistent pair is `{most_similar["model_pair"]}` with average edge distance {most_similar["edge_distance"]:.3f}.'
        )
        lines.append(
            f'- The most divergent pair is `{most_different["model_pair"]}` with average edge distance {most_different["edge_distance"]:.3f}.'
        )
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    personas = load_personas(args.persona_fn)

    condition_records = []
    graphs_by_condition = {}
    verification_rows = []

    for culture in args.cultures:
        for model in args.models:
            print(f'=== Running method={args.method} culture={culture} model={model} ===')
            save_prefix = run_generation(
                method=args.method,
                persona_fn=args.persona_fn,
                model=model,
                num_seeds=args.num_seeds,
                start_seed=args.start_seed,
                mean_choices=args.mean_choices,
                temp=args.temp,
                culture_context=culture,
                prompt_language=args.prompt_language,
                num_iter=args.num_iter,
            )
            graphs_by_condition[save_prefix] = analyze_condition(personas, save_prefix, args.start_seed, args.num_seeds)
            condition_records.append({
                'save_prefix': save_prefix,
                'method': args.method,
                'culture': culture,
                'model': model,
                'prompt_language': args.prompt_language or 'english',
            })
            verification_rows.extend(verify_condition_outputs(save_prefix, args.start_seed, args.num_seeds, expected_nodes=len(personas)))

    condition_summaries, homophily_summary, _ = build_condition_summaries(condition_records)
    dominance_df = build_dominance_df(homophily_summary)
    divergence_df = build_model_divergence(condition_records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])

    output_dir = os.path.join(PATH_TO_STATS_FILES, args.output_subdir)
    save_common_outputs(output_dir, condition_summaries, dominance_df, divergence_df, verification_rows)

    culture_summary = build_focus_summary(condition_summaries, ['culture'])
    culture_summary.to_csv(os.path.join(output_dir, 'culture_summary.csv'), index=False)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(build_research_summary(condition_summaries, dominance_df, divergence_df))


if __name__ == '__main__':
    main()
