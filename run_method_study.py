"""
Step 3 study runner.

This file does not replace Step 2. It complements it.

Step 2 already answered the first version of RQ1 to RQ3 with the sequential
method. Step 3 adds the missing methods, `global`, `local`, and `iterative`,
so the final capstone can say those three research questions were studied
across all four methods together.
"""

import argparse
import os

import pandas as pd

from constants_and_utils import DEFAULT_TEMPERATURE, PATH_TO_STATS_FILES
from study_runner_utils import (
    DEFAULT_CULTURES,
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

STEP3_METHODS = ['global', 'local', 'iterative']


def parse_args():
    parser = argparse.ArgumentParser(description='Run the Step 3 method-expansion study.')
    parser.add_argument('--persona_fn', type=str, default='us_50_gpt4o_w_interests.json')
    parser.add_argument('--methods', nargs='+', default=STEP3_METHODS)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--cultures', nargs='+', default=DEFAULT_CULTURES)
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--mean_choices', type=int, default=5)
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--output_subdir', type=str, default='method_study')
    return parser.parse_args()


def build_research_summary(condition_summaries, dominance_df, model_divergence_df, method_divergence_df, verification_df):
    lines = ['# Step 3 Method Study Summary', '']
    lines.append('## Verification')
    passed = int(verification_df[['graph_exists', 'png_ok', 'homophily_ok', 'network_metrics_ok', 'node_count_ok', 'edge_count_ok']].all(axis=1).sum())
    total = len(verification_df)
    lines.append(f'- {passed} of {total} generated method-study graphs passed the artifact and sanity checks.')
    lines.append('')

    lines.append('## Method effects')
    density_df = condition_summaries[
        (condition_summaries['table'] == 'network')
        & (condition_summaries['metric_name'] == 'density')
    ].copy()
    if len(density_df) > 0:
        method_density = density_df.groupby('method', as_index=False)['_metric_value'].mean().sort_values('_metric_value', ascending=False)
        densest = method_density.iloc[0]
        sparsest = method_density.iloc[-1]
        lines.append(f'- `{densest["method"]}` produced the highest average density ({densest["_metric_value"]:.3f}).')
        lines.append(f'- `{sparsest["method"]}` produced the lowest average density ({sparsest["_metric_value"]:.3f}).')
    if len(method_divergence_df) > 0:
        method_pairs = method_divergence_df.groupby('method_pair', as_index=False)['edge_distance'].mean().sort_values('edge_distance')
        lines.append(f'- The most similar method pair was `{method_pairs.iloc[0]["method_pair"]}` with average edge distance {method_pairs.iloc[0]["edge_distance"]:.3f}.')
        lines.append(f'- The most different method pair was `{method_pairs.iloc[-1]["method_pair"]}` with average edge distance {method_pairs.iloc[-1]["edge_distance"]:.3f}.')
    lines.append('')

    lines.append('## Dominant demographics')
    dominance_counts = dominance_df.groupby(['method', 'top_demo']).size().reset_index(name='count')
    if len(dominance_counts) > 0:
        dominance_counts = dominance_counts.sort_values(['method', 'count', 'top_demo'], ascending=[True, False, True])
        for method in sorted(dominance_counts['method'].unique()):
            top_row = dominance_counts[dominance_counts['method'] == method].iloc[0]
            lines.append(f'- `{method}` most often elevated `{top_row["top_demo"]}` as the strongest homophily dimension ({int(top_row["count"])} conditions).')
    lines.append('')

    lines.append('## Model consistency within methods')
    if len(model_divergence_df) > 0:
        ranked = model_divergence_df.groupby('model_pair', as_index=False)['edge_distance'].mean().sort_values('edge_distance')
        lines.append(f'- Across Step 3, the closest model pair was `{ranked.iloc[0]["model_pair"]}` ({ranked.iloc[0]["edge_distance"]:.3f}).')
        lines.append(f'- The farthest model pair was `{ranked.iloc[-1]["model_pair"]}` ({ranked.iloc[-1]["edge_distance"]:.3f}).')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    personas = load_personas(args.persona_fn)

    condition_records = []
    graphs_by_condition = {}
    verification_rows = []

    # We intentionally run only the three missing methods here because
    # sequential already exists in the Step 2 cultural study outputs.
    for method in args.methods:
        for culture in args.cultures:
            for model in args.models:
                print(f'=== Running method={method} culture={culture} model={model} ===')
                save_prefix = run_generation(
                    method=method,
                    persona_fn=args.persona_fn,
                    model=model,
                    num_seeds=args.num_seeds,
                    start_seed=args.start_seed,
                    mean_choices=args.mean_choices,
                    temp=args.temp,
                    culture_context=culture,
                    prompt_language=None,
                    num_iter=args.num_iter,
                )
                graphs_by_condition[save_prefix] = analyze_condition(personas, save_prefix, args.start_seed, args.num_seeds)
                condition_records.append({
                    'save_prefix': save_prefix,
                    'method': method,
                    'culture': culture,
                    'model': model,
                    'prompt_language': 'english',
                })
                verification_rows.extend(verify_condition_outputs(save_prefix, args.start_seed, args.num_seeds, expected_nodes=len(personas)))

    # These aggregate files are the "method expansion" layer that gets read
    # together with Step 2 when we explain RQ1, RQ2, and RQ3 in the README and notebook.
    condition_summaries, homophily_summary, _ = build_condition_summaries(condition_records)
    dominance_df = build_dominance_df(homophily_summary)
    model_divergence_df = build_model_divergence(condition_records, graphs_by_condition, group_keys=['method', 'culture', 'prompt_language'])
    method_divergence_df = build_pairwise_graph_divergence(
        condition_records,
        graphs_by_condition,
        group_keys=['culture', 'model', 'prompt_language'],
        compare_key='method',
        pair_label='method_pair',
    )

    output_dir = os.path.join(PATH_TO_STATS_FILES, args.output_subdir)
    save_common_outputs(output_dir, condition_summaries, dominance_df, model_divergence_df, verification_rows)
    build_focus_summary(condition_summaries, ['method']).to_csv(os.path.join(output_dir, 'method_summary.csv'), index=False)
    method_divergence_df.to_csv(os.path.join(output_dir, 'method_divergence.csv'), index=False)

    verification_df = pd.DataFrame(verification_rows)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(build_research_summary(condition_summaries, dominance_df, model_divergence_df, method_divergence_df, verification_df))


if __name__ == '__main__':
    main()
