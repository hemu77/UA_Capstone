"""
Step 2 study runner.

This file automates the cultural-context experiment added for the project
proposal. It loops over cultures, models, and seeds, runs generation and
analysis for each condition, then aggregates the results into summary tables.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import networkx as nx

from analyze_networks import compute_edge_distance, load_list_of_graphs, summarize_network_metrics
from constants_and_utils import PATH_TO_STATS_FILES, PATH_TO_TEXT_FILES, DEFAULT_TEMPERATURE
from generate_networks import get_save_prefix_and_demos


DEFAULT_MODELS = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']
DEFAULT_CULTURES = ['us', 'india', 'japan', 'brazil']
DEFAULT_DEMOS = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']


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
    return parser.parse_args()


def make_save_prefix(args, model, culture):
    # Reuse the main generator's naming logic so all files line up cleanly.
    namespace = SimpleNamespace(
        method=args.method,
        persona_fn=args.persona_fn,
        mean_choices=args.mean_choices,
        include_names=False,
        include_interests=False,
        only_interests=False,
        shuffle_all=False,
        shuffle_interests=False,
        include_friend_list=False,
        include_reason=False,
        prompt_all=False,
        model=model,
        num_networks=args.num_seeds,
        start_seed=args.start_seed,
        temp=args.temp,
        num_iter=3,
        culture_context=culture,
        verbose=False,
    )
    save_prefix, _ = get_save_prefix_and_demos(namespace)
    return save_prefix


def run_generation(args, model, culture):
    # Delegate graph creation to generate_networks.py rather than duplicating
    # the generation logic in two places.
    save_prefix = make_save_prefix(args, model, culture)
    command = [
        sys.executable,
        'generate_networks.py',
        args.method,
        '--persona_fn', args.persona_fn,
        '--model', model,
        '--num_networks', str(args.num_seeds),
        '--start_seed', str(args.start_seed),
        '--mean_choices', str(args.mean_choices),
        '--temp', str(args.temp),
        '--culture_context', culture,
    ]
    subprocess.run(command, check=True)
    return save_prefix


def analyze_condition(personas, save_prefix, start_seed, num_seeds):
    # Analyze each condition immediately after generation so failures are local.
    list_of_graphs = load_list_of_graphs(save_prefix, start_seed, start_seed + num_seeds, directed=False)
    summarize_network_metrics(list_of_graphs, personas, DEFAULT_DEMOS, save_name=save_prefix)
    return list_of_graphs


def extract_condition_parts(save_prefix):
    prefix, _, culture = save_prefix.partition('_culture_')
    method, model, _ = prefix.split('_', 2)
    return method, model, culture


def load_condition_metrics(save_prefix):
    homophily_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'homophily.csv')
    network_metrics_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'network_metrics.csv')
    homophily_df = pd.read_csv(homophily_path)
    network_metrics_df = pd.read_csv(network_metrics_path)
    method, model, culture = extract_condition_parts(save_prefix)
    for df in (homophily_df, network_metrics_df):
        df['condition'] = save_prefix
        df['method'] = method
        df['model'] = model
        df['culture'] = culture
    return homophily_df, network_metrics_df


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
    lines.append('')
    return '\n'.join(lines)


def summarize_study(save_prefixes, graphs_by_condition):
    # This converts many condition-level outputs into one study-level view.
    all_homophily = []
    all_network = []
    for save_prefix in save_prefixes:
        homophily_df, network_metrics_df = load_condition_metrics(save_prefix)
        all_homophily.append(homophily_df)
        all_network.append(network_metrics_df)

    all_homophily_df = pd.concat(all_homophily, ignore_index=True)
    all_network_df = pd.concat(all_network, ignore_index=True)

    homophily_summary = (
        all_homophily_df
        .groupby(['condition', 'culture', 'model', 'metric_name', 'demo'], as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )
    homophily_summary['table'] = 'homophily'

    scalar_network_df = all_network_df[pd.isnull(all_network_df.get('node'))].copy()
    network_summary = (
        scalar_network_df
        .groupby(['condition', 'culture', 'model', 'metric_name'], as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )
    network_summary['demo'] = None
    network_summary['table'] = 'network'

    condition_summaries = pd.concat([homophily_summary, network_summary], ignore_index=True)

    same_ratio_df = homophily_summary[homophily_summary['metric_name'] == 'same_ratio'].copy()
    same_ratio_df['rank'] = same_ratio_df.groupby('condition')['_metric_value'].rank(method='dense', ascending=False)
    dominance_df = same_ratio_df.sort_values(['condition', 'rank', 'demo']).groupby('condition', as_index=False).first()
    dominance_df = dominance_df.rename(columns={'demo': 'top_demo', '_metric_value': 'top_demo_same_ratio'})

    divergence_rows = []
    for culture in sorted({extract_condition_parts(prefix)[2] for prefix in save_prefixes}):
        culture_prefixes = sorted([prefix for prefix in save_prefixes if prefix.endswith(f'_culture_{culture}')])
        by_model = {extract_condition_parts(prefix)[1]: graphs_by_condition[prefix] for prefix in culture_prefixes}
        for model_a, model_b in itertools.combinations(sorted(by_model.keys()), 2):
            graphs_a = by_model[model_a]
            graphs_b = by_model[model_b]
            for offset, (graph_a, graph_b) in enumerate(zip(graphs_a, graphs_b)):
                divergence_rows.append({
                    'culture': culture,
                    'seed': offset,
                    'model_pair': f'{model_a} vs {model_b}',
                    'edge_distance': compute_edge_distance(graph_a, graph_b),
                })
    divergence_df = pd.DataFrame(divergence_rows)

    output_dir = os.path.join(PATH_TO_STATS_FILES, 'cultural_study')
    os.makedirs(output_dir, exist_ok=True)
    condition_summaries.to_csv(os.path.join(output_dir, 'condition_summary.csv'), index=False)
    dominance_df.to_csv(os.path.join(output_dir, 'demographic_dominance.csv'), index=False)
    divergence_df.to_csv(os.path.join(output_dir, 'model_divergence.csv'), index=False)
    with open(os.path.join(output_dir, 'research_answers.md'), 'w', encoding='utf-8') as f:
        f.write(build_research_summary(condition_summaries, dominance_df, divergence_df))


def main():
    args = parse_args()
    personas_path = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)

    save_prefixes = []
    graphs_by_condition = {}
    for culture in args.cultures:
        for model in args.models:
            print(f'=== Running culture={culture} model={model} ===')
            save_prefix = run_generation(args, model, culture)
            save_prefixes.append(save_prefix)
            graphs_by_condition[save_prefix] = analyze_condition(personas, save_prefix, args.start_seed, args.num_seeds)

    summarize_study(save_prefixes, graphs_by_condition)


if __name__ == '__main__':
    main()
