"""
Shared helpers for Step 2, Step 3, and Step 4 experiment runners.

This module keeps the orchestration scripts thin:
- launch generation with consistent naming
- run the existing analysis pipeline
- aggregate condition-level outputs into study-level tables
- verify that generated artifacts are complete and readable
"""

import itertools
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import networkx as nx
import pandas as pd
from PIL import Image

from analyze_networks import compute_edge_distance, load_list_of_graphs, summarize_network_metrics
from constants_and_utils import DEFAULT_TEMPERATURE, PATH_TO_STATS_FILES, PATH_TO_TEXT_FILES
from generate_networks import get_save_prefix_and_demos

DEFAULT_MODELS = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']
DEFAULT_CULTURES = ['us', 'india', 'japan', 'brazil']
DEFAULT_LANGUAGES = ['english', 'spanish', 'hindi', 'japanese']
DEFAULT_DEMOS = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']
DEFAULT_METHODS = ['global', 'local', 'sequential', 'iterative']


def load_personas(persona_fn):
    personas_path = os.path.join(PATH_TO_TEXT_FILES, persona_fn)
    with open(personas_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_save_prefix(
    method,
    persona_fn,
    model,
    num_seeds,
    start_seed,
    mean_choices,
    temp,
    culture_context=None,
    prompt_language=None,
    num_iter=3,
):
    effective_mean_choices = mean_choices if method != 'global' else -1
    namespace = SimpleNamespace(
        method=method,
        persona_fn=persona_fn,
        mean_choices=effective_mean_choices,
        include_names=False,
        include_interests=False,
        only_interests=False,
        shuffle_all=False,
        shuffle_interests=False,
        include_friend_list=False,
        include_reason=False,
        prompt_all=False,
        model=model,
        num_networks=num_seeds,
        start_seed=start_seed,
        temp=temp,
        num_iter=num_iter,
        culture_context=culture_context,
        prompt_language=prompt_language,
        verbose=False,
    )
    save_prefix, _ = get_save_prefix_and_demos(namespace)
    return save_prefix


def run_generation(
    method,
    persona_fn,
    model,
    num_seeds,
    start_seed,
    mean_choices,
    temp,
    culture_context=None,
    prompt_language=None,
    num_iter=3,
):
    effective_mean_choices = mean_choices if method != 'global' else -1
    save_prefix = make_save_prefix(
        method=method,
        persona_fn=persona_fn,
        model=model,
        num_seeds=num_seeds,
        start_seed=start_seed,
        mean_choices=effective_mean_choices,
        temp=temp,
        culture_context=culture_context,
        prompt_language=prompt_language,
        num_iter=num_iter,
    )
    command = [
        sys.executable,
        'generate_networks.py',
        method,
        '--persona_fn',
        persona_fn,
        '--model',
        model,
        '--num_networks',
        str(num_seeds),
        '--start_seed',
        str(start_seed),
        '--temp',
        str(temp),
    ]
    if method != 'global':
        command.extend(['--mean_choices', str(mean_choices)])
    if culture_context is not None:
        command.extend(['--culture_context', culture_context])
    if prompt_language is not None:
        command.extend(['--prompt_language', prompt_language])
    if method == 'iterative':
        command.extend(['--num_iter', str(num_iter)])
    subprocess.run(command, check=True)
    return save_prefix


def analyze_condition(personas, save_prefix, start_seed, num_seeds):
    list_of_graphs = load_list_of_graphs(save_prefix, start_seed, start_seed + num_seeds, directed=False)
    summarize_network_metrics(list_of_graphs, personas, DEFAULT_DEMOS, save_name=save_prefix)
    return list_of_graphs


def load_condition_metrics(record):
    save_prefix = record['save_prefix']
    homophily_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'homophily.csv')
    network_metrics_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'network_metrics.csv')
    homophily_df = pd.read_csv(homophily_path)
    network_metrics_df = pd.read_csv(network_metrics_path)
    metadata = {k: v for k, v in record.items() if k != 'save_prefix'}
    for df in (homophily_df, network_metrics_df):
        df['condition'] = save_prefix
        for key, value in metadata.items():
            df[key] = value
    return homophily_df, network_metrics_df


def build_condition_summaries(condition_records):
    all_homophily = []
    all_network = []
    for record in condition_records:
        homophily_df, network_metrics_df = load_condition_metrics(record)
        all_homophily.append(homophily_df)
        all_network.append(network_metrics_df)

    all_homophily_df = pd.concat(all_homophily, ignore_index=True)
    all_network_df = pd.concat(all_network, ignore_index=True)

    metadata_cols = [
        col for col in ['condition', 'method', 'culture', 'model', 'prompt_language']
        if col in all_homophily_df.columns
    ]
    homophily_summary = (
        all_homophily_df
        .groupby(metadata_cols + ['metric_name', 'demo'], as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )
    homophily_summary['table'] = 'homophily'

    scalar_network_df = all_network_df[pd.isnull(all_network_df.get('node'))].copy()
    network_summary = (
        scalar_network_df
        .groupby(metadata_cols + ['metric_name'], as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )
    network_summary['demo'] = None
    network_summary['table'] = 'network'

    condition_summaries = pd.concat([homophily_summary, network_summary], ignore_index=True)
    return condition_summaries, homophily_summary, network_summary


def build_dominance_df(homophily_summary):
    same_ratio_df = homophily_summary[homophily_summary['metric_name'] == 'same_ratio'].copy()
    same_ratio_df['rank'] = same_ratio_df.groupby('condition')['_metric_value'].rank(method='dense', ascending=False)
    dominance_df = same_ratio_df.sort_values(['condition', 'rank', 'demo']).groupby('condition', as_index=False).first()
    return dominance_df.rename(columns={'demo': 'top_demo', '_metric_value': 'top_demo_same_ratio'})


def build_pairwise_graph_divergence(condition_records, graphs_by_condition, group_keys, compare_key, pair_label):
    grouped = {}
    for record in condition_records:
        group_key = tuple(record.get(key) for key in group_keys)
        grouped.setdefault(group_key, []).append(record)

    divergence_rows = []
    for group_key, records in grouped.items():
        by_value = {record[compare_key]: graphs_by_condition[record['save_prefix']] for record in records}
        for value_a, value_b in itertools.combinations(sorted(by_value.keys()), 2):
            graphs_a = by_value[value_a]
            graphs_b = by_value[value_b]
            for offset, (graph_a, graph_b) in enumerate(zip(graphs_a, graphs_b)):
                row = {key: value for key, value in zip(group_keys, group_key)}
                row['seed'] = offset
                row[pair_label] = f'{value_a} vs {value_b}'
                row['edge_distance'] = compute_edge_distance(graph_a, graph_b)
                divergence_rows.append(row)
    return pd.DataFrame(divergence_rows)


def build_model_divergence(condition_records, graphs_by_condition, group_keys):
    return build_pairwise_graph_divergence(
        condition_records,
        graphs_by_condition,
        group_keys=group_keys,
        compare_key='model',
        pair_label='model_pair',
    )


def build_focus_summary(condition_summaries, group_keys):
    focus_cols = [key for key in group_keys if key in condition_summaries.columns]
    metric_cols = focus_cols + ['table', 'metric_name']

    network_rows = condition_summaries[condition_summaries['table'] == 'network'].copy()
    network_rows = (
        network_rows
        .groupby(metric_cols, as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )
    network_rows['demo'] = None

    homophily_rows = condition_summaries[condition_summaries['table'] == 'homophily'].copy()
    homophily_rows = (
        homophily_rows
        .groupby(metric_cols + ['demo'], as_index=False)['_metric_value']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': '_metric_value', 'std': 'std_value'})
    )

    return pd.concat([homophily_rows, network_rows], ignore_index=True)


def verify_condition_outputs(save_prefix, start_seed, num_seeds, expected_nodes=50):
    rows = []
    homophily_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'homophily.csv')
    network_metrics_path = os.path.join(PATH_TO_STATS_FILES, save_prefix, 'network_metrics.csv')
    cost_stats_dir = os.path.join(PATH_TO_STATS_FILES, save_prefix)
    cost_stats_files = [
        fn for fn in os.listdir(cost_stats_dir)
        if fn.startswith('cost_stats_') and fn.endswith('.csv')
    ] if os.path.exists(cost_stats_dir) else []

    homophily_ok = os.path.exists(homophily_path) and not pd.read_csv(homophily_path).empty
    network_metrics_ok = os.path.exists(network_metrics_path)
    metrics_have_required_scalars = False
    if network_metrics_ok:
        network_df = pd.read_csv(network_metrics_path)
        scalar_df = network_df[pd.isnull(network_df.get('node'))]
        required_scalars = {'density', 'avg_clustering_coef', 'prop_nodes_lcc', 'modularity'}
        present_scalars = set(scalar_df['metric_name'].unique())
        metrics_have_required_scalars = required_scalars.issubset(present_scalars) and scalar_df['_metric_value'].notna().all()

    for seed in range(start_seed, start_seed + num_seeds):
        graph_path = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}_{seed}.adj')
        plot_path = os.path.join('plots', f'{save_prefix}_{seed}.png')
        graph_exists = os.path.exists(graph_path)
        png_exists = os.path.exists(plot_path)
        png_ok = False
        node_count = None
        edge_count = None
        if graph_exists:
            G = nx.read_adjlist(graph_path)
            node_count = len(G.nodes())
            edge_count = len(G.edges())
        if png_exists:
            try:
                with Image.open(plot_path) as img:
                    img.verify()
                png_ok = True
            except Exception:
                png_ok = False
        rows.append({
            'condition': save_prefix,
            'seed': seed,
            'graph_exists': graph_exists,
            'png_exists': png_exists,
            'png_ok': png_ok,
            'homophily_ok': homophily_ok,
            'network_metrics_ok': network_metrics_ok,
            'metrics_have_required_scalars': metrics_have_required_scalars,
            'cost_stats_ok': len(cost_stats_files) > 0,
            'node_count': node_count,
            'edge_count': edge_count,
            'node_count_ok': node_count == expected_nodes,
            'edge_count_ok': (edge_count is not None) and (edge_count > 0),
        })
    return rows


def save_common_outputs(output_dir, condition_summaries, dominance_df, divergence_df, verification_rows):
    os.makedirs(output_dir, exist_ok=True)
    condition_summaries.to_csv(os.path.join(output_dir, 'condition_summary.csv'), index=False)
    dominance_df.to_csv(os.path.join(output_dir, 'demographic_dominance.csv'), index=False)
    divergence_df.to_csv(os.path.join(output_dir, 'model_divergence.csv'), index=False)
    pd.DataFrame(verification_rows).to_csv(os.path.join(output_dir, 'verification_summary.csv'), index=False)
