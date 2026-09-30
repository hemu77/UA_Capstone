"""Reanalyze the 192 historical graphs offline, without changing their sources.

The browser consumes these Python-computed measurements. Filenames provide
inferred historical conditions, not proof of the original request settings.
Missing provenance remains missing; this export cannot repair study design.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from analyze_networks import (compute_age_assortativity, compute_coleman_homophily,
                              compute_edge_distance, compute_network_metrics)

ROOT = Path(__file__).resolve().parent
DEMOS = ['gender', 'race/ethnicity', 'religion', 'political affiliation']
PATTERN = re.compile(r'(?P<method>global|local|sequential|iterative)_(?P<model>gpt-4\.1(?:-mini|-nano)?)'
                     r'(?P<budget>_n5)?_culture_(?P<culture>us|india|japan|brazil)'
                     r'(?:_lang_(?P<language>english|spanish|hindi|japanese))?_(?P<seed>[01])')
KEYS = ['study', 'method', 'model', 'culture', 'language']
ANALYSIS_VERSION = 'revision-v1.1-canonical-order'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def inspect_png(path):
    """Image readability is a separate artifact check, never graph validity."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (OSError, ValueError):
        return False


def expected_runs():
    """Enumerate cells, not just a count: a duplicate cannot conceal a missing run."""
    names = set()
    for method, model, culture, seed in itertools.product(
            ['global', 'local', 'sequential', 'iterative'],
            ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'], ['us', 'india', 'japan', 'brazil'], range(2)):
        prefix = f'{method}_{model}' + ('' if method == 'global' else '_n5')
        names.add(f'{prefix}_culture_{culture}_{seed}')
        if culture == 'us':
            for language in ['english', 'spanish', 'hindi', 'japanese']:
                names.add(f'{prefix}_culture_us_lang_{language}_{seed}')
    return names


def pilot_records(root):
    """Expose completed, checked pilots only; never export raw API audit logs."""
    folder = root / 'outputs/revision_budget_v1'
    if not folder.exists():
        return []
    from paid_study import build_cells, verify_completed
    records = []
    for cell in build_cells('pilot') + build_cells('sol-pilot') + build_cells('luna56-pilot'):
        path = folder / (cell['run_id'] + '.json')
        if not path.exists():
            continue
        record = verify_completed(path, cell)
        graph = nx.read_adjlist(path.with_suffix('.adj'))
        scores = record['verified_homophily']
        records.append(dict(run_id=cell['run_id'], study='engineering_pilot',
            method=cell['method'], model=cell['model'], culture=cell['culture'],
            language=cell['language'], seed=cell['seed'], roster_id=record['roster_sha256'],
            source=path.with_suffix('.adj').relative_to(root).as_posix(),
            source_sha256=record['artifact_sha256']['.adj'], status=record['status'],
            png_source=path.with_suffix('.png').relative_to(root).as_posix(),
            png_sha256=record['artifact_sha256']['.png'],
            png_url=f'./data/artifacts/{cell["run_id"]}.png',
            png_provenance='Receipt hash matches; runner saved the PNG and adjacency list from the same graph. Layout differs from the browser.',
            provenance='Engineering pilot receipt, saved artifact hashes and replay parity checked. API-response reconciliation is a separate audit; not confirmatory evidence.',
            prompt_variant=record['verified_source_variant'],
            language_confound=False, metrics=record['verified_metrics'], analysis_version=ANALYSIS_VERSION,
            homophily={demo: scores[demo] for demo in DEMOS}, age_assortativity=scores['age'],
            edges=sorted([sorted(edge) for edge in graph.edges()]), events=record['events'],
            replay_status='actual_recorded_graph_changes'))
    return records


def export(root=ROOT):
    root = Path(root)
    persona_path = root / 'text-files/us_50_gpt4o_w_interests.json'
    personas = json.loads(persona_path.read_text(encoding='utf-8'))
    if len(personas) != 50:
        raise ValueError('Historical export requires the original 50-person roster.')
    paths = sorted(p for p in (root / 'text-files').glob('*_culture_*.adj') if PATTERN.fullmatch(p.stem))
    if {p.stem for p in paths} != expected_runs():
        raise ValueError('Historical matrix incomplete or unexpected: ' + str(sorted(expected_runs() - {p.stem for p in paths})))
    sources = [persona_path] + paths
    original_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sources}
    roster_hash = original_hashes[persona_path.relative_to(root).as_posix()]
    # Fixed sphere coordinates encode identity, not a measured spatial location.
    # Same person occupies the same position in every comparison, with no layout drift.
    positions = {}
    for i, pid in enumerate(sorted(personas)):
        y = 1 - 2 * (i + 0.5) / len(personas)
        radius, angle = math.sqrt(1 - y * y), i * math.pi * (3 - math.sqrt(5))
        positions[pid] = [round(radius * math.cos(angle) * 75, 5), round(y * 75, 5), round(radius * math.sin(angle) * 75, 5)]
    output = root / 'stats/revision_v1'
    public = root / 'viewer/public/data'
    output.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    networks, homophily, groups, checks, graphs, records = [], [], [], [], {}, []
    for path in paths:
        fields = PATTERN.fullmatch(path.stem).groupdict()
        study = 'language' if fields['language'] else ('cultural' if fields['method'] == 'sequential' else 'method')
        meta = dict(run_id=path.stem, study=study, method=fields['method'], model=fields['model'],
                    culture=fields['culture'], language=fields['language'] or 'english', seed=int(fields['seed']))
        graph = nx.read_adjlist(path, create_using=nx.Graph)
        loops = sorted([list(edge) for edge in nx.selfloop_edges(graph)])
        if set(graph) != set(personas) or loops or not graph.number_of_edges():
            # Do not silently "fix" an experiment by deleting invalid model choices.
            # Record the failure and exclude it from measurements and the explorer.
            checks.append(dict(**meta, nodes=len(graph), edges=graph.number_of_edges(),
                               roster_ok=set(graph) == set(personas), self_loops=len(loops),
                               loop_edges=json.dumps(loops), topology_finite=False,
                               historical_png_decodes=None, historical_png_graph_parity='not_checked_quarantined',
                               graph_valid=False, passed=False, exclusion_reason='invalid roster, self-links or empty graph'))
            continue
        metrics = compute_network_metrics(graph)
        required = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'modularity']
        if not all(math.isfinite(metrics[key]) for key in required):
            raise ValueError(f'{path.name}: missing required topology measurement')
        png = root / 'plots' / f'{path.stem}.png'
        png_ok = inspect_png(png)
        checks.append(dict(**meta, nodes=len(graph), edges=graph.number_of_edges(), roster_ok=True,
                           self_loops=0, loop_edges='[]', topology_finite=True, historical_png_decodes=png_ok,
                           historical_png_graph_parity='unknown_not_inferred_from_pixels', graph_valid=True, passed=True))
        scores = {}
        for demo in DEMOS:
            value, group_values = compute_coleman_homophily(graph, personas, demo)
            scores[demo] = finite_or_none(value)
            homophily.append(dict(**meta, demographic=demo, metric='coleman_population_weighted', value=value))
            for row in group_values:
                groups.append(dict(**meta, demographic=demo, **row))
        age = compute_age_assortativity(graph, personas)
        homophily.append(dict(**meta, demographic='age', metric='numeric_assortativity', value=age))
        networks.append(dict(**meta, nodes=len(graph), edges=graph.number_of_edges(), **metrics))
        edges = sorted([sorted(edge) for edge in graph.edges()])
        records.append(dict(**meta, roster_id=roster_hash, source=f'text-files/{path.name}',
                            source_sha256=original_hashes[f'text-files/{path.name}'],
                            status='historical_reanalyzed', provenance='condition inferred from filename; model snapshot and API logs unavailable',
                            language_confound=True, metrics={k: finite_or_none(v) for k, v in metrics.items()},
                            homophily=scores, age_assortativity=finite_or_none(age), edges=edges,
                            events=None, replay_status='unavailable_historical_final_only'))
        graphs[path.stem] = graph
    frame, homo, verification = pd.DataFrame(networks), pd.DataFrame(homophily), pd.DataFrame(checks)
    frame.to_csv(output / 'network_metrics.csv', index=False)
    homo.to_csv(output / 'homophily.csv', index=False)
    pd.DataFrame(groups).to_csv(output / 'homophily_groups.csv', index=False)
    verification.to_csv(output / 'verification_summary.csv', index=False)
    verification[~verification.passed].to_csv(output / 'quarantined_runs.csv', index=False)
    metrics_long = frame.melt(id_vars=KEYS + ['run_id', 'seed'], value_vars=list(metrics), var_name='metric', value_name='value')
    metrics_long.groupby(KEYS + ['metric']).value.agg(['count', 'mean', 'std', 'min', 'max']).reset_index().to_csv(output / 'condition_summary.csv', index=False)
    homo.groupby(KEYS + ['demographic', 'metric']).value.agg(['size', 'count', 'mean', 'std']).reset_index().to_csv(output / 'homophily_summary.csv', index=False)
    # Compare models only inside the same historical study/condition/seed block.
    # No silent pooling of the separate US-English culture and language runs.
    pairs = []
    for key, block in frame.groupby(['study', 'method', 'culture', 'language', 'seed']):
        for (_, left), (_, right) in itertools.combinations(block.iterrows(), 2):
            pairs.append(dict(zip(['study', 'method', 'culture', 'language', 'seed'], key),
                              model_a=left.model, model_b=right.model, run_a=left.run_id, run_b=right.run_id,
                              edge_distance=compute_edge_distance(graphs[left.run_id], graphs[right.run_id])))
    pd.DataFrame(pairs).to_csv(output / 'model_divergence.csv', index=False)
    pilots = pilot_records(root)
    # Publish only verified figures, never the private API ledger or raw responses.
    import shutil
    (public / 'artifacts').mkdir(exist_ok=True)
    for run in pilots:
        shutil.copyfile(root / run['png_source'], public / 'artifacts' / (run['run_id'] + '.png'))
    payload = dict(schema_version=1, analysis_version=ANALYSIS_VERSION, status='historical_and_engineering_only',
                   engineering_pilots=len(pilots),
                   inventoried_runs=len(paths), quarantined_runs=verification.loc[~verification.passed, 'run_id'].tolist(),
                   graph_semantics='undirected union; no historical direction or reciprocity can be recovered',
                   limitations=['One US-structured synthetic roster; two intended seeds per condition, fewer after exclusions.',
                                'Nine personas are under 18, including four under five; political labels require an age-appropriate redesign.',
                                'Historical language prompts also assigned participant language.',
                                'No confidence intervals or causal/cultural-validity claim.',
                                'Four categorical Coleman scores and numeric age assortativity are different measures.',
                                'Names and interests were not supplied by the study runners and are excluded from this export.'],
                   personas=[dict(id=pid, attributes={k: person[k] for k in DEMOS + ['age']}, position=positions[pid])
                             for pid, person in sorted(personas.items())], runs=records + pilots)
    (public / 'networks.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    render_figures(output, frame, graphs, positions)
    render_pilot_comparison(root, output, pilots, personas)
    unchanged = all(sha256(root / name) == value for name, value in original_hashes.items())
    manifest = dict(analysis_version=ANALYSIS_VERSION, expected_runs=192, actual_runs=len(paths), exported_runs=len(frame),
                    engineering_pilots_exported=len(pilots),
                    by_study=verification.groupby('study').size().to_dict(), passed=int(verification.passed.sum()),
                    quarantined=int((~verification.passed).sum()), self_links=int(verification.self_loops.sum()),
                    png_missing_or_corrupt_among_valid_graphs=int((verification.graph_valid & (verification.historical_png_decodes == False)).sum()),
                    sources_unchanged=unchanged, source_sha256=original_hashes,
                    viewer_sha256=sha256(public / 'networks.json'), api_requests=0,
                    undefined_homophily_count=int(homo.value.isna().sum()),
                    roster_age_min=min(int(p['age']) for p in personas.values()),
                    roster_age_max=max(int(p['age']) for p in personas.values()),
                    roster_under_18=sum(int(p['age']) < 18 for p in personas.values()),
                    roster_under_5=sum(int(p['age']) < 5 for p in personas.values()),
                    model_snapshot='unknown', historical_failure_rate='unknown; successful files are not an attempt log',
                    findings_status='descriptive_reanalysis_not_confirmatory_validation')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not unchanged:
        raise ValueError('Source changed during export. Do not use this export.')
    return manifest


def render_figures(output, frame, graphs, positions):
    # This figure fixes model/country/language rather than pooling incompatible cells.
    subset = frame[(frame.model == 'gpt-4.1-mini') & (frame.culture == 'us') & (frame.study != 'language')]
    methods = ['global', 'local', 'sequential', 'iterative']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, metric, label in zip(axes, ['density', 'avg_clustering_coef', 'prop_nodes_lcc'], ['Density', 'Mean clustering', 'Largest component / all nodes']):
        for x, method in enumerate(methods):
            values = subset[subset.method == method].sort_values('seed')[metric]
            ax.scatter(np.linspace(x - .05, x + .05, len(values)), values, color='#137c79')
        ax.set_xticks(range(4), methods, rotation=20)
        ax.set_title(label)
        ax.set_ylim(-.02, 1.04)  # Leave room for markers at exactly zero or one.
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Historical US-English / GPT-4.1-mini / two seeds per method\nDots are individual runs, not confidence intervals')
    fig.savefig(output / 'topology_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    layout = {pid: position[:2] for pid, position in positions.items()}
    for ax, method in zip(axes, methods):
        row = subset[(subset.method == method) & (subset.seed == 0)].iloc[0]
        graph = graphs[row.run_id]
        nx.draw_networkx(graph, pos=layout, ax=ax, with_labels=False, node_size=20, node_color='#137c79',
                         edge_color='#999999', width=.45)
        ax.set_title(f'{method}: {len(graph)} nodes, {graph.number_of_edges()} edges')
        ax.set_axis_off()
    fig.suptitle('Historical US-English / GPT-4.1-mini / seed 0\nShared identity layout, not social or geographic distance')
    fig.savefig(output / 'network_comparison.png', dpi=150)
    plt.close(fig)


def render_pilot_comparison(root, output, pilots, personas):
    """Show each measured model's four US-English methods, never invented cells."""
    rows = [run for run in pilots if run['language'] == 'english']
    if not rows:
        return  # A partial pilot must not create a complete-looking comparison.
    methods = ['global', 'local', 'sequential', 'iterative']
    models = sorted({run['model'] for run in rows})
    graphs = {run['run_id']: nx.read_adjlist(root / run['source']) for run in rows}
    union = nx.compose_all(list(graphs.values()))
    layout = nx.spring_layout(union, seed=0)
    colors = {'Democrat': '#186e71', 'Republican': '#b94e20'}
    fig, axes = plt.subplots(len(models), 4, figsize=(14, 3.5 * len(models)), layout='constrained', squeeze=False)
    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            row = next((run for run in rows if run['model'] == model and run['method'] == method), None)
            if row is None:
                axes[i, j].text(.5, .5, 'NOT COLLECTED', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{model} / {method}', fontsize=10)
                axes[i, j].set_axis_off()
                continue
            graph = graphs[row['run_id']]
            nx.draw_networkx(graph, pos=layout, ax=axes[i, j], with_labels=False, node_size=18,
                node_color=[colors[personas[node]['political affiliation']] for node in graph],
                edge_color='#7b8996', width=.35)
            axes[i, j].set_title(f'{model} / {method}\n50 nodes, {graph.number_of_edges()} edges', fontsize=10)
            axes[i, j].set_axis_off()
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=color, label=label) for label, color in colors.items()],
               loc='lower center', bbox_to_anchor=(.5, -.045), ncol=2)
    fig.suptitle('ENGINEERING PILOT ONLY / US-English / seed 1000 / historical roster includes minors\n'
                 'Shared union layout is algorithmic; prompt variants differ; no confirmatory inference', fontsize=11)
    fig.savefig(output / 'pilot_network_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = export()
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, indent=2))
