"""Post-collection structural context from eight existing friendship networks.

This is not a new confirmatory family, representative human sample, or a
benchmark of culture/language fidelity. No node identities are exported.
"""
from pathlib import Path
import json
import hashlib
import math
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import pandas as pd

from analysis_primary.core import write_json
from analysis_primary.graph_evaluation import _validate_primary

PAPER = 'https://arxiv.org/html/2408.16629v2#A3.SS3'
REFERENCES = (
    ('Galesburg', 'Galesburg2.paj', 'Galesburg_friends.net', 31, 63, 'physician friendship nominations'),
    ('Hi-tech', 'Hi-tech.xml', 'test', 36, 91, 'employee personal-friend nominations'),
    ('Karate', 'karate.xml', 'agent x agent', 34, 78, 'karate club social ties'),
    ('Prison', 'prison.xml', 'ID', 67, 142, 'inmate closest-friend nominations'),
    ('Tailor 1', 'kaptail.xml', 'KAPFTS1', 39, 158, 'sociational ties, wave 1'),
    ('Tailor 2', 'kaptail.xml', 'KAPFTS2', 39, 223, 'sociational ties, wave 2; same workers as wave 1'),
    ('Moreno freshmen', 'moreno_vdb/out.moreno_vdb_vdb', None, 31, 218, 'positive friendship ratings, repeated observations unioned'),
    ('Moreno high school', 'moreno_highschool/out.moreno_highschool_highschool', None, 70, 274, 'positive friendship ratings, repeated observations unioned'),
)
METRICS = ('density', 'avg_clustering_coef', 'global_efficiency')


def source_fingerprints(path):
    data = path.read_bytes()
    return dict(source_sha256=hashlib.sha256(data).hexdigest(),
                source_lf_sha256=hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest())


def read_reference(path, selector):
    """Explicit positive-edge selectors; undirected OR projection, no loops."""
    graph = nx.Graph()
    if path.suffix == '.xml':
        meta = ET.parse(path).getroot().find('MetaNetwork')
        classes = meta.findall('./nodes/nodeclass')
        if len(classes) != 1:
            raise ValueError('Expected one declared node class')
        nodes = [x.attrib['id'] for x in classes[0].findall('node')]
        if len(nodes) != len(set(nodes)):
            raise ValueError('Duplicate declared node')
        graph.add_nodes_from(nodes)
        layers = [x for x in meta.findall('./networks/network') if x.get('id', 'ID') == selector]
        if len(layers) != 1:
            raise ValueError('Expected one exact friendship layer')
        for edge in layers[0].findall('link'):
            a, b, weight = edge.attrib['source'], edge.attrib['target'], float(edge.attrib['value'])
            if a not in graph or b not in graph or not math.isfinite(weight):
                raise ValueError('Invalid reference endpoint or weight')
            if weight > 0 and a != b:
                graph.add_edge(a, b)
    elif path.suffix == '.paj':
        lines = path.read_text().splitlines()
        starts = [i for i, line in enumerate(lines) if line.strip() == '*Network ' + selector]
        if len(starts) != 1:
            raise ValueError('Expected exact Pajek friendship section')
        start = starts[0] + 1
        end = next((i for i in range(start, len(lines)) if lines[i].startswith('*Network')), len(lines))
        source = nx.parse_pajek(lines[start:end])
        graph.add_nodes_from(source)
        for a, b, values in source.edges(data=True):
            if float(values.get('weight', 1)) > 0 and a != b:
                graph.add_edge(a, b)
    else:
        # KONECT tables contain repeated signed ratings; match the archived
        # positive-edge union, not a final-wave or reciprocated-tie estimand.
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith('%'):
                continue
            columns = line.split()
            if len(columns) < 3 or not math.isfinite(float(columns[2])):
                raise ValueError('Expected finite signed KONECT rating')
            if float(columns[2]) > 0 and columns[0] != columns[1]:
                graph.add_edge(columns[0], columns[1])
    return graph


def ks_distance(left, right):
    """ECDF sup distance only: no p-value or independent-sample assumption."""
    left, right = np.sort(left), np.sort(right)
    points = np.unique(np.r_[left, right])
    return float(np.max(np.abs(np.searchsorted(left, points, side='right') / len(left)
                               - np.searchsorted(right, points, side='right') / len(right))))


def run_empirical_evaluation(root: Path, primary: pd.DataFrame, output: Path) -> dict:
    _validate_primary(root, primary)
    rows = []
    for name, file, selector, n, m, ties in REFERENCES:
        path = root / 'real_networks' / file
        graph = read_reference(path, selector)
        if (len(graph), graph.number_of_edges()) != (n, m):
            raise ValueError(f'{name}: preprocessing differs from published node/edge anchor')
        rows.append(dict(reference=name, source_file=path.relative_to(root).as_posix(), **source_fingerprints(path),
            selector=selector, nodes=n, edges=m, tie_definition=ties,
            density=nx.density(graph), avg_clustering_coef=nx.average_clustering(graph),
            global_efficiency=nx.global_efficiency(graph), source_reference=PAPER))
    reference = pd.DataFrame(rows)
    comparisons = []
    for (method, model), group in primary.groupby(['method', 'model'], sort=True):
        for metric in METRICS:
            observed = group.loc[group.valid_graph, metric].dropna().to_numpy(float)
            real = reference[metric].to_numpy(float)
            comparisons.append(dict(method=method, model=model, metric=metric,
                primary_attempts=len(group), primary_defined=len(observed), reference_networks=len(real),
                primary_mean=float(observed.mean()), reference_mean=float(real.mean()),
                reference_min=float(real.min()), reference_max=float(real.max()),
                mean_difference=float(observed.mean()-real.mean()),
                ecdf_sup_distance=ks_distance(observed, real), descriptive_only=True))
    output.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output/'empirical_reference_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output/'empirical_comparisons_descriptive.csv', index=False)
    result = dict(reference_networks=8, published_node_edge_anchors_verified=True,
        selection='Existing eight-network friendship panel from Chang et al. Appendix C.3; chosen post collection, not fitted to primary results.',
        projection='Positive ties, binary simple undirected OR projection; loops removed. XML/Pajek declared isolates retained; Moreno nodes are endpoints of positive ties only.',
        source_fingerprints='source_sha256 records exact local bytes; source_lf_sha256 replaces CRLF with LF in memory only for cross-checkout equivalence. Legacy files are never rewritten. Both hashes are retained; no other normalization is allowed.',
        independence='Eight networks are not eight independent populations: two Tailor waves share workers; Moreno unions repeated ratings.',
        hi_tech_caveat='Archived XML plus undirected OR projection reproduces published 36/91; it is not a newly imposed mutual-nomination filter. Layer semantics and upstream transformations remain limitations.',
        ks_interpretation='ECDF sup distance is descriptive only, no KS p-values, no optimized winner, no new hypothesis family.',
        metric_scope='Density, average clustering, global efficiency; size/density differences remain confounded. Missing primary metrics are omitted with counts, never zero-filled.',
        human_validation=False, representative_sample=False, homophily_ground_truth=False,
        limitations=['Small legacy convenience panel, not a held-out population-matched human benchmark.',
            'No matched persona attributes or political labels; cannot validate national, language or political realism.',
            'Real reference sizes 31-70 versus 50 generated nodes; dimensionless metrics are not size-invariant.',
            'No empirical degree pooling or dyad-level significance. Structural overlap alone is not behavioral fidelity.',
            'No diffusion, longitudinal polarization, held-out prediction or survey validation performed.'],
        source=PAPER, paid_calls=0)
    write_json(output/'empirical_evaluation.json', result)
    return result
