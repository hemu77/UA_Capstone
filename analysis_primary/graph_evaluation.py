"""Bounded offline quality checks and descriptive G(n,m) context for primary runs.

No frozen analysis math is imported. Like make_matched_baselines.py's uniform
edge-count control, references use NetworkX G(n,m), but omit its demographic
and rewiring controls, persona inputs, and legacy collection entry point.
"""

import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath

import networkx as nx
import numpy as np
import pandas as pd


PRIMARY_CSV = Path('outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
PRIMARY_CSV_SHA256 = 'ec24e7867a22d687d40d99253aa823a76bd506a734b6afb022ca359b64aac073'
REFERENCE_REPLICATES = 8
METRICS = ('density', 'avg_clustering_coef', 'global_efficiency')
GROUPS = ['method', 'model', 'culture', 'language', 'wording', 'condition']
LABEL = 'offline_synthetic_G(n,m)_descriptive_only'


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _validate_primary(root, primary):
    data = (root / PRIMARY_CSV).read_bytes()
    if hashlib.sha256(data).hexdigest() != PRIMARY_CSV_SHA256:
        raise ValueError('Primary CSV SHA-256 does not match the fixed primary whitelist.')
    source = pd.read_csv(io.BytesIO(data))
    if not set(source.columns).issubset(primary.columns):
        raise ValueError('Primary input is missing required CSV columns.')
    if (primary.empty or primary.original_run_id.duplicated().any()
            or primary.run_id.duplicated().any()
            or not primary.cohort.eq('primary').all()
            or not primary.primary_member.map(_true).all()
            or not primary.run_id.eq(primary.original_run_id).all()
            or set(primary.original_run_id) != set(source.original_run_id)):
        raise ValueError('Require the exact original primary run-ID whitelist; no supplemental rows.')
    try:
        pd.testing.assert_frame_equal(
            primary.set_index('original_run_id').loc[:, source.columns.drop('original_run_id')]
            .sort_index(),
            source.set_index('original_run_id').sort_index(),
            check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-15,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError('Primary rows differ from their hash-bound CSV records.') from error


def _read_graph(root, row):
    relative = Path(str(row.adj_path))
    windows = PureWindowsPath(str(row.adj_path))
    if (relative.is_absolute() or windows.drive or windows.is_absolute()
            or '..' in relative.parts or '..' in windows.parts):
        raise ValueError(f'{row.run_id}: adjacency path must be root-relative.')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'{row.run_id}: adjacency path escapes root.')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row.adj_sha256:
        raise ValueError(f'{row.run_id}: adjacency SHA-256 mismatch.')
    graph = nx.read_adjlist(io.BytesIO(data), create_using=nx.Graph())
    if set(graph) != {str(i) for i in range(50)} or nx.number_of_selfloops(graph):
        raise ValueError(f'{row.run_id}: require exact IDs 0..49 and no self-loops.')
    if len(graph) != row.nodes or graph.number_of_edges() != row.edges:
        raise ValueError(f'{row.run_id}: node/edge count differs from primary CSV.')
    return graph


def _reference_metrics(run_id, n, m):
    """Eight deterministic equal-n/equal-m draws, including empty/trivial graphs."""
    values = {metric: [] for metric in METRICS}
    for replicate in range(REFERENCE_REPLICATES):
        seed = int.from_bytes(hashlib.sha256(
            f'primary-graph-evaluation-v1:{run_id}:{replicate}'.encode('utf-8')
        ).digest()[:8], 'big')
        graph = nx.gnm_random_graph(n, m, seed=seed)
        values['density'].append(nx.density(graph))
        values['avg_clustering_coef'].append(nx.average_clustering(graph) if n else float('nan'))
        values['global_efficiency'].append(nx.global_efficiency(graph))
    return values


def run_graph_evaluation(root: Path, primary: pd.DataFrame, output: Path) -> dict:
    """Validate all primary graphs; write three CSVs and a JSON manifest to output.

    Returns that manifest (counts, files, provenance, and interpretation limits).
    Raises ValueError on cohort, provenance, structure, or defined metric mismatch.
    Undefined observations and failed attempts remain missing, never zero-filled.
    All validation and computation precede output creation; no graph files written.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    _validate_primary(root, primary)
    quality, references = [], []
    verified = 0
    for row in primary.sort_values('original_run_id').itertuples(index=False):
        valid = _true(row.valid_graph)
        if valid != _true(row.eligible_primary_graph):
            raise ValueError(f'{row.run_id}: valid/eligible flags disagree.')
        labels = {name: getattr(row, name) for name in GROUPS if name != 'condition'}
        labels['condition'] = '|'.join(f'{name}={getattr(row, name)}'
                                      for name in ('culture', 'language', 'wording'))
        labels.update(cohort='primary', primary_member=True, run_id=row.run_id,
                      original_run_id=row.original_run_id, valid_graph=valid,
                      failure=row.failure, empty_graph=(row.edges == 0) if valid else None,
                      record_path=row.record_path, record_sha256=row.record_sha256)
        check = dict(labels, status='verified' if valid else 'failure',
                     adj_path=row.adj_path, adj_sha256=row.adj_sha256,
                     nodes=row.nodes, edges=row.edges, graph_sha256_verified=False)
        reference = {metric: [] for metric in METRICS}
        computed = dict.fromkeys(METRICS, float('nan'))
        if valid:
            graph = _read_graph(root, row)
            check['graph_sha256_verified'] = True
            computed = dict(density=nx.density(graph),
                            avg_clustering_coef=nx.average_clustering(graph),
                            global_efficiency=nx.global_efficiency(graph))
            reference = _reference_metrics(row.original_run_id, len(graph), graph.number_of_edges())
            verified += 1
        for metric in METRICS:
            observed = getattr(row, metric)
            if not pd.isna(observed) and not math.isfinite(float(observed)):
                raise ValueError(f'{row.run_id}: non-finite recorded {metric}.')
            check[f'recorded_{metric}'] = observed
            check[f'recomputed_{metric}'] = computed[metric]
            if metric != 'global_efficiency':
                comparable = valid and not pd.isna(observed)
                delta = abs(computed[metric] - observed) if comparable else float('nan')
                check[f'{metric}_absolute_error'] = delta
                check[f'{metric}_check'] = ('match' if comparable else
                                             'undefined' if valid else 'failure')
                if comparable and delta > 1e-12:
                    raise ValueError(f'{row.run_id}: {metric} mismatch exceeds 1e-12.')
            values = reference[metric]
            references.append(dict(
                labels, evidence_type=LABEL, metric=metric, observed_value=observed,
                reference_replicates=len(values),
                reference_failures=0, reference_skipped_primary_failure=not valid,
                reference_mean=float(np.mean(values)) if values else float('nan'),
                reference_sd=float(np.std(values, ddof=1)) if values else float('nan'),
            ))
        quality.append(check)
    tidy = pd.DataFrame(references)
    aggregate = []
    for keys, group in tidy.groupby(GROUPS + ['metric'], dropna=False, sort=True):
        aggregate.append(dict(
            zip(GROUPS + ['metric'], keys), cohort='primary', evidence_type=LABEL,
            attempts=len(group), valid_graphs=int(group.valid_graph.sum()),
            failed_attempts=int((~group.valid_graph).sum()),
            valid_empty_graphs=int(group.empty_graph.eq(True).sum()),
            reference_failures=int(group.reference_failures.sum()),
            reference_skipped_primary_failures=int(group.reference_skipped_primary_failure.sum()),
            observed_defined=int(group.observed_value.notna().sum()),
            observed_mean=group.observed_value.mean(),
            observed_between_run_sd=group.observed_value.std(ddof=1),
            reference_defined=int(group.reference_mean.notna().sum()),
            reference_mean=group.reference_mean.mean(),
            reference_between_run_sd=group.reference_mean.std(ddof=1),
            mean_within_run_reference_sd=group.reference_sd.mean(),
        ))
    manifest = dict(
        schema_version=1, cohort='primary', primary_csv=PRIMARY_CSV.as_posix(),
        primary_csv_sha256=PRIMARY_CSV_SHA256, attempts=len(primary),
        verified_graphs=verified, failed_attempts=len(primary) - verified,
        synthetic_graphs=verified * REFERENCE_REPLICATES,
        metric_checks={metric: {status: sum(row[f'{metric}_check'] == status for row in quality)
                                for status in ('match', 'undefined', 'failure')}
                       for metric in ('density', 'avg_clustering_coef')},
        reference_replicates=REFERENCE_REPLICATES, reference_model='G(n,m)',
        reference_MC_reps=REFERENCE_REPLICATES, reference_SD_is_CI=False,
        reference_SD_interpretation='Sample SD across eight Monte Carlo draws, not a confidence interval. Between-run SD and mean within-run SD are descriptive only.',
        valid_empty_graphs=int(sum(row['edges'] == 0 for row in quality if row['valid_graph'])),
        reference_empty_graphs=REFERENCE_REPLICATES * int(sum(row['edges'] == 0 for row in quality if row['valid_graph'])),
        reference_failures=0, reference_skipped_primary_failures=len(primary) - verified,
        reference_failure_policy='Fail fast; reference generation exceptions abort without writing new outputs. Primary failures are skipped, not successful zero graphs.',
        record_provenance='record_path and record_sha256 must match the hash-sealed primary CSV; parent core verifies receipt bytes. This lane hashes adjacency bytes only, not raw receipt archives.',
        seed_scheme='first 8 SHA-256 bytes, big-endian: primary-graph-evaluation-v1:{original_run_id}:{replicate}; replicate=0..7',
        metric_absolute_tolerance=1e-12, reference_sd_ddof=1,
        networkx_version=nx.__version__, empirical_ground_truth=False,
        primary_hypothesis_test=False, baseline_pvalues=False,
        interpretation='Equal-n/equal-m synthetic descriptive context only; not empirical ground truth, a realism benchmark, or a new primary hypothesis test. Density is fixed by n,m. Eight draws do not establish null convergence. Efficiency observations are retained from CSV; recomputed efficiency is descriptive, not a frozen-math verification claim.',
        undefined_policy='Preserve missing observations and failure rows; only defined metrics are compared. Empty 50-node graphs have defined zero density, clustering and efficiency.',
        files=['graph_quality.csv', 'synthetic_reference_per_run.csv',
               'synthetic_reference_descriptive.csv', 'graph_evaluation.json'],
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(quality).to_csv(output / manifest['files'][0], index=False)
    tidy.to_csv(output / manifest['files'][1], index=False)
    pd.DataFrame(aggregate).to_csv(output / manifest['files'][2], index=False)
    (output / manifest['files'][3]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return manifest
