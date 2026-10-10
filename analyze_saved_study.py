"""Public, key-free reanalysis of the reviewed V5 calibration, not new LLM data.

Raw receipts/graphs stay immutable. Reference ensembles quantify structure beyond
edge count or degrees; their percentiles are NOT treatment confidence intervals.
"""
import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct, t

import revision224 as study
from analyze_networks import compute_network_metrics, compute_coleman_homophily, compute_age_assortativity
from export_calibration_viewer import load_calibration, sha256
from make_matched_baselines import matched_baselines

OUTPUT = study.ROOT / 'stats/public_v5_reanalysis'
CATEGORICAL = [key for key in study.prompts.DEMOS if key != 'age']
META = ['run_id', 'model', 'method', 'culture', 'language', 'repetition', 'seed', 'roster_sha256']


def measures(graph, roster):
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    gini = np.abs(degrees[:, None] - degrees).sum() / (2 * len(graph) * degrees.sum()) if degrees.sum() else None
    return study.clean_numbers({**compute_network_metrics(graph),
        'degree_one_share': float(np.mean(degrees == 1)),
        'isolate_share': float(np.mean(degrees == 0)), 'degree_gini': gini,
        **{'coleman_' + key: compute_coleman_homophily(graph, roster, key)[0] for key in CATEGORICAL},
        'age_assortativity': compute_age_assortativity(graph, roster)})


def detectable_effect(n, hypotheses=1, alpha=0.05, power=0.8):
    """Two-sided paired-t MDES in SDs of differences; Bonferroni planning bound.

    This assumes independent, approximately normal paired differences. It is a
    sensitivity calculation, not measured power or a guarantee for graph metrics.
    """
    if type(n) is not int or n < 2 or type(hypotheses) is not int or hypotheses < 1:
        raise ValueError('Need at least two pairs and a positive hypothesis count.')
    if not 0 < alpha < 1 or not alpha / hypotheses < power < 1:
        raise ValueError('Require 0 < per-test alpha < power < 1.')
    critical = t.isf(alpha / hypotheses / 2, n - 1)
    def achieved(effect):
        noncentrality = effect * math.sqrt(n)
        return nct.sf(critical, n - 1, noncentrality) + nct.cdf(-critical, n - 1, noncentrality)
    return brentq(lambda effect: achieved(effect) - power, 0, 100)


def decision_rows(record):
    """Recover requested cardinality from the frozen RNG and check actor order.

    The receipt stores actual attempts, not the count prompt. Matching the saved
    actor sequence first makes this reconstruction auditable rather than guessed.
    """
    cell, events, requests = record['cell'], record['events'], record['requests']
    quota_by_actor = {}
    if cell['method'] != 'global':
        rng = np.random.RandomState(cell['seed'])
        ids = list(study.prompts.adult_roster())
        actors = rng.choice(ids, len(ids), replace=False).tolist()
        if [event['persona'] for event in events[:50]] != actors:
            raise ValueError('Cannot reconstruct quotas: saved actor order differs.')
        quota_by_actor = {actor: int(min(max(rng.exponential(5), 1), 20, len(ids)-1)) for actor in actors}
    offset, rows = 0, []
    for index, event in enumerate(events):
        attempts = requests[offset:offset + event['attempts']]
        if len(attempts) != event['attempts'] or not attempts or attempts[-1]['parse'].get('valid') is not True:
            raise ValueError('Decision attempts do not end in a verified valid response.')
        if any(a['parse'].get('valid') is not False for a in attempts[:-1]):
            raise ValueError('Unexpected successful intermediate attempt.')
        count = None if cell['method'] == 'global' else quota_by_actor[event['persona']] if index < 50 else 1
        rows.append({**{key: cell[key] for key in META}, 'step': event['step'], 'actor': event['persona'],
            'decision_method': event['method'], 'requested_count': count,
            'attempts': len(attempts), 'first_attempt_failed': len(attempts) > 1,
            'failure_messages': json.dumps([a['parse'] for a in attempts[:-1]], ensure_ascii=False),
            'count_source': 'frozen_rng_reconstructed_and_actor_order_checked' if count is not None else 'unconstrained'})
        offset += event['attempts']
    if offset != len(requests):
        raise ValueError('Unassigned request attempts in receipt.')
    return rows


def reference_summary(observed, controls):
    rows = []
    metrics = list(measures(nx.empty_graph(study.prompts.adult_roster()), study.prompts.adult_roster()))
    for (run_id, baseline), group in controls.groupby(['run_id', 'baseline']):
        accepted = group[group['rewiring_target_reached']]
        for metric in metrics:
            values = pd.to_numeric(accepted[metric], errors='coerce').dropna()
            value = observed.loc[run_id, metric]
            mean = float(values.mean()) if len(values) else None
            rows.append(dict(run_id=run_id, baseline=baseline, metric=metric,
                attempted_draws=len(group), completed_draws=len(accepted), defined_draws=len(values),
                observed=value, reference_mean=mean,
                observed_minus_reference_mean=value - mean if pd.notna(value) and mean is not None else None,
                reference_p025=values.quantile(.025) if len(values) > 1 else None,
                reference_p975=values.quantile(.975) if len(values) > 1 else None,
                evidence_type='descriptive_reference_not_treatment_CI'))
    return pd.DataFrame(rows)


def analyze(control_repetitions=20):
    if type(control_repetitions) is not int or not 2 <= control_repetitions <= 1000:
        raise ValueError('Control repetitions must be an integer in [2, 1000].')
    # Verify every input before writing a derivative. No private Budget is opened.
    config, roster, verified = load_calibration()
    observed, controls, compliance = [], [], []
    for path, _, record, graph in verified:
        meta = {key: record['cell'][key] for key in META}
        observed.append({**meta, 'nodes': len(graph), 'edges': graph.number_of_edges(), **measures(graph, roster)})
        compliance.extend(decision_rows(record))
        for draw in range(control_repetitions):
            seed = int.from_bytes(hashlib.sha256(f"{path.stem}:reference:{draw}".encode()).digest()[:4], 'big')
            for name, control, complete in matched_baselines(graph, roster, seed=seed, categorical=CATEGORICAL):
                if name == 'equal_weight_demographic_similarity' and draw:
                    continue  # Repeating a deterministic reference adds no evidence.
                if set(control) != set(roster) or control.number_of_edges() != graph.number_of_edges():
                    raise ValueError('Reference changed the roster or edge count.')
                if name == 'degree_preserving_rewire' and dict(control.degree()) != dict(graph.degree()):
                    raise ValueError('Rewiring changed degrees.')
                controls.append({**meta, 'baseline': name, 'draw': draw, 'seed_control': seed,
                    'rewiring_target_reached': complete, 'source_receipt_sha256': sha256(path),
                    'evidence_type': 'offline_reference_not_LLM_output', **measures(control, roster)})
    data, references = pd.DataFrame(observed), pd.DataFrame(controls)
    contrasts = []
    for left, right, dimension in study.contrast_pairs(observed):
        for metric in measures(verified[0][3], roster):
            contrasts.append({**study.contrast_labels(left, right, dimension),
                'reference_run': left['run_id'], 'comparison_run': right['run_id'], 'repetition': left['repetition'],
                'metric': metric, 'comparison_minus_reference': right[metric] - left[metric] if left[metric] is not None and right[metric] is not None else None,
                'evidence_type': 'matched_labels_descriptive_not_confirmatory'})
    power = pd.DataFrame([dict(pairs=n, hypotheses=h, alpha=.05, power=.8,
                              mdes_sd_of_paired_differences=detectable_effect(n, h))
                          for n in [8, 16, 32, 64] for h in [1, 6, 24, 84, 288]])
    tables = {'network_metrics.csv': data, 'reference_draws.csv': references,
              'reference_summary.csv': reference_summary(data.set_index('run_id'), references),
              'decision_compliance.csv': pd.DataFrame(compliance),
              'descriptive_contrasts.csv': pd.DataFrame(contrasts), 'power_sensitivity.csv': power}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = dict(status='WRITING', paid_calls=0, main_authorized=False)
    target = OUTPUT / 'report.json'
    # A failed export must not leave a previous COMPLETE report blessing mixed CSVs.
    target.write_text(json.dumps(report), encoding='utf-8')
    for name, table in tables.items():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in cast', category=RuntimeWarning)
            table.to_csv(OUTPUT / name, index=False)
    report.update(status='VERIFIED_V5_EXPLORATORY_REANALYSIS', networks=len(verified),
        reference_graphs=len(controls), reference_draws_per_stochastic_family=control_repetitions,
        failed_rewiring_draws=sum(not r['rewiring_target_reached'] for r in controls),
        source_protocol_sha256=study.digest(config), source_receipts_sha256={p.name: sha256(p) for p, *_ in verified},
        analysis_source_sha256={name: sha256(study.ROOT / name) for name in ['analyze_saved_study.py', 'export_calibration_viewer.py', 'make_matched_baselines.py', 'analyze_networks.py']},
        tables_sha256={name: sha256(OUTPUT / name) for name in tables},
        limitations=['Not a revised-model run or main study; V5 is exploratory.',
            'No key or private billing ledger required; provider billing is not independently audited.',
            'Reference percentiles are not treatment confidence intervals; completed swaps do not prove mixing.',
            'Undefined metrics remain blank, not zero; failed rewires remain in draw tables but not reference summaries.',
            'Nominally matched seed/repetition labels do not establish paired statistical power.',
            'Instruction-language effects use US-only contrasts, not pooled-country English means.'])
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return {key: report[key] for key in ['status', 'networks', 'reference_graphs', 'failed_rewiring_draws', 'paid_calls', 'main_authorized']}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--control-repetitions', type=int, default=20)
    args = cli.parse_args()
    print(json.dumps(analyze(args.control_repetitions), indent=2))
