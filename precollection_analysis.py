"""Finite-panel, schedule-block analysis; no generation or credential access.

The panel average is undefined if even one model is missing. That prevents a
failed model from quietly changing which model population a contrast describes.
"""
import itertools
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from statistical_review import holm

ROOT = Path(__file__).resolve().parent
MODELS = ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol']
METHODS = ['global', 'local', 'sequential', 'iterative']
CONDITIONS = [('us', 'english'), ('india', 'english'), ('japan', 'english'),
              ('brazil', 'english'), ('us', 'hindi'), ('us', 'japanese'), ('us', 'portuguese')]
ROSTER = '26744d2c8fb08406aef3ccba968799473bf7b47456d38baadd7595a5882e90ac'
SPEC_SHA256 = 'ea4dfddc908bba2494850ee0dc0d0aa312bc4e7617abb7197fe6bdc63fa20e81'
IDENTITY = ('contract', 'model', 'method', 'culture', 'language', 'seed', 'wording', 'roster_sha256')


def load_spec():
    spec = json.loads((ROOT/'precollection_spec.json').read_text(encoding='utf8'))
    validate_spec(spec)
    return spec


def validate_spec(spec):
    encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != SPEC_SHA256:
        raise ValueError('Complete analysis specification changed; version and review it.')
    expected = {m: 'density' if m == 'global' else 'avg_clustering_coef' for m in METHODS}
    if (spec.get('main_authorized') is not False or spec.get('models') != MODELS
            or spec.get('roster_sha256') != ROSTER
            or spec.get('methods') != METHODS or spec.get('conditions') != [list(c) for c in CONDITIONS]
            or spec.get('schedule_seeds') != list(range(44000, 44032))
            or spec.get('primary_family_size') != 24 or spec.get('alpha') != .05
            or spec.get('wording') != 'reference'
            or spec.get('primary_endpoints') != expected):
        raise ValueError('Analysis candidate differs; version and review a new specification.')


def design(spec):
    validate_spec(spec)
    return [dict(contract=spec['version'], model=model, method=method,
                 culture=country, language=language, seed=seed, wording=spec['wording'], roster_sha256=ROSTER)
            for seed, method, model, (country, language) in itertools.product(
                spec['schedule_seeds'], spec['methods'], spec['models'], spec['conditions'])]


def endpoint(row, metric):
    if type(row.get('valid_graph')) is not bool:
        raise ValueError('valid_graph must be an explicit boolean.')
    value = row.get(metric)
    if not row['valid_graph']:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            raise ValueError('An invalid graph cannot carry a measured endpoint.')
        return None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Endpoint must be finite numeric data or explicitly undefined.')
    return float(value)


def denominators(rows, metrics):
    result = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        for metric in metrics:
            values = [endpoint(r, metric) for r in selected]
            result.append(dict(method=method, endpoint=metric, attempted=len(selected),
                valid=sum(r['valid_graph'] for r in selected),
                nonempty=sum(r['valid_graph'] and r.get('edges', 0) > 0 for r in selected),
                endpoint_defined=sum(v is not None for v in values)))
    return result


def primary_contrasts(rows, spec):
    """All 24 hypotheses remain in the family, including suppressed tests.

    Missingness bounds retain observed parts of each four-model average and
    bound each missing metric in [0, 1]. They are identification bounds, NOT CIs.
    """
    planned = {tuple(c[k] for k in IDENTITY) for c in design(spec)}
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in IDENTITY)
        if key in indexed or key not in planned:
            raise ValueError('Duplicate cell or mixed experiment identity.')
        metric = spec['primary_endpoints'][row['method']]
        value = endpoint(row, metric)
        if value is not None and not 0 <= value <= 1:
            raise ValueError('Primary endpoint outside [0, 1].')
        indexed[key] = value
    if set(indexed) != planned:
        raise ValueError('Every planned cell needs an explicit result, including failures.')

    def panel(method, condition, seed):
        values = [indexed[(spec['version'], model, method, *condition, seed, spec['wording'], ROSTER)]
                  for model in MODELS]
        observed = [v for v in values if v is not None]
        lower = sum(observed)/4
        upper = (sum(observed)+4-len(observed))/4
        return (lower if len(observed) == 4 else None), lower, upper, values

    result, probabilities = [], []
    n, m, alpha = len(spec['schedule_seeds']), spec['primary_family_size'], spec['alpha']
    for method, condition in itertools.product(METHODS, CONDITIONS[1:]):
        differences, lower, upper = [], [], []
        model_differences = {model: [] for model in MODELS}
        for seed in spec['schedule_seeds']:
            target, lo, hi, tv = panel(method, condition, seed)
            base, blo, bhi, bv = panel(method, CONDITIONS[0], seed)
            lower.append(lo-bhi)
            upper.append(hi-blo)
            if target is not None and base is not None:
                differences.append(target-base)
            for model, a, b in zip(MODELS, tv, bv):
                if a is not None and b is not None:
                    model_differences[model].append(a-b)
        count = len(differences)
        row = dict(hypothesis_id=f'{method}:{condition[0]}:{condition[1]}', method=method,
            culture=condition[0], language=condition[1], endpoint=spec['primary_endpoints'][method],
            planned_schedule_blocks=n, complete_schedule_blocks=count,
            complete_block_mean=float(np.mean(differences)) if count else None,
            missingness_lower=float(np.mean(lower)), missingness_upper=float(np.mean(upper)),
            status='MISSING_NO_INFERENCE', raw_p=None, holm_p=None, simultaneous_ci=None,
            per_model={model: dict(complete_blocks=len(ds), mean=float(np.mean(ds)) if ds else None)
                       for model, ds in model_differences.items()})
        probability = 1.
        if count == n:
            row['status'] = 'ZERO_VARIANCE_NO_T_INFERENCE'
            sample = np.asarray(differences)
            se = float(sample.std(ddof=1)/np.sqrt(n))
            if np.ptp(sample) > 1e-12 and se > 0:
                probability = float(2*t.sf(abs(float(sample.mean())/se), n-1))
                half = float(t.isf(alpha/(2*m), n-1)*se)
                row.update(status='COMPLETE_ASSUMPTIONS_REQUIRED', raw_p=probability,
                    simultaneous_ci=[max(-1., float(sample.mean())-half), min(1., float(sample.mean())+half)])
        result.append(row)
        probabilities.append(probability)
    for row, adjusted in zip(result, holm(probabilities)):
        if row['raw_p'] is not None:
            row['holm_p'] = float(adjusted)
    return result


def precision_scenarios(spec):
    """Assumed SDs, not power estimates from two calibration schedules."""
    validate_spec(spec)
    return [dict(blocks=n, assumed_block_difference_sd=sd,
                 simultaneous_half_width=float(t.isf(.05/(2*24), n-1)*sd/np.sqrt(n)),
                 planning_target_half_width=spec['precision_scenario_half_width'],
                 meets_target_under_assumed_sd=bool(t.isf(.05/(2*24), n-1)*sd/np.sqrt(n) <= spec['precision_scenario_half_width']),
                 family_size=24, evidence_type='ASSUMPTION_BASED_NOT_MEASURED_POWER')
            for n, sd in itertools.product([8, 16, 32, 64], [.02, .05, .1, .2])]
