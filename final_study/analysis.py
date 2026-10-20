"""Eight-schedule specification using the unchanged, reviewed statistical math.

Load an isolated copy so adapting its specification validator cannot weaken the
historical 32-schedule validator used elsewhere in the repository.

Version 2 adds one predeclared partial-study rule. A complete study gives
exactly the version-1 results. When schedule blocks are incomplete, a whole
block (all four models, both arms) is dropped from that contrast only, and
inference runs on the remaining blocks if at least FALLBACK_MIN_BLOCKS remain.
The fixed four-model panel is never replaced by a three-model average.
"""
import importlib.util
import itertools

import numpy as np
from scipy.stats import t

import precollection_analysis as historical

MODELS, METHODS, CONDITIONS = historical.MODELS, historical.METHODS, historical.CONDITIONS
ROSTER, IDENTITY = historical.ROSTER, historical.IDENTITY
FALLBACK_MIN_BLOCKS = 6


def load_spec():
    spec = historical.load_spec()
    spec.update(version='final-eight-schedule-analysis-v3-terminal-coverage',
                status='ADOPTED_EIGHT_EQUAL_SCHEDULES',
                schedule_seeds=list(range(44000, 44008)),
                precision_scenario_half_width=None,
                fallback_min_complete_blocks=FALLBACK_MIN_BLOCKS,
                missingness='Never zero-fill failures or undefined endpoints. A schedule block is complete '
                    'for a contrast only when all four models have a defined endpoint in both arms. '
                    'With all eight blocks complete, inference is unchanged from analysis v1. Otherwise '
                    'incomplete blocks are dropped from that contrast only and inference uses the '
                    'remaining blocks if at least six remain; with fewer, primary inference is suppressed '
                    'and the hypothesis keeps its place in the family. Dropping a block assumes its '
                    'missingness is unrelated to the contrast outcome given schedule and condition. '
                    'Every result reports excluded blocks, missing cells, [0, 1]-bounded missing-data '
                    'extremes over all eight blocks, and terminal failure/undefined-endpoint counts '
                    'by method, condition and model as sensitivity evidence. Without a terminal receipt, '
                    'execution status is unknown, not unattempted. Fallback results are labelled PARTIAL, '
                    'never COMPLETE. No stopping based on interim p-values.',
                inference='Paired schedule-block t diagnostics on the complete blocks of each contrast '
                    '(n-1 degrees of freedom for n complete blocks), Holm across exactly 24 hypotheses, '
                    'Bonferroni simultaneous intervals. Assumes independent blocks and an adequate '
                    'approximation for their mean. No equivalence or guaranteed power claim.',
                transport='Governed by the execution contract transport_policy (contract v4 and later). '
                    'This analysis specification neither approves nor restricts transport recovery.')
    return spec


def validate_spec(spec):
    if spec != load_spec():
        raise ValueError('Eight-schedule analysis changed; version and review it.')


_module_spec = importlib.util.spec_from_file_location('_final_schedule_math', historical.__file__)
_math = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_math)
_math.validate_spec = validate_spec


def design(spec):
    return _math.design(spec)


def _values(rows, spec):
    """Index terminal results; absence says nothing about partial paid work."""
    planned = {tuple(c[k] for k in IDENTITY) for c in design(spec)}
    indexed, recorded = {}, {}
    for row in rows:
        key = tuple(row[k] for k in IDENTITY)
        if key in indexed or key not in planned:
            raise ValueError('Duplicate cell or mixed experiment identity.')
        if 'attempted' in row:
            raise ValueError('Use terminal_recorded; terminal receipts cannot establish attempted counts.')
        if type(row.get('terminal_recorded', True)) is not bool:
            raise ValueError('terminal_recorded must be an explicit boolean.')
        has_record = row.get('terminal_recorded', True)
        if not has_record and row['valid_graph']:
            raise ValueError('A cell without a terminal receipt cannot be a valid graph.')
        value = _math.endpoint(row, spec['primary_endpoints'][row['method']])
        if value is not None and not 0 <= value <= 1:
            raise ValueError('Primary endpoint outside [0, 1].')
        indexed[key], recorded[key] = value, has_record
    if set(indexed) != planned:
        raise ValueError('Every planned cell needs an explicit result, including failures and missing receipts.')
    return indexed, recorded


def primary_contrasts(rows, spec):
    """All 24 hypotheses stay in the family, including suppressed tests."""
    indexed, recorded = _values(rows, spec)
    seeds = spec['schedule_seeds']
    n, m, alpha = len(seeds), spec['primary_family_size'], spec['alpha']
    minimum = spec['fallback_min_complete_blocks']

    def key(model, method, condition, seed):
        return (spec['version'], model, method, *condition, seed, spec['wording'], ROSTER)

    def panel(method, condition, seed):
        values = [indexed[key(model, method, condition, seed)] for model in MODELS]
        observed = [v for v in values if v is not None]
        lower = sum(observed)/4
        upper = (sum(observed)+4-len(observed))/4
        return (lower if len(observed) == 4 else None), lower, upper, values

    result, probabilities = [], []
    for method, condition in itertools.product(METHODS, CONDITIONS[1:]):
        differences, complete, lower, upper, missing = [], [], [], [], []
        model_differences = {model: [] for model in MODELS}
        for seed in seeds:
            target, lo, hi, tv = panel(method, condition, seed)
            base, blo, bhi, bv = panel(method, CONDITIONS[0], seed)
            lower.append(lo-bhi)
            upper.append(hi-blo)
            if target is not None and base is not None:
                differences.append(target-base)
                complete.append(seed)
            for arm, values in ((condition, tv), (CONDITIONS[0], bv)):
                for model, value in zip(MODELS, values):
                    if value is None:
                        cell = key(model, method, arm, seed)
                        missing.append(dict(model=model, culture=arm[0], language=arm[1], seed=seed,
                                            terminal_recorded=recorded[cell]))
            for model, a, b in zip(MODELS, tv, bv):
                if a is not None and b is not None:
                    model_differences[model].append(a-b)
        count = len(differences)
        row = dict(hypothesis_id=f'{method}:{condition[0]}:{condition[1]}', method=method,
            culture=condition[0], language=condition[1], endpoint=spec['primary_endpoints'][method],
            planned_schedule_blocks=n, complete_schedule_blocks=count,
            excluded_schedule_seeds=[s for s in seeds if s not in complete], missing_cells=missing,
            complete_block_mean=float(np.mean(differences)) if count else None,
            missingness_lower=float(np.mean(lower)), missingness_upper=float(np.mean(upper)),
            status='MISSING_NO_INFERENCE', raw_p=None, holm_p=None, simultaneous_ci=None,
            per_model={model: dict(complete_blocks=len(ds), mean=float(np.mean(ds)) if ds else None)
                       for model, ds in model_differences.items()})
        probability = 1.
        if count >= minimum:
            full = count == n
            row['status'] = 'ZERO_VARIANCE_NO_T_INFERENCE' if full else 'PARTIAL_ZERO_VARIANCE_NO_T_INFERENCE'
            sample = np.asarray(differences)
            se = float(sample.std(ddof=1)/np.sqrt(count))
            if np.ptp(sample) > 1e-12 and se > 0:
                probability = float(2*t.sf(abs(float(sample.mean()))/se, count-1))
                half = float(t.isf(alpha/(2*m), count-1)*se)
                row.update(status='COMPLETE_ASSUMPTIONS_REQUIRED' if full else 'PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED',
                    raw_p=probability,
                    simultaneous_ci=[max(-1., float(sample.mean())-half), min(1., float(sample.mean())+half)])
        result.append(row)
        probabilities.append(probability)
    for row, adjusted in zip(result, _math.holm(probabilities)):
        if row['raw_p'] is not None:
            row['holm_p'] = float(adjusted)
    return result


def missingness_report(rows, spec):
    """Failure and coverage counts, so readers can judge whether missingness was informative."""
    indexed, recorded = _values(rows, spec)
    valid = {tuple(row[k] for k in IDENTITY): row['valid_graph'] for row in rows}
    report = []
    for method, model, condition in itertools.product(METHODS, MODELS, CONDITIONS):
        keys = [(spec['version'], model, method, *condition, seed, spec['wording'], ROSTER)
                for seed in spec['schedule_seeds']]
        terminal = [k for k in keys if recorded[k]]
        report.append(dict(method=method, model=model, culture=condition[0], language=condition[1],
            planned=len(keys), terminal_results=len(terminal), without_terminal_result=len(keys)-len(terminal),
            endpoint_defined=sum(indexed[k] is not None for k in keys),
            terminal_failures=sum(not valid[k] for k in terminal),
            valid_without_endpoint=sum(valid[k] and indexed[k] is None for k in terminal)))
    return report
