"""Offline statistical safety checks, not a scientific-approval certificate.

Paired-t output is assumption-dependent. Bounded-mean intervals provide a
conservative sensitivity analysis without a normality assumption. Both require
independent repetition blocks; no method repairs dependent API generations.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, binomtest

OUTPUT = Path(__file__).resolve().parent / 'outputs/statistical_review_v6/report.json'


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Supply one full family of finite probabilities in [0, 1].')
    order = np.argsort(values, kind='stable')
    result = np.empty_like(values)
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


def family_summary(differences, difference_bound=1.0, alpha=.05):
    """Rows are predeclared hypotheses, columns are matched network repetitions.

    Bounds must be declared from the metric, not estimated from the sample.
    Missing values suppress inference for that row, retaining its place in the
    family as p=1 for adjustment. There is no available-case significance test.
    """
    data = np.asarray(differences, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2:
        raise ValueError('Need a full hypothesis-by-repetition matrix with at least two repetitions.')
    if not np.isfinite(difference_bound) or difference_bound <= 0 or not 0 < alpha < 1:
        raise ValueError('Invalid predeclared bound or alpha.')
    if np.isinf(data).any() or (np.abs(data[np.isfinite(data)]) > difference_bound).any():
        raise ValueError('Difference outside its predeclared metric bounds.')
    m, n = data.shape
    rows, t_p, bounded_p = [], [], []
    radius = 2 * difference_bound * np.sqrt(np.log(2 * m / alpha) / (2 * n))
    for sample in data:
        count = int(np.isfinite(sample).sum())
        row = dict(planned_repetitions=n, defined_repetitions=count, status='MISSING_NO_INFERENCE',
                   mean_difference=None, paired_t_p=None, paired_t_ci=None, bounded_p=None, bounded_ci=None)
        tp = bp = 1.0
        if count == n:
            mean, se = float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n))
            bp = min(1., float(2 * np.exp(-n * mean**2 / (2 * difference_bound**2))))
            row.update(status='COMPLETE_ASSUMPTIONS_REQUIRED', mean_difference=mean, bounded_p=bp,
                       bounded_ci=[max(-difference_bound, mean - radius), min(difference_bound, mean + radius)])
            # Equal floating-point observations can have tiny nonzero sample SD.
            if np.ptp(sample) > 0 and se > 0:
                tp = float(2 * t.sf(abs(mean / se), n - 1))
                half = float(t.isf(alpha / m / 2, n - 1) * se)
                row.update(paired_t_p=tp, paired_t_ci=[max(-difference_bound, mean-half), min(difference_bound, mean+half)])
            else:
                row['status'] = 'ZERO_VARIANCE_NO_T_INFERENCE'
        rows.append(row)
        t_p.append(tp)
        bounded_p.append(bp)
    for row, tp, bp in zip(rows, holm(t_p), holm(bounded_p)):
        row['paired_t_holm_p'] = None if row['paired_t_p'] is None else float(tp)
        row['bounded_holm_p'] = None if row['bounded_p'] is None else float(bp)
    return rows


def simulate(worlds=1000):
    if type(worlds) is not int or worlds < 100:
        raise ValueError('Use at least 100 labelled offline simulation worlds.')
    rng, rows = np.random.default_rng(20261002), []
    for n in [8, 32]:
        for scenario in ['symmetric_bounded', 'skewed_zero_mean', 'bimodal_zero_mean', 'correlated_hypotheses']:
            t_reject, bounded_reject = 0, 0
            for _ in range(worlds):
                if scenario == 'symmetric_bounded':
                    data = rng.uniform(-1, 1, (24, n))
                elif scenario == 'skewed_zero_mean':
                    data = rng.binomial(1, .05, (24, n)) - .05
                elif scenario == 'bimodal_zero_mean':
                    data = rng.choice([-1., 1.], (24, n))
                else:
                    data = np.repeat(rng.uniform(-1, 1, (1, n)), 24, axis=0)
                results = family_summary(data)
                t_reject += any(r['paired_t_holm_p'] is not None and r['paired_t_holm_p'] < .05 for r in results)
                bounded_reject += any(r['bounded_holm_p'] < .05 for r in results)
            for name, failures in [('paired_t', t_reject), ('bounded_mean', bounded_reject)]:
                interval = binomtest(failures, worlds).proportion_ci(method='exact')
                rows.append(dict(scenario=scenario, repetitions=n, hypotheses=24, simulated_worlds=worlds,
                    procedure=name, family_false_positive_fraction=failures/worlds,
                    monte_carlo_p025=interval.low, monte_carlo_p975=interval.high))
    return dict(status='OFFLINE_METHOD_SENSITIVITY_NOT_SCIENTIFIC_APPROVAL', simulation_seed=20261002,
        paid_calls=0, rows=rows, main_authorized=False,
        caveats=['Synthetic bounded differences, not generated networks or power estimates.',
            'All-null FWER stress check is not a proof of validity for arbitrary data distributions.',
            'Hoeffding bounds assume independent bounded repetitions, and can be very wide/low-powered.',
            'No selected meaningful-effect threshold, equivalence margin, final sample allocation, or independent scientific approval.'])


if __name__ == '__main__':
    result = simulate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
