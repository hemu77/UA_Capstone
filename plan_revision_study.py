"""Print the planned request volume without importing an SDK or spending tokens.

A graph is not one API call: iterative generation asks each person to revise
ties repeatedly. This estimate counts attempts, not imaginary measured costs.
Dollar pricing needs verified rates and prompt/output limits before approval.
"""
import argparse
import itertools
import json
from pathlib import Path


def build_plan(config):
    personas = json.loads(Path(config['persona_file']).read_text(encoding='utf-8'))
    n = len(personas)
    if n != config['expected_personas'] or n < 2:
        raise ValueError('Persona count does not match the study protocol.')
    methods = config['methods']
    models = config['models']
    if not methods or not models or set(methods) - {'global', 'local', 'sequential', 'iterative'}:
        raise ValueError('Provide supported methods and at least one model.')
    for field in ('cultures', 'languages'):
        if not config[field] or len(config[field]) != len(set(config[field])):
            raise ValueError(f'{field} must be a nonempty, duplicate-free list.')
    for field in ('confirmatory_rosters', 'confirmatory_repetitions'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer.')
    if len(set(methods)) != len(methods) or len(set(models)) != len(models):
        raise ValueError('Duplicate methods or models would duplicate experimental cells.')
    rounds = config['iterative_rounds']
    attempts = config['max_attempts_per_request']
    if type(rounds) is not int or type(attempts) is not int or rounds < 0 or not 1 <= attempts <= 3:
        raise ValueError('Invalid iteration count or retry limit.')
    calls = {'global': 1, 'local': n, 'sequential': n, 'iterative': n * (1 + 2 * rounds)}
    pilot = []
    for model, method in itertools.product(models, methods):
        pilot.append(dict(model=model, method=method, culture='us', language='english'))
    for language, method in itertools.product([v for v in config['languages'] if v != 'english'], methods):
        pilot.append(dict(model=models[0], method=method, culture='us', language=language))
    for row in pilot:
        row.update(seed=config['pilot_seed'], personas=n, max_requests_before_retries=calls[row['method']])
    conditions = config.get('main_conditions', list(itertools.product(config['cultures'], config['languages'])))
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
           pair[0] not in config['cultures'] + ['neutral'] or pair[1] not in config['languages']
           for pair in conditions):
        raise ValueError('Main conditions require a supported context-language pair.')
    if not conditions or len({tuple(c) for c in conditions}) != len(conditions):
        raise ValueError('Main conditions must be nonempty and unique.')
    per_method = (len(conditions) * len(models)
                  * config['confirmatory_rosters'] * config['confirmatory_repetitions'])
    return {
        'status': 'NOT_AUTHORIZED_NO_API_CALLS',
        'pilot_networks': len(pilot),
        'pilot_requests_before_retries_upper_bound': sum(r['max_requests_before_retries'] for r in pilot),
        'pilot_attempts_upper_bound': sum(r['max_requests_before_retries'] for r in pilot) * attempts,
        'confirmatory_networks_draft': per_method * len(methods),
        'confirmatory_requests_before_retries_upper_bound': per_method * sum(calls[m] for m in methods),
        'dollar_estimate': None,
        'dollar_estimate_status': 'Requires verified model prices, supported settings and token limits; no approval implied.',
        'excluded_from_totals': ['country-grounded rosters', 'robustness panels', 'unselected culture-language interactions'],
        'approval_gates': config['approval_gates'],
        'pilot': pilot,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='study_protocol.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(build_plan(config), indent=2))
