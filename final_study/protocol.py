"""The adopted protocol and executable contract, separate from frozen history.

The statistical specification is reused verbatim. Its old 'candidate' label is
provenance, not permission: this enclosing contract adopts the specification.
Spending still requires an explicit stage authorization and the live gate.
"""
import contextlib
from datetime import date
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import calibration_v6 as engine
import precollection_analysis as analysis
import precollection_candidate as candidate
from precollection_audit import write_json
import repair_v7 as legacy
import repair_v8 as v8
import revision_v7_prompts as prompts

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'final_study/contract.json'
OUTPUT = ROOT / 'outputs/final_study'
VERSION = 'final-instruction-study-v1'
HINDI_OLD = 'मित्रताओं का एक संभावित विन्यास उत्पन्न करें।'
HINDI_NEW = 'मित्रताओं का ऐसा विन्यास उत्पन्न करें जो यथार्थसंगत और विश्वसनीय हो।'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def system(method, language, country, actor=None, count=None):
    text = prompts.treatment_prompt(method, language, country, actor, count,
                                    'reference', 'a0_b0' if method == 'global' else None)
    if method == 'global':
        prefix = v8.CONSTRUCTION[language]
        if language == 'hindi':
            if HINDI_OLD not in prefix:
                raise ValueError('Reviewed Hindi source changed.')
            prefix = prefix.replace(HINDI_OLD, HINDI_NEW)
        text = prefix + text
    return text


def catalog():
    # Render the actual callable used by execution, not a parallel prompt copy.
    rows = []
    for country, language in analysis.CONDITIONS:
        for method in ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']:
            for count in (range(1, 21) if method in {'local', 'sequential'} else [None]):
                actor = None if method == 'global' else '0'
                rows.append(dict(country=country, language=language, method=method, count=count,
                    system=system(method, language, country, actor, count),
                    retry=None if method == 'global' else prompts.treatment_retry(
                        method, language, country, actor, count, wording='reference')))
    return rows


def sources():
    # Includes the historical import closure and the complete new execution package.
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('study_protocol*.json'))
    paths += list((ROOT/'final_study').glob('*.py'))
    paths += [ROOT/n for n in ['precollection_spec.json', 'requirements.lock.txt', '.python-version']]
    return {p.relative_to(ROOT).as_posix(): engine.sha(p) for p in sorted(paths)}


def build_contract():
    return dict(version=VERSION, status='ADOPTED_PENDING_LIVE_CHECK', main_authorized=False,
        analysis=analysis.load_spec(), analysis_adopted=True, sources=sources(),
        roster_sha256=analysis.ROSTER, runtime=engine.frozen.runtime_versions(),
        catalog_sha256=legacy.digest(catalog()),
        rates={m: list(legacy.paid.RATES[m]) for m in analysis.MODELS},
        prices_verified_on='2026-10-18', prices_valid_through='2026-10-25',
        price_sources={m: 'https://developers.openai.com/api/docs/models/'+m for m in analysis.MODELS},
        gpt41_shutdown='2027-01-14', requested_service_tier='default',
        settings={m: {method: engine.settings(m, method) for method in analysis.METHODS}
                  for m in analysis.MODELS},
        main_attempts=3584, schedule_blocks=32, hindi_check_attempts=8,
        hindi_check_seeds=[45000, 45001], execution_order_seed=44032,
        global_parser='Validate every line; collapse exact undirected duplicates; retain raw response and flags; NONE valid.',
        transport_policy='SDK retries zero. Stop on uncertain transport; retain full reservation. Explicit one-request recovery only.',
        retries_per_received_local_decision=3, calibration_reuse=False,
        claim=analysis.load_spec()['claim'], human_bilingual_validation=False,
        live_gate='Eight Hindi global attempts, two schedules per model; all must parse and finish. Empty graphs remain valid, not replaced.')


def verify_contract(spec):
    if spec != build_contract():
        raise ValueError('Frozen contract/source/runtime differs. Do not execute or re-anchor spending.')
    return spec


def load_contract():
    return verify_contract(read(CONTRACT))


def freeze():
    spec = build_contract()
    if CONTRACT.exists() and read(CONTRACT) != spec:
        raise ValueError('An existing freeze differs; do not overwrite it silently.')
    write_json(CONTRACT, spec)
    return dict(contract_sha256=legacy.digest(spec), main_attempts=spec['main_attempts'], main_authorized=False)


def cells(spec, stage):
    if stage == 'main':
        rows = analysis.design(spec['analysis'])
    elif stage == 'hindi_check':
        rows = [dict(model=m, method='global', culture='us', language='hindi', seed=s,
                     wording='reference', roster_sha256=analysis.ROSTER)
                for s, m in itertools.product(spec['hindi_check_seeds'], analysis.MODELS)]
    else:
        raise ValueError('Unknown stage.')
    fingerprint = legacy.digest(spec)
    result = []
    for row in rows:
        row = {k: v for k, v in row.items() if k != 'contract'}
        result.append(dict(**row, personas=50, phase='main', stage=stage,
            repetition=row['seed']-(44000 if stage == 'main' else 45000),
            global_arm='a0_b0' if row['method'] == 'global' else None,
            run_id=f"final_{fingerprint[:12]}_{stage}_{row['method']}_{row['model']}_{row['culture']}_{row['language']}_s{row['seed']}",
            final_contract_sha256=fingerprint, frozen_source_sha256=spec['sources'],
            requested_service_tier='default', runtime_versions=spec['runtime']))
    random.Random(spec['execution_order_seed']).shuffle(result)
    return result


def folder(spec, stage):
    if stage not in {'main', 'hindi_check'}:
        raise ValueError('Unknown stage.')
    return OUTPUT/legacy.digest(spec)[:12]/stage


def check_prices(spec, today=None):
    today = today or date.today()
    if not date.fromisoformat(spec['prices_verified_on']) <= today <= date.fromisoformat(spec['prices_valid_through']):
        raise ValueError('Price verification window expired; refresh evidence before new calls.')
    if today >= date.fromisoformat(spec['gpt41_shutdown']):
        raise ValueError('GPT-4.1 shutdown reached; do not substitute a model silently.')


@contextlib.contextmanager
def adapter(cell, spec):
    original_parse = engine.revised.parse_response
    proxy = SimpleNamespace(**{k: v for k, v in vars(engine.revised).items() if not k.startswith('__')})
    proxy.system_prompt = system
    proxy.retry_prompt = lambda method, language, country, actor=None, count=None, preserving=False: prompts.treatment_retry(
        method, language, country, actor, count, preserving, 'reference', 'a0_b0' if method == 'global' else None)

    def parse(**kwargs):
        if kwargs['method'] == 'global':
            flags = candidate.canonical_global(kwargs['response'], proxy.previous.adult_roster())
            kwargs = {**kwargs, 'response': flags['canonical_response']}
        return original_parse(**kwargs)

    proxy.parse_response = parse
    with patch.object(engine, 'revised', proxy), patch.object(engine, 'contract', load_contract):
        yield
