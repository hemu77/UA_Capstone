"""The adopted protocol and executable contract, separate from frozen history.

The reviewed analysis math is reused with eight schedules, not 32. Previously
paid Hindi checks stay under their original contract and are replayed read-only.
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
from . import analysis
import precollection_candidate as candidate
from precollection_audit import write_json
import repair_v7 as legacy
import repair_v8 as v8
import revision_v7_prompts as prompts

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'final_study/contract.json'
OUTPUT = ROOT / 'outputs/final_study'
VERSION = 'final-instruction-study-v6.3-scheduled-checkpoints'
# Superseded before any main attempt; may be replaced by `freeze`.
UNEXECUTED_V2_SHA = '9d7c929fbff5a41865b741f4c0e053d8c72f00e8135e0213ae2ed6618fc37ecc'
UNEXECUTED_V3_SHA = '830d00cb180d3801cbccd2da7ec10c07987648a76d23ff6bbf5bb7f2115a3b4a'
UNEXECUTED_V4_SHA = 'be0e2e5740060bad8694bdc39d10ee2bcd60c50fc67717499395ca0bd490516e'
UNEXECUTED_V5_SHA = '17c2750fe36a368150c73acc1fd080696684f273b5aa2c446d65695210e6d9c4'
UNEXECUTED_V6_SHA = '04100ffe593c3887cabdd7cc74921bb0c65d2f078e0b1fe001cde7b56bdd9fab'
UNEXECUTED_V61_SHA = 'c54e1a30cbdda201c441453bf725d73dc0aa1d036a059ce707f954c07cb8b990'
UNEXECUTED_V62_SHA = 'ff50df9c176ee6eee1d1dd9802002fc57632a56364c2f719332eaab3f472d1cf'
HISTORY = ROOT / 'final_study/history/v1'
HISTORY_SHA = '5886a306a053579ed95f47011a8a41acd893c8ee75480cd25968c64658790607'
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
    paths += [ROOT/'scripts/hoffman2/run.sbatch']
    paths += [HISTORY/'contract.json'] + list((HISTORY/'final_study').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): engine.sha(p) for p in sorted(paths)}


def build_contract():
    return dict(version=VERSION, status='ADOPTED_EIGHT_EQUAL_SCHEDULES', main_authorized=False,
        analysis=analysis.load_spec(), analysis_adopted=True, sources=sources(),
        roster_sha256=analysis.ROSTER, runtime=engine.frozen.runtime_versions(),
        catalog_sha256=legacy.digest(catalog()),
        rates={m: list(legacy.paid.RATES[m]) for m in analysis.MODELS},
        prices_verified_on='2026-10-25', prices_valid_through='2026-11-01',
        price_sources={m: 'https://developers.openai.com/api/docs/models/'+m for m in analysis.MODELS},
        gpt41_shutdown='2027-01-14', requested_service_tier='default',
        settings={m: {method: engine.settings(m, method) for method in analysis.METHODS}
                  for m in analysis.MODELS},
        main_attempts=896, schedule_blocks=8, main_budget_ceiling_usd=120,
        hindi_evidence_contract_sha256=HISTORY_SHA, hindi_check_attempts=8,
        hindi_check_seeds=[45000, 45001], execution_order_seed=44032,
        execution_order='Schedule-block-major: seeds 44000-44007 in ascending order; the 112 cells '
            'of each block are shuffled by one Random(execution_order_seed) consumed block by block. '
            'A stop leaves complete earlier blocks. Pairing removes shared additive block effects, '
            'not arbitrary provider drift or model-by-condition changes within a block.',
        global_parser='Validate every line; collapse exact undirected duplicates; retain raw response and flags; NONE valid.',
        transport_policy='SDK retries zero. Stop on uncertain transport; retain full reservation. '
            '`recover` (or launch --auto-recover) retires one lost-transport attempt with no saved reply '
            'and permits one replacement of that same request; 4xx, received replies, a failed replacement '
            'or more than 30 retirements stop for human review. Interrupted retirement journals/logs '
            'are reconciled from the unchanged ledger before any new request.',
        max_transport_recoveries=30, replacements_per_request=1,
        local_write_policy='Atomic JSON writes retry PermissionError only, at most ten times '
            'after the initial attempt (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 20 seconds; about 54 s). '
            'Retry identical local data, never transport or parsing. Exhaustion stops for inspection.',
        journal_reconciliation_policy='`recover` (or launch --auto-recover) may rewrite one stale request '
            'journal from the unchanged ledger only when the row is received, its transport receipt matches, '
            'and the journal equals the intent record or the exact pre-parse row digest. Each repair is '
            'logged; at most 30 per run. Existing log provenance is validated against received rows '
            'before further recovery or requests. Any other mismatch stops for human review.',
        max_journal_reconciliations=30,
        scheduler_policy='Optional walltime pauses only between requests, before intent creation, '
            'using one monotonic deadline across recovery loops. Received replies and partial cells '
            'resume under the identical authorization and ledger. PAUSED_AT_CHECKPOINT exits 75; '
            'other failures never trigger scheduler resubmission. No concurrent collectors.',
        partial_analysis='`analyze --stage main` replays every saved receipt offline and reports a stopped '
            'or failed inventory under the predeclared analysis fallback, labelled PARTIAL.',
        retries_per_received_local_decision=3, calibration_reuse=False,
        claim=analysis.load_spec()['claim'], human_bilingual_validation=False,
        live_gate='Replay the eight pinned historical Hindi global checks; identical prompts/parser/settings, no paid recheck. Empty graphs remain valid.')


def verify_contract(spec):
    if spec != build_contract():
        raise ValueError('Frozen contract/source/runtime differs. Do not execute or re-anchor spending.')
    return spec


def load_contract():
    return verify_contract(read(CONTRACT))


def freeze():
    spec = build_contract()
    if CONTRACT.exists() and read(CONTRACT) != spec:
        existing = legacy.digest(read(CONTRACT))
        if existing not in {HISTORY_SHA, UNEXECUTED_V2_SHA, UNEXECUTED_V3_SHA, UNEXECUTED_V4_SHA,
                            UNEXECUTED_V5_SHA, UNEXECUTED_V6_SHA, UNEXECUTED_V61_SHA,
                            UNEXECUTED_V62_SHA} or (OUTPUT/existing[:12]/'main').exists():
            raise ValueError('An existing freeze differs; do not overwrite it silently.')
        if existing == HISTORY_SHA:
            hindi_evidence_contract()
        # Explicit one-time allocation migration. Paid evidence remains at its
        # original hash; an attempted main study may never be re-anchored here.
    write_json(CONTRACT, spec)
    return dict(contract_sha256=legacy.digest(spec), main_attempts=spec['main_attempts'], main_authorized=False)


def cells(spec, stage):
    if stage == 'main':
        rows = analysis.design(spec['analysis'])
    elif stage == 'hindi_check':
        spec = evidence_spec(spec, stage)
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
    rng = random.Random(spec['execution_order_seed'])
    if stage != 'main':
        # Historical Hindi evidence keeps its original identity and order.
        rng.shuffle(result)
        return result
    # Finish one schedule block before starting the next, so a stop leaves
    # complete blocks; shuffle within each block against within-block drift.
    ordered = []
    for seed in spec['analysis']['schedule_seeds']:
        block = [cell for cell in result if cell['seed'] == seed]
        rng.shuffle(block)
        ordered += block
    if len(ordered) != len(result):
        raise ValueError('Every main cell must belong to exactly one schedule block.')
    return ordered


def folder(spec, stage):
    if stage not in {'main', 'hindi_check'}:
        raise ValueError('Unknown stage.')
    return OUTPUT/legacy.digest(evidence_spec(spec, stage))[:12]/stage


def hindi_evidence_contract():
    """Validate the original source archive and unchanged live-check semantics."""
    old = read(HISTORY/'contract.json')
    if legacy.digest(old) != HISTORY_SHA:
        raise ValueError('Historical Hindi contract changed.')
    for name, expected in old['sources'].items():
        path = HISTORY/name if name.startswith('final_study/') else ROOT/name
        if engine.sha(path) != expected:
            raise ValueError('Historical Hindi source evidence changed: '+name)
    if (old['catalog_sha256'] != legacy.digest(catalog())
            or old['runtime'] != engine.frozen.runtime_versions()
            or old['rates'] != {m: list(legacy.paid.RATES[m]) for m in analysis.MODELS}
            or old['settings'] != {m: {method: engine.settings(m, method) for method in analysis.METHODS}
                                  for m in analysis.MODELS}):
        raise ValueError('Hindi evidence is incompatible with current execution semantics.')
    return old


def evidence_spec(spec, stage):
    if stage != 'hindi_check':
        return spec
    old = hindi_evidence_contract()
    if legacy.digest(spec) != HISTORY_SHA and spec.get('hindi_evidence_contract_sha256') != HISTORY_SHA:
        raise ValueError('Unrecognized Hindi evidence bridge.')
    return old


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
