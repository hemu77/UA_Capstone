"""72 first-response checks with immutable authorization and the original ledger.

No correction requests, automatic transport retries, main runs, or new ledger.
Replies are saved even when invalid. A resume uses cached replies, not new calls.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from contextlib import closing
from pathlib import Path

import networkx as nx
from scipy.stats import binomtest

import revision_next as revised
import paid_study as paid

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/wording_probe_v6'
LEDGER = ROOT / 'outputs/revision_budget_v1/budget.sqlite'
MODEL = 'gpt-6-luna'
PRICE_SOURCE = 'https://developers.openai.com/api/docs/models/gpt-6-luna'


def manifest():
    roster = revised.previous.adult_roster()
    graph = nx.empty_graph(roster)
    display = revised.schedule(21000)['display_order']
    rows = []
    for language, count, actor, wording in itertools.product(
            ['english', 'portuguese'], [1, 2, 8], ['0', '7', '14', '21', '28', '35'],
            ['v5_original', 'v6_candidate']):
        system = revised.system_prompt('local', language, 'us', actor, count) if wording == 'v6_candidate' else revised.previous.system_prompt(
            'local', roster, revised.previous.DEMOS, curr_pid=actor, num_choices=count,
            culture_context='us', prompt_language=language)
        rows.append(dict(probe_id=f'{language}_k{count}_p{actor}_{wording}', language=language,
            count=count, actor=actor, wording=wording, system=system,
            user=revised.candidate_payload('local', language, actor, graph, display)))
    random.Random(20261002).shuffle(rows)
    return rows


def contract():
    if tuple(paid.RATES[MODEL]) != (.10, .50):
        raise ValueError('Probe price contract changed; review required.')
    return dict(model=MODEL, manifest_sha256=revised.frozen.digest(manifest()),
        candidate_protocol_sha256=revised.frozen.digest(revised.config()),
        source_sha256={**revised.frozen.source_hashes(), **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['revision_next.py', 'wording_probe.py']}},
        runtime_versions=revised.frozen.runtime_versions(), rates_per_million=list(paid.RATES[MODEL]),
        price_source=PRICE_SOURCE, price_checked='2026-10-02',
        calls=72, max_completion_tokens=512, sdk_retries=0, correction_retries=0,
        analysis='first-response compliance by language/count/wording; descriptive only',
        screen_rule='All 36 revised replies must parse on first response for engineering-screen pass. Any failure requires review. No language-equivalence or main-study authorization follows.')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def authorization():
    value = json.loads((OUTPUT / 'authorization.json').read_text(encoding='utf-8'))
    if value.get('contract') != contract() or value.get('main_authorized') is not False:
        raise ValueError('Probe authorization is stale or changed scope.')
    if json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8')) != manifest():
        raise ValueError('Saved probe manifest differs from the authorized prompts.')
    start, ceiling = value['ledger_start_usd'], value['cumulative_ceiling_usd']
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in [start, ceiling]) or start < 0 or not math.isclose(ceiling - start, 1.0, abs_tol=1e-12):
        raise ValueError('Authorization must retain the original additional $1 ceiling.')
    return value


def authorize():
    """Run only following the owner's explicit $1 probe authorization."""
    with revised.frozen.workflow_lock():
        if (OUTPUT / 'authorization.json').exists():
            return authorization()  # Never reset the spending anchor on resume.
        with closing(paid.Budget(LEDGER, require_existing=True).db) as db:
            if db.execute("SELECT COUNT(*) FROM attempts WHERE status NOT IN ('received','abandoned')").fetchone()[0]:
                raise RuntimeError('Unresolved historical request; do not authorize new calls.')
            start = db.execute('SELECT COALESCE(SUM(cost),0) FROM attempts').fetchone()[0]
        value = dict(contract=contract(), additional_authorized_usd=1.0, ledger_start_usd=start,
            cumulative_ceiling_usd=start + 1.0, main_authorized=False,
            authorization_basis='Owner explicitly authorized up to $1 for this 72-call probe on 2026-10-02.')
        write_json(OUTPUT / 'manifest.json', manifest())
        write_json(OUTPUT / 'authorization.json', value)
        return value


def cell(row, approved):
    return dict(run_id='wording_v6_' + row['probe_id'], phase='main', method='local', model=MODEL,
        culture='us', language=row['language'], probe_id=row['probe_id'], count=row['count'], actor=row['actor'],
        wording=row['wording'], evidence_type='wording_compliance_probe_not_network',
        probe_contract_sha256=revised.frozen.digest(approved['contract']),
        frozen_source_sha256=approved['contract']['source_sha256'])


def collect(client, budget, approved):
    """Shared production caller persists every charge and aborts uncertain outcomes."""
    if budget.total_cap != approved['cumulative_ceiling_usd'] or budget.spent() + 1e-12 < approved['ledger_start_usd']:
        raise ValueError('Original ledger/cap is missing or inconsistent.')
    if budget.unresolved_count():
        raise RuntimeError('Unresolved paid attempt; no requests dispatched.')
    rows = []
    for index, row in enumerate(manifest(), 1):
        if approved['contract'] != contract():
            raise ValueError('Source or protocol changed during probe; stopped.')
        caller = paid.PaidCaller(client, budget, cell(row, approved))
        text = caller(MODEL, [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['user']}])
        error = None
        try:
            revised.parse_response('local', text, nx.empty_graph(revised.previous.adult_roster()),
                curr_pid=row['actor'], num_choices=row['count'])
        except ValueError as failure:
            error = failure
        budget.record_parse(caller.last_request_id, error)
        cached = budget.cached(caller.last_request_id)
        result = dict(**{key: row[key] for key in ['probe_id', 'language', 'count', 'actor', 'wording']},
            contract_sha256=revised.frozen.digest(approved['contract']), request_id=caller.last_request_id,
            evidence_type='paid_first_response_wording_probe_not_network', response=cached,
            valid=error is None, error=None if error is None else str(error))
        write_json(OUTPUT / 'replies' / (row['probe_id'] + '.json'), result)
        rows.append(result)
        print(f'Probe {index}/72; cumulative charged/reserved ${budget.spent():.5f}', flush=True)
    return rows


def summarize(rows):
    expected = {row['probe_id'] for row in manifest()}
    if len(rows) != 72 or {row['probe_id'] for row in rows} != expected:
        raise ValueError('Only an exact completed manifest can be summarized as complete.')
    groups = []
    for language, count, wording in itertools.product(['english', 'portuguese'], [1, 2, 8], ['v5_original', 'v6_candidate']):
        selected = [r for r in rows if (r['language'], r['count'], r['wording']) == (language, count, wording)]
        failures = sum(not row['valid'] for row in selected)
        ci = binomtest(failures, len(selected)).proportion_ci(method='exact')
        groups.append(dict(language=language, count=count, wording=wording, calls=len(selected), failures=failures,
            failure_fraction=failures / len(selected), descriptive_binomial_p025=ci.low, descriptive_binomial_p975=ci.high))
    revised_failures = sum(not r['valid'] for r in rows if r['wording'] == 'v6_candidate')
    return dict(status='COMPLETE', probe_calls=72, new_networks=0, revised_failures=revised_failures,
        engineering_screen='NO_FAILURES_OBSERVED' if not revised_failures else 'REVISE_BEFORE_CALIBRATION',
        main_authorized=False, human_bilingual_validation=False, groups=groups,
        limitations=['Six fixed actors per stratum; binomial intervals assume common independent success probability and are descriptive, not population inference.',
            'Wording bundles include grammatical/context fixes, not an isolated singular-noun causal test.',
            'No correction retries. No language equivalence, cross-model robustness or statistical approval established.'])


def execute(api_key):
    with revised.frozen.workflow_lock():
        approved = authorization()
        if not api_key:
            raise RuntimeError('A credential is required; do not store it in the repository.')
        write_json(OUTPUT / 'report.json', dict(status='RUNNING', main_authorized=False))
        budget = None
        try:
            budget = paid.Budget(LEDGER, total_cap=approved['cumulative_ceiling_usd'],
                pilot_cap=min(5, approved['cumulative_ceiling_usd']), require_existing=True, allow_higher_cap=True)
            if budget.unresolved_count():
                raise RuntimeError('Unresolved paid attempt; no client opened.')
            # Never send the user's credential to an ambient custom API base URL.
            with paid.OpenAI(api_key=api_key, base_url='https://api.openai.com/v1', max_retries=0, timeout=60) as client:
                client.models.retrieve(MODEL)
                rows = collect(client, budget, approved)
            result = summarize(rows)
            result.update(contract=approved['contract'], cumulative_ledger_usd=budget.spent(),
                ledger_change_usd=budget.spent() - approved['ledger_start_usd'],
                probe_conservative_charge_usd=sum(r['response']['conservative_charge_usd'] for r in rows),
                received_prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in rows),
                received_completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in rows),
                replies_sha256={r['probe_id']: hashlib.sha256((OUTPUT / 'replies' / (r['probe_id'] + '.json')).read_bytes()).hexdigest() for r in rows})
            write_json(OUTPUT / 'report.json', result)
            return {k: result[k] for k in ['status', 'probe_calls', 'new_networks', 'engineering_screen', 'revised_failures', 'ledger_change_usd']}
        except Exception as error:
            write_json(OUTPUT / 'report.json', dict(status='STOPPED', error_type=type(error).__name__,
                main_authorized=False, cumulative_ledger_usd=budget.spent() if budget else None,
                unresolved_attempts=budget.unresolved_count() if budget else None))
            raise
        finally:
            if budget is not None:
                budget.db.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument('--authorize-one-dollar', action='store_true')
    action.add_argument('--execute-probe', action='store_true')
    cli.add_argument('--key-stdin', action='store_true', help='Transient credential input; never saved or printed.')
    args = cli.parse_args()
    try:
        if args.authorize_one_dollar:
            approved = authorize()
            print(json.dumps({k: approved[k] for k in ['additional_authorized_usd', 'ledger_start_usd', 'cumulative_ceiling_usd', 'main_authorized']}, indent=2))
        else:
            key = sys.stdin.readline().strip() if args.key_stdin else paid.credential()
            print(json.dumps(execute(key), indent=2))
    except Exception as error:
        cli.exit(1, f'{type(error).__name__}: probe stopped; inspect local report. No automatic paid retry.\n')
