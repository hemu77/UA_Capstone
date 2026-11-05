"""Six separately authorized supplemental draws. Original outcomes stay immutable."""
import contextlib
import hashlib
import json
import math
from pathlib import Path
import socket
import sqlite3
import sys
from unittest.mock import patch

REPO = Path('C:/Users/Hemu/Projects/UA_Captsone_SocN')
BUNDLE = Path(__file__).resolve().parent
BACKUP = BUNDLE.parent / 'retrieved-814020'
ORIGINAL = BACKUP / 'expanded/outputs/final_study/060dcd3a86d5/main'
TARGET = REPO / 'outputs/supplement_20261105_v1'
NAME = 'owner-six-supplemental-attempts-20261105-v1'
CONTRACT = '060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f'
FAILURES = {
    'final_060dcd3a86d5_main_global_gpt-5.6-luna_us_portuguese_s44000': 'invalid_global_response',
    'final_060dcd3a86d5_main_global_gpt-6-luna_us_japanese_s44002': 'invalid_global_response',
    'final_060dcd3a86d5_main_global_gpt-5.6-luna_us_portuguese_s44004': 'invalid_global_response',
    'final_060dcd3a86d5_main_global_gpt-5.6-luna_us_japanese_s44004': 'invalid_global_response',
    'final_060dcd3a86d5_main_global_gpt-6-sol_us_japanese_s44005': 'invalid_global_response',
    'final_060dcd3a86d5_main_iterative_gpt-4.1_us_japanese_s44006': 'local_corrections_exhausted',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def remaining_allowance(original_spent):
    require(type(original_spent) in (int, float) and math.isfinite(original_spent)
            and 0 <= original_spent < 120, 'Invalid original spending anchor')
    return math.floor((120 - original_spent) * 1e8) / 1e8

def supplemental_cells(originals, policy_sha):
    require(set(originals) == set(FAILURES), 'Exactly the six approved failed cells are required')
    require(len(policy_sha) == 64 and all(c in '0123456789abcdef' for c in policy_sha), 'Invalid policy digest')
    cells = []
    for rid, failure in FAILURES.items():
        record = originals[rid]
        require(record['failure'] == failure and record['cell']['run_id'] == rid
                and record['cell']['final_contract_sha256'] == CONTRACT, 'Original failure differs')
        cells.append({**record['cell'], 'run_id': 'supplement_20261105_v1_' + rid,
                      'original_run_id': rid, 'supplement_id': NAME, 'supplement_policy_sha256': policy_sha})
    return cells

def engine():
    # Reuse the exact reviewed billing adapter, without launching its main runner.
    if (BUNDLE / 'policy.json').exists():
        for name, digest in read(BUNDLE / 'policy.json')['adapter_sha256'].items():
            require(sha(BACKUP / 'expanded' / name) == digest, 'Reviewed adapter changed')
    sys.path[:0] = [str(BACKUP / 'expanded'), str(REPO)]
    import amended_runner as r
    import billing_amendment as a
    from final_study import protocol as p
    return r, a, p

def original_anchor():
    audit = read(BACKUP / 'local-audit.json')
    require(audit['status'] == 'DOWNLOADED_CHECKPOINT_AUDITED' and audit['contract_sha256'] == CONTRACT
            and (audit['attempted'], audit['valid_graphs'], audit['failed_attempts']) == (896, 890, 6),
            'Complete original checkpoint audit required')
    require(sha(ORIGINAL / 'budget.sqlite') == audit['ledger_sha256'], 'Original ledger copy changed')
    with contextlib.closing(sqlite3.connect((ORIGINAL / 'budget.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        spent = db.execute('SELECT sum(cost) FROM attempts').fetchone()[0]
    require(spent == audit['ledger_charge_or_reservation_usd'], 'Original accounting anchor changed')
    for item in audit['failures']:
        require(sha(ORIGINAL / 'runs' / (item['run_id'] + '.json')) == item['sha256'], 'Original receipt changed')
    return audit

def load_policy():
    policy = read(BUNDLE / 'policy.json')
    require(policy['supplement_id'] == NAME and policy['contract_sha256'] == CONTRACT, 'Wrong supplement')
    require(policy['runner_sha256'] == sha(__file__), 'Supplement runner changed after clearance')
    require(policy['tests_sha256'] == sha(BUNDLE / 'test_supplement.py'), 'Supplement tests changed')
    audit = original_anchor()
    require(policy['original_ledger_sha256'] == audit['ledger_sha256']
            and policy['original_spent_usd'] == audit['ledger_charge_or_reservation_usd']
            and policy['additional_cap_usd'] == remaining_allowance(policy['original_spent_usd']),
            'Cumulative budget cannot be reset')
    originals = {rid: read(ORIGINAL / 'runs' / (rid + '.json')) for rid in FAILURES}
    return policy, supplemental_cells(originals, sha(BUNDLE / 'policy.json'))

def check():
    r, a, p = engine()
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline supplement check')), \
            patch.object(p.legacy.paid, 'OpenAI', side_effect=AssertionError('Offline supplement check')):
        spec = p.load_contract()
        a.prices(spec)
        audit = original_anchor()
        policy = dict(supplement_id=NAME, contract_sha256=CONTRACT,
                      original_archive_sha256=audit['original_archive_sha256'],
                      original_ledger_sha256=audit['ledger_sha256'],
                      original_spent_usd=audit['ledger_charge_or_reservation_usd'],
                      additional_cap_usd=remaining_allowance(audit['ledger_charge_or_reservation_usd']),
                      cumulative_cap_usd=120, original_attempts=896, original_valid_graphs=890,
                      credential_selection='Owner confirmed existing local key/project on 2026-11-05; never prompt or save a key.',
                      original_failed_attempts=6, original_failures=FAILURES,
                      runner_sha256=sha(__file__), tests_sha256=sha(BUNDLE / 'test_supplement.py'),
                      adapter_sha256={name: sha(BACKUP / 'expanded' / name)
                                      for name in ('amended_runner.py', 'billing_amendment.py', 'checkpoint/verified.json')},
                      owner_authorization='2026-11-05: six separately labeled supplemental local attempts with offline checks and one bounded reviewer; retain all original failures and cumulative $120 cap.',
                      policy='One fresh observation per listed cell; same prompts/settings/seed. No automatic transport recovery or repeat-until-valid. Do not merge into original primary analysis.')
        path = BUNDLE / 'policy.json'
        if path.exists():
            require(read(path) == policy, 'Existing supplement policy differs; do not overwrite')
        else:
            require(not TARGET.exists(), 'Cannot create a new policy over existing results')
            r.write_json(path, policy)
        _, cells = load_policy()
        for cell in cells:
            rid = cell['original_run_id']
            original = read(ORIGINAL / 'runs' / (rid + '.json'))
            r.verify_record(ORIGINAL / 'runs' / (rid + '.json'), original['cell'], spec)
            with p.adapter(cell, spec):
                result, failure = r.observe(cell, p.engine.FixtureCaller(cell))
            require(failure is None and len(result[0]) == 50, 'Offline supplement fixture failed')
        proof = dict(status='SUPPLEMENT_OFFLINE_VERIFIED', policy_sha256=sha(path),
                     original_failure_replays=6, fixture_cases=6, paid_calls=0)
        r.write_json(BUNDLE / 'offline-verified.json', proof)
        print(json.dumps(proof), flush=True)

def verify_record(path, cell, spec, r, p):
    import networkx as nx
    record = read(path)
    require(record.get('cell') == cell and record.get('evidence_type') == 'received_supplemental_study', 'Not supplemental evidence')
    items = record['requests']
    require(bool(items), 'No received evidence')
    for ordinal, item in enumerate(items):
        r.validate_request(item, cell, spec, ordinal)
    replay = r.Replay(items)
    with p.adapter(cell, spec):
        result, failure = r.observe(cell, replay)
    require(replay.ordinal == len(items) and failure == record['failure'], 'Supplement replay differs')
    require(r.normalization(items, cell) == record['normalization'], 'Normalization differs')
    if failure is None:
        graph, events, decisions, schedule = result
        require(p.legacy.paid.clean_numbers(events) == record['events'] and decisions == record['decisions']
                and schedule == record['schedule'] and p.engine.measures(graph, p.engine.revised.previous.adult_roster()) == record['metrics'], 'Graph evidence differs')
        require(set(record['artifacts']) == {'.adj', '.png'}, 'Graph artifacts missing')
        for extension, digest in record['artifacts'].items():
            require(sha(path.with_suffix(extension)) == digest, 'Graph artifact changed')
        saved = nx.read_adjlist(path.with_suffix('.adj'))
        require(set(saved) == set(graph) and {frozenset(e) for e in saved.edges} == {frozenset(e) for e in graph.edges}, 'Adjacency differs')
    else:
        require(record['metrics'] is None and not record['artifacts'], 'Failed supplement cannot carry a graph')
    return record

def accounting(budget, cells, spec, r, p):
    allowed = {c['run_id']: c for c in cells}
    rows = {row[0]: row for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    require(set(rows) == {f.stem for f in (TARGET / 'requests').glob('*.json')}, 'Supplement journal inventory mismatch; no resend')
    require(set(rows) == {f.stem for f in (TARGET / 'transport').glob('*.json')}, 'Unresolved or orphan transport; no resend')
    for rid, row in rows.items():
        require(row[1] == 'main' and row[3] == 'received', 'Unresolved supplemental request; retain reservation and stop')
        request, response = json.loads(row[4]), json.loads(row[5])
        require(allowed.get(request['cell']['run_id']) == request['cell'], 'Unapproved supplementary request')
        require(read(TARGET / 'requests' / (rid + '.json')) == dict(request_id=rid, row_sha256=p.legacy.digest(row)), 'Changed journal')
        r.validate_request(dict(request_id=rid, request=request, response=response), request['cell'], spec, request['ordinal'])
        r.a.resolved(request['cell']['model'], response['resolved_model'])
        require(row[2] == response['conservative_charge_usd'], 'Recorded cost differs')
        p.legacy.verify_transport(TARGET, rid, response)
    require(not ({f.stem for f in (TARGET / 'runs').glob('*.json')} - set(allowed)), 'Unplanned terminal result')
    for path in (TARGET / 'runs').glob('*.json'):
        record = read(path)
        expected = {rid for rid, row in rows.items() if json.loads(row[4])['cell']['run_id'] == path.stem}
        require({item['request_id'] for item in record['requests']} == expected, 'Terminal result omits paid evidence')
        for item in record['requests']:
            row = rows[item['request_id']]
            require(json.loads(row[4]) == item['request'] and json.loads(row[5]) == item['response'], 'Terminal paid evidence changed')

def launch():
    r, a, p = engine()
    with r.output_lock(), r.workflow_lock(BUNDLE):
        policy, cells = load_policy()
        for name, digest in policy['adapter_sha256'].items():
            require(sha(BACKUP / 'expanded' / name) == digest, 'Reviewed adapter changed')
        policy_sha = sha(BUNDLE / 'policy.json')
        proof, review = read(BUNDLE / 'offline-verified.json'), read(BUNDLE / 'review.json')
        require(proof == dict(status='SUPPLEMENT_OFFLINE_VERIFIED', policy_sha256=policy_sha,
                              original_failure_replays=6, fixture_cases=6, paid_calls=0), 'Offline clearance missing')
        require(review['status'] == 'CONDITIONAL_PAID_GO' and review['policy_sha256'] == policy_sha, 'Matching reviewer clearance missing')
        spec = p.load_contract()
        a.prices(spec)
        approval = dict(supplement_id=NAME, policy_sha256=policy_sha, additional_usd=policy['additional_cap_usd'],
                        original_spent_usd=policy['original_spent_usd'], cumulative_cap_usd=120)
        if TARGET.exists():
            require(read(TARGET / 'authorization.json') == approval and (TARGET / 'budget.sqlite').is_file(), 'Existing budget cannot be reset')
        else:
            TARGET.mkdir(parents=True)
            initial = p.legacy.paid.Budget(TARGET / 'budget.sqlite', total_cap=approval['additional_usd'],
                                          pilot_cap=approval['additional_usd'], allow_higher_cap=True)
            try:
                initial.db.execute('CREATE TABLE authorization (value TEXT NOT NULL)')
                initial.db.execute('INSERT INTO authorization VALUES (?)', (json.dumps(approval, sort_keys=True),))
                initial.db.commit()
                r.write_json(TARGET / 'authorization.json', approval)
            finally:
                initial.db.close()
        budget = p.legacy.paid.Budget(TARGET / 'budget.sqlite', total_cap=approval['additional_usd'],
                                     pilot_cap=approval['additional_usd'], allow_higher_cap=True, require_existing=True)
        try:
            saved = budget.db.execute('SELECT value FROM authorization').fetchall()
            require(len(saved) == 1 and json.loads(saved[0][0]) == approval, 'Budget authorization changed')
            accounting(budget, cells, spec, r, p)
            records = []
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(p.engine, 'write_json', r.write_json))
                stack.enter_context(patch.object(p.legacy, 'write_json', r.write_json))
                client = None
                for cell in cells:
                    path = TARGET / 'runs' / (cell['run_id'] + '.json')
                    if not path.exists():
                        if client is None:
                            key = p.legacy.paid.credential()
                            require(bool(key.strip()), 'No configured key; no provider call')
                            client = stack.enter_context(p.legacy.paid.OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                            del key
                            for model in sorted({c['model'] for c in cells}):
                                require(client.models.retrieve(model).id == model, 'Requested model unavailable')
                            print('SUPPLEMENT_MODEL_ACCESS_VERIFIED; no generation calls in access check', flush=True)
                        with p.adapter(cell, spec):
                            live = r.Caller(client, budget, cell, spec, TARGET)
                            caller = r.Replay(r.requests_from_ledger(budget, cell), live)
                            result, failure = r.observe(cell, caller)
                        items = r.requests_from_ledger(budget, cell)
                        if failure is None:
                            def marker(destination, record):
                                r.write_json(destination, {**record, 'failure': None, 'normalization': r.normalization(items, cell)})
                            with patch.object(p.engine, 'write_json', marker):
                                p.engine.save_run(path, cell, result, items, 'received_supplemental_study')
                        else:
                            r.write_json(path, dict(cell=cell, evidence_type='received_supplemental_study', requests=items,
                                                   failure=failure, normalization=r.normalization(items, cell), metrics=None, artifacts={}))
                    records.append(verify_record(path, cell, spec, r, p))
                    r.write_json(TARGET / 'progress.json', dict(terminal=len(records), planned=6,
                                 valid_graphs=sum(q['failure'] is None for q in records), supplemental_charge_or_reservation_usd=budget.spent(),
                                 cumulative_charge_or_reservation_usd=policy['original_spent_usd'] + budget.spent()))
                    print(f"SUPPLEMENT: {len(records)}/6 terminal; {sum(q['failure'] is None for q in records)} valid", flush=True)
            accounting(budget, cells, spec, r, p)
            original_anchor()
            summary = dict(status='SUPPLEMENT_COMPLETE', supplement_id=NAME, policy_sha256=policy_sha,
                           original_attempts=896, original_valid_graphs=890, original_failed_attempts=6,
                           supplemental_attempts=len(records), supplemental_valid_graphs=sum(q['failure'] is None for q in records),
                           supplemental_failed_attempts=sum(q['failure'] is not None for q in records),
                           supplemental_charge_or_reservation_usd=budget.spent(),
                           cumulative_charge_or_reservation_usd=policy['original_spent_usd'] + budget.spent(),
                           interpretation='Separate post-collection supplement; not replacement observations or original primary analysis.')
            require(summary['cumulative_charge_or_reservation_usd'] <= 120, 'Cumulative cap exceeded')
            r.write_json(TARGET / 'summary.json', summary)
            print(json.dumps(summary), flush=True)
        finally:
            budget.db.close()

def main():
    try:
        require(sys.argv[1:] in (['check'], ['launch']), 'Expected check or launch')
        if sys.argv[1] == 'check':
            check()
        else:
            launch()
        return 0
    except Exception:
        # Never print SDK bodies, chained exceptions, or credential-related values.
        print('SUPPLEMENT_STOPPED: gate, provider, or persistence check failed. Preserve all evidence and reservations; no automatic retry.', flush=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
