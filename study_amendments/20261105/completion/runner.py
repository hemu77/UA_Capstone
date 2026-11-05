"""Bounded, selectively completed supplement; never part of primary inference."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parent
UPSTREAM = BUNDLE.parent / 'supplement-20261105-v1/runner.py'
UPSTREAM_SHA = 'd7dac09925abdf85df49cf1f400b43384ba6aea1bcc8fde6c5500267117fa008'
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != UPSTREAM_SHA:
    raise ValueError('Reviewed upstream changed')
loader = importlib.util.spec_from_file_location('fixed_supplement', UPSTREAM)
s = importlib.util.module_from_spec(loader)
loader.loader.exec_module(s)
r, a, p = s.engine()
read, sha, require = s.read, s.sha, s.require
PRIOR = s.TARGET
COPY = BUNDLE / 'checkpoint'
TARGET = s.REPO / 'outputs/supplement_completion_20261105_v1'
NAME = 'owner-bounded-supplement-completion-20261105-v1'

def manifest(folder):
    require(folder.is_dir() and not folder.is_symlink(), 'Missing or redirected checkpoint')
    entries = {}
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'Symlinks are not evidence')
        if path.is_file():
            require(not path.name.endswith(('-wal', '-journal', '-shm')), 'Checkpoint is still active')
            entries[path.relative_to(folder).as_posix()] = sha(path)
    return entries

def baseline():
    require(read(PRIOR / 'summary.json')['status'] == 'SUPPLEMENT_COMPLETE', 'Prior six must finish first')
    require(manifest(PRIOR) == manifest(COPY), 'Prior evidence changed')
    spec = p.load_contract()
    old_policy, cells = s.load_policy()
    with contextlib.closing(sqlite3.connect((COPY / 'budget.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        approval = dict(supplement_id=s.NAME, policy_sha256=sha(s.BUNDLE / 'policy.json'),
                        additional_usd=old_policy['additional_cap_usd'], original_spent_usd=old_policy['original_spent_usd'], cumulative_cap_usd=120)
        require(read(COPY / 'authorization.json') == approval and db.execute('SELECT value FROM authorization').fetchall()
                == [(json.dumps(approval, sort_keys=True),)], 'Prior authorization differs')
        with patch.object(s, 'TARGET', COPY):
            s.accounting(SimpleNamespace(db=db), cells, spec, r, p)
        records = {c['original_run_id']: s.verify_record(COPY / 'runs' / (c['run_id'] + '.json'), c, spec, r, p) for c in cells}
        spent = db.execute('SELECT coalesce(sum(cost),0) FROM attempts').fetchone()[0]
    summary = read(COPY / 'summary.json')
    require(summary['supplemental_attempts'] == 6 and summary['supplemental_charge_or_reservation_usd'] == spent,
            'Prior accounting differs')
    require(summary['supplemental_valid_graphs'] == sum(v['failure'] is None for v in records.values())
            and summary['supplemental_failed_attempts'] == sum(v['failure'] is not None for v in records.values()), 'Prior counts differ')
    original = s.original_anchor()['ledger_charge_or_reservation_usd']
    require(summary['cumulative_charge_or_reservation_usd'] == original + spent, 'Original spending differs')
    return spec, records, original + spent

def plan(records, policy_sha):
    require(set(records) == set(s.FAILURES), 'All six prior outcomes required')
    cells = []
    for oid in sorted(records):
        if records[oid]['failure'] is None:
            continue
        original = read(s.ORIGINAL / 'runs' / (oid + '.json'))['cell']
        for attempt in range(1, 4):
            cells.append({**original, 'run_id': 'supp_complete_20261105_v1_' + oid + '_a' + str(attempt),
                          'original_run_id': oid, 'supplement_id': NAME, 'supplement_policy_sha256': policy_sha,
                          'completion_attempt': attempt})
    return cells

def eligible(cells, saved, paid_runs=()):
    """Yield only the first unsent/failed chain until one success per original cell."""
    successful = set()
    for cell in cells:
        oid = cell['original_run_id']
        if oid in successful:
            require(cell['run_id'] not in saved and cell['run_id'] not in paid_runs, 'Evidence exists after successful completion; refuse')
            continue
        record = saved.get(cell['run_id'])
        if record is None:
            yield cell
            record = saved.get(cell['run_id'])
            require(record is not None, 'Missing durable terminal result')
        if record['failure'] is None:
            successful.add(oid)

def validate_chains(cells, saved, paid_runs):
    states = {}
    for cell in cells:
        oid, rid = cell['original_run_id'], cell['run_id']
        state = states.get(oid)
        require(state not in {'success', 'unfinished'} or (rid not in saved and rid not in paid_runs),
                'Existing chain has evidence after success or an unfinished earlier attempt')
        record = saved.get(rid)
        if record is None:
            states[oid] = state or 'unfinished'
        elif record['failure'] is None:
            states[oid] = 'success'

def check():
    with r.output_lock(), r.workflow_lock(BUNDLE), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline completion check')), \
            patch.object(p.legacy.paid, 'OpenAI', side_effect=AssertionError('Offline completion check')):
        if not COPY.exists():
            require(not TARGET.exists(), 'Cannot rebuild baseline after execution')
            before = manifest(PRIOR)
            require(read(PRIOR / 'summary.json')['status'] == 'SUPPLEMENT_COMPLETE', 'Prior run unfinished')
            shutil.copytree(PRIOR, COPY)
            require(before == manifest(PRIOR) == manifest(COPY), 'Copy preservation failed')
        spec, records, anchor = baseline()
        a.prices(spec)
        policy = dict(supplement_id=NAME, contract_sha256=s.CONTRACT, runner_sha256=sha(__file__),
                      tests_sha256=sha(BUNDLE / 'test_completion.py'), upstream_sha256=UPSTREAM_SHA,
                      prior_manifest_sha256=p.legacy.digest(manifest(COPY)), original_and_prior_usd=anchor,
                      additional_cap_usd=s.remaining_allowance(anchor), cumulative_cap_usd=120,
                      missing_originals=sorted(k for k,v in records.items() if v['failure'] is not None),
                      max_additional_per_cell=3, target_unique_valid_cells=6,
                      interpretation='Adaptive post-collection supplement. Keep every failure; no primary substitution or repeat beyond three additional draws per missing cell.',
                      owner_authorization='2026-11-05: bounded supplemental completion with review and offline checks; existing intended project key, $120 cumulative ceiling.')
        path = BUNDLE / 'policy.json'
        if path.exists():
            require(read(path) == policy, 'Never overwrite different clearance')
        else:
            require(not TARGET.exists(), 'Cannot reset policy over paid evidence')
            r.write_json(path, policy)
        cells = plan(records, sha(path))
        for cell in cells:
            with p.adapter(cell, spec):
                result, failure = r.observe(cell, p.engine.FixtureCaller(cell))
            require(failure is None and len(result[0]) == 50, 'Completion fixture failed')
        proof = dict(status='COMPLETION_OFFLINE_VERIFIED', policy_sha256=sha(path), prior_replays=6,
                     fixture_cases=len(cells), paid_calls=0)
        r.write_json(BUNDLE / 'offline-verified.json', proof)
        print(json.dumps(proof), flush=True)

def launch():
    with r.output_lock(), r.workflow_lock(BUNDLE):
        policy = read(BUNDLE / 'policy.json')
        require(policy['supplement_id'] == NAME and policy['contract_sha256'] == s.CONTRACT
                and policy['upstream_sha256'] == UPSTREAM_SHA and policy['cumulative_cap_usd'] == 120, 'Wrong completion policy')
        require(policy['runner_sha256'] == sha(__file__) and policy['tests_sha256'] == sha(BUNDLE / 'test_completion.py'), 'Code changed')
        spec, prior, anchor = baseline()
        require(policy['original_and_prior_usd'] == anchor and policy['additional_cap_usd'] == s.remaining_allowance(anchor)
                and policy['prior_manifest_sha256'] == p.legacy.digest(manifest(COPY)), 'Cumulative anchor changed')
        a.prices(spec)
        digest = sha(BUNDLE / 'policy.json')
        cells = plan(prior, digest)
        require(policy['missing_originals'] == sorted({c['original_run_id'] for c in cells})
                and policy['max_additional_per_cell'] == 3 and policy['target_unique_valid_cells'] == 6, 'Scope changed')
        require(read(BUNDLE / 'offline-verified.json') == dict(status='COMPLETION_OFFLINE_VERIFIED', policy_sha256=digest,
                prior_replays=6, fixture_cases=len(cells), paid_calls=0), 'Offline proof differs')
        review = read(BUNDLE / 'review.json')
        require(review['status'] == 'CONDITIONAL_PAID_GO' and review['policy_sha256'] == digest, 'Review required')
        approval = dict(supplement_id=NAME, policy_sha256=digest, original_and_prior_usd=anchor,
                        additional_usd=policy['additional_cap_usd'], cumulative_cap_usd=120)
        if TARGET.exists():
            require((TARGET / 'budget.sqlite').is_file() and read(TARGET / 'authorization.json') == approval, 'Never reset an existing budget')
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
            require(budget.db.execute('SELECT value FROM authorization').fetchall() == [(json.dumps(approval, sort_keys=True),)], 'Authorization changed')
            with patch.object(s, 'TARGET', TARGET):
                s.accounting(budget, cells, spec, r, p)
            saved = {c['run_id']: s.verify_record(TARGET / 'runs' / (c['run_id'] + '.json'), c, spec, r, p)
                     for c in cells if (TARGET / 'runs' / (c['run_id'] + '.json')).exists()}
            paid_runs = {json.loads(row[0])['cell']['run_id'] for row in budget.db.execute('SELECT request FROM attempts')}
            validate_chains(cells, saved, paid_runs)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(p.engine, 'write_json', r.write_json))
                stack.enter_context(patch.object(p.legacy, 'write_json', r.write_json))
                client = None
                for cell in eligible(cells, saved, paid_runs):
                    if client is None:
                        key = p.legacy.paid.credential()
                        client = stack.enter_context(p.legacy.paid.OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                        del key
                        for model in sorted({c['model'] for c in cells}):
                            require(client.models.retrieve(model).id == model, 'Requested model unavailable')
                    with p.adapter(cell, spec):
                        caller = r.Replay(r.requests_from_ledger(budget, cell), r.Caller(client, budget, cell, spec, TARGET))
                        result, failure = r.observe(cell, caller)
                    items = r.requests_from_ledger(budget, cell)
                    path = TARGET / 'runs' / (cell['run_id'] + '.json')
                    if failure is None:
                        def marker(destination, record):
                            r.write_json(destination, {**record, 'failure': None, 'normalization': r.normalization(items, cell)})
                        with patch.object(p.engine, 'write_json', marker):
                            p.engine.save_run(path, cell, result, items, 'received_supplemental_study')
                    else:
                        r.write_json(path, dict(cell=cell, evidence_type='received_supplemental_study', requests=items,
                            failure=failure, normalization=r.normalization(items, cell), metrics=None, artifacts={}))
                    saved[cell['run_id']] = s.verify_record(path, cell, spec, r, p)
                    write_summary(prior, saved, anchor, budget, digest, final=False)
            with patch.object(s, 'TARGET', TARGET):
                s.accounting(budget, cells, spec, r, p)
            baseline()
            write_summary(prior, saved, anchor, budget, digest, final=True)
        finally:
            budget.db.close()

def write_summary(prior, saved, anchor, budget, digest, *, final):
    selected = {oid: dict(run_id=record['cell']['run_id'], source='supplement_20261105_v1')
                for oid,record in prior.items() if record['failure'] is None}
    for record in saved.values():
        if record['failure'] is None:
            oid = record['cell']['original_run_id']
            require(oid not in selected, 'More than one selected graph per original')
            selected[oid] = dict(run_id=record['cell']['run_id'], source='supplement_completion_20261105_v1')
    report = dict(status=('COMPLETION_TARGET_REACHED' if len(selected) == 6 else 'BOUNDED_LIMIT_REACHED') if final else 'IN_PROGRESS',
        supplement_id=NAME, policy_sha256=digest, valid_supplemental_cells=len(selected), target_valid_cells=6,
        additional_attempts=len(saved), additional_valid_graphs=sum(v['failure'] is None for v in saved.values()),
        additional_failed_attempts=sum(v['failure'] is not None for v in saved.values()), selected=selected,
        cumulative_charge_or_reservation_usd=anchor + budget.spent(), additional_charge_or_reservation_usd=budget.spent(),
        original_attempts=896, original_valid_graphs=890, original_failed_attempts=6,
        all_attempts=902+len(saved), all_valid_graphs=890+len(selected),
        interpretation='Selectively completed supplementary observations only; primary analysis unchanged.')
    require(report['cumulative_charge_or_reservation_usd'] <= 120, 'Cumulative ceiling exceeded')
    r.write_json(TARGET / ('summary.json' if final else 'progress.json'), report)
    print(json.dumps(report), flush=True)

def main():
    try:
        require(sys.argv[1:] in (['check'], ['launch']), 'Expected check or launch')
        check() if sys.argv[1] == 'check' else launch()
        return 0
    except Exception as error:
        status, code = getattr(error, 'status_code', None), getattr(error, 'code', None)
        print(json.dumps(dict(status='COMPLETION_STOPPED', http_status=status if type(status) is int else None,
            error_code=code if type(code) is str and code in {'invalid_api_key','model_not_found','credit_balance_exhausted','insufficient_quota'} else None,
            message='Preserve evidence/reservations; no automatic retry.')), flush=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
