"""One guarded restart after a rolled-back, pre-transport SQLite reservation."""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

CONTRACT = '060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f'
WORK = Path('/u/scratch/d/dalharth/ua-capstone-main-20261026-6c94')
BUNDLE = WORK/'private/lock-resume-20261104-v2'
BASE = WORK/'private/billing-resume-20261103-v2'
ANCHOR = BUNDLE/'checkpoint'
TARGET = WORK/'repo/outputs/final_study'/CONTRACT[:12]/'main'
REFERENCE = 'owner-eight-repetitions-120'
ORPHAN = 'e6a61184019d126aa703e8177f68581b11b8746c8e294a1eec41f0c80d31a587'
ORPHAN_SHA = 'a1ec59a2a52346a2972e751cb33f2f144a7734012ab83a5e6e3bc735be9a7625'
ORPHAN_RUN = 'final_060dcd3a86d5_main_iterative_gpt-4.1_us_hindi_s44006'
ORPHAN_ORDINAL = 167
FAILURE_HASHES = {
    '763678': '822f3450919800bcd8bced3397c0a72e4819bc83c02c6e5547aaaa0c9667db3c',
    '788115': '379427df6e5904913f9640d2c0434c1f92070b4d332bf3448da52172c54b2a81',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync_directory(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def archive_intent(db, target):
    """Preserve one proven unsent intent; never modify a ledger row or receipt."""
    target = Path(target)
    source = target/'requests'/(ORPHAN+'.json')
    archive = target/'pretransport_rollbacks'/(ORPHAN+'.json')
    receipt = target/'transport'/(ORPHAN+'.json')
    require(target.is_dir() and target.resolve() == target, 'Redirected target')
    for path in (source.parent, archive.parent, receipt.parent, source, archive, receipt):
        require(not path.is_symlink() and path.resolve() == path, 'Redirected reconciliation path')
    require(db.execute('SELECT 1 FROM attempts WHERE id=?',(ORPHAN,)).fetchone() is None
            and not receipt.exists(), 'Orphan now has paid evidence; never resend')
    require(source.exists() or archive.exists(), 'Missing pinned intent evidence')
    for path in (source, archive):
        if path.exists():
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == ORPHAN_SHA
                    and json.loads(raw) == dict(request_id=ORPHAN,status='INTENT_BEFORE_TRANSPORT'),
                    'Changed pinned intent bytes')
    rows = {row[0] for row in db.execute('SELECT id FROM attempts')}
    journals = {path.stem for path in source.parent.glob('*.json')}
    require(journals == rows | ({ORPHAN} if source.exists() else set()), 'Unexpected inventory difference')
    if archive.parent.exists():
        require({p.name for p in archive.parent.iterdir()} <= {archive.name}, 'Unexpected archival evidence')
    archive.parent.mkdir(mode=0o700, exist_ok=True)
    if not archive.exists():
        os.link(source, archive)  # Exclusive destination; an interruption retains both identical names.
    with archive.open('r+b') as stream:
        os.fsync(stream.fileno())
    sync_directory(archive.parent)
    sync_directory(target)
    if source.exists():
        require(source.read_bytes() == archive.read_bytes(), 'Archive changed before reconciliation')
        source.unlink()
        sync_directory(source.parent)
    require(not source.exists() and hashlib.sha256(archive.read_bytes()).hexdigest() == ORPHAN_SHA,
            'Archival reconciliation incomplete')


def reconciled_manifest(before):
    source = 'requests/'+ORPHAN+'.json'
    archive = 'pretransport_rollbacks/'+ORPHAN+'.json'
    require(before.get(source) == ORPHAN_SHA and archive not in before, 'Unexpected original intent manifest')
    after = dict(before)
    after[archive] = after.pop(source)
    return after


def require_snapshot(expected_manifest, report, manifest, state, schema):
    require(manifest == expected_manifest, 'Live evidence changed since verification')
    require(json.loads(json.dumps(state)) == report['ledger_state'],
            'Live ledger changed since verification')
    require(json.loads(json.dumps(schema)) == report['schema'], 'Live schema changed since verification')


def baseline(summary, state, terminal, receipts, unresolved):
    require(summary.get('status') == 'STOPPED' and summary.get('error_type') == 'OperationalError'
            and summary.get('completed') == 751 and summary.get('contract_sha256') == CONTRACT,
            'Stopped checkpoint differs')
    require(state['rows'] == 84464 and math.isclose(state['cost'], 85.16933377, rel_tol=0, abs_tol=1e-8)
            and terminal == 751 and receipts == 84457 and not unresolved, 'Baseline differs or unresolved request')


def allowance(start, now):
    require(math.isfinite(start) and math.isfinite(now) and start <= now, 'Invalid allocation clock')
    remaining = min(80000, int(start + 86400 - now - 1800))
    require(remaining > 900, 'Insufficient allocation remains; no paid launch')
    return remaining


def approval(report, job, manifest_sha):
    expected = dict(status='LOCK_RESUME_COPY_VERIFIED', job=job, contract=CONTRACT,
                    terminal=751, valid=746, receipts=84457, rows=84464,
                    manifest_sha256=manifest_sha, paid_calls=0,
                    orphan_request_id=ORPHAN, orphan_sha256=ORPHAN_SHA)
    require(all(report.get(k) == v for k, v in expected.items())
            and math.isclose(report.get('conservative_usd', -1), 85.16933377, rel_tol=0, abs_tol=1e-8),
            'Missing or mismatched job-specific copy verification')


def command(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def runtime():
    require(Path(__file__).resolve().parent == BUNDLE and WORK.resolve() == WORK
            and socket.gethostname().split('.')[0] == 'n6106', 'Wrong bundle/workspace/node')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit() and os.environ.get('SLURM_JOB_NAME') == 'ua-main-896', 'Expected scheduled allocation')
    for rel, digest in json.loads((BUNDLE/'upstream.json').read_text()).items():
        path = WORK/rel
        require(not path.is_symlink() and path.resolve() == path and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Changed upstream: '+rel)
    sys.path[:0] = [str(BASE), str(WORK/'repo')]
    import gate as previous
    return job, previous, previous.helpers()


def scheduler(job):
    for prior in ('763678','788115'):
        require(command('sacct','-X','-n','-P','-j',prior,'--format=JobID,State,ExitCode')
                == prior+'|FAILED|1:0', 'Prior job does not match diagnosed failure')
    require(command('squeue','-h','-u','dalharth','-o','%i|%j') == job+'|ua-main-896', 'Another account job exists')
    state = command('sacct','-X','-n','-P','-j',job,'--format=JobID,State,Start,TimelimitRaw,NodeList').split('|')
    require(len(state) == 5 and state[:2] == [job,'RUNNING'] and state[3:] == ['1440','n6106'],
            'Unexpected allocation identity/limit/node')
    return datetime.fromisoformat(state[2]).replace(tzinfo=timezone.utc).timestamp()


def lock_test():
    code = '''import fcntl,sys
for name in sys.argv[1:]:
    with open(name,'rb') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: continue
        raise AssertionError('Required lock was not exclusively held: '+name)
print('BOTH_REAL_LOCKS_VERIFIED',flush=True)
'''
    subprocess.run([sys.executable,'-I','-B','-c',code,str(WORK/'collector.lock'),
                    str(WORK/'repo/outputs/final_study/workflow.lock')], check=True, timeout=20)


def prefix(r, p, db, target, cell, spec, *, allow_pinned_intent=False):
    class Boundary(BaseException):
        pass
    result = {}
    items = r.requests_from_ledger(SimpleNamespace(db=db, billing_target=target), cell)
    require(bool(items), 'Expected a saved partial prefix')
    for ordinal, item in enumerate(items):
        r.validate_request(item, cell, spec, ordinal)
        p.legacy.verify_transport(target, item['request_id'], item['response'])
    class Missing:
        def __call__(self, model, messages):
            rid = r.BASE_REQUEST_ID(cell, self.ordinal, messages)
            require(model == cell['model'] and self.ordinal == len(items), 'Wrong reconstructed boundary')
            require(rid == ORPHAN and self.ordinal == ORPHAN_ORDINAL and cell['run_id'] == ORPHAN_RUN,
                    'Pinned orphan is not the reconstructed next request')
            journal = target/'requests'/(rid+'.json')
            require(db.execute('SELECT 1 FROM attempts WHERE id=?',(rid,)).fetchone() is None
                    and not (target/'transport'/(rid+'.json')).exists(), 'Next request already has evidence')
            if allow_pinned_intent:
                require(journal.is_file() and not journal.is_symlink()
                        and hashlib.sha256(journal.read_bytes()).hexdigest() == ORPHAN_SHA
                        and p.read(journal) == dict(request_id=rid,status='INTENT_BEFORE_TRANSPORT'),
                        'Missing or changed exact unsent intent')
            else:
                require(not journal.exists(), 'Next request still has an active intent')
            require(r.mapped_request_id(SimpleNamespace(db=db))(cell, self.ordinal, messages) == rid,
                    'Reconstructed request would use an unexpected replacement ID')
            result.update(run_id=cell['run_id'], ordinal=self.ordinal, request_id=rid,
                          messages_sha256=p.legacy.digest(messages), saved_replies=len(items))
            raise Boundary()
    replay = r.Replay(items, Missing())
    with p.adapter(cell, spec):
        try:
            r.observe(cell, replay)
        except Boundary:
            require(replay.ordinal == len(items), 'Prefix was not fully consumed')
            return result
    raise ValueError('Expected first unsent request was not reached')


def verify():
    require('OPENAI_API_KEY' not in os.environ, 'Offline gate must not receive a provider credential')
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('OFFLINE_GATE_NO_NETWORK')):
        job, old, c = runtime()
        r, p, a = old.r, old.p, old.a
        scheduler(job)
        with c.exclusive_existing(WORK/'repo/outputs/final_study/workflow.lock'):
            lock_test()
            spec = p.load_contract()
            a.prices(spec)
            prior = p.read(BASE/'checkpoint/verified.json')
            require(prior['status'] == 'BILLING_COPY_VERIFIED' and str(prior['copy_job']) == '759889'
                    and prior['contract'] == CONTRACT
                    and prior['manifest_sha256'] == c.file_hash(BASE/'checkpoint/manifest.json'), 'Prior proof changed')
            log = (WORK/'logs/main.763678.log').read_text()
            for prior, digest in FAILURE_HASHES.items():
                require(c.file_hash(WORK/('logs/main.'+prior+'.log')) == digest, 'Failure evidence changed')
            require('sqlite3.OperationalError: database is locked' in log
                    and 'self.budget.reserve(request_id' in log and 'self.db.commit()' in log
                    and 'RecoveryRefused: 0 unresolved attempts; exactly one can be recovered.' in log,
                    'Failure is not the diagnosed pre-transport reservation rollback')
            print('GATE_PHASE=source_inventory utc='+datetime.now(timezone.utc).isoformat(), flush=True)
            before = c.evidence_manifest(TARGET)
            expected_after = reconciled_manifest(before)
            terminal = {name for name in before if name.startswith('runs/') and name.endswith('.json')}
            receipts = sum(name.startswith('transport/') and name.endswith('.json') for name in before)
            with contextlib.closing(c.indexes.connect(TARGET/'budget.sqlite')) as db:
                state = c.indexes.logical_state(db)
                schema = db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall()
                baseline(p.read(TARGET/'summary.json'), state, len(terminal), receipts,
                         r._unresolved(SimpleNamespace(db=db)))
                a.pair(SimpleNamespace(db=db), TARGET)
                require('COVERING INDEX ua_budget_cost_v1' in str(db.execute(
                    'EXPLAIN QUERY PLAN '+c.indexes.COST_SQL).fetchall()), 'Budget index changed')
            ANCHOR.mkdir(mode=0o700, exist_ok=False)
            saved = ANCHOR/'main'
            saved.mkdir(mode=0o700)
            needed = sum((TARGET/name).stat().st_size for name in before)+(TARGET/'budget.sqlite').stat().st_size
            require(shutil.disk_usage(ANCHOR).free > needed*1.1+1024**3, 'Insufficient checkpoint disk space')
            print('FRESH_CHECKPOINT_COPY_BEGIN files='+str(len(before)), flush=True)
            for n, name in enumerate(sorted(before), 1):
                destination = saved/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(TARGET/name, destination)
                if n % 10000 == 0 or n == len(before):
                    print(f'COPY_FILES={n}/{len(before)} utc='+datetime.now(timezone.utc).isoformat(), flush=True)
            c.indexes.backup(TARGET/'budget.sqlite', saved/'budget.sqlite', max_seconds=1800)
            require(c.evidence_manifest(saved) == before, 'Copied evidence differs')
            with contextlib.closing(c.indexes.connect(saved/'budget.sqlite')) as db:
                c.indexes.assert_preserved(state, c.indexes.logical_state(db))
                require(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall() == schema,
                        'Copied schema differs')
                budget = SimpleNamespace(db=db, billing_target=saved)
                cells = p.cells(spec, 'main')
                print('RECONSTRUCTING_UNSENT_BOUNDARY saved_replies=167', flush=True)
                boundary = prefix(r, p, db, saved, cells[751], spec, allow_pinned_intent=True)
                archive_intent(db, saved)
                print('COPY_INTENT_ARCHIVED; CHECKING_ACCOUNTING', flush=True)
                r.accounting(budget, saved, spec, 'main')
                require(terminal == {'runs/'+cell['run_id']+'.json' for cell in cells[:751]}, 'Terminal schedule differs')
                active = {row[0] for row in db.execute("SELECT DISTINCT json_extract(request,'$.cell.run_id') FROM attempts")}
                require(active <= {cell['run_id'] for cell in cells[:752]}, 'Unaccounted future-cell evidence')
                valid = 0
                for n, cell in enumerate(cells[:751], 1):
                    record = r.verify_record(saved/'runs'/(cell['run_id']+'.json'), cell, spec)
                    valid += record['failure'] is None
                    if n % 100 == 0 or n == 751:
                        print(f'OFFLINE_REPLAY={n}/751', flush=True)
                require(valid == 746, 'Valid/failed graph counts differ')
                require(prefix(r, p, db, saved, cells[751], spec) == boundary, 'Reconciled boundary changed')
                c.indexes.assert_preserved(state, c.indexes.logical_state(db))
            require(c.evidence_manifest(saved) == expected_after and c.evidence_manifest(TARGET) == before,
                    'Evidence changed beyond exact intent archival')
            p.verify_contract(spec)
            scheduler(job)
            c.indexes.report(ANCHOR/'original_manifest.json', before)
            c.indexes.report(ANCHOR/'manifest.json', expected_after)
            c.indexes.report(ANCHOR/'reconciliation.json', dict(job=job, parent='788115',
                request_id=ORPHAN, original_sha256=ORPHAN_SHA, original='requests/'+ORPHAN+'.json',
                archive='pretransport_rollbacks/'+ORPHAN+'.json', boundary=boundary,
                baseline_logical_sha256=state['logical_sha256'], date_utc=datetime.now(timezone.utc).isoformat(),
                reason='Exact pre-transport reservation rollback; retain bytes, original identity and all paid evidence'))
            with contextlib.closing(c.indexes.connect(TARGET/'budget.sqlite')) as db:
                c.indexes.assert_preserved(state, c.indexes.logical_state(db))
                require(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall() == schema,
                        'Live schema changed')
                archive_intent(db, TARGET)
                c.indexes.assert_preserved(state, c.indexes.logical_state(db))
            require(c.evidence_manifest(TARGET) == expected_after, 'Live evidence differs after exact archive')
            p.verify_contract(spec)
            scheduler(job)
            report = dict(status='LOCK_RESUME_COPY_VERIFIED', job=job, contract=CONTRACT, terminal=751,
                          valid=valid, receipts=receipts, rows=state['rows'], conservative_usd=state['cost'],
                          logical_sha256=state['logical_sha256'], ledger_state=state, schema=schema, prefix=boundary,
                          manifest_sha256=c.file_hash(ANCHOR/'manifest.json'),
                          bundle_sha256=c.file_hash(BUNDLE/'bundle.sha256'),
                          prior_proof_sha256=c.file_hash(BASE/'checkpoint/verified.json'),
                          failed_log_sha256=c.file_hash(WORK/'logs/main.763678.log'), node='n6106', paid_calls=0,
                          orphan_request_id=ORPHAN, orphan_sha256=ORPHAN_SHA,
                          reconciliation_sha256=c.file_hash(ANCHOR/'reconciliation.json'),
                          original_manifest_sha256=c.file_hash(ANCHOR/'original_manifest.json'))
            approval(report, job, report['manifest_sha256'])
            c.indexes.report(ANCHOR/'verified.json', report)
            print('LOCK_RESUME_COPY_VERIFIED job='+job+' paid_calls=0', flush=True)


def launch():
    job, old, c = runtime()
    start = scheduler(job)
    report = old.p.read(ANCHOR/'verified.json')
    approval(report, job, c.file_hash(ANCHOR/'manifest.json'))
    require(report['node'] == 'n6106' and report['bundle_sha256'] == c.file_hash(BUNDLE/'bundle.sha256')
            and report['prior_proof_sha256'] == c.file_hash(BASE/'checkpoint/verified.json')
            and report['reconciliation_sha256'] == c.file_hash(ANCHOR/'reconciliation.json')
            and report['original_manifest_sha256'] == c.file_hash(ANCHOR/'original_manifest.json')
            and c.file_hash(TARGET/'pretransport_rollbacks'/(ORPHAN+'.json')) == ORPHAN_SHA,
            'Proof binding changed')
    with c.exclusive_existing(WORK/'repo/outputs/final_study/workflow.lock'):
        old.p.verify_contract(old.p.load_contract())
        manifest = c.evidence_manifest(TARGET)
        with contextlib.closing(c.indexes.connect(TARGET/'budget.sqlite')) as db:
            state = c.indexes.logical_state(db)
            schema = db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY type,name').fetchall()
        require_snapshot(old.p.read(ANCHOR/'manifest.json'), report, manifest, state, schema)
        scheduler(job)
        print('LIVE_CHECKPOINT_MATCHES_VERIFIED_COPY job='+job, flush=True)
    # The original runner rechecks all accounting and cached records before transport.
    result = old.r.run('main', 120, REFERENCE, auto_recover=True, walltime_seconds=allowance(start, time.time()))
    print('LOCK_RESUME_COLLECTION_STATUS='+result['status'], flush=True)
    if result['status'] == 'PAUSED_AT_CHECKPOINT':
        raise SystemExit(75)
    require(result['status'] in ('COMPLETE','COMPLETE_WITH_FAILURES'), 'Collection not complete')


if __name__ == '__main__':
    require(sys.argv[1:] in (['verify'], ['launch']), 'Expected verify or launch')
    verify() if sys.argv[1] == 'verify' else launch()
