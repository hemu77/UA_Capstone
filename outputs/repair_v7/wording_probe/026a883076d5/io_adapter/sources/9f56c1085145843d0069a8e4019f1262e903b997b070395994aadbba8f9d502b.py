"""Bounded local checkpoint-write recovery; never retry an API request.

The generation source, prompt contract, request identities and stage allowance
stay frozen. Reconciliation below is restricted to one inspected received reply.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import socket
import sqlite3
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import repair_v8 as v
import calibration_v6_io as local_io
import recover_calibration_v6_checkpoint as checkpoint_recovery

FROZEN_RUNNER = 'bba9050afadcdd6866495474c22ba81756f25010ad498472a128184adfd3d474'
IMPORTED_SHA = v.r.base.sha(__file__)
IO_SHA = v.r.base.sha(local_io.__file__)
RECOVERY_SHA = v.r.base.sha(checkpoint_recovery.__file__)
REQUEST_ID = '269b666d116214835f3fcc00e9dc573c4fc9fb4f20702ca452fcb85c396fb585'
CONTRACT = '026a883076d535553285b80395c83f55755f8a651da0a854a31631dfe8759f13'
BEFORE = '6e53352e2a873e1041c85a366a17104aa1011169e7a411fee490f81f38f3f62e'
AFTER = '160e316a25bfab01d55d2b98a6d5fd1d82b8443d69ae87a2fee4955cb4343703'


def check_source():
    if (v.IMPORTED_SHA != FROZEN_RUNNER or v.r.base.sha(v.__file__) != FROZEN_RUNNER
            or v.r.base.sha(__file__) != IMPORTED_SHA or v.r.base.sha(local_io.__file__) != IO_SHA
            or v.r.base.sha(checkpoint_recovery.__file__) != RECOVERY_SHA):
        raise ValueError('I/O or generation source changed; no automatic continuation.')


@contextlib.contextmanager
def local_writes(target, operation):
    check_source()
    path = target/'io_adapter'/(uuid4().hex+'.json')
    events = []
    audit = dict(status='STARTED', operation=operation, started_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=IMPORTED_SHA, checkpoint_helper_sha256=IO_SHA,
        recovery_helper_sha256=RECOVERY_SHA, frozen_runner_sha256=FROZEN_RUNNER,
        policy='At most five retries of the identical local request-journal write on PermissionError only. No API retries.',
        main_authorized=False, write_retries=events)
    writer = v.r.base.write_json
    for source, sha in [(Path(__file__), IMPORTED_SHA), (Path(local_io.__file__), IO_SHA),
            (Path(checkpoint_recovery.__file__), RECOVERY_SHA)]:
        snapshot = target/'io_adapter'/'sources'/(sha+'.py')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            with snapshot.open('xb') as handle:
                handle.write(source.read_bytes())
        except FileExistsError:
            pass
        if v.r.base.sha(snapshot) != sha:
            raise ValueError('I/O source snapshot differs.')
    writer(path, audit)
    def guarded(path, value):
        check_source()
        return local_io.write_checkpoint(path, value, target/'requests', events, writer,
            persist=lambda: writer(audit_path, audit))
    audit_path = path
    try:
        with patch.object(v.r.base, 'write_json', guarded):
            yield
        audit['status'] = 'COMPLETED'
    except Exception as error:
        audit.update(status='STOPPED', error_type=type(error).__name__)
        raise
    finally:
        audit['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        writer(audit_path, audit)


def reconcile(apply=False):
    with v.r.base.frozen.workflow_lock(), v.isolated(), \
            patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('Offline reconciliation')), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline reconciliation')):
        check_source()
        spec = v.contract('wording_probe')
        if v.r.digest(spec) != CONTRACT:
            raise ValueError('This recovery only applies to the inspected wording contract.')
        target = v.r.folder(spec)
        v.stage_review(spec)
        approved = v.read(target.parent/'authorization.json')
        if approved.get('contract_sha256') != CONTRACT:
            raise ValueError('Wrong authorization.')
        budget = v.r.paid.Budget.__new__(v.r.paid.Budget)
        budget.db = sqlite3.connect(v.r.LEDGER.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            rows = {row[0]: row for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            if budget.unresolved_count() or any(v.r.digest(rows.get(rid)) != sha for rid,sha in approved['baseline'].items()):
                raise ValueError('Uncertain requests or changed historical ledger.')
            row = rows[REQUEST_ID]
            path = target/'requests'/(REQUEST_ID+'.json')
            saved = v.read(path)
            if saved.get('row_sha256') != BEFORE or v.r.digest(row) != AFTER:
                raise ValueError('Not the exact inspected pre/post-parse checkpoint.')
            replacement = checkpoint_recovery.validate_transition(row, saved)
            journal = {}
            for other in (target/'requests').glob('*.json'):
                item = v.read(other)
                expected = AFTER if other.stem == REQUEST_ID else item.get('row_sha256')
                if item.get('request_id') != other.stem or v.r.digest(rows.get(other.stem)) != expected:
                    raise ValueError('Another checkpoint differs; stop for inspection.')
                journal[other.name] = v.r.base.sha(other)
            relevant = {rid for rid,value in rows.items() if json.loads(value[4]).get('cell',{}).get('repair_contract_sha256') == CONTRACT}
            if relevant != {name[:-5] for name in journal}:
                raise ValueError('Stage ledger and journal sets differ.')
            cell = json.loads(row[4])['cell']
            if cell not in v.r.cells(spec):
                raise ValueError('Unapproved cell.')
            prefix = v.receipts(budget, cell, spec, target)
            if len(prefix) != 23 or prefix[-1]['request_id'] != REQUEST_ID:
                raise ValueError('Expected the inspected 23-response prefix.')
            replay = v.ResumingCaller(prefix)
            with v.adapter(cell, spec):
                try:
                    v.run_observation(cell, replay)
                except ValueError as error:
                    if str(error) != 'Replay lacks a recorded request; verify-only cannot purchase one.' or replay.replay.ordinal != 23:
                        raise
                else:
                    raise ValueError('Expected an incomplete cell.')
            report = dict(status='VERIFIED_OFFLINE_NOT_APPLIED', request_id=REQUEST_ID, contract_sha256=CONTRACT,
                source_sha256=IMPORTED_SHA, original_checkpoint=saved, replacement_checkpoint=replacement,
                ledger_rows_sha256=v.r.digest(rows), ledger_usd=budget.spent(), replayed_responses=23,
                paid_requests=0, main_authorized=False,
                cause='Observed Windows PermissionError during local replace; file holder unknown.')
            if apply:
                recovery = target/'checkpoint_recovery'/REQUEST_ID/'recovery.json'
                with local_writes(target, 'reconcile_one_post_parse_checkpoint'):
                    checkpoint_recovery.replace_checkpoint(path, recovery, report, replacement)
                v.accounting(budget, approved, spec)
                after = {row[0]: row for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
                changed = [p.name for p in (target/'requests').glob('*.json') if journal.get(p.name) != v.r.base.sha(p)]
                if after != rows or changed != [path.name]:
                    raise ValueError('Recovery changed more than the inspected checkpoint.')
                report.update(status='RECONCILED_NO_PAID_CALLS', ledger_rows_unchanged=True, changed_journals=changed)
                v.r.write_json(recovery, report)
            return report
        finally:
            budget.db.close()


def execute(stage):
    target = v.r.folder(v.contract(stage))
    with local_writes(target, 'execute_'+stage):
        return v.execute(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check-recovery', 'apply-recovery', 'execute'])
    parser.add_argument('stage', nargs='?', choices=['wording_probe', 'calibration'], default='wording_probe')
    args = parser.parse_args()
    try:
        result = execute(args.stage) if args.action == 'execute' else reconcile(args.action == 'apply-recovery')
        print(json.dumps(result, indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__+': stopped; inspect saved evidence; no automatic API retry.\n')
