"""Second exact owner-approved transport replacement; neither reservation is refunded."""
import argparse
import contextlib
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import repair_v8_transport as first

v, io = first.v, first.io
RID = 'f577020a10e58e11a3fee44bfce1fda5bbf6feca9f613c11afc8f57942fb213a'
BEFORE = '915a6d7ae4b973b162197908e5d91a90eeae7dc4e7da48f37a6a65b9fba2456b'
NONCE = 'owner-oct13-v8-http520-one-replacement'
REASON = 'Owner authorized one exact HTTP-520 replacement on 2026-10-13; both original reservations retained; unchanged $15 cap; no main study.'
SOURCE_SHA = v.r.base.sha(__file__)
FIRST_SOURCE_SHA = '6e76f96658322882d0a2ab9465bb4e2a65aae018489c8d663abf6747ae18f2a7'
INSPECTION_ROOT = v.r.DESTINATION/'calibration'/first.CONTRACT[:12]


def target():
    if v.r.base.sha(__file__) != SOURCE_SHA or v.r.base.sha(first.__file__) != FIRST_SOURCE_SHA:
        raise ValueError('Recovery source changed.')
    return first.target()


def replacement_id():
    return hashlib.sha256(json.dumps([RID, 'replacement', NONCE]).encode()).hexdigest()


def validate_original(row):
    if v.r.digest(row) != BEFORE or row[0] != RID or row[3] != 'uncertain':
        raise ValueError('Not the explicitly inspected second uncertain request.')
    request, response = json.loads(row[4]), json.loads(row[5])
    if (request['cell']['repair_contract_sha256'] != first.CONTRACT
            or first.BASE_ID(request['cell'], request['ordinal'], request['messages']) != RID
            or request.get('replaces_request_id')
            or response.get('error_type') != 'InternalServerError'
            or response['diagnostics'].get('http_status') != 520):
        raise ValueError('Second recovery identity or HTTP-520 evidence differs.')


def evidence():
    previous = first.evidence()
    record = v.read(target()/'transport_recovery_second'/'recovery.json')
    original, retired = record['original_row'], record['retired_row']
    validate_original(original)
    response = json.loads(retired[5])
    if (record.get('source_sha256') != SOURCE_SHA or record.get('nonce') != NONCE
            or record.get('first_recovery_sha256') != v.r.digest(previous)
            or record.get('replacement_id') != replacement_id()
            or retired[:3] != original[:3] or retired[3] != 'abandoned' or retired[4] != original[4]
            or response.get('previous_response') != json.loads(original[5])
            or response.get('previous_status') != 'uncertain'
            or response.get('authorization') != REASON
            or response.get('replacement_authorization') != NONCE
            or response.get('retained_reservation_usd') != original[2]
            or response.get('accounting') != 'full_reservation_retained_no_replay'
            or response.get('billing_verified') is not False):
        raise ValueError('Second retirement evidence differs.')
    return [previous, record]


def check_retired(budget, records):
    for record in records:
        row = v.r.base.ledger_row(budget, record['original_row'][0])
        if v.r.paid.clean_numbers(row) != record['retired_row']:
            raise ValueError('Retired ledger row differs.')


def retire():
    """Checkpoint before the one allowed mutation, so a local crash costs no new call."""
    with v.r.base.frozen.workflow_lock():
        root, previous = target(), first.evidence()
        approved = v.read(root.parent/'authorization.json')
        if (approved['contract_sha256'] != first.CONTRACT or approved['additional_usd'] != 15.
                or approved['cumulative_ceiling_usd'] != approved['ledger_start_usd']+15.):
            raise ValueError('Original calibration allowance differs.')
        budget = v.r.paid.Budget(v.r.LEDGER, total_cap=approved['cumulative_ceiling_usd'],
            pilot_cap=5., require_existing=True, allow_higher_cap=True)
        try:
            first.check_retired(budget, previous)
            folder = root/'transport_recovery_second'
            snapshot = folder/'private_snapshot.json'
            rows = {r[0]: r for r in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            journal_path = root/'requests'/(RID+'.json')
            if not snapshot.exists():
                validate_original(rows[RID])
                if budget.unresolved_count() != 1 or v.read(journal_path).get('row_sha256') != BEFORE:
                    raise ValueError('Another unresolved request or changed checkpoint.')
                v.r.write_json(snapshot, dict(original_row=rows[RID], original_journal=v.read(journal_path),
                    original_summary=v.read(root/'summary.json'), source_sha256=SOURCE_SHA,
                    first_recovery_sha256=v.r.digest(previous),
                    other_rows={rid: v.r.digest(row) for rid, row in rows.items() if rid != RID}))
            saved = v.read(snapshot)
            validate_original(saved['original_row'])
            if (saved['source_sha256'] != SOURCE_SHA or saved['first_recovery_sha256'] != v.r.digest(previous)
                    or any(v.r.digest(rows.get(rid)) != sha for rid, sha in saved['other_rows'].items())):
                raise ValueError('Historical ledger or recovery source changed.')
            if rows[RID][3] == 'uncertain':
                validate_original(rows[RID])
                budget.abandon(RID, REASON, replacement_authorization=NONCE)
            retired = v.r.base.ledger_row(budget, RID)
            record = dict(original_row=saved['original_row'], retired_row=retired,
                source_sha256=SOURCE_SHA, nonce=NONCE, replacement_id=replacement_id(),
                first_recovery_sha256=v.r.digest(previous), main_authorized=False)
            path = folder/'recovery.json'
            if path.exists() and v.read(path) != v.r.paid.clean_numbers(record):
                raise ValueError('Existing second retirement record differs.')
            v.r.write_json(path, record)
            records = evidence()
            current_journal = v.read(journal_path)
            expected = dict(request_id=RID, row_sha256=v.r.digest(retired))
            if current_journal not in [saved['original_journal'], expected]:
                raise ValueError('Unexpected original journal transition.')
            with io.local_writes(root, 'retire_exact_http520'):
                v.r.base.write_json(journal_path, expected)
            return dict(status='SECOND_EXACT_REPLACEMENT_AUTHORIZED',
                retained_reservation_usd=sum(r['retired_row'][2] for r in records),
                replacement_id=replacement_id(), paid_calls=0, main_authorized=False)
        finally:
            budget.db.close()


def accounting(budget, approved, spec):
    records = evidence()
    check_retired(budget, records)
    retired = {r['original_row'][0] for r in records}
    replacements = {r['replacement_id']: r['original_row'][0] for r in records}
    current = {row[0]: v.r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    if budget.unresolved_count() or any(current.get(rid) != sha for rid, sha in approved['baseline'].items()):
        raise ValueError('Historical ledger changed or unresolved requests remain.')
    relevant, allowed = set(), {cell['run_id'] for cell in v.r.cells(spec)}
    contract_sha = v.r.digest(spec)
    for rid, status, raw in budget.db.execute('SELECT id,status,request FROM attempts'):
        request = json.loads(raw)
        cell = request.get('cell', {})
        if cell.get('repair_contract_sha256') == contract_sha:
            if cell.get('run_id') not in allowed or (status != 'received' and rid not in retired):
                raise ValueError('Unapproved or unresolved stage request.')
            if rid in replacements and request.get('replaces_request_id') != replacements[rid]:
                raise ValueError('Replacement provenance missing.')
            relevant.add(rid)
    journal = {}
    for path in (target()/'requests').glob('*.json'):
        item = v.read(path)
        if item.get('request_id') != path.stem or item.get('row_sha256') != current.get(path.stem):
            raise ValueError('Request journal differs; no automatic replacement.')
        journal[path.stem] = item
    if relevant != set(journal):
        raise ValueError('Ledger/journal request set differs.')


@contextlib.contextmanager
def recovered():
    records = evidence()
    by_id = {record['original_row'][0]: record for record in records}

    def request_id(cell, ordinal, messages):
        rid = first.BASE_ID(cell, ordinal, messages)
        if rid in by_id:
            original = json.loads(by_id[rid]['original_row'][4])
            if any(original[k] != val for k, val in [('cell', cell), ('ordinal', ordinal), ('messages', messages)]):
                raise ValueError('Replacement request differs.')
            return by_id[rid]['replacement_id']
        return rid

    def receipts(budget, cell, spec, root):
        check_retired(budget, records)
        # Only logical replay excludes the two retired originals; spending includes both.
        class ReceiptRows:
            def execute(self, sql, params):
                expected = "SELECT id,phase,cost,status,request,response FROM attempts WHERE json_extract(request,'$.cell.run_id')=?"
                if sql != expected:
                    raise ValueError('Frozen receipt query changed.')
                return budget.db.execute(sql+' AND id NOT IN (?,?)', (*params, *by_id))
        return first.BASE_RECEIPTS(SimpleNamespace(db=ReceiptRows()), cell, spec, root)

    with patch.object(v.r.base, 'request_id', request_id), patch.object(v, 'receipts', receipts), \
            patch.object(v, 'ORIGINAL_ACCOUNTING', accounting):
        yield records


def _run(action):
    if action == 'retire':
        return retire()
    with recovered() as records:
        if action == 'inspect':
            import inspect_repair_v8 as inspector
            writer = v.r.write_json
            def publish(path, report):
                if path == target()/'inspection.json' and report.get('status') == 'PUBLIC_REPLAY_VERIFIED':
                    retained = sum(r['retired_row'][2] for r in records)
                    report.update(retained_uncertain_reservation_usd=retained,
                        total_charge_or_reservation_usd=report['conservative_usd']+retained,
                        transport_recovery_sha256=[v.r.digest(r) for r in records])
                return writer(path, report)
            with patch.object(v.r, 'write_json', publish):
                return inspector.inspect('calibration')
        if action == 'execute':
            return io.execute('calibration')
        if action == 'verify':
            return v.execute('calibration', verify_only=True)
        raise ValueError('Unknown action.')


def run(action):
    if action != 'inspect':
        return _run(action)
    with v.r.base.frozen.workflow_lock():
        v.r.write_json(INSPECTION_ROOT/'inspection.json', dict(status='VALIDATION_PENDING', main_authorized=False))
        # The outer lock also covers recovery evidence; the inspector must not reacquire it.
        with patch.object(v.r.base.frozen, 'workflow_lock', return_value=contextlib.nullcontext()):
            return _run(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['retire', 'execute', 'verify', 'inspect'])
    print(json.dumps(run(parser.parse_args().action), indent=2))
