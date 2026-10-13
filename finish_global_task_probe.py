"""Finish the approved eight first responses, retaining malformed replies as failures.

This recovery entry point never changes prompts, parsing, identities or quotas.
A received invalid reply is not an empty graph and never gets a replacement.
The original stopped summary remains history; completion.json reports all attempts.
"""
import argparse
import contextlib
import json
import socket
from unittest.mock import patch

import global_task_probe as p

r = p.r
IMPORTED_SHA = r.base.sha(__file__)
FROZEN_PROBE_SHA = 'f3a87d7148bbd41fa1b9a8fe6cb0b4bed9315de930796d27c49a5526a17bd06a'


class TerminalReplay(r.base.ReplayCaller):
    """Keep failure replay read-only and terminal, like the paid global caller."""
    def __init__(self, requests):
        super().__init__(requests)
        self.request_ids = [item['request_id'] for item in requests]

    def parse_response(self, **kwargs):
        try:
            return super().parse_response(**kwargs)
        except ValueError as error:
            saved = self.requests[self.ordinal-1]['response']['parse']
            if saved.get('duplicate_edges', []) != r.paid.clean_numbers(getattr(error, 'duplicate_edges', [])):
                raise RuntimeError('Duplicate-edge evidence changed.') from error
            raise r.DiagnosticInvalid('Recorded global parse failure; no replacement.') from error


def received_request(budget, rid, cell, spec, target):
    row = r.base.ledger_row(budget, rid)
    if row is None or row[3] != 'received':
        raise ValueError('Only received replies can become terminal observations.')
    request, response = json.loads(row[4]), json.loads(row[5])
    usage = response['usage']
    if any(type(usage[key]) is not int or usage[key] < 0 for key in ['prompt_tokens', 'completion_tokens']):
        raise ValueError('Invalid recorded usage.')
    input_rate, output_rate = spec['rates'][cell['model']]
    charge = (usage['prompt_tokens']*input_rate*1.25 + usage['completion_tokens']*output_rate)/1e6
    if (request['cell'] != cell or request['ordinal'] != 0
            or request['settings'] != r.base.settings(cell['model'], 'global')
            or rid != r.base.request_id(cell, 0, request['messages'])
            or row[2] != charge or response['conservative_charge_usd'] != charge):
        raise ValueError('Terminal request identity or accounting differs.')
    r.verify_transport(target, rid, response)
    return dict(request_id=rid, request=request, response=response)


def execute(verify_only=False):
    with r.base.frozen.workflow_lock():
        if p.IMPORTED_SHA != FROZEN_PROBE_SHA or r.base.sha(__file__) != IMPORTED_SHA:
            raise ValueError('Recovery source changed; inspect before proceeding.')
        spec = p.contract()
        target = p.folder(spec)
        report_path = target / 'completion.json'
        r.write_json(report_path, dict(status='VALIDATION_PENDING', contract_sha256=r.digest(spec), main_authorized=False))
        origin, summary_sha = p.origin_evidence()
        approved = json.loads((target.parent / 'authorization.json').read_text(encoding='utf8'))
        if (approved.get('contract_sha256') != r.digest(spec)
                or approved.get('main_authorized') is not False
                or approved.get('origin_authorization_sha256') != r.digest(origin)
                or approved.get('origin_summary_sha256') != summary_sha
                or any(approved.get(key) != origin[key] for key in
                       ['ledger_start_usd', 'cumulative_ceiling_usd', 'additional_usd'])):
            raise ValueError('Original allowance or extension evidence differs.')
        with p.isolated():
            r.stage_review(spec)
            stopped = json.loads((target / 'summary.json').read_text(encoding='utf8'))
            if (stopped.get('status') != 'STOPPED' or stopped.get('error_type') != 'DiagnosticInvalid'
                    or stopped.get('contract_sha256') != r.digest(spec)):
                raise ValueError('Original parser-stop evidence required.')
            evidence = dict(version='terminal-invalid-global-1', source_sha256=IMPORTED_SHA,
                frozen_probe_sha256=FROZEN_PROBE_SHA, contract_sha256=r.digest(spec),
                stopped_summary_sha256=r.digest(stopped),
                change='Continue remaining approved first responses after a received parser failure; never replace it.',
                prompts_changed=False, parser_changed=False, request_identity_changed=False, main_authorized=False)
            sidecar = target / 'terminal_failure_adapter.json'
            if sidecar.exists() and json.loads(sidecar.read_text(encoding='utf8')) != evidence:
                raise ValueError('Recovery adapter evidence changed.')
            r.write_json(sidecar, evidence)
            budget = r.paid.Budget(r.LEDGER, total_cap=approved['cumulative_ceiling_usd'],
                pilot_cap=5., require_existing=True, allow_higher_cap=True)
            results = []
            try:
                r.accounting(budget, approved, spec)
                manifest = r.cells(spec)
                recorded_ids = {cell['run_id']: [row[0] for row in budget.db.execute(
                    "SELECT id FROM attempts WHERE json_extract(request,'$.cell.run_id')=?", (cell['run_id'],))]
                    for cell in manifest}
                counts = [len(recorded_ids[cell['run_id']]) for cell in manifest]
                if any(count > 1 for count in counts) or (verify_only and any(count != 1 for count in counts)):
                    raise ValueError('Expected exactly one first response per cell; no replacements.')
                for audit in (target / 'transport').glob('*.json'):
                    if r.base.ledger_row(budget, audit.stem) is None:
                        raise ValueError('Orphan transport evidence blocks replacement.')
                for failure in (target / 'failures').glob('*.json'):
                    if len(recorded_ids.get(failure.stem, [])) != 1:
                        raise ValueError('Orphan failure evidence blocks replacement.')
                needs_calls = any(count == 0 for count in counts)
                before = {row[0]: r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
                with contextlib.ExitStack() as stack:
                    if not needs_calls:
                        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Cached-only replay')))
                    client = stack.enter_context(r.paid.OpenAI(api_key=r.paid.credential() if needs_calls else 'cached-only-no-network',
                        base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                    if needs_calls:
                        for model in r.MODELS:
                            client.models.retrieve(model)
                    for cell in manifest:
                        if r.base.sha(__file__) != IMPORTED_SHA:
                            raise ValueError('Recovery source changed during execution.')
                        path = target / 'runs' / (cell['run_id'] + '.json')
                        failure_path = target / 'failures' / path.name
                        if path.exists():
                            if failure_path.exists():
                                raise ValueError('Conflicting graph and failure evidence.')
                            record = p.checked.verify_run(path, cell, spec, budget)
                            valid = True
                        else:
                            cached = [received_request(budget, rid, cell, spec, target)
                                for rid in recorded_ids[cell['run_id']]]
                            caller = TerminalReplay(cached) if cached else r.Caller(client, budget, cell, spec, target)
                            valid = True
                            with p.adapter(cell, spec):
                                try:
                                    result = r.base.run_cell(cell, caller)
                                except r.DiagnosticInvalid:
                                    valid = False
                            requests = [received_request(budget, rid, cell, spec, target) for rid in caller.request_ids]
                            if len(requests) != 1:
                                raise ValueError('Diagnostic must retain exactly one first response.')
                            if valid:
                                if failure_path.exists():
                                    raise ValueError('Previously invalid response now parsed; inspect before proceeding.')
                                record = r.base.save_run(path, cell, result, requests, 'received_repair_v7')
                                p.checked.verify_run(path, cell, spec, budget)
                            else:
                                response = requests[0]['response']
                                if response['parse']['valid'] is not False or response['finish_reason'] != 'stop':
                                    raise ValueError('Only proven parser failures may be retained here.')
                                r.verify_transport(target, requests[0]['request_id'], response)
                                record = dict(cell=cell, evidence_type='received_invalid_global_diagnostic',
                                    requests=requests, parse_error=response['parse']['error'])
                                if failure_path.exists() and json.loads(failure_path.read_text(encoding='utf8')) != record:
                                    raise ValueError('Invalid observation changed.')
                                r.write_json(failure_path, record)
                        requests = record['requests']
                        results.append(dict(run_id=cell['run_id'], model=cell['model'], repetition=cell['repetition'],
                            valid_graph=valid, metrics=record.get('metrics'), parse_error=record.get('parse_error'),
                            conservative_usd=sum(item['response']['conservative_charge_usd'] for item in requests),
                            api_seconds=sum(item['response']['diagnostics']['elapsed_ms']/1000 for item in requests)))
                        print(f"Accounted {len(results)}/8 first responses; valid graphs {sum(row['valid_graph'] for row in results)}", flush=True)
                r.accounting(budget, approved, spec)
                after = {row[0]: r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
                if any(after.get(rid) != digest for rid, digest in before.items()) or (not needs_calls and after != before):
                    raise ValueError('Continuation changed existing ledger evidence.')
                invalid = sum(not row['valid_graph'] for row in results)
                report = dict(status='COMPLETE_WITH_INVALID_RESPONSES' if invalid else 'COMPLETE_REVIEW_REQUIRED',
                    contract_sha256=r.digest(spec), main_authorized=False, completed_requests=len(results),
                    completed_graphs=len(results)-invalid, invalid_responses=invalid, results=results,
                    stage_charge_usd=sum(row['conservative_usd'] for row in results),
                    api_seconds=sum(row['api_seconds'] for row in results),
                    existing_ledger_rows_unchanged=True,
                    cached_replay_ledger_unchanged=not needs_calls and before == after)
                r.write_json(report_path, report)
                return {key: value for key, value in report.items() if key != 'results'}
            finally:
                budget.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Require all eight charged responses; block network calls.')
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.verify_only), indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__ + ': stopped; preserve evidence, no automatic replacement.\n')
