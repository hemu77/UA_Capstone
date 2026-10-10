"""Resume the frozen probe, retaining received abnormal replies as failures.

The network generator must reject incomplete responses. This diagnostic screen
instead counts a received truncation/refusal as a failed observation. It never
repairs the text, changes request identity, or buys a replacement.
"""
import getpass
import hashlib
import json

import wording_probe as probe


def verify_partition(budget, approved):
    """One bounded recovery from the inspected 61/11 checkpoint, not a retry loop."""
    expected = []
    for row in probe.manifest():
        settings = {'max_completion_tokens': 512, 'reasoning_effort': 'none'}
        messages = [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['user']}]
        # Match the FROZEN caller's identity, without reserving or dispatching.
        identity = [probe.cell(row, approved), 0, probe.MODEL, messages, settings]
        expected.append(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
    rows = budget.db.execute('SELECT id,status,cost,request,response FROM attempts').fetchall()
    received = {}
    for rid, status, cost, request, response in rows:
        request = json.loads(request)
        if request.get('cell', {}).get('probe_contract_sha256') != probe.revised.frozen.digest(approved['contract']):
            continue
        saved = json.loads(response) if response else {}
        actual_id = hashlib.sha256(json.dumps([request['cell'], request['ordinal'], request['model'],
            request['messages'], request['settings']], sort_keys=True).encode()).hexdigest()
        if rid != actual_id or status != 'received' or cost != saved.get('conservative_charge_usd') or not saved.get('response_id'):
            raise ValueError('Historical probe receipt is inconsistent; no transport allowed.')
        received[rid] = saved
    if set(received) != set(expected[:61]) or any(budget.cached(rid) is not None for rid in expected[61:]):
        raise ValueError('Recovery requires exactly 61 received original IDs and 11 missing original IDs.')
    if received[expected[60]]['finish_reason'] != 'length':
        raise ValueError('This recovery is only for the inspected truncated 61st reply.')
    return dict(received_ids=expected[:61], missing_ids=expected[61:])


def collect(client, budget, approved):
    if budget.total_cap != approved['cumulative_ceiling_usd'] or budget.spent() + 1e-12 < approved['ledger_start_usd']:
        raise ValueError('Original ledger/cap is missing or inconsistent.')
    if budget.unresolved_count():
        raise RuntimeError('Unresolved paid attempt; no requests dispatched.')
    source = probe.Path(__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for index, row in enumerate(probe.manifest(), 1):
        if approved['contract'] != probe.contract() or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError('Source or protocol changed during probe; stopped.')
        caller = probe.paid.PaidCaller(client, budget, probe.cell(row, approved),
            execution_source_sha256={source.name: source_hash})
        error = None
        try:
            text = caller(probe.MODEL, [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['user']}])
        except ValueError:
            # Only a durable received abnormal reply is an observable failure.
            # All pre-dispatch, ledger, budget and transport errors still stop.
            cached = budget.cached(caller.last_request_id) if caller.last_request_id else None
            if not cached or cached.get('finish_reason') not in {'length', 'content_filter'}:
                raise
            error = ValueError('Received abnormal finish: ' + cached['finish_reason'])
        if error is None:
            try:
                probe.revised.parse_response('local', text, probe.nx.empty_graph(probe.revised.previous.adult_roster()),
                    curr_pid=row['actor'], num_choices=row['count'])
            except ValueError as failure:
                error = failure
        budget.record_parse(caller.last_request_id, error)
        result = dict(**{key: row[key] for key in ['probe_id', 'language', 'count', 'actor', 'wording']},
            contract_sha256=probe.revised.frozen.digest(approved['contract']), request_id=caller.last_request_id,
            evidence_type='paid_first_response_wording_probe_not_network', response=budget.cached(caller.last_request_id),
            valid=error is None, error=None if error is None else str(error))
        path = probe.OUTPUT / 'replies' / (row['probe_id'] + '.json')
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != result:
            raise ValueError('Existing exported reply changed; stop instead of overwriting evidence.')
        probe.write_json(path, result)
        rows.append(result)
        print(f'Probe {index}/72; cumulative charged/reserved ${budget.spent():.5f}', flush=True)
    return rows


def execute(api_key):
    with probe.revised.frozen.workflow_lock():
        approved = probe.authorization()
        if not api_key:
            raise RuntimeError('A credential is required; never save it in the repository.')
        budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'],
            pilot_cap=min(5, approved['cumulative_ceiling_usd']), require_existing=True, allow_higher_cap=True)
        recovery = dict(source_sha256=hashlib.sha256(probe.Path(__file__).read_bytes()).hexdigest(),
            policy='Count received length/content_filter replies as failures; no retries or request changes.')
        previous = probe.OUTPUT / 'stopped_before_resume.json'
        if not previous.exists() and (probe.OUTPUT / 'report.json').exists():
            probe.write_json(previous, json.loads((probe.OUTPUT / 'report.json').read_text(encoding='utf-8')))
        try:
            if budget.unresolved_count():
                raise RuntimeError('Unresolved paid attempt; no client opened.')
            recovery['initial_partition'] = verify_partition(budget, approved)
            with probe.paid.OpenAI(api_key=api_key, base_url='https://api.openai.com/v1', max_retries=0, timeout=60) as client:
                client.models.retrieve(probe.MODEL)
                rows = collect(client, budget, approved)
            result = probe.summarize(rows)
            result.update(contract=approved['contract'], recovery=recovery,
                cumulative_ledger_usd=budget.spent(), ledger_change_usd=budget.spent() - approved['ledger_start_usd'],
                probe_conservative_charge_usd=sum(r['response']['conservative_charge_usd'] for r in rows),
                received_prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in rows),
                received_completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in rows),
                replies_sha256={r['probe_id']: hashlib.sha256((probe.OUTPUT / 'replies' / (r['probe_id'] + '.json')).read_bytes()).hexdigest() for r in rows})
            probe.write_json(probe.OUTPUT / 'report.json', result)
            return {k: result[k] for k in ['status', 'probe_calls', 'new_networks', 'engineering_screen', 'revised_failures', 'ledger_change_usd']}
        except Exception as error:
            probe.write_json(probe.OUTPUT / 'report.json', dict(status='STOPPED', error_type=type(error).__name__,
                recovery=recovery, main_authorized=False, cumulative_ledger_usd=budget.spent(),
                unresolved_attempts=budget.unresolved_count()))
            raise
        finally:
            budget.db.close()


def read_key():
    try:
        return probe.paid.credential()
    except RuntimeError:
        return getpass.getpass('API key (not saved): ')


if __name__ == '__main__':
    try:
        print(json.dumps(execute(read_key()), indent=2))
    except Exception as error:
        raise SystemExit(type(error).__name__ + ': probe stopped; inspect local report. No automatic paid retry.') from None
