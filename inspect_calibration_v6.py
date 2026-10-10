"""Offline inspection of revised calibration; never authorizes new collection."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean


def decisions_from_record(record):
    base = {k: record['cell'][k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'repetition']}
    return [dict(**base, decision_method=d['method'], prompt_method=d['prompt_method'],
                 requested_count=d['requested_count'], attempt=d['attempt'], valid=d['valid'])
            for d in record['decisions']]


def failure_rates(decisions):
    """Count initial decisions separately from correction attempts."""
    fields = ['model', 'method', 'culture', 'language', 'prompt_method', 'requested_count']
    groups = defaultdict(list)
    for row in decisions:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, rows in groups.items():
        first = [r for r in rows if r['attempt'] == 1]
        failed = sum(not r['valid'] for r in first)
        result.append(dict(zip(fields, key), first_decisions=len(first), first_failures=failed,
                           correction_attempts=len(rows)-len(first),
                           first_failure_rate=failed/len(first) if first else None))
    return result


def forecasts(rows, complete):
    """Hypothetical fresh studies, not permission or automatic calibration reuse."""
    if not complete:
        return []
    import calibration_v6 as c
    cfg = c.revised.config()
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ['model', 'method', 'culture', 'language'])].append(row)
    if not groups:
        raise ValueError('No measured cells for a complete calibration forecast.')
    averages = {key: {metric: mean(r[metric] for r in values)
                      for metric in ['conservative_usd', 'api_seconds']}
                for key, values in groups.items()}
    result = []
    for global_repeats in [8, 32]:
        cost, seconds, networks, transferred = 0., 0., 0, 0
        for model in cfg['models']:
            for method in cfg['methods']:
                repeats = global_repeats if method == 'global' else 8
                for country, language in cfg['settings']:
                    key = (model, method, country, language)
                    values = averages.get(key)
                    if values is None:
                        # Only the unmeasured non-US country cells may use this proxy.
                        if model == 'gpt-6-luna' or country == 'us' or language != 'english':
                            raise ValueError('Missing directly measured calibration cell.')
                        try:
                            base = averages[(model, method, 'us', 'english')]
                            luna_base = averages[('gpt-6-luna', method, 'us', 'english')]
                            luna_country = averages[('gpt-6-luna', method, country, language)]
                            values = {k: base[k]*luna_country[k]/luna_base[k] for k in base}
                        except (KeyError, ZeroDivisionError) as error:
                            raise ValueError('Country-transfer forecast lacks a valid baseline.') from error
                        transferred += repeats
                    if any(not math.isfinite(v) or v < 0 for v in values.values()):
                        raise ValueError('Invalid cost or duration in forecast.')
                    networks += repeats
                    cost += repeats*values['conservative_usd']
                    seconds += repeats*values['api_seconds']
        result.append(dict(scenario=f'global_{global_repeats}_others_8', networks=networks,
            conservative_usd=cost, cost_with_25_percent_allowance=cost*1.25,
            serial_api_hours=seconds/3600, api_hours_with_50_percent_allowance=seconds/2400,
            unmeasured_country_transfer_networks=transferred, calibration_reuse=False,
            main_authorized=False))
    return result


def inspect():
    import networkx as nx
    from PIL import Image
    import calibration_v6 as c

    with c.frozen.workflow_lock():
        spec = c.contract()
        target = c.folder(spec)
        if json.loads((target/'contract.json').read_text(encoding='utf-8')) != spec:
            raise ValueError('Saved calibration contract differs from current source/runtime.')
        c.verify_probe()
        manifest = c.cells(spec)
        expected = {cell['run_id'] for cell in manifest}
        if any(p.stem not in expected for p in (target/'runs').glob('*.json')):
            raise ValueError('Unexpected receipt in calibration directory.')
        rows, decisions, hashes, all_ids, starts, ends = [], [], {}, set(), [], []
        for cell in manifest:
            path = target/'runs'/(cell['run_id']+'.json')
            if not path.exists():
                continue
            record = c.verify_receipt(path, cell)
            graph = nx.read_adjlist(path.with_suffix('.adj'))
            if set(graph) != set(c.revised.previous.adult_roster()) or nx.number_of_selfloops(graph):
                raise ValueError('Wrong roster or self-links in calibration graph.')
            with Image.open(path.with_suffix('.png')) as png:
                if png.info.get('Description') != 'received_revised_calibration':
                    raise ValueError('PNG is not labelled as received calibration evidence.')
                png.verify()
            ids = [r['request_id'] for r in record['requests']]
            if all_ids.intersection(ids):
                raise ValueError('A paid request is counted in multiple networks.')
            all_ids.update(ids)
            base = {k: cell[k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'repetition']}
            requests = record['requests']
            starts.extend(datetime.fromisoformat(r['request']['date_utc']) for r in requests)
            ends.extend(datetime.fromisoformat(r['response']['date_utc']) for r in requests)
            rows.append(dict(**base, nodes=len(graph), edges=graph.number_of_edges(),
                **record['metrics'], requests=len(requests),
                conservative_usd=sum(r['response']['conservative_charge_usd'] for r in requests),
                prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in requests),
                completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in requests),
                api_seconds=sum(r['response']['diagnostics']['elapsed_ms']/1000 for r in requests)))
            decisions.extend(decisions_from_record(record))
            hashes[path.name] = c.sha(path)
        complete = len(rows) == len(manifest)
        if not complete and (target/'hypothetical_main_forecasts.csv').exists():
            raise ValueError('Existing forecast conflicts with missing receipts; reconcile evidence first.')
        rates = failure_rates(decisions)
        for name, values in [('inspection_networks.csv', rows), ('decision_compliance.csv', rates)]:
            if not values and (target/name).exists():
                raise ValueError('Existing table conflicts with empty evidence; reconcile receipts first.')
        first_total = sum(r['first_decisions'] for r in rates)
        first_failed = sum(r['first_failures'] for r in rates)
        adapters = []
        for path in sorted((target/'io_adapter').glob('*.json')):
            adapter = json.loads(path.read_text(encoding='utf-8'))
            version = adapter['adapter_source_sha256']
            if not re.fullmatch('[0-9a-f]{64}',version):
                raise ValueError('Invalid I/O-adapter source version.')
            if (adapter['contract_sha256'] != c.frozen.digest(spec)
                    or version != c.sha(target/'io_adapter'/'sources'/(version+'.py'))):
                raise ValueError('I/O-adapter provenance differs from inspected source.')
            adapters.append(dict(file=path.name, sha256=c.sha(path), status=adapter['status'],
                source_unchanged=adapter.get('source_unchanged_during_execution'),
                local_file_retry_count=len(adapter['write_retries'])))
        report = dict(status='COMPLETE_CALIBRATION' if complete else 'PARTIAL_CALIBRATION',
            inspected_at_utc=datetime.now(timezone.utc).isoformat(),
            contract_sha256=c.frozen.digest(spec), inspector_sha256=c.sha(Path(__file__)),
            expected_networks=len(manifest), verified_networks=len(rows), unique_requests=len(all_ids),
            receipt_sha256=hashes, networks=rows, decision_compliance=rates,
            first_decisions=first_total, first_failures=first_failed,
            first_failure_rate=first_failed/first_total if first_total else None,
            correction_attempts=sum(r['correction_attempts'] for r in rates),
            empty_networks=[r['run_id'] for r in rows if r['edges']==0],
            undefined_metrics={metric: sum(r[metric] is None for r in rows)
                               for metric in (record['metrics'] if rows else [])},
            completed_network_conservative_usd=sum(r['conservative_usd'] for r in rows),
            summed_api_hours=sum(r['api_seconds'] for r in rows)/3600,
            observed_span_hours_including_pauses=(max(ends)-min(starts)).total_seconds()/3600 if starts else None,
            io_adapter_sessions=adapters,
            checkpoint_recovery_sha256=c.sha(target/'checkpoint_recovery.json') if (target/'checkpoint_recovery.json').exists() else None,
            hypothetical_main_forecasts=forecasts(rows, complete),
            main_authorized=False, human_bilingual_validation=False, api_calls_by_inspection=0,
            assumptions=[
                'All graphs are exploratory calibration, never automatic confirmation-sample reuse.',
                'Empty graphs remain observed outcomes; undefined metrics remain null.',
                'Failure-rate denominators are initial decisions, not all attempts or independent people.',
                'Costs use recorded tokens, frozen rates, 25 percent input uplift, and no cache discount; not invoices.',
                'Completed-network totals exclude unfinished calls; reconcile the private ledger separately.',
                'Forecasts assume unchanged prompts/settings and fresh samples, not approved main-study designs.',
                'Other-model non-US country costs/times transfer Luna country-to-US ratios; not directly measured.',
                'Serial API hours exclude local processing, inter-request work, downtime and analysis.',
                'Cost/runtime allowances are planning margins, not confidence intervals or guarantees.',
                'Small unequal calibration counts cannot establish language equivalence or statistical power.',
                'PNG checks establish metadata/file integrity, not independent visual reconstruction.'
            ])
        c.write_csv(target/'inspection_networks.csv', rows)
        c.write_csv(target/'decision_compliance.csv', rates)
        c.write_csv(target/'hypothetical_main_forecasts.csv', report['hypothetical_main_forecasts'])
        table_names = ['inspection_networks.csv','decision_compliance.csv']
        if complete:
            table_names.append('hypothetical_main_forecasts.csv')
        report['tables_sha256'] = {name:c.sha(target/name) for name in table_names if (target/name).exists()}
        c.write_json(target/'inspection.json', report)
        return report


if __name__ == '__main__':
    report = inspect()
    print(json.dumps({k: report[k] for k in ['status', 'verified_networks', 'unique_requests',
        'first_failures', 'first_decisions', 'completed_network_conservative_usd',
        'summed_api_hours', 'hypothetical_main_forecasts', 'main_authorized']}, indent=2))
