"""Key-free public receipt replay and descriptive V8 reports, never generation."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import socket
from statistics import mean
from unittest.mock import patch

import networkx as nx
from PIL import Image

import repair_v8 as v
from inspect_calibration_v6 import forecasts
from analyze_saved_study import detectable_effect


class DecisionReplay(v.ResumingCaller):
    def __init__(self, requests):
        super().__init__(requests)
        self.decisions = []

    def parse_response(self, **kwargs):
        saved = self.replay.requests[self.replay.ordinal-1]['response']['parse']
        attempt = self.decisions[-1]['attempt']+1 if self.decisions and not self.decisions[-1]['valid'] else 1
        self.decisions.append(dict(attempt=attempt, valid=saved['valid'], actor=kwargs.get('curr_pid'),
            requested_count=kwargs.get('num_choices'), parser_method=kwargs['method']))
        return super().parse_response(**kwargs)


def inspect(stage):
    with v.r.base.frozen.workflow_lock(), v.isolated(), \
            patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('Public replay cannot open API client')), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Public replay cannot use network')):
        spec = v.contract(stage)
        target = v.r.folder(spec)
        v.r.write_json(target/'inspection.json', dict(status='VALIDATION_PENDING',
            contract_sha256=v.r.digest(spec), main_authorized=False, billing_verified=False))
        if v.read(target/'contract.json') != spec:
            raise ValueError('Saved experiment identity differs.')
        summary = v.read(target/'summary.json')
        if (summary.get('contract_sha256') != v.r.digest(spec)
                or summary.get('status') not in {'COMPLETE_REVIEW_REQUIRED', 'COMPLETE_WITH_INVALID_RESPONSES'}):
            raise ValueError('A completed, exact-version summary is required.')
        expected = {cell['run_id'] for cell in v.r.cells(spec)}
        for kind in ['runs', 'failures']:
            if any(p.stem not in expected for p in (target/kind).glob('*.json')):
                raise ValueError('Unexpected observation outside the planned design.')
        rows, decisions, ids, starts, ends = [], [], set(), [], []
        edges = {}
        recomputed = []
        for cell in v.r.cells(spec):
            path = target/'runs'/(cell['run_id']+'.json')
            failure_path = target/'failures'/path.name
            if path.exists() == failure_path.exists():
                raise ValueError('Expected exactly one graph or failure per planned cell.')
            if path.exists():
                record = v.checked.verify_run(path, cell, spec)
                graph = nx.read_adjlist(path.with_suffix('.adj'))
                if set(graph) != set(v.r.v6.previous.adult_roster()) or nx.number_of_selfloops(graph):
                    raise ValueError('Graph roster or self-link check failed.')
                with Image.open(path.with_suffix('.png')) as png:
                    if png.info.get('Description') != 'received_repair_v7':
                        raise ValueError('PNG source metadata differs.')
                    png.verify()
                extra = dict(nodes=len(graph), edges=graph.number_of_edges(), **record['metrics'])
                edges[cell['run_id']] = {frozenset(pair) for pair in graph.edges}
            else:
                record = v.read(failure_path)
                if record['cell'] != cell or record.get('evidence_type') != 'received_invalid_v8_observation':
                    raise ValueError('Failure identity differs.')
                extra = dict(nodes=None, edges=None)
            caller = DecisionReplay(record['requests'])
            with v.adapter(cell, spec):
                result, failure = v.run_observation(cell, caller)
            if failure != record.get('failure') or caller.replay.ordinal != len(record['requests']):
                raise ValueError('Terminal outcome or request count differs from replay.')
            recomputed.append(v.summary_row(cell, record, failure))
            identity = {k: cell[k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'repetition', 'wording']}
            first = [d for d in caller.decisions if d['attempt'] == 1]
            rows.append(dict(**identity, valid_graph=failure is None, failure=failure,
                **extra, requests=len(record['requests']), first_decisions=len(first),
                first_failures=sum(not d['valid'] for d in first),
                correction_requests=len(caller.decisions)-len(first),
                conservative_usd=sum(q['response']['conservative_charge_usd'] for q in record['requests']),
                api_seconds=sum(q['response']['diagnostics']['elapsed_ms']/1000 for q in record['requests'])))
            decisions.extend(dict(**identity, **d) for d in caller.decisions)
            for ordinal, q in enumerate(record['requests']):
                if (q['request_id'] in ids or q['request']['cell'] != cell or q['request']['ordinal'] != ordinal
                        or q['request']['settings'] != v.r.base.settings(cell['model'], cell['method'])
                        or q['request_id'] != v.r.base.request_id(cell, ordinal, q['request']['messages'])):
                    raise ValueError('Repeated or invalid public request identity.')
                ids.add(q['request_id'])
                usage = q['response']['usage']
                if any(type(usage[k]) is not int or usage[k] < 0 for k in ['prompt_tokens', 'completion_tokens']):
                    raise ValueError('Invalid public usage evidence.')
                rates = spec['rates'][cell['model']]
                cost = (usage['prompt_tokens']*rates[0]*1.25 + usage['completion_tokens']*rates[1])/1e6
                if not math.isclose(cost, q['response']['conservative_charge_usd'], rel_tol=0, abs_tol=1e-12):
                    raise ValueError('Public usage and cost disagree.')
                starts.append(datetime.fromisoformat(q['request']['date_utc']))
                ends.append(datetime.fromisoformat(q['response']['date_utc']))
        if recomputed != summary['results'] or summary['completed_cells'] != len(rows):
            raise ValueError('Summary differs from underlying observations.')
        valid = sum(row['valid_graph'] for row in rows)
        charge = sum(row['conservative_usd'] for row in rows)
        if (summary['completed_graphs'] != valid or summary['invalid_cells'] != len(rows)-valid
                or abs(summary['stage_charge_usd']-charge) > 1e-10):
            raise ValueError('Summary counts or charge differ.')
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in ['model', 'method', 'culture', 'language', 'wording'])].append(row)
        conditions = []
        for key, group in sorted(groups.items()):
            good = [row for row in group if row['valid_graph']]
            count = sum(row['first_decisions'] for row in group)
            failed = sum(row['first_failures'] for row in group)
            conditions.append(dict(zip(['model', 'method', 'culture', 'language', 'wording'], key),
                attempted_cells=len(group), valid_graphs=len(good), failed_cells=len(group)-len(good),
                empty_graphs=sum(row['edges'] == 0 for row in good),
                mean_edges=mean(row['edges'] for row in good) if good else None,
                mean_density=mean(row['density'] for row in good) if good else None,
                first_decisions=count, first_failures=failed, first_failure_rate=failed/count if count else None,
                correction_requests=sum(row['correction_requests'] for row in group),
                conservative_usd=sum(row['conservative_usd'] for row in group)))
        # Union columns keep failed cells explicit without silently dropping missing metrics.
        columns = list(dict.fromkeys(k for row in rows for k in row))
        v.r.base.write_csv(target/'inspection_networks.csv', [{k: row.get(k) for k in columns} for row in rows])
        v.r.base.write_csv(target/'condition_summary.csv', conditions)
        v.r.base.write_csv(target/'decision_compliance.csv', decisions)
        if stage == 'wording_probe':
            matches = {tuple(row[k] for k in ['model', 'language', 'repetition', 'wording']): row for row in rows}
            sensitivity = []
            for row in rows:
                if row['wording'] == 'reference':
                    continue
                reference = matches[tuple(row[k] for k in ['model', 'language', 'repetition'])+('reference',)]
                a, b = edges.get(reference['run_id']), edges.get(row['run_id'])
                valid_pair = a is not None and b is not None
                sensitivity.append(dict(model=row['model'], language=row['language'], repetition=row['repetition'],
                    wording=row['wording'], valid_pair=valid_pair,
                    edge_jaccard=(len(a & b)/len(a | b) if a | b else 1.) if valid_pair else None,
                    density_difference=row['density']-reference['density'] if valid_pair else None,
                    interpretation='Matched within-diagnostic sensitivity; no causal or significance claim.'))
            v.r.base.write_csv(target/'wording_sensitivity.csv', sensitivity)
        forecast = forecasts(rows, complete=True) if stage == 'calibration' else []
        if forecast:
            for row in forecast:
                row['interpretation'] = 'Cost of attempted cells, including observed failures; not a guarantee of this many valid graphs.'
            v.r.base.write_csv(target/'hypothetical_main_forecasts.csv', forecast)
            v.r.base.write_csv(target/'detectable_effect_scenarios.csv', [dict(
                independent_pairs=n, hypothesis_family=h, standardized_detectable_effect=detectable_effect(n,h),
                assumptions='Two-sided paired t, normal independent differences, alpha .05, power .80, Bonferroni; not measured study power.')
                for n in [8, 16, 32] for h in [1, 3, 12]])
        report = dict(status='PUBLIC_REPLAY_VERIFIED', contract_sha256=v.r.digest(spec),
            observations=len(rows), valid_graphs=valid, invalid_cells=len(rows)-valid, requests=len(ids),
            empty_graphs=sum(row['edges'] == 0 for row in rows if row['valid_graph']),
            conservative_usd=charge, api_seconds=sum(row['api_seconds'] for row in rows),
            observed_elapsed_hours=(max(ends)-min(starts)).total_seconds()/3600,
            summary_sha256=v.r.digest(summary), main_authorized=False, human_bilingual_validation=False,
            forecasts=forecast, billing_verified=False,
            limitations='Public replay is not provider invoice reconciliation. Calibration is exploratory; missing graphs are not imputed.')
        v.r.write_json(target/'inspection.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['wording_probe', 'calibration'])
    print(json.dumps(inspect(parser.parse_args().stage), indent=2))
