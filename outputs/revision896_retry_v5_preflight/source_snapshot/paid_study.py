"""Pilot-first OpenAI runner with durable dollar reservations.

The ledger counts unresolved requests at their full reservation. A timeout is
not proof that the provider did no work. Restarting must never erase that cost.
Only this runner may make paid requests; the old unbounded API path stays shut.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sqlite3
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
from openai import OpenAI

import constants_and_utils as shared
import generate_networks as generation
from analyze_networks import compute_network_metrics, compute_coleman_homophily, compute_age_assortativity

ROOT = Path(__file__).resolve().parent
MODELS = ['gpt-6-luna', 'gpt-4.1-mini']
METHODS = ['global', 'local', 'sequential', 'iterative']
LANGUAGES = ['english', 'hindi', 'japanese', 'portuguese']
RATES = {'gpt-6-luna': (.10, .50), 'gpt-4.1-mini': (.40, 1.60), 'gpt-4.1': (2., 8.), 'gpt-6-sol': (2., 10.), 'gpt-5.6-luna': (.20, 1.20)}
DEMOS = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """SQLite makes the check and reservation indivisible across processes."""
    def __init__(self, path, total_cap=50, pilot_cap=5, require_existing=False, allow_higher_cap=False):
        if (any(type(value) not in {int, float} or not math.isfinite(value) for value in [total_cap, pilot_cap])
                or not 0 < pilot_cap <= total_cap or (not allow_higher_cap and total_cap > 50)):
            raise ValueError('Require finite positive pilot/total caps; legacy runs retain the $50 maximum.')
        if require_existing:
            # mode=rw refuses a missing database atomically; an exists() check
            # followed by a normal connection could still recreate a lost ledger.
            self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=rw', uri=True, timeout=30)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=30)
        self.total_cap, self.pilot_cap = total_cap, pilot_cap
        try:
            if require_existing:
                self.db.execute('SELECT id,phase,cost,status,request,response FROM attempts LIMIT 1')
            else:
                self.db.execute('CREATE TABLE IF NOT EXISTS attempts '
                                '(id TEXT PRIMARY KEY, phase TEXT, cost REAL, status TEXT, request TEXT, response TEXT)')
                self.db.commit()
        except Exception:
            self.db.close()
            raise

    def spent(self, phase=None):
        query = 'SELECT COALESCE(SUM(cost),0) FROM attempts'
        args = ()
        if phase:
            query += ' WHERE phase=?'
            args = (phase,)
        return self.db.execute(query, args).fetchone()[0]

    def unresolved_count(self):
        return self.db.execute("SELECT COUNT(*) FROM attempts WHERE status NOT IN ('received','abandoned')").fetchone()[0]

    def abandon(self, request_id, reason):
        """Explicitly retire an uncertain request at FULL reserved cost, not a refund.

        This is conservative accounting authorized by the operator, never a
        provider usage receipt. The original request remains permanently cached
        as non-replayable. No generation path calls this automatically.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Abandonment requires an explicit authorization reason.')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT status,cost,response FROM attempts WHERE id=?', (request_id,)).fetchone()
            if row is None or row[0] not in {'reserved', 'uncertain'}:
                raise ValueError('Only an unresolved request can be abandoned.')
            record = dict(accounting='full_reservation_retained_no_replay', billing_verified=False,
                          retained_reservation_usd=row[1], previous_status=row[0],
                          previous_response=json.loads(row[2]) if row[2] else None,
                          authorization=reason, date_utc=datetime.now(timezone.utc).isoformat())
            self.db.execute('UPDATE attempts SET status=?,response=? WHERE id=?',
                            ('abandoned', json.dumps(record), request_id))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def reserve(self, request_id, phase, amount, request):
        if phase not in {'pilot', 'main'} or not math.isfinite(amount) or amount <= 0:
            raise ValueError('Invalid phase or reservation.')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            if self.spent() + amount > self.total_cap or (
                    phase == 'pilot' and self.spent('pilot') + amount > self.pilot_cap):
                raise BudgetExceeded('Next request would exceed the configured spending cap.')
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,NULL)',
                            (request_id, phase, amount, 'reserved', request))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def cached(self, request_id):
        row = self.db.execute('SELECT status,response FROM attempts WHERE id=?', (request_id,)).fetchone()
        if row is None:
            return None
        if row[0] == 'abandoned':
            raise RuntimeError('Abandoned paid attempt must never be replayed; full reservation retained.')
        if row[0] != 'received':
            raise RuntimeError('Unresolved paid attempt: reservation retained; reconcile provider usage before continuing.')
        return json.loads(row[1])

    def settle(self, request_id, amount, response):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('Invalid measured charge.')
        try:
            # Lock before reading: a concurrent abandonment must not be
            # overwritten by a settlement that checked an older reserved row.
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT cost,status FROM attempts WHERE id=?', (request_id,)).fetchone()
            if row is None or row[1] != 'reserved' or amount > row[0]:
                raise ValueError('Usage cannot exceed its reservation or settle an unresolved failure.')
            self.db.execute('UPDATE attempts SET cost=?,status=?,response=? WHERE id=?',
                            (amount, 'received', json.dumps(response, ensure_ascii=False), request_id))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def fail(self, request_id, error_type):
        # Store the exception type only: SDK error messages can contain headers.
        with self.db:
            self.db.execute('UPDATE attempts SET status=?,response=? WHERE id=? AND status=?',
                            ('uncertain', json.dumps({'error_type': error_type}), request_id, 'reserved'))

    def record_parse(self, request_id, error=None):
        record = self.cached(request_id)
        if record is None:
            raise RuntimeError('Cannot validate a response missing from the paid ledger.')
        record['parse'] = {'valid': error is None, 'error': str(error)[:200] if error else None,
                           'duplicate_edges': getattr(error, 'duplicate_edges', [])}
        with self.db:
            self.db.execute('UPDATE attempts SET response=? WHERE id=?',
                            (json.dumps(record, ensure_ascii=False), request_id))


def build_cells(phase):
    if phase not in {'pilot', 'main', 'sol-pilot', 'luna56-pilot'}:
        raise ValueError('Unknown phase.')
    protocol = json.loads((ROOT / 'study_protocol.json').read_text(encoding='utf-8'))
    if protocol['models'] != MODELS or protocol['methods'] != METHODS or protocol['languages'] != LANGUAGES:
        raise ValueError('Runner settings differ from the versioned protocol.')
    fixed = {'expected_personas': 50, 'pilot_seed': 1000, 'iterative_rounds': 3,
             'max_attempts_per_request': 3, 'confirmatory_rosters': 5,
             'confirmatory_repetitions': 2, 'total_budget_usd': 50, 'pilot_budget_usd': 5}
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError('Protocol parameters differ from the approved budgeted runner.')
    cells = []
    if phase == 'luna56-pilot':
        # Add four new runs without rewriting earlier receipts or their budget.
        settings = [('gpt-5.6-luna', 'us', 'english', 0, 0)]
    elif phase == 'sol-pilot':
        # This requested model extension shares the ORIGINAL $5 pilot ledger.
        # It does not enlarge or silently change the proposed main experiment.
        if protocol.get('sol_pilot') != {'model': 'gpt-6-sol', 'networks': 4, 'culture': 'us', 'language': 'english'}:
            raise ValueError('Sol pilot differs from the explicitly scoped protocol extension.')
        settings = [('gpt-6-sol', 'us', 'english', 0, 0)]
    elif phase == 'pilot':
        settings = [(m, 'us', 'english', 0, 0) for m in MODELS]
        settings += [(MODELS[0], 'us', lang, 0, 0) for lang in LANGUAGES[1:]]
    else:
        conditions = protocol['main_conditions']
        settings = [(m, c, lang, roster, rep) for m, (c, lang), roster, rep
                    in itertools.product(MODELS, conditions, range(5), range(2))]
    for model, culture, language, roster, rep in settings:
        for method in METHODS:
            run_id = f'revision-budget-v1_{phase}_{method}_{model}_{culture}_{language}_r{roster}_s{rep}'
            cells.append(dict(run_id=run_id, phase='main' if phase == 'main' else 'pilot', model=model, method=method, culture=culture,
                              language=language, roster=roster, repetition=rep, personas=50,
                              seed=1000 + roster * 100 + rep))
    return cells


class PaidCaller:
    def __init__(self, client, budget, cell):
        self.client, self.budget, self.cell, self.ordinal = client, budget, cell, 0
        self.parse_method = generation.update_graph_from_response
        self.last_request_id = None
        self.request_ids = []

    def parse_response(self, **kwargs):
        # Parsing never refunds a paid reply. The strict graph parser remains
        # unchanged; this wrapper records its decision for failure-rate audits.
        try:
            result = self.parse_method(**kwargs)
        except (ValueError, AssertionError) as error:
            self.budget.record_parse(self.last_request_id, error)
            raise
        self.budget.record_parse(self.last_request_id)
        return result

    def __call__(self, model, messages, savename=None, temp=None, verbose=False):
        if model != self.cell['model'] or model not in RATES:
            raise ValueError('Requested model differs from the approved cell or price table.')
        ordinal = self.ordinal
        self.ordinal += 1
        # Byte count is a conservative upper bound for byte-level text tokens.
        # Overhead allowance includes role framing and provider-added tokens.
        input_bound = sum(len(m['content'].encode('utf-8')) + 256 for m in messages) + 4096
        if input_bound > 100000:
            raise ValueError('Prompt exceeds the approved short-context limit.')
        # A local answer may list all 49 peers; 128 tokens can truncate valid IDs.
        max_output = 8192 if self.cell['method'] == 'global' else 512
        settings = {'max_completion_tokens': max_output}
        if model in {'gpt-6-luna', 'gpt-6-sol', 'gpt-5.6-luna'}:
            settings['reasoning_effort'] = 'none'
        else:
            settings['temperature'] = .8 if temp is None else temp
        request = dict(cell=self.cell, ordinal=ordinal, model=model, messages=messages, settings=settings,
                       date_utc=datetime.now(timezone.utc).isoformat(), rates_per_million=RATES[model])
        identity = json.dumps([self.cell['run_id'], ordinal, model, messages, settings], sort_keys=True)
        if 'frozen_source_sha256' in self.cell:
            # Fresh-study cache entries are bound to the whole frozen contract.
            # Keep earlier engineering cache identities backward compatible.
            identity = json.dumps([self.cell, ordinal, model, messages, settings], sort_keys=True)
        request_id = hashlib.sha256(identity.encode()).hexdigest()
        self.last_request_id = request_id
        self.request_ids.append(request_id)
        cached = self.budget.cached(request_id)
        if cached:
            if cached['finish_reason'] != 'stop':
                raise ValueError('Saved response did not finish normally; paid usage retained.')
            return cached['text']
        input_rate, output_rate = RATES[model]
        # 1.25 input multiplier also covers a provider cache-write premium.
        reservation = (input_bound * input_rate * 1.25 + max_output * output_rate) / 1e6
        self.budget.reserve(request_id, self.cell['phase'], reservation, json.dumps(request, ensure_ascii=False))
        try:
            response = self.client.chat.completions.create(model=model, messages=messages, extra_body=settings)
            if response.usage is None or not response.choices:
                raise RuntimeError('Provider returned no usage or choices.')
            usage = response.usage.model_dump()
            charge = (usage['prompt_tokens'] * input_rate * 1.25 + usage['completion_tokens'] * output_rate) / 1e6
            record = dict(text=response.choices[0].message.content or '', usage=usage,
                          response_id=response.id, resolved_model=response.model,
                          fingerprint=getattr(response, 'system_fingerprint', None),
                          finish_reason=response.choices[0].finish_reason,
                          date_utc=datetime.now(timezone.utc).isoformat(),
                          conservative_charge_usd=charge)
            self.budget.settle(request_id, charge, record)
        except Exception as exc:
            self.budget.fail(request_id, type(exc).__name__)
            # The retry helper catches transport errors. A timeout may already
            # have been billed, so stop rather than pay for an uncertain resend.
            raise RuntimeError('Paid attempt unresolved; reservation retained. Reconcile usage before continuing.') from exc
        # Usage is durably charged even when the model refuses or gets cut off.
        if record['finish_reason'] != 'stop':
            raise ValueError('Response did not finish normally; paid usage recorded.')
        return record['text']


def credential():
    key = os.getenv('OPENAI_API_KEY')
    if not key and os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as environment:
                key = winreg.QueryValueEx(environment, 'OPENAI_API_KEY')[0]
        except FileNotFoundError:
            pass
    if not key:
        raise RuntimeError('Set OPENAI_API_KEY in the process or Windows user environment.')
    return key


def clean_numbers(value):
    if isinstance(value, dict):
        return {k: clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_numbers(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def verify_completed(path, cell):
    """A marker alone cannot prove that its graph and figure still exist."""
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('cell') != cell or record.get('status') != 'ENGINEERING_PILOT_NOT_CONFIRMATORY':
        raise ValueError('Saved run identity or status does not match this protocol.')
    source_keys = ['generation_source_sha256', 'retry_source_sha256']
    hashes = [record.get(key) for key in source_keys]
    if any(value is not None and (not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)) for value in hashes):
        raise ValueError('Malformed generation or retry source hash.')
    if any(hashes) and not all(hashes):
        raise ValueError('Source hash pair is incomplete; cannot label it a legacy variant.')
    # Engineering-only resume preserves older evidence, not a frozen protocol.
    # A future confirmatory runner must require exact protocol/source identity.
    record['verified_source_variant'] = ':'.join(hashes) if all(hashes) else 'legacy_pre_source_hash'
    for suffix in ('.adj', '.png'):
        artifact = path.with_suffix(suffix)
        if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != record.get('artifact_sha256', {}).get(suffix):
            raise ValueError('Completed-run artifact missing or changed; resume stopped before spending.')
    graph = nx.read_adjlist(path.with_suffix('.adj'))
    roster_path = ROOT / 'text-files/us_50_gpt4o_w_interests.json'
    personas = json.loads(roster_path.read_text(encoding='utf-8'))
    if record.get('roster_sha256') != hashlib.sha256(roster_path.read_bytes()).hexdigest():
        raise ValueError('Saved run used a different roster; resume stopped.')
    if set(graph) != set(personas) or len(graph) != 50 or nx.number_of_selfloops(graph) or not graph.number_of_edges():
        raise ValueError('Saved graph no longer satisfies the pilot contract.')
    replay = nx.empty_graph(list(personas))
    for event in record['events']:
        for a, b in event['removed']:
            if not replay.has_edge(a, b):
                raise ValueError('Event history removes an absent edge.')
            replay.remove_edge(a, b)
        for a, b in event['added']:
            if a == b or a not in personas or b not in personas or replay.has_edge(a, b):
                raise ValueError('Event history contains an invalid addition.')
            replay.add_edge(a, b)
    canonical = lambda g: {tuple(sorted(edge)) for edge in g.edges()}
    if canonical(replay) != canonical(graph):
        raise ValueError('Recorded decisions do not reproduce the saved graph.')
    recomputed = compute_network_metrics(graph)
    record['verified_metrics'] = clean_numbers(recomputed)
    # Stored scores are historical observations, not an analysis authority.
    # Recompute mixing too, while keeping the original JSON evidence untouched.
    scores = {demo: compute_coleman_homophily(graph, personas, demo)[0] for demo in DEMOS if demo != 'age'}
    scores['age'] = compute_age_assortativity(graph, personas)
    record['verified_homophily'] = clean_numbers(scores)
    for metric in ['density', 'avg_clustering_coef', 'prop_nodes_lcc']:
        if not math.isclose(record['metrics'][metric], recomputed[metric], abs_tol=1e-12):
            raise ValueError('Saved measurements do not match the graph.')
    from PIL import Image
    with Image.open(path.with_suffix('.png')) as figure:
        figure.verify()
    return record


def verify_pilot():
    """Recheck saved evidence and costs offline, without opening an API client."""
    destination = ROOT / 'outputs/revision_budget_v1'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = build_cells('pilot') + build_cells('sol-pilot') + build_cells('luna56-pilot')
    for cell in cells:
        path = destination / (cell['run_id'] + '.json')
        row = dict(cell, completed=path.exists(), verified=False)
        if path.exists():
            record = verify_completed(path, cell)
            row.update(verified=True, nodes=50, edges=nx.read_adjlist(path.with_suffix('.adj')).number_of_edges(),
                       prompt_variant=record['verified_source_variant'],
                       **{key: record['verified_metrics'][key] for key in ['density', 'avg_clustering_coef', 'prop_nodes_lcc']})
        rows.append(row)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (destination / 'verification_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ledger = destination / 'budget.sqlite'
    usage_rows = []
    if ledger.exists():
        with closing(sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            for request_id, phase, cost, status, request, response in db.execute('SELECT * FROM attempts ORDER BY rowid'):
                request, response = json.loads(request), json.loads(response) if response else {}
                usage = response.get('usage', {})
                usage_rows.append(dict(request_id=request_id, run_id=request['cell']['run_id'], phase=phase,
                    model=request['model'], resolved_model=response.get('resolved_model'), ordinal=request['ordinal'],
                    status=status, prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                    conservative_charge_or_reservation_usd=cost, finish_reason=response.get('finish_reason'),
                    parse_valid=response.get('parse', {}).get('valid'), parse_error=response.get('parse', {}).get('error')))
    if usage_rows:
        with (destination / 'usage_summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(usage_rows[0]))
            writer.writeheader()
            writer.writerows(usage_rows)
    model_totals = {model: dict(
        verified_graphs=sum(row['verified'] for row in rows if row['model'] == model),
        attempts=sum(row['model'] == model for row in usage_rows),
        conservative_charge_or_reservation_usd=sum(row['conservative_charge_or_reservation_usd']
            for row in usage_rows if row['model'] == model)) for model in sorted({cell['model'] for cell in cells})}
    summary = dict(status='ENGINEERING_PILOT_NOT_CONFIRMATORY', target=len(cells), by_model=model_totals,
        original_target=20, sol_extension_target=4, luna56_extension_target=4,
        verified=sum(row['verified'] for row in rows), attempts=len(usage_rows),
        conservative_charge_or_reservation_usd=sum(row['conservative_charge_or_reservation_usd'] for row in usage_rows),
        unresolved_attempts=sum(row['status'] != 'received' for row in usage_rows),
        observed_parse_failures=sum(row['parse_valid'] is False for row in usage_rows),
        missing_parse_annotations=sum(row['parse_valid'] is None for row in usage_rows),
        limitations=['Earlier engineering responses lack parse annotations and source-version hashes.',
                     'Initial/retry prompts changed after observed failures; pilot variants are not confirmatory replicates.',
                     'Historical roster includes nine minors; translations await bilingual review.',
                     'These checks do not establish real-world validity, statistical power or exact invoiced cost.'])
    (destination / 'pilot_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def run(args):
    cells = build_cells(args.phase)
    if not args.execute:
        print(json.dumps({'status': 'OFFLINE_PLAN', 'phase': args.phase, 'networks': len(cells),
                          'total_cap_usd': 50, 'pilot_cap_usd': 5, 'cells': cells}, indent=2))
        return
    if args.phase == 'main':
        raise RuntimeError('Main collection awaits Astra protocol review, five validated rosters and pilot-informed sample size.')
    # Capture the batch's source version before requests begin. Reading it only
    # after a long run could record an editor change made during generation.
    source_hashes = {key: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() for key, filename in [
        ('generation_source_sha256', 'generate_networks.py'),
        ('retry_source_sha256', 'constants_and_utils.py'),
        ('analysis_source_sha256', 'analyze_networks.py')]}
    client = OpenAI(api_key=credential(), max_retries=0, timeout=60)
    # These read-only lookups establish account access before a paid reservation.
    # A model name in public documentation does not prove this key can use it.
    for model in sorted({cell['model'] for cell in cells}):
        client.models.retrieve(model)
    print('Model access preflight passed for this batch.')
    destination = ROOT / 'outputs/revision_budget_v1'
    destination.mkdir(parents=True, exist_ok=True)
    budget = Budget(destination / 'budget.sqlite')
    persona_path = ROOT / 'text-files/us_50_gpt4o_w_interests.json'
    personas = json.loads(persona_path.read_text(encoding='utf-8'))
    if len(personas) != 50:
        raise ValueError('Pilot requires exactly fifty personas.')
    completed = 0
    for cell in cells:
        result_path = destination / (cell['run_id'] + '.json')
        if result_path.exists():
            saved = verify_completed(result_path, cell)
            print('Reusing engineering-only source variant: ' + saved['verified_source_variant'][:72])
            completed += 1
            continue
        random.seed(cell['seed'])
        np.random.seed(cell['seed'])
        events = []
        caller = PaidCaller(client, budget, cell)
        # The existing retry parser remains the authority on eligible choices.
        with patch.object(shared, 'get_llm_response', caller), \
                patch.object(generation, 'update_graph_from_response', caller.parse_response):
            graph, *_ = generation.generate_network(cell['method'], DEMOS, personas, list(personas),
                cell['model'], mean_choices=5, num_iter=3, culture_context=cell['culture'],
                prompt_language=cell['language'], events=events)
        if set(graph) != set(personas) or nx.number_of_selfloops(graph) or not graph.number_of_edges():
            raise ValueError('Generated graph failed roster, self-link or nonempty-edge checks.')
        metrics = compute_network_metrics(graph)
        for metric in ['density', 'avg_clustering_coef', 'prop_nodes_lcc']:
            if not math.isfinite(metrics[metric]):
                raise ValueError('Generated graph has an undefined primary topology metric.')
        homophily = {demo: compute_coleman_homophily(graph, personas, demo)[0] for demo in DEMOS if demo != 'age'}
        homophily['age'] = compute_age_assortativity(graph, personas)
        result = dict(cell=cell, status='ENGINEERING_PILOT_NOT_CONFIRMATORY', events=events,
                      **source_hashes,
                      roster_sha256=hashlib.sha256(persona_path.read_bytes()).hexdigest(),
                      metrics=clean_numbers(metrics), homophily=clean_numbers(homophily),
                      cumulative_conservative_charge_usd=budget.spent())
        nx.write_adjlist(graph, destination / (cell['run_id'] + '.adj'))
        import matplotlib.pyplot as plt
        nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=0), node_size=55, font_size=6, width=.4)
        plt.axis('off')
        plt.savefig(destination / (cell['run_id'] + '.png'), dpi=180, bbox_inches='tight')
        plt.close()
        result['artifact_sha256'] = {suffix: hashlib.sha256(result_path.with_suffix(suffix).read_bytes()).hexdigest()
                                    for suffix in ('.adj', '.png')}
        temporary = result_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        temporary.replace(result_path)
        verify_completed(result_path, cell)
        completed += 1
        print(f'{completed}/{len(cells)} pilot networks; conservative charged/reserved ${budget.spent():.4f}')
    print('Pilot complete. ASTRA REVIEW POINT: inspect actual usage, failures, translations and final study design before main collection.')
    budget.db.close()
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'sol-pilot', 'luna56-pilot', 'main'], default='pilot')
    parser.add_argument('--execute', action='store_true', help='Make paid pilot calls inside the $5/$50 limits.')
    parser.add_argument('--verify', action='store_true', help='Recheck completed pilot artifacts and summarize usage offline.')
    arguments = parser.parse_args()
    try:
        if arguments.verify:
            if arguments.execute or arguments.phase != 'pilot':
                raise ValueError('--verify is offline and supports only the pilot phase.')
            verify_pilot()
        else:
            run(arguments)
    except (BudgetExceeded, RuntimeError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    except Exception as error:
        parser.exit(1, f'{type(error).__name__}: stopped; consult the local ledger. Unresolved charges remain reserved.\n')
