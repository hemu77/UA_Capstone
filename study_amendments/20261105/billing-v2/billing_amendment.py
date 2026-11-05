"""Execution-only exception. Every earlier row, journal, receipt and charge stays intact."""
import hashlib
import json
import re
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

from final_study import protocol as p

CONTRACT = '060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f'
ORIGINAL = '5cc90a3ffed4916eeb11189684bb797b022d15ad241e712db3f5b0b0db424eeb'
FAILED = '15fa933730603cf4276afb497c6da05369804ee83fe9922db16c59c52aaac110'
NAME = 'owner-billing-resume-20261102-v1'
NEW = hashlib.sha256((NAME + ':' + ORIGINAL + ':' + FAILED).encode('ascii')).hexdigest()
HASHES = {ORIGINAL: 'c46c1a21463124caa1f42439a3588da31ee2d5840f3ed0405d26e0944e5e85f5',
          FAILED: 'dba7e2d106558365e3d19eed6a74b565e41b643631db55b6097b8bc063f870c8'}
RUN = 'final_060dcd3a86d5_main_sequential_gpt-4.1_brazil_english_s44004'
RATES = {'gpt-4.1': [2., 8.], 'gpt-5.6-luna': [.2, 1.2],
         'gpt-6-luna': [.1, .5], 'gpt-6-sol': [2., 10.]}


def require(condition, msg):
    if not condition:
        raise ValueError('Billing amendment: ' + msg)


def prices(spec, today=None):
    require(p.legacy.digest(spec) == CONTRACT and spec['rates'] == RATES and
            spec['requested_service_tier'] == 'default', 'different contract/rates/tier')
    # The existing price AND shutdown checks run on separately dated evidence.
    p.check_prices({**spec, 'prices_verified_on': '2026-11-02', 'prices_valid_through': '2026-11-08'},
                   today=today or datetime.now(timezone.utc).date())


def pair(budget, target):
    result = {}
    for rid in (ORIGINAL, FAILED):
        row = p.engine.ledger_row(budget, rid)
        require(row and p.legacy.digest(row) == HASHES[rid], 'missing/changed rejected row')
        j = target/'requests'/(rid + '.json')
        raw = target/'transport'/(rid + '.json')
        require(not j.is_symlink() and p.read(j) == dict(request_id=rid, row_sha256=HASHES[rid])
                and not raw.exists() and not raw.is_symlink(), 'journal or raw receipt differs')
        req, resp = json.loads(row[4]), json.loads(row[5])
        if rid == ORIGINAL:
            resp = resp['previous_response']
            if isinstance(resp, str):
                resp = json.loads(resp)
        d = resp['diagnostics']
        require(row[1] == 'main' and row[2] == .022816 and
                row[3] == ('abandoned' if rid == ORIGINAL else 'uncertain') and
                req['cell']['run_id'] == RUN and req['ordinal'] == 44 and
                req['cell']['final_contract_sha256'] == CONTRACT and
                req.get('replaces_request_id') == (None if rid == ORIGINAL else ORIGINAL) and
                (d['http_status'], d['error_code'], d['stage'], d['client_request_id']) ==
                (429, 'credit_balance_exhausted', 'transport', rid), 'unexpected billing evidence')
        result[rid] = req
    keys = ('cell', 'ordinal', 'model', 'messages', 'settings')
    require(all(result[ORIGINAL][k] == result[FAILED][k] for k in keys), 'billing retries differ')
    return result[ORIGINAL]


def billing_code(code):
    return str(code).lower() in {'credit_balance_exhausted', 'insufficient_quota', 'billing_hard_limit_reached'}


@lru_cache(maxsize=1)
def expected_models():
    return p.read(Path(__file__).resolve().parent/'checkpoint/verified.json')['resolved_models']


def resolved(model, actual):
    require(isinstance(actual, str) and re.fullmatch(re.escape(model)+r'(?:-\d{4}-\d{2}-\d{2})?', actual)
            and expected_models().get(model) == actual, 'provider model differs from saved version; stop')


def bind_budget(budget, target):
    budget.billing_target = target
    original_fn = budget.replacement_id
    def replacement_id(base, request):
        resolved = original_fn(base, request)
        if base != ORIGINAL:
            return resolved
        prior = pair(budget, target)
        require(resolved == FAILED and all(request[k] == prior[k] for k in
                ('cell', 'ordinal', 'messages', 'model', 'settings')), 'replacement semantics changed')
        require(not (target/'transport'/(NEW+'.json')).exists()
                and not (target/'transport'/(NEW+'.json')).is_symlink()
                and not p.engine.ledger_row(budget, NEW), 'new attempt already exists; never resend')
        request['billing_amendment'] = NAME
        request['billing_failed_replacement'] = FAILED
        return NEW
    budget.replacement_id = replacement_id
    return budget


def replay_id(base, request, rid, ordinary):
    if rid == NEW or 'billing_amendment' in request or 'billing_failed_replacement' in request:
        require((base, rid, request.get('billing_amendment'), request.get('billing_failed_replacement'),
                 request.get('replaces_request_id'), request.get('client_request_id')) ==
                (ORIGINAL, NEW, NAME, FAILED, ORIGINAL, NEW), 'unknown amendment receipt')
        return NEW
    return ordinary


def check_new_row(row):
    if row[0] == NEW:
        require(row[3] == 'received', 'amended request already attempted but not received; human review required')

