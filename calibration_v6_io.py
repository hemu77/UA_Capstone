"""Logged, I/O-only retry adapter. Frozen requests and API retry policy stay unchanged."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from unittest.mock import patch
from uuid import uuid4

import calibration_v6 as c


def write_checkpoint(path, value, journal, events, writer, sleep=time.sleep, persist=lambda: None):
    path = Path(path)
    if path.is_symlink() or journal.resolve() != journal.absolute():
        raise ValueError('Checkpoint links or redirected journal directories are not permitted.')
    if (path.parent.resolve() != journal.resolve() or path.suffix != '.json'
            or not re.fullmatch('[0-9a-f]{64}',path.stem) or value.get('request_id') != path.stem):
        return writer(path,value)
    delays = [.05, .1, .25, .5, 1.]
    for attempt in range(len(delays)+1):
        try:
            result = writer(path,value)
        except PermissionError as error:
            events.append(dict(request_id=path.stem, write_attempt=attempt+1,
                error_type='PermissionError', winerror=getattr(error,'winerror',None),
                outcome='exhausted' if attempt == len(delays) else 'retrying'))
            persist()  # An audit-write failure stops here; it never retries paid work.
            if attempt == len(delays):
                raise
            sleep(delays[attempt])
        else:
            if attempt:
                events[-1]['outcome'] = 'write_succeeded'
                persist()
            return result


def execute(limit=None):
    spec = c.contract()
    approved = c.review(spec)
    target = c.folder(spec)
    report_path = target/'io_adapter'/(uuid4().hex+'.json')
    events = []
    audit = dict(status='STARTED', started_at_utc=datetime.now(timezone.utc).isoformat(),
        adapter_source_sha256=c.sha(__file__), contract_sha256=c.frozen.digest(spec),
        additional_ceiling_usd=approved['additional_usd'], main_authorized=False,
        policy='At most five retries of an identical local journal write on PermissionError only; never retry an API call.',
        frozen_generation_sources_unchanged=True, write_retries=events)
    source_path = target/'io_adapter'/'sources'/(audit['adapter_source_sha256']+'.py')
    source_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with source_path.open('xb') as handle:
            handle.write(Path(__file__).read_bytes())
    except FileExistsError:
        pass
    if c.sha(source_path) != audit['adapter_source_sha256']:
        raise ValueError('I/O-adapter source snapshot differs from this execution.')
    writer = c.write_json
    writer(report_path,audit)
    try:
        with patch.object(c,'write_json',lambda path,value: write_checkpoint(
                path,value,target/'requests',events,writer,persist=lambda:writer(report_path,audit))):
            result = c.execute(limit)
        audit['status'] = 'COMPLETED_EXECUTION'
        return result
    except Exception as error:
        audit.update(status='STOPPED',error_type=type(error).__name__)
        raise
    finally:
        audit['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        audit['source_unchanged_during_execution'] = c.sha(__file__) == audit['adapter_source_sha256']
        writer(report_path,audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.limit),indent=2))
    except Exception as error:
        parser.exit(1,type(error).__name__+': stopped; inspect saved evidence; no automatic API retry.\n')
