"""Canonical offline V8 audit. Never executes a model or edits historical evidence.

Run with: python -X utf8 -B precollection_audit.py
Only outputs/precollection_review is written. The frozen inspector runs in a
temporary copy; its old output-writing behavior cannot replace the originals.
"""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/precollection_review'
PIN_FILE = ROOT/'precollection_pins.json'
CALIBRATION = Path('outputs/repair_v7/calibration/176533db7041')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


@contextlib.contextmanager
def workflow_lock(output):
    """Hold an OS lock: a crashed process must not leave a permanent busy sentinel."""
    with (output/'workflow.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError('An offline audit already holds the output lock.') from error
        try:
            yield
        finally:
            handle.seek(0)
            release()


def write_json(path, value):
    """Publish a complete report or leave the previous pending state, never half JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    handle, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf8', newline='\n') as stream:
            stream.write(data+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_files(root, inventory):
    root = Path(root).resolve()
    for name, expected in inventory.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError('Pinned file changed or missing: '+name)


def public_artifacts(root):
    """Exclude private journals, never public orphan receipts, from this public audit."""
    result = set()
    for directory, names, files in os.walk(root/CALIBRATION):
        names[:] = [name for name in names if name not in {'requests', 'transport', '__pycache__'}]
        for name in files:
            if name not in {'authorization.json', 'review.json', 'upstream_review.json',
                            'private_snapshot.json', 'workflow.lock'}:
                result.add((Path(directory)/name).relative_to(root).as_posix())
    return result


def check_pins():
    pins = read(PIN_FILE)
    if pins.get('version') != 'post-review-pins-1' or pins.get('main_authorized') is not False:
        raise ValueError('Invalid independent review manifest.')
    required = {p.name for p in ROOT.glob('*.py')} | {p.name for p in ROOT.glob('study_protocol*.json')}
    required |= {'precollection_spec.json', 'requirements.lock.txt', '.python-version'}
    if set(pins['sources']) != required:
        raise ValueError('Pinned source inventory is incomplete or unexpected.')
    verify_files(ROOT, pins['sources'])
    verify_files(ROOT, pins['artifacts'])
    # Extra receipts must not quietly become part of the experiment.
    current = public_artifacts(ROOT)
    if current != set(pins['artifacts']):
        raise ValueError('Historical artifact inventory changed.')
    return pins


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse an empty report table.')
    with Path(path).open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_denominators(path, rows):
    """Show which denominators can support analysis, not a success-only plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    selected = [r for r in rows if r['endpoint'] == 'modularity']
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, row in zip(axes, selected):
        counts = [row[k] for k in ['attempted', 'valid', 'nonempty', 'endpoint_defined']]
        bars = ax.bar(range(4), counts, color=['#8e969b', '#254e5f', '#497c83', '#88a7a5'], width=.68)
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_title(row['method'].capitalize(), loc='left', fontsize=12)
        ax.set_xticks(range(4), ['Attempted', 'Valid', 'Nonempty', 'Modularity\ndefined'], rotation=45, ha='right')
        ax.set_ylim(0, 30)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel('Calibration observations')
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={'Description': 'Offline V8 audit; counts from saved observations, not new generation'})
    plt.close(fig)


def replay_copy(destination):
    # Import only AFTER independent hashes pass, not after trusting self-hashes.
    import repair_v8_transport_second as recovery
    import inspect_repair_v8 as inspector
    v = recovery.v
    copied = destination/CALIBRATION
    for name in public_artifacts(ROOT):
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    with patch.object(v.r, 'DESTINATION', destination/'outputs/repair_v7'), \
            patch.object(v.r.base.frozen, 'workflow_lock', return_value=contextlib.nullcontext()), \
            patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('Offline audit: no API client')), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline audit: no network')), \
            recovery.recovered() as records:
        report = inspector.inspect('calibration')
        report['retained_uncertain_reservation_usd'] = sum(r['retired_row'][2] for r in records)
        report['total_charge_or_reservation_usd'] = report['conservative_usd']+report['retained_uncertain_reservation_usd']
    return report, copied


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output/'report.json'
    state_path = output/'audit_state.json'
    with workflow_lock(output):
        attempt = uuid.uuid4().hex
        try:
            # Separate state also invalidates a report held open by a Windows reader.
            write_json(state_path, dict(status='VALIDATION_PENDING', attempt=attempt))
            report_path.unlink(missing_ok=True)
            write_json(report_path, dict(status='VALIDATION_PENDING', main_authorized=False))
            pins = check_pins()
            import precollection_analysis as analysis
            spec = analysis.load_spec()
            with tempfile.TemporaryDirectory(prefix='ua-precollection-') as directory:
                public, copied = replay_copy(Path(directory))
                # These new CSVs are copied from the freshly replayed shadow, not old summaries.
                for name in ['inspection_networks.csv', 'condition_summary.csv']:
                    shutil.copyfile(copied/name, output/name)
                rows = []
                for kind in ['runs', 'failures']:
                    for path in sorted((copied/kind).glob('*.json')):
                        record = read(path)
                        rows.append(dict(**record['cell'], valid_graph=kind == 'runs',
                                         **record.get('metrics', {})))
                # Edge count is a graph property, not the length of its event history.
                import networkx as nx
                for row in rows:
                    row['edges'] = nx.read_adjlist(copied/'runs'/(row['run_id']+'.adj')).number_of_edges() if row['valid_graph'] else None
                metrics = ['density', 'avg_clustering_coef', 'modularity', 'coleman_gender',
                           'coleman_religion', 'coleman_political orientation', 'age_assortativity']
                counts = analysis.denominators(rows, metrics)
                write_csv(output/'endpoint_denominators.csv', counts)
                condition_counts = []
                keys = ['model', 'method', 'culture', 'language']
                for identity in sorted({tuple(r[k] for k in keys) for r in rows}):
                    group = [r for r in rows if tuple(r[k] for k in keys) == identity]
                    for count in analysis.denominators(group, metrics):
                        if count['attempted']:
                            condition_counts.append({**dict(zip(keys, identity)), **count})
                write_csv(output/'condition_endpoint_denominators.csv', condition_counts)
                plot_denominators(output/'calibration_denominators.png', counts)
                write_csv(output/'precision_scenarios.csv', analysis.precision_scenarios(spec))
                write_json(output/'planned_analysis_cells.json', analysis.design(spec))
                from precollection_candidate import candidate_catalog, canonical_global
                write_json(output/'prompt_candidate.json', candidate_catalog())
                parser_results = []
                for path in sorted([*(copied/'failures').glob('*.json'), *(copied/'runs').glob('*.json')]):
                    record = read(path)
                    if record['cell']['method'] != 'global':
                        continue
                    raw = record['requests'][-1]['response']['text']
                    try:
                        parsed = canonical_global(raw, [str(n) for n in range(50)])
                        outcome = dict(candidate_valid=True, candidate_edges=len(parsed['edges']),
                                       repeated_pairs=parsed['repeated_pairs'], reason=None)
                    except ValueError as error:
                        outcome = dict(candidate_valid=False, candidate_edges=None,
                                       repeated_pairs=None, reason=str(error))
                    historical_valid = path.parent.name == 'runs'
                    if historical_valid:
                        original_edges = {frozenset(pair) for pair in nx.read_adjlist(path.with_suffix('.adj')).edges}
                        if not outcome['candidate_valid'] or original_edges != {frozenset(pair) for pair in parsed['edges']}:
                            raise ValueError('Candidate changes a previously valid global graph.')
                    parser_results.append(dict(run_id=record['cell']['run_id'],
                        historical_status='RETAINED_VALID_UNCHANGED' if historical_valid else 'RETAINED_INVALID_UNCHANGED',
                        evidence_type='OFFLINE_REINTERPRETATION_NOT_NEW_OBSERVATION', **outcome))
                write_csv(output/'parser_candidate_diagnostic.csv', parser_results)
                old_forecasts = public['forecasts']
                base = next(r for r in old_forecasts if r['scenario'] == 'global_8_others_8')
                forecasts = []
                for blocks in [8, 16, 32]:
                    factor = blocks/8
                    forecasts.append(dict(schedule_blocks=blocks, attempts=112*blocks,
                        usage_usd=base['conservative_usd']*factor,
                        usage_with_25_percent_allowance=base['cost_with_25_percent_allowance']*factor,
                        serial_api_hours=base['serial_api_hours']*factor,
                        api_hours_with_50_percent_allowance=base['api_hours_with_50_percent_allowance']*factor,
                        main_authorized=False,
                        assumptions='V8 reference-prompt rates/usage; historical prices, country transfer and failures; not a current quote.'))
                write_csv(output/'allocation_scenarios.csv', forecasts)
            # A concurrent manual edit during replay must invalidate this report too.
            if check_pins() != pins:
                raise ValueError('Approval manifest changed during verification.')
            result = dict(status='OFFLINE_AUDIT_PASSED_NOT_MAIN_READY', main_authorized=False,
                attempt=attempt, paid_calls=0, historical_files_unchanged=len(pins['artifacts']),
                sources_checked=len(pins['sources']), pins_sha256=sha(PIN_FILE),
                public_replay=public, endpoint_denominators=counts,
                analysis_candidate_sha256=sha(ROOT/'precollection_spec.json'),
                planned_candidate_attempts=len(analysis.design(spec)),
                output_sha256={p.name: sha(p) for p in sorted(output.iterdir())
                               if p.is_file() and p.suffix in {'.csv', '.json', '.png'} and p not in {report_path, state_path}},
                remaining=['Approve narrowed named-contract claim and analysis candidate.',
                    'Version and live-check adopted prompt/parser changes; never rewrite calibration.',
                    'Review the final execution manifest and transport policy, verify current prices and fresh credential.',
                    'Authorize an explicit main run; this offline command cannot do so.'])
            write_json(report_path, result)
            write_json(state_path, dict(status='COMPLETE', attempt=attempt, report_sha256=sha(report_path)))
            return result
        except Exception as error:
            failed = dict(status='FAILED', error_type=type(error).__name__, attempt=attempt, main_authorized=False)
            for path in [state_path, report_path]:
                try:
                    write_json(path, failed)
                except OSError:
                    # A read-only/full filesystem cannot promise a new durable status.
                    # The failed command is authoritative; never continue verification.
                    if path == report_path:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
            raise


def verified_report(output=OUTPUT):
    """Consumer gate: report status alone is not evidence of a completed audit."""
    output = Path(output)
    state, report = read(output/'audit_state.json'), read(output/'report.json')
    if (state.get('status') != 'COMPLETE' or state.get('attempt') != report.get('attempt')
            or state.get('report_sha256') != sha(output/'report.json')
            or report.get('status') != 'OFFLINE_AUDIT_PASSED_NOT_MAIN_READY'):
        raise ValueError('Audit is failed, pending or stale; no current clearance.')
    check_pins()
    if report.get('pins_sha256') != sha(PIN_FILE):
        raise ValueError('Audit pin manifest changed.')
    verify_files(output, report['output_sha256'])
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ['status', 'paid_calls', 'historical_files_unchanged', 'planned_candidate_attempts']}, indent=2))
