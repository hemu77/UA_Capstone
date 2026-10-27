#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --job-name=ua-validate
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
work=$(realpath -- "${1:?Pass dedicated workspace}")
unset OPENAI_API_KEY PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export HOME="$work/home" XDG_CACHE_HOME="$work/cache"
# Disposable fixtures use node-local storage; actual study evidence stays shared.
TMPDIR=$(mktemp -d /tmp/ua-validation-XXXXXXXX)
printf 'Allocated test directory: %s\n' "$TMPDIR"
TMPDIR=$(realpath -- "$TMPDIR")
export TMPDIR
printf 'Canonical test directory: %s\n' "$TMPDIR"
export MPLCONFIGDIR="$work/cache/matplotlib" MPLBACKEND=Agg PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$work/repo"
python="$work/env/bin/python"
printf '{"status":"VALIDATION_PENDING"}\n' > "$work/compute-validation.json"
"$python" -m pip check
"$python" -X utf8 -B - <<'PY'
import hashlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from final_study import protocol, runner

temporary = Path(tempfile.gettempdir())
assert temporary.is_absolute() and temporary.resolve() == temporary, 'Test root must not contain a filesystem alias'
spec = protocol.load_contract()
assert not list(protocol.OUTPUT.glob('*/main')), 'Main stage must be absent during initial deployment'
for line in Path('requirements.lock.txt').read_text().splitlines():
    if '==' in line and not line.startswith('#'):
        name, expected = line.split('==')
        assert version(name) == expected, name
approval = protocol.folder(protocol.evidence_spec(spec, 'hindi_check'), 'hindi_check')
def private_snapshot():
    paths = [approval/'authorization.json', approval/'budget.sqlite']
    paths += [p for name in ('requests', 'transport') for p in (approval/name).rglob('*') if p.is_file()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

before = private_snapshot()
with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline validation')), \
        patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('Offline validation')):
    suite = unittest.defaultTestLoader.discover('tests')
    expected = suite.countTestCases()
    if expected < 1:
        raise RuntimeError('No tests discovered')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != expected or result.skipped:
        raise SystemExit(1)
    preflight = runner.preflight()
    ready = runner.readiness(spec, require_credential=False)
    assert before == private_snapshot()
    report = dict(status='COMPUTE_VALIDATION_PASSED', job_id=os.environ['SLURM_JOB_ID'], tests_run=result.testsRun,
        skipped=len(result.skipped), preflight_cases=preflight['cases'],
        contract_sha256=protocol.legacy.digest(spec), main_attempts=896,
        hindi_graphs=ready['hindi_check']['valid_graphs'], private_evidence_unchanged=True,
        runtime=spec['runtime'], installed_packages={d.metadata['Name']: d.version for d in distributions()},
        paid_calls=0, credential_checked=False)
    protocol.write_json(Path('../compute-validation.json'), report)
    print(json.dumps(report, indent=2))
PY
