"""Run the offline robustness checks into a NEW directory. No provider calls, no generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures.publication import protected_files, verify_hashes
from analysis_primary.core import sha, write_json
from analysis_robustness.core import run
from analysis_robustness.figures import render

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    destination = parser.parse_args().output.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / 'outputs/research_extension'):
        raise SystemExit('Use a NEW directory inside outputs/research_extension; no overwrites')
    protected = protected_files(ROOT)
    protocol = json.loads((ROOT / 'analysis_extension/protocol.json').read_text())
    code = sorted((ROOT / 'analysis_robustness').glob('*.py')) + [Path(__file__)]
    with TemporaryDirectory(dir=destination.parent) as temporary:
        folder = Path(temporary) / 'release'
        folder.mkdir()
        run(ROOT, folder, protocol)
        render(folder)
        verify_hashes(protected)
        write_json(folder / 'manifest.json', dict(
            status='COMPLETE_EXPLORATORY_ROBUSTNESS_CHECKS', created_utc=datetime.now(timezone.utc).isoformat(),
            python_version=platform.python_version(), primary_attempts=896, primary_valid=890, primary_empty=12,
            primary_failed=6, supplemental_graphs=0, paid_calls=0, ledger_queries=0, new_generation=0,
            preregistered=False, permutations_per_graph=500, snowball_draws_per_record=32,
            artifacts={p.name: sha(p) for p in sorted(folder.iterdir())},
            code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in code}))
        folder.rename(destination)
    print('ROBUSTNESS_VERIFIED=' + destination.relative_to(ROOT).as_posix())
