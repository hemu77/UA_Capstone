"""Build new offline diagnostics atomically, without overwriting historical outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import sys

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_extension.core import build_mixing
from analysis_extension.sampling import build_sampling
from analysis_extension.figures import CAPTIONS, render
from analysis_figures.publication import protected_files, verify_hashes
from analysis_figures.review_v5 import verify as verify_v5
from analysis_primary.core import sha, write_json


def build(destination):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT/'outputs/research_extension') or destination.exists():
        raise ValueError('Use a NEW directory inside outputs/research_extension; no overwrites')
    protected = protected_files(ROOT)
    newest = verify_v5(ROOT)
    for version in ('20261105_v2', '20261105_v3', '20261111_v4', '20261112_v5'):
        for path in (ROOT/'outputs/publication_figures'/version).iterdir():
            if path.is_file():
                protected[path] = sha(path)
    protocol_file = ROOT/'analysis_extension/protocol.json'
    protocol = json.loads(protocol_file.read_text())
    code = [*sorted((ROOT/'analysis_extension').glob('*.py')), protocol_file,
            Path(__file__), ROOT/'scripts/fetch_copenhagen_public.py']
    pins = {p: sha(p) for p in code}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        folder = Path(temporary)/'release'
        folder.mkdir()
        write_json(folder/'execution_protocol.json', protocol)
        primary = build_mixing(ROOT, folder)
        print('RQ2_MIXING_COMPLETE', flush=True)
        source = build_sampling(ROOT, folder, primary, protocol)
        render(folder)
        verify_hashes(protected)
        verify_hashes(pins)
        artifacts = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        write_json(folder/'manifest.json', dict(status='COMPLETE_EXPLORATORY_EXTENSION_WITH_CAVEATS',
            created_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
            primary_attempts=896, primary_valid=890, primary_empty=12, primary_failed=6,
            supplemental_graphs=0, paid_calls=0, ledger_queries=0, new_primary_tests=0,
            copenhagen_scope=source['suitability'], artifacts=artifacts, captions=CAPTIONS,
            code_sha256={p.relative_to(ROOT).as_posix(): h for p, h in pins.items()},
            preserved_sha256={p.relative_to(ROOT).as_posix(): h for p, h in protected.items()}))
        folder.rename(destination)
    print('EXTENSION_VERIFIED='+destination.relative_to(ROOT).as_posix(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
