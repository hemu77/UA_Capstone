"""Build presentation v2 without rerunning or editing a sealed analysis."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures.publication import OUTPUT, DRAWERS, protected_files, render, sha, verify_hashes


def main():
    output = ROOT / OUTPUT
    seal = output / 'figure_manifest.json'
    if seal.exists() and json.loads(seal.read_text()).get('status') == 'COMPLETE_PUBLICATION_FIGURES_V2':
        raise ValueError('Published revision is immutable; use a new version for later changes')
    protected = protected_files(ROOT)
    code = [ROOT / 'analysis_figures/publication.py', Path(__file__),
            ROOT / 'scripts/check_publication_notebooks.py', ROOT / 'docs/PUBLICATION_FIGURES_V2.md']
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in code}
    captions_text = (ROOT / 'docs/PUBLICATION_FIGURES_V2.md').read_text()
    captions = {name: captions_text.split('## ' + name + '\n', 1)[1].split('\n## ', 1)[0].strip()
                for name in DRAWERS}
    output.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps({'status': 'IN_PROGRESS'}), encoding='utf8')
    try:
        paths = render(ROOT, output)
        if len(paths) != 3 * len(DRAWERS):
            raise ValueError('Incomplete figure set')
        verify_hashes(protected)
        verify_hashes({ROOT / p: h for p, h in hashes.items()})
        result = dict(status='COMPLETE_PUBLICATION_FIGURES_V2',
            created_utc=datetime.now(timezone.utc).isoformat(), primary_attempts=896,
            primary_valid=890, supplemental_graphs=0, paid_calls=0, ledger_queries=0,
            code_sha256=hashes, protected_sha256={p.relative_to(ROOT).as_posix(): h for p, h in protected.items()},
            artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths}, captions=captions)
        seal.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
        print(f'PUBLICATION_V2_VERIFIED: {len(DRAWERS)} figures, {len(paths)} files; old evidence unchanged.')
    except Exception:
        seal.write_text(json.dumps({'status': 'FAILED_INCOMPLETE'}), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
