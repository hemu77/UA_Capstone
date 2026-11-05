"""Build a separately sealed manuscript figure set; preserve earlier evidence."""
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_primary.core import sha,write_json
from analysis_heldout.benchmark import validate_previous
from analysis_figures.manuscript import render_manuscript


def main():
    output=ROOT/'outputs/manuscript_figures/20261105_v1'
    output.mkdir(parents=True,exist_ok=True)
    seal=output/'figure_manifest.json'
    protected=validate_previous(ROOT)
    code=[*sorted((ROOT/'analysis_figures').glob('*.py')),Path(__file__),ROOT/'docs/MANUSCRIPT_FIGURE_CAPTIONS.md']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in code}
    write_json(seal,dict(status='IN_PROGRESS',supplemental_graphs=0,paid_calls=0))
    try:
        files=render_manuscript(ROOT,output)
        if len(files)!=9: raise ValueError('Require three PNG/SVG/PDF figure sets')
        if before!={p.relative_to(ROOT).as_posix():sha(p) for p in code}: raise ValueError('Figure code changed while rendering')
        for name,expected in protected.items():
            if sha(ROOT/name)!=expected: raise ValueError('Previous evidence changed: '+name)
        write_json(seal,dict(status='COMPLETE_MANUSCRIPT_FIGURES_WITH_CAVEATS',created_utc=datetime.now(timezone.utc).isoformat(),
            primary_attempts=896,valid_primary_graphs=890,supplemental_graphs=0,paid_calls=0,ledger_queries=0,
            code_sha256=before,previous_primary_seal_sha256=protected['outputs/primary_analysis/20261105/evaluation_manifest.json'],
            artifacts={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
            runtime=dict(python=platform.python_version(),matplotlib=importlib.metadata.version('matplotlib')),
            interpretation='New presentation only. Original data, tests, intervals, failures and earlier plots remain unchanged.'))
        print('MANUSCRIPT_FIGURES_COMPLETE=3; FILES=9; OLD_EVIDENCE_UNCHANGED; PAID_CALLS=0',flush=True)
    except Exception:
        write_json(seal,dict(status='FAILED_INCOMPLETE',paid_calls=0,supplemental_graphs=0))
        raise


if __name__=='__main__': main()
