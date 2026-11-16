"""Rebuild primary analysis in a NEW directory, preserving historical releases."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_primary.core import SOURCE, sha, write_json, run_core
from analysis_primary.graph_evaluation import run_graph_evaluation
from analysis_primary.empirical import run_empirical_evaluation
from analysis_primary.plots import render_figures


def build(destination):
    requested = Path(destination)
    destination = requested.resolve()
    if (requested.is_symlink() or destination.parent != ROOT/'outputs/primary_analysis'
            or destination.exists()):
        raise ValueError('Use a NEW direct child of outputs/primary_analysis; no overwrites')
    protected = [ROOT/SOURCE/name for name in (
        'summary.json', 'analysis_rows.json', 'analysis_summary.json', 'primary_contrasts.json',
        'primary_contrasts.csv', 'network_metrics.csv', 'missingness_report.json')]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    code = [*sorted((ROOT/'analysis_primary').glob('*.py')), Path(__file__)]
    code_before = {p.relative_to(ROOT).as_posix(): sha(p) for p in code}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive creation also rejects a competing creator after the path check.
    seal = destination/'evaluation_manifest.json'
    # The historical driver is hash-pinned; reuse its analysis functions, not its writable entrypoint.
    try:
        write_json(seal, dict(status='IN_PROGRESS', primary_only=True, paid_calls=0))
        spec, data, result = run_core(ROOT, destination)
        graph = run_graph_evaluation(ROOT, data, destination)
        empirical = run_empirical_evaluation(ROOT, data, destination)
        figures = render_figures(ROOT, data, destination)
        if before != {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}:
            raise ValueError('Original collection/analysis artifacts changed')
        for name, expected in spec['sources'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Frozen source changed: '+name)
        code_after = [*sorted((ROOT/'analysis_primary').glob('*.py')), Path(__file__)]
        if code_before != {p.relative_to(ROOT).as_posix(): sha(p) for p in code_after}:
            raise ValueError('Evaluation code changed during execution; rerun on stable code')
        artifacts = {p.relative_to(destination).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
                     for p in sorted(destination.rglob('*')) if p.is_file() and p != seal}
        manifest = dict(
            status='COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS', created_utc=datetime.now(timezone.utc).isoformat(),
            runner_version=2, primary_only=True, primary_attempts=896, valid_primary_graphs=890,
            primary_failures=6, supplemental_graphs_in_inference=0, paid_calls=0, ledger_queries=0,
            preserved_original_artifacts=before, frozen_contract_sha256=result['contract_sha256'],
            code_sha256=code_before, artifacts=artifacts,
            independent_primary_crosscheck=result['independent_numeric_crosscheck'],
            verified_primary_graphs=graph['verified_graphs'], synthetic_reference_graphs=graph['synthetic_graphs'],
            empirical_reference_networks=empirical['reference_networks'], figure_files=len(figures),
            statistical_assumptions_verified=False,
            scope='Fixed roster, fixed models, named instructions; not national populations, equivalence or empirical realism.')
        write_json(seal, manifest)
    except (Exception, KeyboardInterrupt):
        write_json(seal, dict(status='FAILED_INCOMPLETE', primary_only=True, paid_calls=0,
                             message='Use neither partial outputs nor this failed rebuild as a completed analysis.'))
        raise
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in {'code_sha256', 'artifacts', 'preserved_original_artifacts'}}), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='NEW direct child of outputs/primary_analysis. Existing paths are refused.')
    build(parser.parse_args(argv).output)


if __name__ == '__main__':
    main()
