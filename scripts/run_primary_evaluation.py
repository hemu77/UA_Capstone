"""Reproduce offline primary-only statistics, graph evaluations and figures."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_primary.core import SOURCE, sha, write_json, run_core
from analysis_primary.graph_evaluation import run_graph_evaluation
from analysis_primary.empirical import run_empirical_evaluation
from analysis_primary.plots import render_figures

OUTPUT=ROOT/'outputs/primary_analysis/20261105'

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    seal=OUTPUT/'evaluation_manifest.json'
    write_json(seal,dict(status='IN_PROGRESS',primary_only=True,paid_calls=0))
    protected=[ROOT/SOURCE/name for name in ('summary.json','analysis_rows.json','analysis_summary.json',
               'primary_contrasts.json','primary_contrasts.csv','network_metrics.csv','missingness_report.json')]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    code=[*sorted((ROOT/'analysis_primary').glob('*.py')),Path(__file__)]
    code_before={p.relative_to(ROOT).as_posix():sha(p) for p in code}
    try:
        spec,data,result=run_core(ROOT,OUTPUT)
        graph=run_graph_evaluation(ROOT,data,OUTPUT)
        empirical=run_empirical_evaluation(ROOT,data,OUTPUT)
        figures=render_figures(ROOT,data,OUTPUT)
        if before!={p.relative_to(ROOT).as_posix():sha(p) for p in protected}:
            raise ValueError('Original collection/analysis artifacts changed')
        for name,expected in spec['sources'].items():
            if sha(ROOT/name)!=expected: raise ValueError('Frozen source changed: '+name)
        code_after=[*sorted((ROOT/'analysis_primary').glob('*.py')),Path(__file__)]
        if code_before!={p.relative_to(ROOT).as_posix():sha(p) for p in code_after}:
            raise ValueError('Evaluation code changed during execution; rerun on stable code')
        artifacts={p.relative_to(OUTPUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
                   for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p!=seal}
        manifest=dict(status='COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS',created_utc=datetime.now(timezone.utc).isoformat(),
            primary_only=True,primary_attempts=896,valid_primary_graphs=890,primary_failures=6,
            supplemental_graphs_in_inference=0,paid_calls=0,ledger_queries=0,
            preserved_original_artifacts=before,frozen_contract_sha256=result['contract_sha256'],
            code_sha256=code_before,artifacts=artifacts,
            independent_primary_crosscheck=result['independent_numeric_crosscheck'],
            verified_primary_graphs=graph['verified_graphs'],synthetic_reference_graphs=graph['synthetic_graphs'],
            empirical_reference_networks=empirical['reference_networks'],
            figure_files=len(figures),statistical_assumptions_verified=False,
            scope='Fixed roster, fixed models, named instructions; not national populations, equivalence or empirical realism.')
        write_json(seal,manifest)
        print(json.dumps({k:v for k,v in manifest.items() if k not in {'code_sha256','artifacts','preserved_original_artifacts'}}),flush=True)
    except Exception:
        write_json(seal,dict(status='FAILED_INCOMPLETE',primary_only=True,paid_calls=0,
                            message='Do not use partial derived outputs as a completed analysis. Original evidence was not intentionally modified.'))
        raise

if __name__=='__main__': main()
