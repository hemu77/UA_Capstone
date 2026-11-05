"""Run the fixed public structural benchmark offline from hash-verified downloads."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_primary.core import sha,write_json
from analysis_heldout.benchmark import run_benchmark
from analysis_heldout.figures import render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    output=ROOT/'outputs/heldout_evaluation/20261105_v1'
    output.mkdir(parents=True,exist_ok=True)
    seal=output/'evaluation_manifest.json'
    code=[*sorted((ROOT/'analysis_heldout').glob('*.py')),ROOT/'analysis_heldout/protocol.json',
        ROOT/'analysis_heldout/clarifications.json',Path(__file__),ROOT/'scripts/fetch_heldout_public.py']
    code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in code}
    write_json(seal,dict(status='IN_PROGRESS',paid_calls=0,supplemental_graphs=0))
    try:
        result=run_benchmark(ROOT,args.cache.resolve(),output)
        figures=render(output)
        if code_hashes!={p.relative_to(ROOT).as_posix():sha(p) for p in code}:
            raise ValueError('Benchmark code changed during execution; rerun stable reviewed code')
        artifacts={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(output.rglob('*')) if p.is_file() and p!=seal}
        write_json(seal,dict(status='COMPLETE_HELDOUT_STRUCTURAL_EVALUATION_WITH_CAVEATS',
            created_utc=datetime.now(timezone.utc).isoformat(),paid_calls=0,ledger_queries=0,supplemental_graphs=0,
            code_sha256=code_hashes,artifacts=artifacts,previous_primary_seal_sha256=result['preserved_previous_artifacts']['outputs/primary_analysis/20261105/evaluation_manifest.json'],
            source_sha256={name:info['sha256'] for name,info in result['sources'].items()},
            public_studies=2,cohort_records=result['cohort_records'],valid_primary_graphs=890,
            graph_metrics=6,figures=len(figures),causal_or_population_inference=False))
        print(json.dumps(dict(status='COMPLETE',cohorts=result['cohort_records'],node_draws=result['sampled_graphs'],
            baseline_graphs=result['baseline_graphs'],conditional_draws=result['conditional_graph_draws'],
            conditional_failures=result['failed_conditional_draws'],figure_files=len(figures),paid_calls=0)),flush=True)
    except Exception:
        write_json(seal,dict(status='FAILED_INCOMPLETE',paid_calls=0,supplemental_graphs=0))
        raise


if __name__=='__main__': main()
