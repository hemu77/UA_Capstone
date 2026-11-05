"""Primary-only inference and explicitly exploratory sensitivity diagnostics."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

CONTRACT='060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f'
SOURCE='outputs/final_study/060dcd3a86d5/main'
READY='outputs/analysis_ready/060dcd3a86d5'

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')

def validate_primary(frame, spec):
    require(len(frame)==896 and frame['run_id'].is_unique,'Need exactly 896 unique original observations')
    require(frame['cohort'].eq('primary').all() and frame['primary_member'].eq(True).all()
            and frame['run_id'].str.startswith('final_060dcd3a86d5_main_').all(), 'Supplemental or mixed data refused')
    require(frame['valid_graph'].map(type).eq(bool).all(),'Graph status must be boolean')
    a=spec['analysis']
    expected={(m,t,c,l,s) for m,t,(c,l),s in itertools.product(a['models'],a['methods'],a['conditions'],a['schedule_seeds'])}
    actual=set(frame[['model','method','culture','language','seed']].itertuples(index=False,name=None))
    require(actual==expected and len(expected)==896,'Primary schedule coverage changed')
    require(int(frame['valid_graph'].sum())==890 and frame['empty_graph'].eq(True).sum()==12,'Primary validity/empty counts changed')
    for method,metric in a['primary_endpoints'].items():
        group=frame[frame['method'].eq(method)]
        require(group.loc[~group.valid_graph,metric].isna().all(),'Failures cannot carry endpoints')
        observed=group[metric].dropna()
        require(np.isfinite(observed).all() and observed.between(0,1).all(),'Endpoint out of bounds')

def holm_full(probabilities):
    values=np.array([1. if p is None else p for p in probabilities],dtype=float)
    require(len(values)==24 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'Keep the full 24-hypothesis family')
    order=np.argsort(values,kind='stable'); result=np.empty(24)
    result[order]=np.minimum(1,np.maximum.accumulate(values[order]*np.arange(24,0,-1)))
    return [None if p is None else float(v) for p,v in zip(probabilities,result)]

def exact_sign_flip(values):
    values=np.asarray(values,dtype=float)
    require(1<=len(values)<=8 and np.isfinite(values).all(),'Require one to eight finite blocks')
    signs=np.array(list(itertools.product((-1,1),repeat=len(values))))
    statistics=np.abs((signs*values).mean(axis=1))
    return float(np.mean(statistics>=abs(values.mean())-1e-12))

def load_inputs(root):
    spec=json.loads((root/'final_study/contract.json').read_text(encoding='utf8'))
    encoded=json.dumps(spec,sort_keys=True,ensure_ascii=False).encode()
    require(hashlib.sha256(encoded).hexdigest()==CONTRACT,'Frozen contract changed')
    for name,expected in spec['sources'].items(): require(sha(root/name)==expected,'Frozen source changed: '+name)
    readiness=json.loads((root/READY/'readiness.json').read_text())
    path=root/READY/'primary_attempts.csv'
    require(sha(path)==readiness['generated_files']['primary_attempts.csv']['sha256'],'Prepared primary CSV changed')
    data=pd.read_csv(path)
    validate_primary(data,spec)
    rows=json.loads((root/SOURCE/'analysis_rows.json').read_text())
    require(len(rows)==896 and {r['run_id'] for r in rows}==set(data.run_id),'Original analysis rows differ')
    indexed=data.set_index('run_id')
    for row in rows:
        original=indexed.loc[row['run_id']]
        require(bool(original.valid_graph)==row['valid_graph'],'Analysis validity mismatch')
        for key in ('model','method','culture','language','seed'): require(original[key]==row[key],'Analysis identity mismatch')
        for metric in readiness['metric_columns']:
            value=row.get(metric)
            require((value is None and pd.isna(original[metric])) or
                    (value is not None and np.isclose(float(value),original[metric],rtol=0,atol=1e-12,equal_nan=False)), 'Analysis metric mismatch: '+metric)
    return spec,data,rows

def block_table(frame,spec):
    records=[]; a=spec['analysis']
    index=frame.set_index(['model','method','culture','language','seed'])
    for method,(culture,language),seed in itertools.product(a['methods'],a['conditions'][1:],a['schedule_seeds']):
        metric=a['primary_endpoints'][method]
        target=[index.loc[(model,method,culture,language,seed),metric] for model in a['models']]
        base=[index.loc[(model,method,'us','english',seed),metric] for model in a['models']]
        observed_target=[float(v) for v in target if pd.notna(v)]
        observed_base=[float(v) for v in base if pd.notna(v)]
        complete=len(observed_target)==len(observed_base)==4
        records.append(dict(hypothesis_id=f'{method}:{culture}:{language}',method=method,culture=culture,language=language,
            endpoint=metric,seed=int(seed),complete=complete,target_defined=len(observed_target),baseline_defined=len(observed_base),
            difference=float(np.mean(target)-np.mean(base)) if complete else None,
            lower=(sum(observed_target)-sum(observed_base)-(4-len(observed_base)))/4,
            upper=(sum(observed_target)+(4-len(observed_target))-sum(observed_base))/4))
    return pd.DataFrame(records)

def independent_contrasts(blocks,spec):
    results=[]; a=spec['analysis']; alpha=a['alpha']; family=a['primary_family_size']
    for hid,g in blocks.groupby('hypothesis_id',sort=False):
        sample=g.loc[g.complete,'difference'].to_numpy(dtype=float); n=len(sample)
        row=dict(hypothesis_id=hid,n=n,mean=float(sample.mean()) if n else None,
                 raw_p=None,ci=None,sign_flip_p=None,sign_consistency=float(max((sample>0).sum(),(sample<0).sum())/n) if n else None,
                 sample_sd=float(sample.std(ddof=1)) if n>1 else None,
                 minimum_exact_two_sided_p=2/2**n if n else None,
                 bounded_missingness_lower=float(g.lower.mean()),bounded_missingness_upper=float(g.upper.mean()))
        if n>=a['fallback_min_complete_blocks'] and np.ptp(sample)>1e-12:
            se=sample.std(ddof=1)/np.sqrt(n)
            pvalue=float(2*t.sf(abs(sample.mean()/se),n-1))
            half=float(t.isf(alpha/(2*family),n-1)*se)
            row.update(raw_p=pvalue,ci=[max(-1,float(sample.mean()-half)),min(1,float(sample.mean()+half))],sign_flip_p=exact_sign_flip(sample))
        results.append(row)
    for key,out in (('raw_p','holm_p'),('sign_flip_p','sign_flip_holm_p')):
        for row,value in zip(results,holm_full([r[key] for r in results])): row[out]=value
    return results

def leave_one_block_out(blocks,spec):
    rows=[]; a=spec['analysis']
    for omitted in a['schedule_seeds']:
        candidates=[]
        for hid,g in blocks[blocks.seed.ne(omitted)].groupby('hypothesis_id',sort=False):
            sample=g.loc[g.complete,'difference'].to_numpy(dtype=float); n=len(sample); pv=None
            if n>=a['fallback_min_complete_blocks'] and np.ptp(sample)>1e-12:
                pv=float(2*t.sf(abs(sample.mean()/(sample.std(ddof=1)/np.sqrt(n))),n-1))
            candidates.append(dict(omitted_seed=omitted,hypothesis_id=hid,n=n,
                                   mean=float(sample.mean()) if n else None,raw_p=pv))
        for row,pv in zip(candidates,holm_full([r['raw_p'] for r in candidates])):
            row['holm_p']=pv; rows.append(row)
    return pd.DataFrame(rows)

def compare_frozen(frozen,independent):
    other={r['hypothesis_id']:r for r in independent}
    require(len(frozen)==len(other)==24,'Contrast family changed')
    for row in frozen:
        check=other[row['hypothesis_id']]
        require(row['complete_schedule_blocks']==check['n'],'Complete blocks differ')
        for key,second in (('complete_block_mean','mean'),('raw_p','raw_p'),('holm_p','holm_p')):
            left,right=row[key],check[second]
            require((left is None and right is None) or (left is not None and right is not None and np.isclose(left,right,rtol=0,atol=1e-12)), 'Independent check differs: '+key)
        if row['simultaneous_ci'] is not None: require(np.allclose(row['simultaneous_ci'],check['ci'],rtol=0,atol=1e-12),'Simultaneous interval differs')

def archived_agreement(actual,expected):
    differences=[]
    def walk(left,right,path):
        if isinstance(left,dict):
            require(isinstance(right,dict) and set(left)==set(right),'Archived keys differ: '+path)
            for key in left: walk(left[key],right[key],path+'/'+key)
        elif isinstance(left,list):
            require(isinstance(right,list) and len(left)==len(right),'Archived list differs: '+path)
            for i,(a,b) in enumerate(zip(left,right)): walk(a,b,path+'/'+str(i))
        elif isinstance(left,(float,np.floating)):
            require(isinstance(right,(float,int)) and np.isfinite(left) and np.isfinite(right),'Archived numeric type differs: '+path)
            delta=abs(float(left)-float(right))
            require(delta<=1e-12,'Archived numeric discrepancy: '+path)
            if delta: differences.append(dict(field=path,absolute_difference=delta))
        else: require(type(left)==type(right) and left==right,'Archived identity/status differs: '+path)
    walk(actual,expected,'primary_contrasts')
    return dict(absolute_tolerance=1e-12,numeric_fields_differing=len(differences),
                maximum_absolute_difference=max((v['absolute_difference'] for v in differences),default=0),
                identity_and_status_exact=True)

def descriptive_tables(data,output):
    groups=['model','method','culture','language']
    metrics=['density','avg_clustering_coef','prop_nodes_lcc','global_efficiency','num_components','modularity',
             'degree_gini','coleman_gender','coleman_religion','coleman_political orientation','age_assortativity']
    tidy=[]
    for keys,g in data.groupby(groups,sort=True):
        for metric in metrics:
            values=g.loc[g.valid_graph,metric].dropna()
            tidy.append(dict(zip(groups,keys),metric=metric,attempts=len(g),valid_graphs=int(g.valid_graph.sum()),
                terminal_failures=int((~g.valid_graph).sum()),endpoint_defined=len(values),
                mean=float(values.mean()) if len(values) else None,sd=float(values.std(ddof=1)) if len(values)>1 else None,
                minimum=float(values.min()) if len(values) else None,maximum=float(values.max()) if len(values) else None))
    pd.DataFrame(tidy).to_csv(output/'descriptive_metrics.csv',index=False)
    quality=data.groupby(groups).agg(attempts=('run_id','size'),valid_graphs=('valid_graph','sum'),valid_empty=('empty_graph',lambda s:int(s.eq(True).sum())))
    quality['failures']=quality.attempts-quality.valid_graphs
    quality.to_csv(output/'coverage_by_cell.csv')

def model_contrasts(data,blocks,spec,output):
    index=data.set_index(['model','method','culture','language','seed']); rows=[]; a=spec['analysis']
    for hid,g in blocks.groupby('hypothesis_id',sort=False):
        first=g.iloc[0]; metric=first.endpoint; common=set(g.loc[g.complete,'seed'])
        for model in a['models']:
            available=[]; shared=[]
            for seed in a['schedule_seeds']:
                x=index.loc[(model,first.method,first.culture,first.language,seed),metric]
                y=index.loc[(model,first.method,'us','english',seed),metric]
                if pd.notna(x) and pd.notna(y):
                    available.append(float(x-y))
                    if seed in common: shared.append(float(x-y))
            rows.append(dict(hypothesis_id=hid,method=first.method,culture=first.culture,language=first.language,
                model=model,endpoint=metric,available_model_pairs=len(available),common_panel_blocks=len(shared),
                available_pair_mean=float(np.mean(available)) if available else None,
                common_panel_mean=float(np.mean(shared)) if shared else None,descriptive_only=True))
    pd.DataFrame(rows).to_csv(output/'model_contrasts_descriptive.csv',index=False)

def resource_tables(root,data,output):
    summary=json.loads((root/SOURCE/'summary.json').read_text())
    source_summary={r['run_id']:r for r in summary['results']}
    resources=[]
    for number,row in enumerate(data.itertuples(index=False),1):
        path=root/row.record_path
        require(sha(path)==row.record_sha256,'Primary receipt changed')
        record=json.loads(path.read_text(encoding='utf8'))
        inputs=outputs=cached=0; cache_known=True; received_cost=api_ms=0.; parse_errors=0
        for request in record['requests']:
            usage=request['response']['usage']; inputs+=usage['prompt_tokens']; outputs+=usage['completion_tokens']
            received_cost+=request['response']['conservative_charge_usd']
            api_ms+=request['response']['diagnostics']['elapsed_ms']
            parse_errors+=request['response'].get('parse',{}).get('valid') is False
            detail=usage.get('prompt_tokens_details') or {}
            if type(detail.get('cached_tokens')) is int: cached+=detail['cached_tokens']
            else: cache_known=False
        receipt=source_summary[row.run_id]
        require(receipt['requests']==len(record['requests'])==row.request_receipts,'Request count differs')
        require(np.isclose(received_cost,receipt['conservative_usd'],atol=1e-10,rtol=0) and parse_errors==receipt['parse_failures'], 'Per-receipt resource total differs')
        require(np.isclose(api_ms/1000,receipt['api_seconds'],atol=1e-6,rtol=0),'API-time total differs')
        resources.append(dict(run_id=row.run_id,model=row.model,method=row.method,culture=row.culture,language=row.language,seed=int(row.seed),
            valid_graph=bool(row.valid_graph),requests=len(record['requests']),parse_failures=receipt['parse_failures'],
            prompt_tokens=inputs,completion_tokens=outputs,cached_prompt_tokens=cached if cache_known else None,
            received_conservative_usd=receipt['conservative_usd'],api_seconds=receipt['api_seconds']))
        if number%200==0: print(f'PRIMARY_RESOURCE_RECEIPTS={number}/896',flush=True)
    frame=pd.DataFrame(resources); frame.to_csv(output/'resources_per_attempt.csv',index=False)
    for dimensions,name in ((['method','model'],'resources_by_method_model.csv'),
                            (['method','model','culture','language'],'resources_by_cell.csv')):
        grouped=frame.groupby(dimensions).agg(attempts=('run_id','size'),valid_graphs=('valid_graph','sum'),
            requests=('requests','sum'),prompt_tokens=('prompt_tokens','sum'),completion_tokens=('completion_tokens','sum'),
            cached_prompt_tokens=('cached_prompt_tokens',lambda x:x.sum() if x.notna().all() else np.nan),
            received_conservative_usd=('received_conservative_usd','sum'),api_seconds=('api_seconds','sum'),parse_failures=('parse_failures','sum'))
        grouped['received_cost_per_attempt']=grouped.received_conservative_usd/grouped.attempts
        grouped['received_cost_per_valid_graph']=grouped.received_conservative_usd/grouped.valid_graphs.replace(0,np.nan)
        grouped.to_csv(output/name)
    require(np.isclose(frame.received_conservative_usd.sum(),summary['conservative_usd'],atol=1e-9,rtol=0),'Received cost total differs')
    result=dict(received_request_count=int(frame.requests.sum()),received_conservative_usd=summary['conservative_usd'],
                total_charge_or_reservation_usd=summary['ledger_charge_or_reservation_usd'],
                retained_reservations_usd=summary['ledger_charge_or_reservation_usd']-summary['conservative_usd'],
                api_seconds=summary['api_seconds'],cost_grouping_excludes_unallocated_retained_reservations=True,
                note='Received-cost groupings charge failed graph attempts. Retained reservations are reported only in the study total, not fabricated method allocations. API seconds are serial provider time, not wall time or a speedup claim. No invoice or equal-compute comparison.')
    write_json(output/'resource_accounting.json',result)
    return result

def run_core(root,output):
    spec,data,rows=load_inputs(root)
    sys.path.insert(0,str(root))
    from final_study import analysis as frozen_analysis
    require(frozen_analysis.load_spec()==spec['analysis'],'Analysis specification differs')
    frozen=frozen_analysis.primary_contrasts(rows,spec['analysis'])
    old=json.loads((root/SOURCE/'primary_contrasts.json').read_text())
    agreement=archived_agreement(frozen,old)
    blocks=block_table(data,spec); independent=independent_contrasts(blocks,spec)
    compare_frozen(frozen,independent)
    blocks.to_csv(output/'paired_schedule_blocks.csv',index=False)
    write_json(output/'primary_contrasts.json',frozen)
    flat=[]
    for row in frozen:
        item={k:v for k,v in row.items() if k not in {'simultaneous_ci','per_model','missing_cells','excluded_schedule_seeds'}}
        item['ci_lower']=row['simultaneous_ci'][0] if row['simultaneous_ci'] else None
        item['ci_upper']=row['simultaneous_ci'][1] if row['simultaneous_ci'] else None
        flat.append(item)
    pd.DataFrame(flat).to_csv(output/'primary_contrasts.csv',index=False)
    write_json(output/'sensitivity.json',independent)
    pd.DataFrame(independent).drop(columns=['ci']).to_csv(output/'sensitivity.csv',index=False)
    leave_one_block_out(blocks,spec).to_csv(output/'leave_one_seed_out.csv',index=False)
    write_json(output/'missingness.json',frozen_analysis.missingness_report(rows,spec['analysis']))
    descriptive_tables(data,output)
    model_contrasts(data,blocks,spec,output)
    resources=resource_tables(root,data,output)
    result=dict(contract_sha256=CONTRACT,primary_attempts=896,valid_primary_graphs=890,primary_failures=6,valid_empty_primary_graphs=12,
        family_size=24,independent_numeric_crosscheck=True,archived_primary_agreement=agreement,
        primary_holm_significant=[r['hypothesis_id'] for r in frozen if r['holm_p'] is not None and r['holm_p']<.05],
        exploratory_sign_flip_holm_significant=[r['hypothesis_id'] for r in independent if r['sign_flip_holm_p'] is not None and r['sign_flip_holm_p']<.05],
        status_counts={k:int(v) for k,v in pd.Series([r['status'] for r in frozen]).value_counts().items()},resources=resources,
        caveats=['Single fictional roster; four fixed models; eight schedule blocks; conditional t assumptions.',
                 'Sign flips are a post-collection symmetry sensitivity diagnostic, not causal randomization inference.',
                 'Missingness bounds are identification bounds, not confidence intervals. Complete blocks need ignorable missingness.',
                 'No equivalence margin; nonsignificance is not evidence of sameness. Supplementary graphs excluded.'])
    write_json(output/'summary.json',result)
    return spec,data,result
