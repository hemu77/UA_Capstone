import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {validateData,replayFrames,conditionCoverage,layerCandidates} from './graph-data.js';
const data=validateData(JSON.parse(readFileSync(new URL('./public/data/primary-viewer.json',import.meta.url))));
test('final primary cohort preserves failures and all empty graphs; no supplements',()=>{
  assert.equal(data.runs.length,890);assert.equal(data.failures.length,6);
  assert.equal(data.runs.filter(r=>!r.edges.length).length,12);
  assert.equal(data.personas.length,50);
  assert.ok(data.runs.every(r=>r.study==='primary'&&r.run_id.startsWith('final_060dcd3a86d5_main_')));
  assert.equal(new Set([...data.runs,...data.failures].map(r=>r.run_id)).size,896);
  assert.ok(data.runs.every(r=>Number.isFinite(r.metrics.density)&&Number.isFinite(r.metrics.avg_clustering_coef)));
  assert.ok(data.failures.every(r=>!('edges' in r)));
  assert.equal(layerCandidates(data.runs,{collection:'primary'}).length,890);
});
test('all 890 whitelisted histories reconstruct their exact final graph',()=>{
  for(const run of data.runs){
    const bytes=readFileSync(new URL('./public/'+run.events_url.slice(2),import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'),run.events_sha256);
    const events=JSON.parse(bytes);
    assert.ok(events.every(e=>Object.keys(e).sort().join(',')==='added,attempts,method,persona,removed,step'));
    assert.equal(replayFrames({...run,events},data.personas.map(p=>p.id)).at(-1).edges.length,run.edges.length);
  }
});
test('112 primary cells reconcile planned, valid and failed attempts',()=>{
  let planned=0,saved=0,failed=0;
  for(const model of data.models)for(const method of data.methods)for(const [culture,language] of data.settings){
    const r=conditionCoverage(data,model,method,culture,language);
    assert.equal(r.planned,8);assert.equal(r.saved.length+r.failed.length,8);
    planned+=r.planned;saved+=r.saved.length;failed+=r.failed.length;
  }
  assert.deepEqual([planned,saved,failed],[896,890,6]);
});
