import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {validateData,replayFrames,conditionCoverage,layerCandidates} from './graph-data.js';
import {loadPrimaryDataset} from './layers-data-all.js';
// `data` is the pinned cross-check fixture; `compact` is what the viewer loads (./data/graphs/).
const data=validateData(JSON.parse(readFileSync(new URL('./public/data/primary-viewer.json',import.meta.url))));
const fetchImpl=async url=>{const bytes=readFileSync(new URL('./public/'+String(url).slice(2),import.meta.url));return {ok:true,status:200,json:async()=>JSON.parse(bytes),arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};};
const compact=await loadPrimaryDataset({fetchImpl});
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
test('compact dataset keeps the same primary population and helper behaviour',()=>{
  assert.equal(compact.runs.length,890);assert.equal(compact.failures.length,6);
  assert.equal(compact.runs.filter(r=>!r.edges.length).length,12);
  assert.deepEqual(compact.runs.map(r=>r.run_id).sort(),data.runs.map(r=>r.run_id).sort());
  assert.deepEqual(compact.failures.map(r=>r.run_id).sort(),data.failures.map(r=>r.run_id).sort());
  assert.equal(layerCandidates(compact.runs,{collection:'primary'}).length,890);
});
test('all 890 whitelisted histories reconstruct their exact final graph from edges.bin',()=>{
  for(const run of compact.runs){
    assert.equal(run.events_url,`./data/primary-events/${run.run_id}.json.gz`);
    const bytes=readFileSync(new URL(`./public/data/primary-events/${run.run_id}.json`,import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'),run.events_sha256);
    const events=JSON.parse(bytes);
    assert.ok(events.every(e=>Object.keys(e).sort().join(',')==='added,attempts,method,persona,removed,step'));
    assert.equal(replayFrames({...run,events},compact.personas.map(p=>p.id)).at(-1).edges.length,run.edges.length);
  }
});
function reconcile(dataset) {
  let planned=0,saved=0,failed=0;
  for(const model of dataset.models)for(const method of dataset.methods)for(const [culture,language] of dataset.settings){
    const r=conditionCoverage(dataset,model,method,culture,language);
    assert.equal(r.planned,8);assert.equal(r.saved.length+r.failed.length,8);
    planned+=r.planned;saved+=r.saved.length;failed+=r.failed.length;
  }
  assert.deepEqual([planned,saved,failed],[896,890,6]);
}
test('112 primary cells reconcile planned, valid and failed attempts in fixture and compact data',()=>{
  reconcile(data);reconcile(compact);
});
