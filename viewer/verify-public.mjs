// Check the exact deployment output, not just a development manifest.
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';
import {replayFrames,edgeKey} from './graph-data.js';
import {loadPrimaryDataset,loadMetrics,loadSupplements,COUNTS} from './layers-data-all.js';
import {OUTPUT_FILES as GRAPH_FILES} from './build-primary-graphs.mjs';
import {SAVED_NETWORKS_SHA256} from './network-catalog.js';
const root=new URL('./dist/',import.meta.url);
const publicData=new URL('./public/data/',import.meta.url);
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const files=(await readdir(root,{recursive:true})).map(f=>f.replaceAll('\\','/'));
// Serve dist/ exactly as the static host would; record what the page code would request.
const requested=[];
const fetchImpl=async url=>{
  const path=String(url).replace(/^\.\//,'');requested.push(path);
  try{const bytes=await readFile(new URL(path,root));return {ok:true,status:200,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};}
  catch{return {ok:false,status:404};}
};

// Only the compact files are deployed; the monolithic fixtures stay repo-only cross-checks.
for(const name of ['primary-viewer.json','saved-networks.json','saved-plots.json'])assert.ok(!files.includes(`data/${name}`),`${name} must not be deployed`);
assert.deepEqual(files.filter(f=>f.startsWith('data/graphs/')).sort(),GRAPH_FILES.map(n=>`data/graphs/${n}`).sort());
for(const name of GRAPH_FILES)assert.deepEqual(await readFile(new URL(`data/graphs/${name}`,root)),await readFile(new URL(`graphs/${name}`,publicData)),`dist graphs/${name} differs from the committed file`);

const primary=await loadPrimaryDataset({fetchImpl});
assert.deepEqual(requested,['data/graphs/index.json','data/graphs/edges.bin']);
assert.equal(primary.runs.length,COUNTS.valid);
assert.equal(primary.failures.length,COUNTS.failures);
assert.equal(primary.runs.filter(r=>r.edges.length===0).length,COUNTS.empty);
assert.equal(primary.slots.length,COUNTS.attempts);
assert.equal(new Set([...primary.runs,...primary.failures].map(r=>r.run_id)).size,COUNTS.attempts);
assert.ok(primary.failures.every(r=>!('edges' in r)));
await loadMetrics(primary,{fetchImpl});
assert.ok(primary.runs.every(r=>Object.keys(r.metrics).length===16&&Number.isFinite(r.metrics.density)));
const ids=primary.personas.map(p=>p.id);
for(const r of primary.runs){
  assert.match(r.run_id,/^final_060dcd3a86d5_main_[a-zA-Z0-9._-]+$/);
  assert.equal(r.events_url,`./data/primary-events/${r.run_id}.json.gz`);
  const bytes=gunzipSync(await readFile(new URL(r.events_url,root)));
  assert.deepEqual(bytes,await readFile(new URL(`primary-events/${r.run_id}.json`,publicData)),'Compression changed recorded bytes');
  assert.equal(sha256(bytes),r.events_sha256);
  const events=JSON.parse(bytes);
  assert.ok(Array.isArray(events)&&events.length>0,'A valid empty graph still requires its recorded no-change event');
  for(const e of events)assert.deepEqual(Object.keys(e).sort(),['added','attempts','method','persona','removed','step']);
  const final=replayFrames({...r,events},ids).at(-1).edges;
  assert.deepEqual(final.map(edgeKey).sort(),r.edges.map(edgeKey).sort(),'Recorded events do not rebuild the edges.bin graph');
}
assert.deepEqual(files.filter(f=>f.startsWith('data/primary-events/')).sort(),primary.runs.map(r=>r.events_url.slice(2)).sort(),'Missing or orphan event exports');

// Supplements: labelled, separate, and identical to the pinned saved-network fixture.
const supplements=await loadSupplements({fetchImpl});
const savedBytes=await readFile(new URL('saved-networks.json',publicData));
assert.equal(sha256(savedBytes),SAVED_NETWORKS_SHA256);
const saved=new Map(JSON.parse(savedBytes).graphs.map(g=>[g.run_id,g]));
const primaryIds=new Set(primary.runs.map(r=>r.run_id));
for(const r of primary.runs)assert.deepEqual(saved.get(r.run_id).edges.map(edgeKey).sort(),r.edges.map(edgeKey).sort());
assert.deepEqual(supplements.graphs.map(g=>g.run_id).sort(),[...saved.keys()].filter(id=>!primaryIds.has(id)).sort());
for(const g of supplements.graphs)assert.deepEqual(g.edges.map(edgeKey).sort(),saved.get(g.run_id).edges.map(edgeKey).sort());
for(const name of ['index.json','metrics.json']){
  const text=await readFile(new URL(`data/graphs/${name}`,root),'utf8');
  assert.ok(supplements.graphs.every(g=>!text.includes(g.run_id)),`Supplement leaked into ${name}`);
}

assert.match(await readFile(new URL('index.html',root),'utf8'),/id="saved-plots"/);
assert.ok(!files.some(f=>f.startsWith('data/saved-plots/')),'PNG gallery unexpectedly deployed');
for(const path of ['index.html','layers.html','study.html','layers.js','layers.css','graph-data.js','layers-data-all.js','layers-app.js','verified-events.js','formation-charts.js','navigation.js','research-plots.js','vendor/three.module.js','vendor/three.core.js','vendor/addons/controls/OrbitControls.js','data/study-figures/14_demographic_mixing.png','data/study-figures/15_efficiency_density.png'])assert.ok((await readFile(new URL(path,root))).length>0,path);
for(const page of ['index.html','study.html','archive.html'])assert.equal(await readFile(new URL(page,root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
const blocked=/(^|[/\\])(?:\.env[^/\\]*|api-key\.txt|authorization\.json|requests|ledger|billing)([/\\.]|$)/i;
for(const file of files)assert.ok(!blocked.test(file),`Private path in public build: ${file}`);
for(const file of files.filter(name=>/\.(json|js|html|css|adj|svg|txt)$/.test(name))){
  const text=await readFile(new URL(file,root),'utf8');
  assert.ok(!/sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}/.test(text),`Credential pattern in ${file}`);
}
const legacy=/(^|\/)(?:archive\/|calibration|revised-calibration|presentation|artifacts|networks\.json|main\.js|style\.css)/;
assert.ok(!files.some(file=>legacy.test(file)),'Historical viewer payload leaked into deployment');
const dataBytes=async names=>{let total=0;for(const name of names)total+=(await readFile(new URL(name,root))).length;return total;};
console.log(JSON.stringify({status:'PASS',primary_attempts:primary.slots.length,primary_graphs:primary.runs.length,empty_graphs:COUNTS.empty,failures:primary.failures.length,
  verified_event_logs:primary.runs.length,supplements_separate:supplements.graphs.length,startup_data_bytes:await dataBytes(['data/graphs/index.json','data/graphs/edges.bin']),
  monolith_files_deployed:0,archived_payloads:0,credential_patterns:0,paid_calls:0}));
