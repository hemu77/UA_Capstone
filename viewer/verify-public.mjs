// Check the exact deployment output, not just a development manifest.
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';
import {validateData,replayFrames} from './graph-data.js';
import {verifiedPlotProjection} from './plot-catalog-build.mjs';
import {SAVED_NETWORKS_SHA256,validateNetworkCatalog} from './network-catalog.js';
const root=new URL('./dist/',import.meta.url);
const primary=validateData(JSON.parse(await readFile(new URL('data/primary-viewer.json',root),'utf8')));
assert.equal(primary.runs.length,890);
assert.equal(primary.failures.length,6);
assert.equal(new Set([...primary.runs,...primary.failures].map(r=>r.run_id)).size,896);
assert.ok(primary.failures.every(r=>!('edges' in r)));
assert.equal(primary.runs.filter(r=>r.edges.length===0).length,12);
for(const r of primary.runs){
  assert.match(r.run_id,/^final_060dcd3a86d5_main_[a-zA-Z0-9._-]+$/);
  assert.equal(r.events_url,`./data/primary-events/${r.run_id}.json.gz`);
  const bytes=gunzipSync(await readFile(new URL(r.events_url,root)));
  assert.deepEqual(bytes,await readFile(new URL(`public/data/primary-events/${r.run_id}.json`,import.meta.url)),'Compression changed recorded bytes');
  assert.equal(createHash('sha256').update(bytes).digest('hex'),r.events_sha256);
  const events=JSON.parse(bytes);
  assert.ok(Array.isArray(events)&&events.length>0,'A valid empty graph still requires its recorded no-change event');
  for(const e of events)assert.deepEqual(Object.keys(e).sort(),['added','attempts','method','persona','removed','step']);
  assert.ok(replayFrames({...r,events},primary.personas.map(p=>p.id)),'Missing replay');
}
const files=(await readdir(root,{recursive:true})).map(f=>f.replaceAll('\\','/'));
const plots=verifiedPlotProjection(await readFile(new URL('data/saved-plots.json',root)));
assert.deepEqual(plots.filter(p=>p.cohort==='primary').map(p=>p.run_id).sort(),primary.runs.map(p=>p.run_id).sort());
assert.ok(plots.filter(p=>p.cohort!=='primary').every(p=>!primary.runs.some(r=>r.run_id===p.run_id)));
assert.ok(!files.some(f=>f.startsWith('data/saved-plots/')),'PNG gallery unexpectedly deployed');
const rawNetworks=await readFile(new URL('data/saved-networks.json',root));
assert.equal(createHash('sha256').update(rawNetworks).digest('hex'),SAVED_NETWORKS_SHA256);
const networks=validateNetworkCatalog(JSON.parse(rawNetworks),plots,primary);
for(const r of primary.runs)assert.deepEqual(networks.get(r.run_id).edges,r.edges);
assert.match(await readFile(new URL('index.html',root),'utf8'),/id="saved-plots"/);
assert.ok((await readFile(new URL('plot-gallery.js',root))).length);
const expected=primary.runs.map(r=>r.events_url.slice(2)).sort();
assert.deepEqual(files.filter(f=>f.startsWith('data/primary-events/')).sort(),expected,'Missing or orphan event exports');
for(const path of ['index.html','layers.html','study.html','layers.js','layers.css','graph-data.js','formation-charts.js','navigation.js','research-plots.js','vendor/three.module.js','vendor/three.core.js','vendor/addons/controls/OrbitControls.js','data/study-figures/14_demographic_mixing.png','data/study-figures/15_efficiency_density.png'])assert.ok((await readFile(new URL(path,root))).length>0,path);
assert.equal(await readFile(new URL('index.html',root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
assert.equal(await readFile(new URL('study.html',root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
const blocked=/(^|[/\\])(?:\.env[^/\\]*|api-key\.txt|authorization\.json|requests|ledger|billing)([/\\.]|$)/i;
for(const file of files)assert.ok(!blocked.test(file),`Private path in public build: ${file}`);
for(const file of files.filter(name=>/\.(json|js|html|css|adj|svg|txt)$/.test(name))){
  const text=await readFile(new URL(file.replaceAll('\\','/'),root),'utf8');
  assert.ok(!/sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}/.test(text),`Credential pattern in ${file}`);
}
const legacy=/(^|\/)(?:archive\/|calibration|revised-calibration|presentation|artifacts|networks\.json|main\.js|style\.css)/;
assert.ok(!files.some(file=>legacy.test(file)),'Historical viewer payload leaked into deployment');
assert.equal(await readFile(new URL('archive.html',root),'utf8'),await readFile(new URL('layers.html',root),'utf8'));
console.log(JSON.stringify({status:'PASS',primary_graphs:primary.runs.length,primary_attempts:896,verified_event_logs:primary.runs.length,saved_3d_graphs:networks.size,supplemental_plots:6,archived_payloads:0,credential_patterns:0,paid_calls:0}));
