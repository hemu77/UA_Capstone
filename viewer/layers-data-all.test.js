import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {loadPrimaryDataset,loadMetrics,loadSupplements,decodeEdges,COUNTS,slotOf,settingKey} from './layers-data-all.js';
import {generateGraphFiles,parseCsv,parseAdjacency,pythonJson,OUTPUT_FILES,METRIC_COLUMNS,SOURCES} from './build-primary-graphs.mjs';
import {validateData,layerComparison,MAX_LAYERS} from './graph-data.js';

const publicFile=name=>readFileSync(new URL('./public/data/'+name,import.meta.url));
const repoFile=path=>readFileSync(new URL('../'+path,import.meta.url));
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
// Serves viewer/public/data the way the static host serves dist/data; `edit` swaps one file's bytes.
const server=(edit={})=>{
  const requests=[];
  const fetchImpl=async url=>{
    const name=String(url).replace('./data/','');requests.push(name);
    let bytes;
    try{bytes=name in edit?Buffer.from(edit[name]):publicFile(name);}catch{return {ok:false,status:404};}
    return {ok:true,status:200,json:async()=>JSON.parse(bytes),arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
  };
  return {fetchImpl,requests};
};
const editIndex=mutate=>{const index=JSON.parse(publicFile('graphs/index.json'));mutate(index);return {'graphs/index.json':JSON.stringify(index)};};
const fixture=JSON.parse(publicFile('primary-viewer.json'));
const catalog=JSON.parse(publicFile('saved-networks.json'));
const csv=parseCsv(repoFile(SOURCES.attempts).toString('utf8'));
const numeric=edges=>edges.map(e=>e.map(Number).sort((a,b)=>a-b)).sort((x,y)=>x[0]-y[0]||x[1]-y[1]);
const {fetchImpl,requests}=server();
const data=await loadPrimaryDataset({fetchImpl});

test('loader exposes exactly 896 slots: 890 valid (12 empty) and 6 failures, no supplements',()=>{
  assert.equal(data.slots.length,COUNTS.attempts);
  assert.deepEqual([data.slots.filter(s=>s.status!=='failed').length,data.slots.filter(s=>s.status==='empty').length,data.slots.filter(s=>s.status==='failed').length],[890,12,6]);
  assert.equal(new Set(data.runs.map(r=>r.run_id)).size,890);
  assert.ok(data.runs.every(r=>r.run_id.startsWith('final_060dcd3a86d5_main_')&&data.slots[r.slot].run_id===r.run_id&&r.setting===settingKey(r.culture,r.language)));
  assert.ok(data.slots.every((s,i)=>s.slot===i));
  assert.equal(slotOf(data.design,{model:data.models[0],method:data.methods[0],culture:'us',language:'english',seed:44000}),0);
  assert.equal(slotOf(data.design,{model:'gpt-6-sol',method:'iterative',culture:'us',language:'portuguese',seed:44007}),895);
  assert.throws(()=>slotOf(data.design,{model:'x',method:'global',culture:'us',language:'english',seed:44000}));
  assert.throws(()=>slotOf(data.design,{model:'gpt-4.1',method:'global',culture:'japan',language:'hindi',seed:44000}));
  assert.deepEqual(data.counts,COUNTS);
  assert.deepEqual(COUNTS,{attempts:896,valid:890,empty:12,failures:6,supplements_excluded:6,personas:50,cells:112,repetitions:8});
  assert.deepEqual(requests,['graphs/index.json','graphs/edges.bin'],'start-up reads only the index and the edge buffer');
});

test('dataset is a validated superset of primary-viewer.json for every run and failure',()=>{
  assert.equal(validateData(data),data);
  for(const key of ['dataset','roster_id','models','methods','settings','planned_networks','planned_repetitions','analysis_version','evidence_notice','calibration_note','limitations','verification'])assert.deepEqual(data[key],fixture[key],key);
  assert.deepEqual(data.design.seeds,[44000,44001,44002,44003,44004,44005,44006,44007]);
  const runs=new Map(data.runs.map(r=>[r.run_id,r]));
  for(const published of fixture.runs){
    const run=runs.get(published.run_id);
    for(const [key,value] of Object.entries(published)){
      if(key==='edges')assert.deepEqual(numeric(run.edges),numeric(value));
      else if(key==='events_url')assert.equal(run.events_url,value+'.gz','deployed logs are gzip transported');
      else if(key==='metrics')assert.ok(Math.abs(run.metrics.density-value.density)<1e-12&&Math.abs(run.metrics.avg_clustering_coef-value.avg_clustering_coef)<1e-12);
      else if(key!=='homophily')assert.deepEqual(run[key],value,key);
    }
  }
  const failures=new Map(data.failures.map(f=>[f.run_id,f]));
  for(const published of fixture.failures)assert.deepEqual({...failures.get(published.run_id),slot:undefined,setting:undefined,record_sha256:undefined},{...published,slot:undefined,setting:undefined,record_sha256:undefined});
  assert.ok(data.failures.every(f=>!('edges' in f)&&data.slots[f.slot].status==='failed'));
});

test('roster equals primary-viewer.json personas and reproduces the canonical roster hash',()=>{
  const roster=JSON.parse(repoFile(SOURCES.roster));
  assert.equal(createHash('sha256').update(pythonJson(roster)).digest('hex'),data.roster_id);
  assert.ok(csv.every(row=>row.roster_sha256===data.roster_id));
  assert.equal(data.personas.length,COUNTS.personas);
  data.personas.forEach((person,i)=>{
    const published=fixture.personas[i];
    assert.equal(person.id,published.id);assert.deepEqual(person.attributes,published.attributes);assert.deepEqual(person.attributes,roster[person.id]);
    person.position.forEach((v,k)=>assert.ok(Math.abs(v-published.position[k])<1e-9));
  });
  assert.equal(data.layout.points.size,50);
  assert.ok([...data.layout.points.values()].every(([x,y])=>Math.hypot(x,y)<=75.1));
  assert.match(data.layout.basis,/not geographic/);
});

test('all 890 edge lists from edges.bin equal their .adj files, the CSV counts and primary-viewer.json',()=>{
  const bySlot=new Map(data.runs.map(r=>[r.run_id,r]));
  const published=new Map(fixture.runs.map(r=>[r.run_id,r]));
  let ties=0;
  for(const row of csv.filter(r=>r.valid_graph==='True')){
    const run=bySlot.get(row.run_id),adj=repoFile(run.source);
    assert.equal(sha256(adj),row.adj_sha256);assert.equal(run.source_sha256,row.adj_sha256);
    assert.deepEqual(numeric(run.edges),parseAdjacency(adj.toString('utf8'),50));
    assert.deepEqual(numeric(run.edges),numeric(published.get(row.run_id).edges));
    assert.equal(run.edges.length,Number(row.edges));
    assert.equal(run.edges.length===0,row.empty_graph==='True');
    for(const [a,b] of run.edges)assert.ok(typeof a==='string'&&Number(a)<Number(b)&&Number(b)<50);
    assert.equal(new Set(run.edges.map(e=>e.join('-'))).size,run.edges.length);
    ties+=run.edges.length;
  }
  assert.equal(ties,156730);
  assert.equal(data.edgeData.bytes.length,2*ties);
  assert.equal(data.edgeData.offsets.length,897);
  for(const run of data.runs)assert.deepEqual(decodeEdges(data.edgeData.bytes,data.edgeData.offsets[run.slot],data.edgeData.offsets[run.slot+1]-data.edgeData.offsets[run.slot]),run.edges);
  for(const failure of data.failures)assert.equal(data.edgeData.offsets[failure.slot+1],data.edgeData.offsets[failure.slot]);
});

test('index.json rows reconcile with the attempts CSV, the event logs and the integrity split',()=>{
  const index=JSON.parse(publicFile('graphs/index.json'));
  assert.equal(index.attempts.length,896);
  assert.equal(index.source.attempts_csv_sha256,sha256(repoFile(SOURCES.attempts)));
  assert.equal(index.source.roster_file_sha256,sha256(repoFile(SOURCES.roster)));
  const rows=new Map(csv.map(r=>[r.run_id,r]));
  for(const slot of data.slots){
    const row=rows.get(slot.run_id),attempt=index.attempts[slot.slot];
    assert.equal(attempt[6],row.record_sha256);
    if(slot.status==='failed'){assert.equal(row.valid_graph,'False');assert.equal(attempt[8],row.failure);continue;}
    const run=data.runs.find(r=>r.slot===slot.slot);
    assert.equal(attempt[2],Number(row.edges));
    assert.equal(run.events_sha256,sha256(publicFile(`primary-events/${run.run_id}.json`)));
  }
  assert.equal(index.edges.bytes,313460);
  assert.equal(index.edges.sha256,sha256(publicFile('graphs/edges.bin')));
  assert.equal(index.metrics.sha256,sha256(publicFile('graphs/metrics.json')));
  assert.equal(index.supplements.sha256,sha256(publicFile('graphs/supplements.json')));
});

test('metrics load once on request and equal the attempts CSV at full precision',async()=>{
  const {fetchImpl:counted,requests:log}=server();
  const fresh=await loadPrimaryDataset({fetchImpl:counted});
  assert.deepEqual(Object.keys(fresh.runs[0].metrics),['density','avg_clustering_coef']);
  assert.deepEqual(fresh.runs[0].homophily,{});
  const [first,second]=await Promise.all([loadMetrics(fresh,{fetchImpl:counted}),loadMetrics(fresh,{fetchImpl:counted})]);
  assert.equal(first,fresh);assert.equal(second,fresh);
  await loadMetrics(fresh,{fetchImpl:counted});
  assert.equal(log.filter(name=>name==='graphs/metrics.json').length,1,'metrics.json is fetched once');
  const rows=new Map(csv.map(r=>[r.run_id,r]));
  for(const run of fresh.runs){
    const row=rows.get(run.run_id);
    assert.deepEqual(Object.keys(run.metrics),[...METRIC_COLUMNS]);
    for(const column of METRIC_COLUMNS)assert.equal(run.metrics[column],row[column]===''?null:Number(row[column]),`${column} of ${run.run_id}`);
    assert.deepEqual(run.homophily,{gender:run.metrics.coleman_gender,religion:run.metrics.coleman_religion,'political orientation':run.metrics['coleman_political orientation']});
  }
});

test('supplements load separately: exactly the 6 excluded graphs, equal to saved-networks.json and absent from the primary files',async()=>{
  const supplements=await loadSupplements({fetchImpl});
  assert.match(supplements.note,/^Excluded from the primary study/);
  const primary=new Set(fixture.runs.map(r=>r.run_id)),excluded=catalog.graphs.filter(g=>!primary.has(g.run_id));
  assert.equal(excluded.length,6);
  assert.deepEqual(supplements.graphs.map(g=>g.run_id).sort(),excluded.map(g=>g.run_id).sort());
  for(const graph of supplements.graphs){
    const saved=excluded.find(g=>g.run_id===graph.run_id);
    assert.deepEqual(numeric(graph.edges),numeric(saved.edges));
    assert.equal(graph.source_sha256,saved.source_sha256);assert.equal(graph.source,saved.source);
    assert.equal(sha256(repoFile(graph.source)),graph.source_sha256);
    assert.ok(!data.runs.some(r=>r.run_id===graph.run_id)&&!data.failures.some(f=>f.run_id===graph.run_id));
  }
  for(const name of ['graphs/index.json','graphs/metrics.json','graphs/edges.bin']){
    const text=publicFile(name).toString('latin1');
    for(const graph of supplements.graphs)assert.ok(!text.includes(graph.run_id),`${graph.run_id} leaked into ${name}`);
    assert.ok(!/supp(lement)?_(complete_)?2026/.test(text),`supplement id pattern in ${name}`);
  }
});

test('generator is byte-deterministic and matches the committed files',async()=>{
  const [first,second]=[await generateGraphFiles(),await generateGraphFiles()];
  assert.deepEqual([...first.keys()],[...OUTPUT_FILES]);
  for(const name of OUTPUT_FILES){
    assert.ok(first.get(name).equals(second.get(name)),`${name} differs between two runs`);
    assert.ok(first.get(name).equals(publicFile('graphs/'+name)),`${name} differs from the committed file; run node build-primary-graphs.mjs`);
  }
});

test('a whole condition (8 seeds) stacks as one comparison; a 9th layer is refused',()=>{
  const condition=data.runs.filter(r=>r.model==='gpt-4.1'&&r.method==='local'&&r.setting==='us-english');
  assert.equal(condition.length,8);assert.equal(MAX_LAYERS,8);
  assert.ok(layerComparison(condition).union>0);
  assert.throws(()=>layerComparison([...condition,data.runs.find(r=>!condition.includes(r))]),/between 1 and 8/);
});

test('decodeEdges returns sorted string pairs and rejects malformed buffers',()=>{
  const bytes=Uint8Array.from([0,1,0,49,3,7,2,2,5,4,1,0]);
  assert.deepEqual(decodeEdges(bytes,0,3),[['0','1'],['0','49'],['3','7']]);
  assert.deepEqual(decodeEdges(bytes,1,0),[]);
  assert.throws(()=>decodeEdges(bytes,3,1),/integrity/,'self tie');
  assert.throws(()=>decodeEdges(bytes,4,1),/integrity/,'a > b');
  assert.throws(()=>decodeEdges(Uint8Array.from([0,2,0,1]),0,2),/integrity/,'unsorted');
  assert.throws(()=>decodeEdges(Uint8Array.from([0,1,0,1]),0,2),/integrity/,'duplicate');
  assert.throws(()=>decodeEdges(bytes,5,2),/integrity/,'range');
  assert.throws(()=>decodeEdges(bytes,-1,1),/integrity/);
});

test('loader refuses HTTP errors, tampered bytes, wrong counts and leaked supplements; nothing is substituted',async()=>{
  const reject=(edit,pattern)=>assert.rejects(loadPrimaryDataset({fetchImpl:server(edit).fetchImpl}),pattern);
  const missing=async()=>({ok:false,status:503});
  await assert.rejects(loadPrimaryDataset({fetchImpl:missing}),/^Error: Data request failed: 503\. No substitute data is loaded\.$/);
  const edges=Buffer.from(publicFile('graphs/edges.bin'));edges[10]^=1;
  await reject({'graphs/edges.bin':edges},/edges\.bin does not match its SHA-256/);
  await reject({'graphs/edges.bin':publicFile('graphs/edges.bin').subarray(2)},/wrong length/);
  await reject(editIndex(i=>{i.counts.valid=891;}),/index count valid/);
  await reject(editIndex(i=>{i.attempts.pop();}),/896 rows/);
  await reject(editIndex(i=>{[i.attempts[0],i.attempts[1]]=[i.attempts[1],i.attempts[0]];}),/slot order/);
  await reject(editIndex(i=>{i.attempts[214][1]='valid';}),/slot 214/);
  await reject(editIndex(i=>{i.schema_version=1;}),/schema 2/);
  await reject(editIndex(i=>{i.shared.run_id_template='supplement_{method}_{model}_{culture}_{language}_s{seed}';}),/identity is invalid/);
  await reject(editIndex(i=>{const e=i.attempts.findIndex(r=>r[1]==='empty');i.attempts[e][1]='valid';}),/row is invalid/);
  const metrics=JSON.parse(publicFile('graphs/metrics.json'));metrics.rows[0][5]=0.5;
  const tampered=server({'graphs/metrics.json':JSON.stringify(metrics)}).fetchImpl;
  await assert.rejects(loadMetrics(await loadPrimaryDataset({fetchImpl:tampered}),{fetchImpl:tampered}),/metrics\.json does not match its SHA-256/);
  const supplements=JSON.parse(publicFile('graphs/supplements.json'));
  for(const mutate of [s=>{s.graphs[0].run_id=fixture.runs[0].run_id;},s=>{s.graphs.pop();},s=>{s.note='Supplements';},s=>{s.graphs[0].edges[0]=['1','0'];},s=>{s.graphs[0].edges[0]=['0','50'];}]){
    const copy=structuredClone(supplements);mutate(copy);
    await assert.rejects(loadSupplements({fetchImpl:server({'graphs/supplements.json':JSON.stringify(copy)}).fetchImpl}),/integrity/);
  }
});

test('generator parsers refuse malformed sources instead of repairing them',()=>{
  const nodes=n=>Array.from({length:n},(_,i)=>String(i));
  assert.deepEqual(parseAdjacency(`# header\n${['0 2 1','1','2'].join('\n')}\n`,3),[[0,1],[0,2]]);
  assert.throws(()=>parseAdjacency('0 0\n1\n',2),/self tie/);
  assert.throws(()=>parseAdjacency('0 1\n1 0\n',2),/duplicate tie/);
  assert.throws(()=>parseAdjacency('0 2\n1\n',2),/outside the roster/);
  assert.throws(()=>parseAdjacency('0 01\n1\n',2),/outside the roster/);
  assert.throws(()=>parseAdjacency(nodes(2).join('\n'),3),/expected 3/);
  assert.throws(()=>parseAdjacency('0\n0\n1\n',2),/twice/);
  assert.deepEqual(parseCsv('a,b\r\n"x, ""y""",2\n'),[{a:'x, "y"',b:'2'}]);
  assert.throws(()=>parseCsv('a,b\n1\n'),/1 fields/);
  assert.throws(()=>parseCsv('a,b\n"1,2\n'),/unterminated/);
  assert.equal(pythonJson({b:[1,'x'],a:null}),'{"a": null, "b": [1, "x"]}');
  assert.throws(()=>pythonJson({a:1.5}),/canonical/);
});
