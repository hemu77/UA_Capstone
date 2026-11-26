// Compact primary-study loader (docs/qa/viewer-3d-2026-11-25/SPEC.md, section C.3).
// Reads ./data/graphs/ written by build-primary-graphs.mjs: index.json and edges.bin at start,
// metrics.json and supplements.json only on request. Every file is checked; any mismatch throws and
// no substitute data is ever loaded. The returned Dataset is a superset of what validateData() accepts.
import {validateData} from './graph-data.js';
// Study facts. Tests pin these; a mismatch with the loaded data must throw, never be papered over.
export const COUNTS=Object.freeze({attempts:896,valid:890,empty:12,failures:6,supplements_excluded:6,personas:50,cells:112,repetitions:8});
export const settingKey=(culture,language)=>`${culture}-${language}`;
// Slot = position of an attempt in the fixed 4 x 4 x 7 x 8 design; addresses all 896 attempts.
export function slotOf(design,{model,method,culture,language,seed}) {
  const m=design.models.indexOf(model),p=design.methods.indexOf(method),s=design.settings.findIndex(([c,l])=>c===culture&&l===language),b=design.seeds.indexOf(Number(seed));
  if([m,p,s,b].some(i=>i<0))throw new Error('Attempt is outside the primary design.');
  return ((m*design.methods.length+p)*design.settings.length+s)*design.seeds.length+b;
}
const INDEX_COLUMNS=['slot','status','edges','density','avg_clustering_coef','adj_sha256','record_sha256','events_sha256','failure'];
const METRIC_COLUMNS=['density','avg_clustering_coef','prop_nodes_lcc','global_efficiency','num_components','modularity','radius_lcc','diameter_lcc',
  'avg_shortest_path_lcc','degree_one_share','isolate_share','degree_gini','coleman_gender','coleman_religion','coleman_political orientation','age_assortativity'];
// Coleman indices double as the per-run `homophily` object the comparison tables read.
const HOMOPHILY=[['gender','coleman_gender'],['religion','coleman_religion'],['political orientation','coleman_political orientation']];
const PRIMARY_PREFIX='final_060dcd3a86d5_main_';
const ID_TEXT=Array.from({length:256},(_,i)=>String(i));
const HEX=/^[0-9a-f]{64}$/;
const integrity=detail=>new Error(`Primary data failed its integrity check: ${detail}. No substitute data is loaded.`);
const check=(ok,detail)=>{if(!ok)throw integrity(detail);};
const fill=(template,values)=>template.replace(/\{(\w+)\}/g,(_,key)=>String(values[key]));
const graphsUrl=(base,file)=>`${base}graphs/${file}`;

// Revalidated with the host on each load (cheap 304s), so a redeploy never mixes old and new files.
async function request(fetchImpl,url) {
  const response=await fetchImpl(url,{cache:'no-cache'});
  if(!response.ok)throw new Error(`Data request failed: ${response.status}. No substitute data is loaded.`);
  return new Uint8Array(await response.arrayBuffer());
}
// JSON is read as bytes so its SHA-256 can be checked before parsing.
const requestJson=async(fetchImpl,url,sha256)=>{
  const bytes=await request(fetchImpl,url);
  if(sha256)check(await digest(bytes)===sha256,`${url.split('/').pop()} does not match its SHA-256`);
  return JSON.parse(new TextDecoder().decode(bytes));
};
async function digest(bytes) {
  if(!globalThis.crypto?.subtle)throw new Error('This browser cannot verify the study data (Web Crypto is unavailable). Open the viewer over https or from localhost.');
  return [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
}

// Ties of one graph from edges.bin: `offsetEdges` and `countEdges` count ties (2 bytes each), not bytes.
// Returns [[a,b],...] with string ids, numeric a < b, sorted by a then b; anything else throws.
export function decodeEdges(bytes,offsetEdges,countEdges) {
  if(!Number.isInteger(offsetEdges)||!Number.isInteger(countEdges)||offsetEdges<0||countEdges<0||2*(offsetEdges+countEdges)>bytes.length)throw integrity('edge range lies outside edges.bin');
  const edges=new Array(countEdges);
  let previous=-1;
  for(let i=0;i<countEdges;i++){
    const a=bytes[2*(offsetEdges+i)],b=bytes[2*(offsetEdges+i)+1],order=a*256+b;
    check(a<b&&order>previous,'ties are not ordered pairs a < b in ascending order');
    previous=order;edges[i]=[ID_TEXT[a],ID_TEXT[b]];
  }
  return edges;
}

function validateIndex(index) {
  check(index&&index.schema_version===2&&index.dataset==='primary','index.json is not the primary schema 2 index');
  for(const [key,value] of Object.entries(COUNTS))check(index.counts?.[key]===value,`index count ${key} is not ${value}`);
  const {models,methods,settings,seeds}=index.design||{};
  check([models,methods,settings,seeds].every(Array.isArray)&&models.length*methods.length*settings.length===COUNTS.cells&&seeds.length===COUNTS.repetitions,'design is not 4 x 4 x 7 x 8');
  check(settings.every(s=>Array.isArray(s)&&s.length===2)&&new Set(settings.map(([c,l])=>settingKey(c,l))).size===settings.length,'settings are not distinct culture/language pairs');
  check(JSON.stringify(index.columns)===JSON.stringify(INDEX_COLUMNS),'index columns differ');
  check(Array.isArray(index.attempts)&&index.attempts.length===COUNTS.attempts&&index.attempts.every((row,slot)=>Array.isArray(row)&&row.length===INDEX_COLUMNS.length&&row[0]===slot),'attempt rows are not 896 rows in slot order');
  check(Array.isArray(index.personas)&&index.personas.length===COUNTS.personas&&index.personas.every((p,i)=>p.id===String(i)),'roster is not personas 0-49');
  check(Array.isArray(index.layout?.xy)&&index.layout.xy.length===2*COUNTS.personas&&index.layout.xy.every(Number.isFinite),'atlas layout is incomplete');
  check(index.edges?.file==='edges.bin'&&HEX.test(index.edges.sha256)&&index.metrics?.file==='metrics.json'&&HEX.test(index.metrics.sha256),'data file references are invalid');
  check(HEX.test(index.roster_id)&&index.shared&&index.notes,'roster identity or shared fields missing');
}

// Primary dataset from ./data/graphs/: 890 runs with edges, 6 failures, 896 slots, roster and atlas layout.
export async function loadPrimaryDataset({fetchImpl=globalThis.fetch,base='./data/'}={}) {
  const index=await requestJson(fetchImpl,graphsUrl(base,'index.json'));
  validateIndex(index);
  const bytes=await request(fetchImpl,graphsUrl(base,index.edges.file));
  check(bytes.length===index.edges.bytes,'edges.bin has the wrong length');
  check(await digest(bytes)===index.edges.sha256,'edges.bin does not match its SHA-256');

  const {design,shared}=index,runs=[],failures=[],slots=[],offsets=new Uint32Array(COUNTS.attempts+1);
  let offset=0;
  for(const [slot,status,count,density,clustering,adj,record,events,failure] of index.attempts){
    offsets[slot]=offset;
    const s=slot%design.seeds.length,setting=Math.floor(slot/design.seeds.length)%design.settings.length,
      method=Math.floor(slot/(design.seeds.length*design.settings.length))%design.methods.length,model=Math.floor(slot/(design.seeds.length*design.settings.length*design.methods.length));
    const [culture,language]=design.settings[setting];
    const identity={model:design.models[model],method:design.methods[method],culture,language,seed:design.seeds[s]};
    const run_id=fill(shared.run_id_template,identity);
    check(run_id.startsWith(PRIMARY_PREFIX)&&HEX.test(record),`slot ${slot} identity is invalid`);
    if(status==='failed'){
      check(count===0&&typeof failure==='string'&&failure&&adj===null&&events===null,`failed slot ${slot} carries graph data`);
      failures.push({run_id,...identity,failure,slot,setting:settingKey(culture,language),record_sha256:record});
      slots.push({slot,status,run_id,failure});
      continue;
    }
    check((status==='valid'&&count>0||status==='empty'&&count===0)&&failure===null&&HEX.test(adj)&&HEX.test(events)&&Number.isFinite(density)&&Number.isFinite(clustering),`slot ${slot} row is invalid`);
    runs.push({run_id,...identity,study:shared.study,status:'valid',roster_id:index.roster_id,prompt_variant:shared.prompt_variant,
      source:fill(shared.source_template,{run_id}),source_sha256:adj,record_sha256:record,protocol_sha256:shared.protocol_sha256,
      metrics:{density,avg_clustering_coef:clustering},homophily:{},analysis_version:shared.analysis_version,
      edges:decodeEdges(bytes,offset,count),events:[],events_url:`${base}primary-events/${run_id}.json.gz`,events_sha256:events,
      replay_status:shared.replay_status,slot,setting:settingKey(culture,language)});
    slots.push({slot,status,run_id});
    offset+=count;
  }
  offsets[COUNTS.attempts]=offset;
  check(2*offset===bytes.length,'edges.bin length differs from the tie counts');
  check(runs.length===COUNTS.valid&&runs.filter(r=>!r.edges.length).length===COUNTS.empty&&failures.length===COUNTS.failures,'population differs from 896 attempts / 890 valid / 12 empty / 6 failures');
  check(new Set(slots.map(s=>s.run_id)).size===COUNTS.attempts,'run ids are not unique');

  // `position` is the legacy roster circle that validateData() requires; layouts are computed elsewhere.
  const personas=index.personas.map(({id,...attributes},i)=>{const angle=2*Math.PI*i/COUNTS.personas;return {id,attributes,position:[75*Math.cos(angle),0,75*Math.sin(angle)]};});
  const xy=index.layout.xy;
  const data=validateData({
    schema_version:1, // shape version of the in-memory dataset accepted by validateData(); index.json itself is schema 2
    dataset:'primary',roster_id:index.roster_id,personas,runs,failures,
    models:design.models,methods:design.methods,settings:design.settings,
    planned_networks:COUNTS.attempts,planned_repetitions:COUNTS.repetitions,
    ...index.notes,
    design,counts:COUNTS,slots,
    layout:{basis:index.layout.basis,xy,points:new Map(personas.map(({id},i)=>[id,[xy[2*i],xy[2*i+1]]]))},
    edgeData:{bytes,offsets},
    files:{base,metrics:index.metrics,supplements:index.supplements},
    source:index.source
  });
  return data;
}

// Fills run.metrics (all 16 saved metrics) and run.homophily for every run. Fetches once per dataset;
// later calls return the same promise. Values are the saved CSV values; nothing is recomputed.
const metricLoads=new WeakMap();
export function loadMetrics(dataset,{fetchImpl=globalThis.fetch,base=dataset.files.base}={}) {
  if(!metricLoads.has(dataset)){
    const load=(async()=>{
      const file=await requestJson(fetchImpl,graphsUrl(base,dataset.files.metrics.file),dataset.files.metrics.sha256);
      check(file.schema_version===2&&JSON.stringify(file.columns)===JSON.stringify(METRIC_COLUMNS),'metrics.json columns differ');
      const runs=[...dataset.runs].sort((a,b)=>a.slot-b.slot);
      check(Array.isArray(file.rows)&&file.rows.length===runs.length,'metrics.json does not have one row per valid graph');
      const rows=file.rows.map((row,i)=>{
        check(Array.isArray(row)&&row.length===METRIC_COLUMNS.length&&row.every(v=>v===null||Number.isFinite(v)),`metrics row ${i} is invalid`);
        check(row[0]===runs[i].metrics.density&&row[1]===runs[i].metrics.avg_clustering_coef,`metrics row ${i} disagrees with index.json`);
        return row;
      });
      runs.forEach((run,i)=>{
        METRIC_COLUMNS.forEach((column,k)=>{run.metrics[column]=rows[i][k];});
        for(const [attribute,column] of HOMOPHILY)run.homophily[attribute]=run.metrics[column];
      });
      return dataset;
    })();
    metricLoads.set(dataset,load);
    load.catch(()=>metricLoads.delete(dataset)); // a failed request may be retried
  }
  return metricLoads.get(dataset);
}

// The 6 supplemental graphs, for the separately labelled panel only. They are never part of a Dataset.
export async function loadSupplements({fetchImpl=globalThis.fetch,base='./data/'}={}) {
  const file=await requestJson(fetchImpl,graphsUrl(base,'supplements.json'));
  check(file.schema_version===1&&typeof file.note==='string'&&file.note.startsWith('Excluded from the primary study'),'supplements.json is not labelled as excluded');
  check(Array.isArray(file.graphs)&&file.graphs.length===COUNTS.supplements_excluded,`supplements.json does not hold ${COUNTS.supplements_excluded} graphs`);
  const ids=new Set();
  for(const graph of file.graphs){
    check(typeof graph.run_id==='string'&&!graph.run_id.startsWith(PRIMARY_PREFIX)&&!ids.has(graph.run_id),'a supplement reuses a primary or duplicate run id');
    ids.add(graph.run_id);
    check(HEX.test(graph.source_sha256)&&Array.isArray(graph.edges),`supplement ${graph.run_id} is incomplete`);
    let previous=-1;
    for(const edge of graph.edges){
      const [a,b]=edge.map(Number);
      check(edge.length===2&&edge.every((id,k)=>id===ID_TEXT[k?b:a])&&a<b&&b<COUNTS.personas&&a*256+b>previous,`supplement ${graph.run_id} has an invalid tie`);
      previous=a*256+b;
    }
  }
  return file;
}
