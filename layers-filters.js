// WORKSTREAM 3 (filters). Contract: docs/qa/viewer-3d-2026-11-25/SPEC.md sections B.4 and C.5.
// One filter system for the 890 valid primary graphs: live multi-select facets, an always-visible
// active-filter bar that also states what is on stage, a deduplicated virtualised result list, and a
// short readable URL grammar that still opens legacy ?view=<json> links.
// The pure functions at the top are DOM-free so node:test can import this file; mountFilters() is the
// only part that touches the document, and it talks to the rest of the page only through the store.
import {EMPTY_FILTERS,GRAPH_STATES,VIEWS,LENSES,MAX_STACK_LAYERS,LAYER_COLORS,INITIAL_STATE} from './layers-app.js?v=d068babb8c4c';
import {evidenceDataset} from './graph-data.js?v=d068babb8c4c';
import {countryNames,runLabel,layerLabel,modelName} from './formation-charts.js?v=d068babb8c4c';
import {replayStart} from './layers-replay.js?v=d068babb8c4c';

export const FACETS=Object.freeze(['model','method','setting','seed']);
export const FACET_NAMES=Object.freeze({model:'Model',method:'Method',setting:'Setting',seed:'Seed',graph:'Graph',search:'Search'});
const FACET_TITLES=Object.freeze({model:'Model',method:'Method',setting:'Country framing / language',seed:'Seed (repetition)',graph:'Graph'});
const GRAPH_LABELS=Object.freeze({any:'Any',nonempty:'With ties',empty:'Empty'});
const GRAPH_CHIPS=Object.freeze({nonempty:'With ties only',empty:'Empty graphs only'});
// Today's wording in layers.js; kept identical so old links fail the same way everywhere.
export const ARCHIVED_LINK_MESSAGE='This link selects archived research. The public demo now shows only the final primary study. Open the default comparison; historical files remain in the repository archive.';
export const ABSENT_RUN_MESSAGE='The saved selection contains a run absent from this dataset. Open the default comparison to choose again.';
export const SORTS=Object.freeze([['design','Design order'],['ties-desc','Most ties'],['ties-asc','Fewest ties'],['density-desc','Highest density'],['density-asc','Lowest density']]);
const SOURCE='layers-filters';
const SHORT_KEYS=Object.freeze(['v','m','p','s','b','g','q','l','a','n','c','x','i','f','k']);
const MAX_SEARCH=120;
// Captured when the module is first evaluated, i.e. before layers.js rewrites the address bar.
const INITIAL_SEARCH=(()=>{try{return globalThis.location?.search||'';}catch{return '';}})();

const capital=text=>text?text[0].toUpperCase()+text.slice(1):text;
const plural=(n,one,many=`${one}s`)=>`${n.toLocaleString('en-US')} ${n===1?one:many}`;
const settingOf=run=>run.setting||`${run.culture}-${run.language}`;
const splitSetting=key=>{const at=String(key).indexOf('-');return at<0?[String(key),'']:[key.slice(0,at),key.slice(at+1)];};
const valueOf=(run,facet)=>facet==='setting'?settingOf(run):facet==='seed'?Number(run.seed):run[facet];
const isEmptyRun=run=>run.edges.length===0;

// A run belongs to the primary set only if it is a saved valid graph with a primary run_id.
// Supplements (supp*_...) and failures (no edges, failure code) can never enter results or counts.
export function isPrimaryRun(run) {
  return !!run&&typeof run==='object'&&Array.isArray(run.edges)&&!run.failure&&evidenceDataset(null,[String(run.run_id)])==='primary';
}

export function facetValues(dataset,facet) {
  const design=dataset.design;
  if(facet==='model')return [...design.models];
  if(facet==='method')return [...design.methods];
  if(facet==='setting')return design.settings.map(([culture,language])=>`${culture}-${language}`);
  if(facet==='seed')return design.seeds.map(Number);
  if(facet==='graph')return [...GRAPH_STATES];
  throw new Error(`Unknown filter facet: ${facet}`);
}

export function valueLabel(facet,value) {
  if(facet==='model')return modelName(String(value));
  if(facet==='method')return capital(String(value));
  if(facet==='setting'){const [culture,language]=splitSetting(value);return `${countryNames[culture]||capital(culture)} / ${capital(language)}`;}
  if(facet==='graph')return GRAPH_LABELS[value]||String(value);
  return String(value);
}

// Canonical filters: known facets only, unique values in design order, seeds as numbers.
export function normalizeFilters(filters=EMPTY_FILTERS,dataset=null) {
  const source=filters&&typeof filters==='object'?filters:EMPTY_FILTERS,out={};
  for(const facet of FACETS){
    let values=Array.isArray(source[facet])?source[facet]:[];
    values=facet==='seed'?values.map(Number).filter(Number.isSafeInteger):values.map(String);
    if(dataset){const known=facetValues(dataset,facet);out[facet]=known.filter(value=>values.includes(value));}
    else out[facet]=[...new Set(values)];
  }
  out.graph=GRAPH_STATES.includes(source.graph)?source.graph:'any';
  out.search=typeof source.search==='string'?source.search.replace(/\s+/g,' ').trim().slice(0,MAX_SEARCH):'';
  return out;
}

export function filtersActive(filters=EMPTY_FILTERS) {
  const f=normalizeFilters(filters);
  return FACETS.some(facet=>f[facet].length>0)||f.graph!=='any'||f.search!=='';
}

// Whole-word search over model, method, country name, language, seed, setting and run_id.
const wordCache=new WeakMap();
function wordsOf(record) {
  let words=wordCache.get(record);
  if(words)return words;
  words=new Set();
  const add=value=>{
    const text=String(value??'').toLowerCase().trim();if(!text)return;
    words.add(text);
    for(const part of text.split(/[^a-z0-9.]+/))if(part)words.add(part);
  };
  for(const value of [record.model,record.method,record.culture,countryNames[record.culture],record.language,record.seed,`${record.culture}-${record.language}`,record.run_id])add(value);
  for(const part of String(record.run_id||'').toLowerCase().split('_'))if(part)words.add(part);
  wordCache.set(record,words);
  return words;
}
export function searchTokens(search='') {
  return String(search).toLowerCase().split(/[\s,;|/·]+/).map(token=>token.replace(/^["'“”]+|["'“”]+$/g,'')).filter(Boolean);
}
export function matchesSearch(record,search) {
  const tokens=searchTokens(search);
  if(!tokens.length)return true;
  const words=wordsOf(record);
  return tokens.every(token=>words.has(token));
}

function compile(filters) {
  const f=normalizeFilters(filters);
  return {f,sets:Object.fromEntries(FACETS.map(facet=>[facet,f[facet].length?new Set(f[facet]):null])),tokens:searchTokens(f.search)};
}
function facetHits(record,compiled) {
  return FACETS.map(facet=>{const set=compiled.sets[facet];return !set||set.has(valueOf(record,facet));});
}
function searchHit(record,compiled) {
  if(!compiled.tokens.length)return true;
  const words=wordsOf(record);return compiled.tokens.every(token=>words.has(token));
}
const graphHit=(run,graph)=>graph==='any'||(graph==='empty')===isEmptyRun(run);

// Valid primary graphs matching the filters, in the order given. Facet values are OR, facets are AND.
export function applyFilters(runs,filters=EMPTY_FILTERS) {
  const compiled=compile(filters);
  return (runs||[]).filter(run=>isPrimaryRun(run)&&graphHit(run,compiled.f.graph)&&searchHit(run,compiled)&&facetHits(run,compiled).every(Boolean));
}

// For each facet value: how many graphs would match if that value were the facet's only choice,
// holding every other active facet, the graph state and the search fixed.
export function facetCounts(dataset,filters=EMPTY_FILTERS) {
  const compiled=compile(filters),out={graph:{nonempty:0,empty:0}};
  for(const facet of FACETS)out[facet]=Object.fromEntries(facetValues(dataset,facet).map(value=>[value,0]));
  for(const run of dataset.runs){
    if(!isPrimaryRun(run)||!searchHit(run,compiled))continue;
    const hits=facetHits(run,compiled),graph=graphHit(run,compiled.f.graph);
    FACETS.forEach((facet,index)=>{
      if(!graph||!hits.every((hit,other)=>other===index||hit))return;
      const value=valueOf(run,facet);
      if(Object.hasOwn(out[facet],value))out[facet][value]++;
    });
    if(hits.every(Boolean))out.graph[isEmptyRun(run)?'empty':'nonempty']++;
  }
  return out;
}

// Failed attempts whose design cell is inside the filters. Failures have no graph, so a graph-state
// filter (with ties / empty) always excludes them.
export function failuresInScope(dataset,filters=EMPTY_FILTERS) {
  const compiled=compile(filters);
  if(compiled.f.graph!=='any')return 0;
  return (dataset.failures||[]).filter(failure=>searchHit(failure,compiled)&&facetHits(failure,compiled).every(Boolean)).length;
}

// 896-slot mask for the atlas: 0 outside the filters, 1 matching graph, 2 failed attempt in scope.
export function slotMask(dataset,filters=EMPTY_FILTERS) {
  const mask=new Uint8Array(dataset.slots.length);
  for(const run of applyFilters(dataset.runs,filters))mask[run.slot]=1;
  const compiled=compile(filters);
  if(compiled.f.graph==='any')for(const slot of dataset.slots){
    if(slot?.status!=='failed')continue;
    const failure=(dataset.failures||[]).find(item=>item.run_id===slot.run_id);
    if(failure&&searchHit(failure,compiled)&&facetHits(failure,compiled).every(Boolean))mask[slot.slot]=2;
  }
  return mask;
}

export function activeChips(dataset,filters=EMPTY_FILTERS) {
  const f=normalizeFilters(filters,dataset),chips=[];
  for(const facet of FACETS)for(const value of f[facet])chips.push({facet,value,label:`${FACET_NAMES[facet]}: ${valueLabel(facet,value)}`});
  if(f.graph!=='any')chips.push({facet:'graph',value:f.graph,label:`${FACET_NAMES.graph}: ${GRAPH_CHIPS[f.graph]}`});
  if(f.search)chips.push({facet:'search',value:f.search,label:`${FACET_NAMES.search}: “${f.search}”`});
  return chips;
}

export function withoutChip(filters,chip) {
  const f=normalizeFilters(filters);
  if(chip.facet==='graph')return {...f,graph:'any'};
  if(chip.facet==='search')return {...f,search:''};
  return {...f,[chip.facet]:f[chip.facet].filter(value=>value!==chip.value)};
}
export function toggleValue(filters,facet,value,dataset=null) {
  const f=normalizeFilters(filters,dataset);
  if(facet==='graph')return {...f,graph:f.graph===value?'any':value};
  const has=f[facet].includes(value);
  return normalizeFilters({...f,[facet]:has?f[facet].filter(item=>item!==value):[...f[facet],value]},dataset);
}

export function sortRuns(runs,sort='design') {
  const ties=run=>run.edges.length,density=run=>Number.isFinite(run.metrics?.density)?run.metrics.density:-1;
  const by={
    'design':(a,b)=>a.slot-b.slot,
    'ties-desc':(a,b)=>ties(b)-ties(a)||a.slot-b.slot,
    'ties-asc':(a,b)=>ties(a)-ties(b)||a.slot-b.slot,
    'density-desc':(a,b)=>density(b)-density(a)||a.slot-b.slot,
    'density-asc':(a,b)=>density(a)-density(b)||a.slot-b.slot
  }[sort]||((a,b)=>a.slot-b.slot);
  // One row per run_id, whatever the input contains.
  const seen=new Set();
  return runs.filter(run=>!seen.has(run.run_id)&&seen.add(run.run_id)).sort(by);
}

// One sentence for the bar; the same numbers drive the result list, so they cannot disagree.
export function scopeSummary(dataset,filters=EMPTY_FILTERS) {
  const counts=dataset.counts,valid=dataset.runs.filter(isPrimaryRun);
  if(!filtersActive(filters))return {active:false,shown:valid.length,text:`All ${counts.valid} valid graphs · no filters set`};
  const shown=applyFilters(valid,filters),empty=shown.filter(isEmptyRun).length,failed=failuresInScope(dataset,filters);
  const parts=[`Showing ${shown.length} of ${counts.valid} valid graphs`];
  if(empty)parts.push(`${empty} empty`);
  if(failed)parts.push(`${plural(failed,'failed attempt')} in scope (not rendered)`);
  return {active:true,shown:shown.length,empty,failed,text:parts.join(' · ')};
}

// --- URL grammar (B.4) -------------------------------------------------------
const indexCache=new WeakMap();
function runIndex(dataset) {
  let index=indexCache.get(dataset);
  if(!index){index=new Map(dataset.runs.map(run=>[run.run_id,run]));indexCache.set(dataset,index);}
  return index;
}
const attributeKeys=dataset=>Object.keys(dataset.personas?.[0]?.attributes||{});

export function encodeState(state,dataset) {
  const params=new URLSearchParams(),f=normalizeFilters(state.filters||EMPTY_FILTERS,dataset),index=runIndex(dataset);
  const slotOfId=id=>{const run=index.get(id);if(!run)throw new Error(`Run ${id} is not in the primary dataset.`);return run.slot;};
  if(VIEWS.includes(state.view))params.set('v',state.view);
  for(const [key,facet] of [['m','model'],['p','method'],['s','setting'],['b','seed']])if(f[facet].length)params.set(key,f[facet].join(','));
  if(f.graph!=='any')params.set('g',f.graph);
  if(f.search)params.set('q',f.search);
  const layers=[...new Set(state.layers||[])];
  if(layers.length)params.set('l',layers.map(slotOfId).join('.'));
  if(state.active&&index.has(state.active))params.set('a',String(slotOfId(state.active)));
  if(state.persona)params.set('n',String(state.persona));
  if(state.colorBy&&state.colorBy!==INITIAL_STATE.colorBy)params.set('c',state.colorBy);
  if(state.lens&&state.lens!=='none')params.set('x',state.lens);
  if(state.groups?.length)params.set('i',state.groups.join(','));
  if(state.consensus&&state.view==='stack')params.set('k',`${state.consensus.scope==='stack'?'s':'f'}${Math.round(state.consensus.min*100)}`);
  if(state.view==='replay'&&Number.isSafeInteger(state.frame)&&state.frame>=0&&state.frame<Number.MAX_SAFE_INTEGER)params.set('f',String(state.frame));
  return params;
}
// Readable form of encodeState: commas and dots stay literal, spaces become '+'.
export function stateQuery(state,dataset) {
  return [...encodeState(state,dataset)].map(([key,value])=>`${key}=${encodeURIComponent(value).replace(/%2C/gi,',').replace(/%20/g,'+')}`).join('&');
}

function linkError(message){const error=new Error(message);error.name='LinkError';return error;}
function parseList(raw,known,what) {
  const values=String(raw).split(',').map(value=>value.trim()).filter(Boolean);
  for(const value of values)if(!known.includes(value))throw linkError(`This link names an unknown ${what} “${value}”.`);
  return known.filter(value=>values.includes(value));
}
function runAtSlot(raw,dataset) {
  if(!/^\d{1,4}$/.test(raw))throw linkError(`“${raw}” is not a slot number (0–${dataset.slots.length-1}).`);
  const number=Number(raw),slot=dataset.slots[number];
  if(!slot)throw linkError(`Slot ${number} is outside the ${dataset.slots.length} planned attempts.`);
  if(slot.status==='failed')throw linkError(`Slot ${number} is a failed attempt (${slot.failure||'no graph saved'}); failed attempts have no graph to show.`);
  const run=runIndex(dataset).get(slot.run_id);
  if(!run||!isPrimaryRun(run))throw linkError(`Slot ${number} has no valid primary graph.`);
  return run;
}

// Returns a partial state. Throws a LinkError with a readable message on anything it cannot honour,
// so the caller can show the message and keep the default view instead of a silent substitute.
export function decodeState(params,dataset) {
  const p=params instanceof URLSearchParams?params:new URLSearchParams(String(params||''));
  const requested=p.get('dataset');
  if(requested&&requested!=='primary')throw linkError(ARCHIVED_LINK_MESSAGE);
  if(!SHORT_KEYS.some(key=>p.has(key)))return p.has('view')?decodeLegacy(p.get('view'),dataset):{};
  const out={},filters={...normalizeFilters(EMPTY_FILTERS)};
  const lists=[['m','model','model'],['p','method','method'],['s','setting','setting'],['b','seed','seed']];
  for(const [key,facet,what] of lists){
    if(!p.has(key))continue;
    const known=facetValues(dataset,facet).map(String);
    filters[facet]=parseList(p.get(key),known,what);
    if(facet==='seed')filters.seed=filters.seed.map(Number);
  }
  if(p.has('g')){const graph=p.get('g')||'any';if(!GRAPH_STATES.includes(graph))throw linkError(`This link names an unknown graph state “${graph}”.`);filters.graph=graph;}
  if(p.has('q'))filters.search=normalizeFilters({search:p.get('q')}).search;
  out.filters=normalizeFilters(filters,dataset);
  if(p.has('v')){const view=p.get('v');if(!VIEWS.includes(view))throw linkError(`This link names an unknown view “${view}”.`);out.view=view;}
  let layers=null;
  if(p.has('l')&&p.get('l')!==''){
    const runs=[...new Set(p.get('l').split('.'))].map(raw=>runAtSlot(raw,dataset));
    if(runs.length>MAX_STACK_LAYERS)throw linkError(`This link stacks ${runs.length} runs; the stack holds at most ${MAX_STACK_LAYERS}.`);
    layers=runs.map(run=>run.run_id);
  }
  if(p.has('a')&&p.get('a')!==''){
    const active=runAtSlot(p.get('a'),dataset).run_id;
    if(layers&&!layers.includes(active))throw linkError(`The active run (slot ${p.get('a')}) is not in the linked stack.`);
    if(!layers)layers=[active];
    out.active=active;
  }
  if(layers){out.layers=layers;if(!out.active)out.active=layers[0];}
  const persona=p.get('n')||'';
  if(persona&&!dataset.personas.some(person=>String(person.id)===persona))throw linkError(`This link names an unknown persona “${persona}”.`);
  out.persona=persona;
  const color=p.get('c')||INITIAL_STATE.colorBy;
  if(color!=='layer'&&!attributeKeys(dataset).includes(color))throw linkError(`This link colours by an unknown attribute “${color}”.`);
  out.colorBy=color;
  const lens=p.get('x')||'none';
  if(!LENSES.includes(lens))throw linkError(`This link names an unknown lens “${lens}”.`);
  out.lens=lens;
  out.groups=[...new Set((p.get('i')||'').split(',').map(value=>value.trim()).filter(Boolean))];
  if(p.has('k')&&p.get('k')!==''){
    const match=/^([sf])(\d{1,3})$/.exec(p.get('k')),share=match?Number(match[2]):0;
    if(!match||share<1||share>100)throw linkError(`“${p.get('k')}” is not an aggregate plane setting (for example f50: filtered graphs, ties in at least 50%).`);
    out.consensus={scope:match[1]==='s'?'stack':'filters',min:share/100};
  }
  if(p.has('f')&&p.get('f')!==''){
    if(!/^\d{1,9}$/.test(p.get('f')))throw linkError(`“${p.get('f')}” is not a recorded update number.`);
    out.frame=Number(p.get('f'));
  }
  return out;
}

// Legacy ?view=<json> written by the deployed site before 2026-11-25.
function decodeLegacy(raw,dataset) {
  let saved;try{saved=JSON.parse(raw||'null');}catch{return {};}
  if(!saved||typeof saved!=='object'||Array.isArray(saved)||!Array.isArray(saved.runs))return {};
  if(evidenceDataset(null,saved.runs)!=='primary')throw linkError(ARCHIVED_LINK_MESSAGE);
  const index=runIndex(dataset),ids=[...new Set(saved.runs.map(String))];
  if(ids.some(id=>!index.has(id)))throw linkError(ABSENT_RUN_MESSAGE);
  if(ids.length>MAX_STACK_LAYERS)throw linkError(`This link stacks ${ids.length} runs; the stack holds at most ${MAX_STACK_LAYERS}.`);
  const out={};
  if(ids.length){out.layers=ids;out.active=ids.includes(saved.active)?saved.active:ids[0];}
  out.view=saved.mode==='formation'?'replay':'stack';
  if(Number.isSafeInteger(saved.frame)&&saved.frame>=0)out.frame=saved.frame;
  if(typeof saved.person==='string'&&(saved.person===''||dataset.personas.some(person=>String(person.id)===saved.person)))out.persona=saved.person;
  if(typeof saved.color==='string'&&(saved.color==='layer'||attributeKeys(dataset).includes(saved.color)))out.colorBy=saved.color;
  const old=saved.filters&&typeof saved.filters==='object'?saved.filters:{},filters={...normalizeFilters(EMPTY_FILTERS)};
  if(old.model&&dataset.design.models.includes(String(old.model)))filters.model=[String(old.model)];
  if(old.method&&dataset.design.methods.includes(String(old.method)))filters.method=[String(old.method)];
  if(old.seed!==undefined&&old.seed!==''&&dataset.design.seeds.includes(Number(old.seed)))filters.seed=[Number(old.seed)];
  if(old.culture||old.language)filters.setting=dataset.design.settings.filter(([culture,language])=>(!old.culture||culture===String(old.culture))&&(!old.language||language===String(old.language))).map(([culture,language])=>`${culture}-${language}`);
  out.filters=normalizeFilters(filters,dataset);
  return out;
}

// Replay a run that is already stacked: one update carrying both the active run and the view.
export function replayRun(app,id,source=SOURCE) {
  const s=app.getState();
  if(!s.layers.includes(id))throw new Error('Only a stacked run can be replayed; add it to the stack first.');
  // Replay opens at the start of the record (event 0); a single-response run opens on its saved graph.
  app.update({active:id,view:'replay',frame:replayStart(app.data?.runs.find(run=>run.run_id===id))},source);
}

// Keeps the address bar in the short grammar. Enabled by mountFilters({writeUrl:true}) once layers.js
// no longer writes ?view= itself (SPEC C.5: WS3 owns URL writing after integration).
export function syncUrl(app,{history=globalThis.history,location=globalThis.location,delay=250}={}) {
  const keys=['filters','view','layers','active','frame','persona','colorBy','lens','groups','consensus'];
  let timer=null;
  const write=()=>{
    timer=null;
    if(!app.data)return;
    try{
      const url=new URL(location.href),query=stateQuery(app.getState(),app.data);
      url.search=query?`?${query}`:'';
      if(url.href!==location.href)history.replaceState(history.state,'',url.href);
    }catch(error){console.warn('Address bar not updated:',error.message);}
  };
  const stop=app.subscribe((state,changed)=>{if(keys.some(key=>changed.has(key))){clearTimeout(timer);timer=setTimeout(write,delay);}});
  return ()=>{clearTimeout(timer);stop();};
}

// --- DOM ---------------------------------------------------------------------
function h(tag,attributes={},...children) {
  const node=document.createElement(tag);
  for(const [key,value] of Object.entries(attributes)){
    if(value===null||value===undefined||value===false)continue;
    if(key==='className')node.className=value;
    else if(key==='text')node.textContent=value;
    else if(key==='dataset')Object.assign(node.dataset,value);
    else if(key==='style')for(const [name,v] of Object.entries(value))node.style.setProperty(name,v);
    else if(key.startsWith('on'))node.addEventListener(key.slice(2),value);
    else node.setAttribute(key,value===true?'':String(value));
  }
  for(const child of children.flat())if(child!==null&&child!==undefined&&child!==false)node.append(child);
  return node;
}

// Builds its own DOM inside the given roots and keeps it in sync with the store.
export function mountFilters(app,{chips,root,results,search=INITIAL_SEARCH,writeUrl=false}={}) {
  let destroyed=false,teardown=[];
  const begin=()=>{
    if(destroyed)return;
    teardown=build(app,{chips,root,results,search,writeUrl});
  };
  if(app.data)begin();else app.ready.then(begin);
  return {destroy(){destroyed=true;for(const fn of teardown)fn();teardown=[];}};
}

function build(app,{chips,root,results,search,writeUrl}) {
  const data=app.data,cleanup=[],uid=`flt-${Math.random().toString(36).slice(2,8)}`;
  let sort='design',matches=[],matchIds=new Set(),counts=null,current=0,announceTimer=null,searchTimer=null;
  const state=()=>app.getState();
  const setFilters=next=>{const changed=app.update({filters:normalizeFilters(next,data)},SOURCE);if(changed.size)scheduleAnnounce();};
  const scheduleAnnounce=()=>{clearTimeout(announceTimer);announceTimer=setTimeout(()=>app.announce(scopeSummary(data,state().filters).text+'.'),450);};
  cleanup.push(()=>{clearTimeout(announceTimer);clearTimeout(searchTimer);});

  // ----- Active-filter bar --------------------------------------------------
  const bar={};
  if(chips){
    bar.summary=h('p',{className:'flt-summary',id:`${uid}-summary`,tabindex:'-1'});
    bar.list=h('ul',{className:'flt-chips','aria-label':'Active filters'});
    bar.reset=h('button',{type:'button',className:'flt-reset',onclick:()=>{setFilters(EMPTY_FILTERS);bar.summary.focus();}},'Reset filters');
    bar.view=h('span',{className:'flt-view'});
    bar.stackTitle=h('span',{className:'flt-stack-title'});
    bar.stack=h('ul',{className:'flt-stack','aria-label':'Stacked runs'});
    bar.noticeText=h('span');
    bar.notice=h('p',{className:'flt-notice',hidden:true},bar.noticeText,h('button',{type:'button',className:'flt-notice-close',onclick:()=>{bar.notice.hidden=true;bar.summary.focus();}},'Dismiss'));
    bar.outside=h('span',{className:'flt-outside-note'});
    chips.replaceChildren(h('div',{className:'flt-bar'},
      h('div',{className:'flt-head'},bar.summary,bar.reset),
      h('div',{className:'flt-rail'},bar.list,
        h('div',{className:'flt-stage',role:'group','aria-label':'On stage'},h('span',{className:'flt-stage-label'},'On stage'),bar.view,bar.stackTitle,bar.stack,bar.outside))),bar.notice);
    chips.dataset.ready='true';
  }
  const chipKey=chip=>`${chip.facet}:${chip.value}`;
  function renderBar() {
    if(!chips)return;
    const s=state(),summary=scopeSummary(data,s.filters),list=activeChips(data,s.filters);
    chips.dataset.active=String(summary.active);
    bar.summary.textContent=summary.text;
    const focused=bar.list.contains(document.activeElement)?document.activeElement.closest('li')?.dataset.key:null;
    bar.list.replaceChildren(...list.map(chip=>h('li',{dataset:{key:chipKey(chip),facet:chip.facet}},
      h('button',{type:'button',className:'flt-chip','aria-label':`Remove filter ${chip.label}`,title:`Remove filter ${chip.label}`,onclick:event=>removeChip(chip,event.currentTarget)},
        h('span',{className:'flt-chip-text'},chip.label),h('span',{className:'flt-chip-x','aria-hidden':'true'},'×')))));
    bar.list.hidden=!list.length;bar.reset.hidden=!summary.active;
    if(focused)bar.list.querySelector(`li[data-key="${CSS.escape(focused)}"] button`)?.focus();
    renderStage();
  }
  function removeChip(chip,button) {
    const items=[...bar.list.querySelectorAll('button')],at=items.indexOf(button);
    setFilters(withoutChip(state().filters,chip));
    const rest=[...bar.list.querySelectorAll('button')];
    (rest[Math.min(at,rest.length-1)]||(bar.reset.hidden?bar.summary:bar.reset)).focus();
  }
  function renderStage() {
    if(!chips)return;
    const s=state(),index=runIndex(data),runs=s.layers.map(id=>index.get(id)).filter(Boolean);
    const at=runs.findIndex(run=>run.run_id===s.active);
    bar.view.textContent=s.view==='atlas'?'Atlas · every attempt slot':s.view==='replay'?`Replay · run ${at+1} of ${runs.length}`:`Stack · ${runs.length} of ${MAX_STACK_LAYERS}`;
    bar.view.dataset.view=s.view;
    bar.stackTitle.textContent=s.view==='stack'?'':`Your stack · ${runs.length} of ${MAX_STACK_LAYERS}`;
    bar.stackTitle.hidden=s.view==='stack';
    const focused=bar.stack.contains(document.activeElement)?[document.activeElement.closest('li')?.dataset.runId,document.activeElement.dataset.role]:null;
    bar.stack.replaceChildren(...runs.map((run,i)=>{
      const outside=filtersActive(s.filters)&&!matchIds.has(run.run_id),full=runLabel(run),short=layerLabel(runs,i);
      return h('li',{dataset:{runId:run.run_id},style:{'--flt-swatch':LAYER_COLORS[i%LAYER_COLORS.length]}},
        h('button',{type:'button',className:'flt-stack-chip','aria-pressed':String(run.run_id===s.active),dataset:{role:'select',outside:String(outside)},
          title:`${full}${outside?' (outside the current filters)':''}`,'aria-label':`Inspect stacked run ${i+1}: ${full}${outside?', outside the current filters':''}`,
          onclick:()=>app.update({active:run.run_id},SOURCE)},
          h('span',{className:'flt-swatch','aria-hidden':'true'}),h('span',{className:'flt-stack-text'},short)),
        h('button',{type:'button',className:'flt-stack-x',dataset:{role:'remove'},'aria-label':`Remove stacked run ${i+1} from the stack: ${full}`,title:runs.length<2?'The stack keeps at least one run':`Remove from stack`,
          'aria-disabled':String(runs.length<2),onclick:()=>removeFromStack(run.run_id)},h('span',{'aria-hidden':'true'},'×')));
    }));
    const outside=filtersActive(s.filters)?runs.filter(run=>!matchIds.has(run.run_id)).length:0;
    bar.outside.textContent=outside?(outside===runs.length?(runs.length===1?'Outside the filters':`All ${runs.length} outside the filters`):`${outside} of ${runs.length} outside the filters`):'';
    bar.outside.hidden=!outside;
    if(focused?.[0]){const li=bar.stack.querySelector(`li[data-run-id="${CSS.escape(focused[0])}"]`);(li?.querySelector(`[data-role="${focused[1]}"]`)||bar.stack.querySelector('button')||bar.summary).focus();}
  }

  // ----- Stack actions (shared by bar, rows and "Stack these") --------------
  function removeFromStack(id) {
    const s=state();
    if(s.layers.length<2){app.announce('The stack keeps at least one run. Add another before removing this one.');return;}
    const layers=s.layers.filter(item=>item!==id);
    app.update({layers,active:layers.includes(s.active)?s.active:layers[0]},SOURCE);
    app.announce(`Removed from the stack. ${plural(layers.length,'run')} stacked of ${MAX_STACK_LAYERS}.`);
  }
  function addToStack(run) {
    const s=state();
    if(s.layers.includes(run.run_id)){removeFromStack(run.run_id);return;}
    if(s.layers.length>=MAX_STACK_LAYERS){app.announce(`The stack is full (${MAX_STACK_LAYERS} of ${MAX_STACK_LAYERS}). Remove a run before adding ${runLabel(run)}.`);return;}
    const layers=[...s.layers,run.run_id];
    app.update({layers,active:s.active&&s.layers.includes(s.active)?s.active:run.run_id},SOURCE);
    app.announce(`Added ${runLabel(run)} to the stack. ${layers.length} of ${MAX_STACK_LAYERS} stacked.`);
  }
  function replay(run) {
    const s=state();
    if(s.layers.includes(run.run_id)){replayRun(app,run.run_id);app.announce(`Replaying ${runLabel(run)} in recorded generation order.`);return;}
    if(s.layers.length>=MAX_STACK_LAYERS){app.announce(`The stack is full (${MAX_STACK_LAYERS} of ${MAX_STACK_LAYERS}). Remove a run before replaying ${runLabel(run)}.`);return;}
    app.update({layers:[...s.layers,run.run_id],active:run.run_id,view:'replay',frame:replayStart(run)},SOURCE);
    app.announce(`Replaying ${runLabel(run)} in recorded generation order. It was added to the stack.`);
  }
  function stackThese() {
    if(!matches.length||matches.length>MAX_STACK_LAYERS)return;
    const ids=sortRuns(matches,'design').map(run=>run.run_id),s=state();
    app.update({layers:ids,active:ids.includes(s.active)?s.active:ids[0],view:'stack'},SOURCE);
    app.announce(`Stacked ${plural(ids.length,'matching graph')} in design order.`);
  }

  // ----- Facet panel ---------------------------------------------------------
  const panel={groups:new Map()};
  if(root){
    panel.search=h('input',{type:'search',id:`${uid}-search`,className:'flt-search-input',autocomplete:'off',spellcheck:'false',placeholder:'e.g. gpt-4.1 hindi 44003',maxlength:String(MAX_SEARCH),'aria-describedby':`${uid}-search-hint`});
    panel.search.dataset.fltSearch='true';
    panel.search.addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>setFilters({...state().filters,search:panel.search.value}),120);});
    panel.search.addEventListener('keydown',event=>{if(event.key==='Escape'&&panel.search.value){event.preventDefault();event.stopPropagation();panel.search.value='';clearTimeout(searchTimer);setFilters({...state().filters,search:''});}else if(event.key==='Enter'){clearTimeout(searchTimer);setFilters({...state().filters,search:panel.search.value});}});
    const searchBox=h('div',{className:'flt-search'},h('label',{for:`${uid}-search`,className:'flt-search-label'},'Search'),panel.search,
      h('p',{className:'flt-hint',id:`${uid}-search-hint`},'Whole words, all must match: model, method, country, language, seed or run id.'));
    const groups=[...FACETS,'graph'].map(facet=>{
      const headId=`${uid}-${facet}`,stateLabel=h('span',{className:'flt-facet-state'});
      const clear=h('button',{type:'button',className:'flt-facet-clear',hidden:true,'aria-label':`Clear ${FACET_TITLES[facet].toLowerCase()} filter`,onclick:()=>{
        setFilters(facet==='graph'?{...app.getState().filters,graph:'any'}:{...app.getState().filters,[facet]:[]});
        group.querySelector('.flt-option[tabindex="0"]')?.focus();
      }},'Clear');
      const options=h('div',{className:'flt-options',dataset:{facet}});
      const values=facetValues(data,facet);
      for(const value of values){
        options.append(h('button',{type:'button',className:'flt-option','aria-pressed':'false',tabindex:'-1',dataset:{value:String(value)},
          onclick:()=>{const s=app.getState();setFilters(facet==='graph'?{...s.filters,graph:value}:toggleValue(s.filters,facet,value,data));}},
          h('span',{className:'flt-check','aria-hidden':'true'}),h('span',{className:'flt-option-label'},valueLabel(facet,value)),h('span',{className:'flt-option-count'})));
      }
      options.addEventListener('keydown',event=>roveOptions(event,options));
      const group=h('div',{className:'flt-facet',role:'group','aria-labelledby':headId,dataset:{facet}},
        h('div',{className:'flt-facet-head'},h('h3',{id:headId,className:'flt-facet-title'},FACET_TITLES[facet]),stateLabel,clear),options);
      panel.groups.set(facet,{group,stateLabel,clear,options});
      return group;
    });
    root.replaceChildren(h('div',{className:'flt-panel'},searchBox,...groups));
    root.dataset.ready='true';
  }
  function roveOptions(event,options) {
    const items=[...options.querySelectorAll('.flt-option')],at=items.indexOf(document.activeElement);
    if(at<0)return;
    const columns=Math.max(1,items.filter(item=>item.offsetTop===items[0].offsetTop).length);
    const next={ArrowRight:at+1,ArrowDown:at+columns,ArrowLeft:at-1,ArrowUp:at-columns,Home:0,End:items.length-1}[event.key];
    if(next===undefined)return;
    event.preventDefault();
    const target=items[Math.max(0,Math.min(items.length-1,next))];
    for(const item of items)item.tabIndex=item===target?0:-1;
    target.focus();
  }
  function renderPanel() {
    if(!root)return;
    const s=state(),f=normalizeFilters(s.filters,data);
    if(document.activeElement!==panel.search&&panel.search.value!==f.search)panel.search.value=f.search;
    for(const [facet,{group,stateLabel:label,clear,options}] of panel.groups){
      const items=[...options.querySelectorAll('.flt-option')];
      const graphCounts=counts.graph,any=graphCounts.nonempty+graphCounts.empty;
      let selected=0;
      for(const item of items){
        const raw=item.dataset.value,value=facet==='seed'?Number(raw):raw;
        const pressed=facet==='graph'?f.graph===value:f[facet].includes(value);
        const n=facet==='graph'?(value==='any'?any:graphCounts[value]):counts[facet][value]??0;
        if(pressed&&!(facet==='graph'&&value==='any'))selected++;
        item.setAttribute('aria-pressed',String(pressed));
        item.disabled=!pressed&&n===0;
        item.querySelector('.flt-option-count').textContent=n.toLocaleString('en-US');
        const label=valueLabel(facet,value);
        item.setAttribute('aria-label',`${label}, ${plural(n,'graph')}`);
        item.title=n===0&&!pressed?'No graph matches together with the other active filters':`${plural(n,'graph')} with the other active filters`;
      }
      const narrowed=facet==='graph'?f.graph!=='any':f[facet].length>0;
      group.dataset.active=String(narrowed);
      label.textContent=narrowed?(facet==='graph'?GRAPH_CHIPS[f.graph]:`${selected} of ${items.length}`):'';
      clear.hidden=!narrowed;
      // Roving tab stop: keep it on the focused or first selected, else first enabled option.
      if(!options.contains(document.activeElement)||document.activeElement.disabled){
        const stop=items.find(item=>item.getAttribute('aria-pressed')==='true'&&!item.disabled)||items.find(item=>!item.disabled)||items[0];
        for(const item of items)item.tabIndex=item===stop?0:-1;
      }
    }
  }

  // ----- Result list (virtualised, one row per run_id) -----------------------
  const list={rows:new Map(),rowHeight:90,viewport:0};
  if(results){
    list.count=h('span',{className:'flt-results-count'});
    list.sort=h('select',{className:'flt-sort-select',id:`${uid}-sort`,onchange:()=>{sort=list.sort.value;recompute(false);list.scroll.scrollTop=0;renderRows();}},
      ...SORTS.map(([value,text])=>h('option',{value},text)));
    list.action=h('div',{className:'flt-results-action'});
    list.stackButton=h('button',{type:'button',className:'flt-stack-these primary',onclick:stackThese});
    list.guide=h('p',{className:'flt-guide'});
    list.action.append(list.stackButton,list.guide);
    list.spacer=h('div',{className:'flt-spacer'});
    list.scroll=h('div',{className:'flt-scroll',role:'list','aria-labelledby':`${uid}-results-title`},list.spacer);
    list.none=h('div',{className:'flt-none',hidden:true},h('p',{},'No valid graph matches these filters.'),h('button',{type:'button',onclick:()=>{setFilters(EMPTY_FILTERS);list.sort.focus();}},'Reset filters'));
    results.replaceChildren(h('div',{className:'flt-results'},
      h('div',{className:'flt-results-head'},h('h2',{className:'flt-results-title',id:`${uid}-results-title`},'Matching graphs'),
        h('label',{className:'flt-sort',for:`${uid}-sort`},'Sort'),list.sort),
      list.action,list.scroll,list.none));
    results.dataset.ready='true';
    let frame=0;
    const onScroll=()=>{if(!frame)frame=requestAnimationFrame(()=>{frame=0;renderRows();});};
    list.scroll.addEventListener('scroll',onScroll,{passive:true});
    list.scroll.addEventListener('keydown',onListKey);
    list.scroll.addEventListener('pointerover',event=>{const row=event.target.closest?.('.flt-row');if(row)focusSlot(Number(row.dataset.slot));});
    list.scroll.addEventListener('pointerleave',()=>focusSlot(null));
    list.scroll.addEventListener('focusin',event=>{const row=event.target.closest('.flt-row');if(row){current=Number(row.dataset.index);syncTabStops();focusSlot(Number(row.dataset.slot));}});
    list.scroll.addEventListener('focusout',event=>{if(!list.scroll.contains(event.relatedTarget))focusSlot(null);});
    const measure=()=>{
      const css=parseFloat(getComputedStyle(list.scroll).getPropertyValue('--flt-row-h'));
      list.rowHeight=Number.isFinite(css)&&css>0?css:90;
      list.viewport=list.scroll.clientHeight;
      list.spacer.style.height=`${matches.length*list.rowHeight}px`;
      for(const [id,row] of list.rows)row.style.transform=`translateY(${Number(row.dataset.index)*list.rowHeight}px)`;
      renderRows();
    };
    if(typeof ResizeObserver==='function'){const observer=new ResizeObserver(measure);observer.observe(list.scroll);cleanup.push(()=>observer.disconnect());}
    else{addEventListener('resize',measure);cleanup.push(()=>removeEventListener('resize',measure));}
    list.measure=measure;
    cleanup.push(()=>{if(frame)cancelAnimationFrame(frame);list.scroll.removeEventListener('scroll',onScroll);});
  }
  const focusSlot=slot=>{if(state().focusSlot!==slot)app.update({focusSlot:slot},SOURCE);};
  function makeRow(run) {
    const ties=run.edges.length,density=run.metrics?.density;
    const meta=[plural(ties,'tie'),Number.isFinite(density)?`density\u00a0${density.toFixed(3)}`:null].filter(Boolean).join(' · ');
    const label=runLabel(run);
    const row=h('div',{className:'flt-row',role:'listitem',dataset:{runId:run.run_id,slot:String(run.slot)}},
      h('span',{className:'flt-swatch','aria-hidden':'true'}),
      h('span',{className:'flt-row-label',title:label},label),
      h('span',{className:'flt-row-meta'},meta,ties===0?h('span',{className:'flt-badge'},'Empty graph'):null,h('span',{className:'flt-row-stacked'})),
      h('div',{className:'flt-row-actions'},
        h('button',{type:'button',className:'flt-replay',dataset:{col:'0'},tabindex:'-1','aria-label':`Replay ${label}`,onclick:()=>replay(run)},'Replay'),
        h('button',{type:'button',className:'flt-add',dataset:{col:'1'},tabindex:'-1',onclick:()=>addToStack(run)})));
    return row;
  }
  function updateRow(row,run,index) {
    const s=state(),at=s.layers.indexOf(run.run_id),stacked=at>=0,full=!stacked&&s.layers.length>=MAX_STACK_LAYERS,label=runLabel(run);
    row.dataset.index=String(index);
    row.setAttribute('aria-posinset',String(index+1));row.setAttribute('aria-setsize',String(matches.length));
    row.style.transform=`translateY(${index*list.rowHeight}px)`;
    row.dataset.stacked=String(stacked);row.dataset.active=String(run.run_id===s.active);
    row.style.setProperty('--flt-swatch',stacked?LAYER_COLORS[at%LAYER_COLORS.length]:'transparent');
    const swatch=row.querySelector('.flt-swatch');swatch.textContent=stacked?String(at+1):'';
    row.querySelector('.flt-row-stacked').textContent=stacked?(run.run_id===s.active?` · stack ${at+1}, active`:` · stack ${at+1}`):'';
    const add=row.querySelector('.flt-add');
    if(add.dataset.stacked!==String(stacked)){add.dataset.stacked=String(stacked);add.replaceChildren(...(stacked?['Remove']:[h('span',{className:'flt-long'},'Add to stack'),h('span',{className:'flt-short','aria-hidden':'true'},'+ Stack')]));}
    add.setAttribute('aria-label',stacked?`Remove ${label} from the stack`:`Add ${label} to the stack`);
    add.setAttribute('aria-pressed',String(stacked));
    add.setAttribute('aria-disabled',String(full||(stacked&&s.layers.length<2)));
    add.title=full?`The stack is full (${MAX_STACK_LAYERS} runs)`:stacked&&s.layers.length<2?'The stack keeps at least one run':'';
    const replayButton=row.querySelector('.flt-replay');
    replayButton.setAttribute('aria-disabled',String(full));
    replayButton.title=full?`The stack is full (${MAX_STACK_LAYERS} runs); remove one to replay another`:'';
  }
  function syncTabStops() {
    for(const [,row] of list.rows){const on=Number(row.dataset.index)===current;for(const button of row.querySelectorAll('button'))button.tabIndex=on?0:-1;}
  }
  function renderRows() {
    if(!results)return;
    const n=matches.length,height=list.rowHeight,top=list.scroll.scrollTop,view=list.viewport||list.scroll.clientHeight||height*8;
    const first=Math.max(0,Math.floor(top/height)-4),last=Math.min(n-1,Math.ceil((top+view)/height)+4);
    const keep=new Set();
    for(let i=first;i<=last;i++)keep.add(i);
    if(current<n)keep.add(current);
    const wanted=new Map([...keep].map(i=>[matches[i].run_id,i]));
    const focusedId=list.scroll.contains(document.activeElement)?document.activeElement.closest('.flt-row')?.dataset.runId:null;
    for(const [id,row] of list.rows)if(!wanted.has(id)&&id!==focusedId){row.remove();list.rows.delete(id);}
    const index=runIndex(data);
    for(const [id,i] of wanted){
      let row=list.rows.get(id);
      if(!row){row=makeRow(index.get(id));list.rows.set(id,row);list.spacer.append(row);}
      updateRow(row,index.get(id),i);
    }
    syncTabStops();
  }
  function onListKey(event) {
    const button=event.target.closest?.('.flt-row button');if(!button)return;
    const row=button.closest('.flt-row'),at=Number(row.dataset.index),col=Number(button.dataset.col),page=Math.max(1,Math.floor((list.viewport||list.rowHeight)/list.rowHeight)-1);
    const moves={ArrowDown:[at+1,col],ArrowUp:[at-1,col],PageDown:[at+page,col],PageUp:[at-page,col],Home:[0,col],End:[matches.length-1,col],ArrowRight:[at,Math.min(1,col+1)],ArrowLeft:[at,Math.max(0,col-1)]};
    const move=moves[event.key];if(!move)return;
    event.preventDefault();
    focusRow(Math.max(0,Math.min(matches.length-1,move[0])),move[1]);
  }
  function focusRow(index,col=0) {
    current=index;
    const height=list.rowHeight,top=index*height,view=list.viewport||list.scroll.clientHeight;
    if(top<list.scroll.scrollTop)list.scroll.scrollTop=top;
    else if(top+height>list.scroll.scrollTop+view)list.scroll.scrollTop=top+height-view;
    renderRows();
    list.rows.get(matches[index].run_id)?.querySelector(`button[data-col="${col}"]`)?.focus({preventScroll:true});
  }
  function renderResultsHead() {
    if(!results)return;
    const n=matches.length,s=state(),ids=sortRuns(matches,'design').map(run=>run.run_id);
    const same=ids.length===s.layers.length&&ids.every((id,i)=>id===s.layers[i]);
    list.stackButton.hidden=!(n>=1&&n<=MAX_STACK_LAYERS);
    list.stackButton.textContent=n===1?'Stack this graph':`Stack these ${n}`;
    list.stackButton.disabled=same;
    list.stackButton.title=same?'These graphs are already the stack':`Replace the stack with the ${plural(n,'matching graph')}, in design order`;
    list.guide.hidden=n<=MAX_STACK_LAYERS;
    list.guide.textContent=`To stack, narrow to ${MAX_STACK_LAYERS} or fewer. One condition (a model, a method and a setting) is exactly ${MAX_STACK_LAYERS} seeds.`;
    list.action.hidden=n===0;
    list.none.hidden=n>0;list.scroll.hidden=n===0;
  }

  // ----- Store wiring ----------------------------------------------------------
  function recompute(resetScroll=true) {
    const s=state(),previous=matches[current]?.run_id;
    matches=sortRuns(applyFilters(data.runs,s.filters),sort);
    matchIds=new Set(matches.map(run=>run.run_id));
    counts=facetCounts(data,s.filters);
    if(results){
      const keep=matches.findIndex(run=>run.run_id===previous);
      current=keep>=0?keep:0;
      for(const [,row] of list.rows)row.remove();list.rows.clear();
      list.spacer.style.height=`${matches.length*list.rowHeight}px`;
      if(resetScroll&&keep<0)list.scroll.scrollTop=0;
      results.dataset.count=String(matches.length);
    }
  }
  function render(changed) {
    const all=!changed;
    if(all||changed.has('filters')){recompute();renderPanel();renderRows();}
    else if(changed.has('layers')||changed.has('active'))for(const [id,row] of list.rows)updateRow(row,runIndex(data).get(id),Number(row.dataset.index));
    if(all||['filters','layers','active','view'].some(key=>changed.has(key))){renderBar();renderResultsHead();}
  }
  const unsubscribe=app.subscribe((s,changed)=>{if(['filters','layers','active','view'].some(key=>changed.has(key)))render(changed);});
  cleanup.push(unsubscribe);

  // Open the link the page was loaded with (short grammar or legacy ?view=).
  let linkMessage='';
  try{const patch=decodeState(new URLSearchParams(search),data);if(Object.keys(patch).length)applyLink(patch);}
  catch(error){linkMessage=`This link could not be opened: ${error.message} Showing the default view instead.`;}
  render(null);
  if(results)list.measure();
  if(linkMessage){
    if(chips){bar.noticeText.textContent=linkMessage;bar.notice.hidden=false;}
    app.announce(linkMessage);
  }
  if(writeUrl)cleanup.push(syncUrl(app));
  return cleanup;

  function applyLink(patch) {
    // Filters first so the result list is right even if layers.js rejects the stack.
    if(patch.filters)app.update({filters:patch.filters},SOURCE);
    const {filters,...rest}=patch;
    if(Object.keys(rest).length)app.update(rest,SOURCE);
  }
}
