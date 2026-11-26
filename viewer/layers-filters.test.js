// Unit tests for the filter model, counts, chips and URL grammar (SPEC.md B.4, C.5).
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {EMPTY_FILTERS,INITIAL_STATE,MAX_STACK_LAYERS} from './layers-app.js';
import {loadPrimaryDataset,COUNTS} from './layers-data-all.js';
import {applyFilters,facetCounts,failuresInScope,activeChips,encodeState,decodeState,stateQuery,normalizeFilters,filtersActive,
  toggleValue,withoutChip,sortRuns,scopeSummary,slotMask,isPrimaryRun,matchesSearch,facetValues,valueLabel,FACETS,ARCHIVED_LINK_MESSAGE,ABSENT_RUN_MESSAGE,mountFilters,replayRun,syncUrl} from './layers-filters.js';
import {createStore} from './layers-app.js';

// Serve public/data/ to the real loader, whatever files it asks for (monolith today, compact graphs/ after WS1).
const fetchImpl=async url=>{
  const path=String(url).replace(/^\.\/data\//,'').replace(/^\.?\/?/,'');
  try{const bytes=await readFile(new URL(`./public/data/${path}`,import.meta.url));
    return {ok:true,status:200,json:async()=>JSON.parse(bytes.toString('utf8')),text:async()=>bytes.toString('utf8'),arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};}
  catch{return {ok:false,status:404,json:async()=>{throw new Error('404');},text:async()=>'',arrayBuffer:async()=>new ArrayBuffer(0)};}
};
const data=await loadPrimaryDataset({fetchImpl,base:'./data/'});
const F=(patch={})=>normalizeFilters({...EMPTY_FILTERS,...patch},data);
const brute=(filters)=>data.runs.filter(run=>
  (!filters.model.length||filters.model.includes(run.model))&&(!filters.method.length||filters.method.includes(run.method))&&
  (!filters.setting.length||filters.setting.includes(`${run.culture}-${run.language}`))&&(!filters.seed.length||filters.seed.includes(run.seed))&&
  (filters.graph==='any'||(filters.graph==='empty')===(run.edges.length===0))&&matchesSearch(run,filters.search));
const supplementIds=async()=>{
  const saved=JSON.parse(await readFile(new URL('./public/data/saved-networks.json',import.meta.url),'utf8'));
  const primary=new Set(data.runs.map(run=>run.run_id));
  return saved.graphs.map(graph=>graph.run_id).filter(id=>!primary.has(id));
};

test('study facts: 890 valid graphs, 12 empty, 6 failures in scope with no filters',()=>{
  assert.equal(COUNTS.attempts,896);
  assert.equal(applyFilters(data.runs,EMPTY_FILTERS).length,890);
  assert.equal(applyFilters(data.runs,F({graph:'empty'})).length,12);
  assert.equal(applyFilters(data.runs,F({graph:'nonempty'})).length,878);
  assert.equal(failuresInScope(data,EMPTY_FILTERS),6);
  assert.equal(failuresInScope(data,F({graph:'nonempty'})),0,'failures are not graphs');
  assert.equal(failuresInScope(data,F({model:['gpt-5.6-luna']})),3);
  assert.equal(failuresInScope(data,F({setting:['us-japanese'],method:['global']})),3);
  const summary=scopeSummary(data,EMPTY_FILTERS);
  assert.equal(summary.text,'All 890 valid graphs · no filters set');
  assert.equal(summary.active,false);
});

test('supplements and failures can never enter the filtered set',async()=>{
  const ids=await supplementIds();
  assert.equal(ids.length,6);
  const fake=ids.map(run_id=>({...data.runs[0],run_id}));
  assert.equal(applyFilters([...data.runs,...fake],EMPTY_FILTERS).length,890);
  assert.equal(fake.some(isPrimaryRun),false);
  const failed=data.failures.map(failure=>({...failure,edges:[]}));
  assert.equal(applyFilters(failed,EMPTY_FILTERS).length,0);
  for(const id of ids)assert.equal(applyFilters(data.runs,F({search:id})).length,0);
  assert.equal(applyFilters(data.runs,F({search:'supplement'})).length,0);
  assert.equal(new Set(applyFilters(data.runs,EMPTY_FILTERS).map(run=>run.run_id)).size,890,'no run twice');
});

test('values within a facet are OR, facets are AND',()=>{
  const two=applyFilters(data.runs,F({method:['local','global']}));
  assert.ok(two.every(run=>['local','global'].includes(run.method)));
  assert.equal(two.length,applyFilters(data.runs,F({method:['local']})).length+applyFilters(data.runs,F({method:['global']})).length);
  const condition=applyFilters(data.runs,F({model:['gpt-4.1'],method:['local'],setting:['us-english']}));
  assert.equal(condition.length,8,'one condition is 8 seeds');
  assert.deepEqual(condition.map(run=>run.seed).sort(),[...data.design.seeds]);
  assert.equal(applyFilters(data.runs,F({model:['gpt-5.6-luna'],method:['global'],setting:['us-portuguese']})).length,6,'two failures in that cell');
});

test('facet counts equal applyFilters for 20 random filter combinations',()=>{
  let seed=20261125;const random=()=>((seed=(seed*1103515245+12345)%2147483648)/2147483648);
  const pick=values=>values.filter(()=>random()<0.3);
  for(let round=0;round<20;round++){
    const filters=F({model:pick(data.design.models),method:pick(data.design.methods),setting:pick(facetValues(data,'setting')),seed:pick(data.design.seeds),
      graph:['any','any','nonempty','empty'][Math.floor(random()*4)],search:random()<0.25?['english','us','local','gpt-6-sol','44003 brazil'][Math.floor(random()*5)]:''});
    const counts=facetCounts(data,filters);
    for(const facet of FACETS)for(const value of facetValues(data,facet)){
      const expected=applyFilters(data.runs,{...filters,[facet]:[value]}).length;
      assert.equal(counts[facet][value],expected,`round ${round} ${facet}=${value}`);
    }
    assert.equal(counts.graph.empty,applyFilters(data.runs,{...filters,graph:'empty'}).length);
    assert.equal(counts.graph.nonempty,applyFilters(data.runs,{...filters,graph:'nonempty'}).length);
    assert.deepEqual(applyFilters(data.runs,filters).map(run=>run.run_id),brute(filters).map(run=>run.run_id));
    const summary=scopeSummary(data,filters);
    assert.equal(summary.shown,applyFilters(data.runs,filters).length,'count line equals result length');
  }
});

test('unfiltered counts per facet',()=>{
  const counts=facetCounts(data,EMPTY_FILTERS);
  assert.deepEqual(counts.graph,{nonempty:878,empty:12});
  assert.equal(Object.values(counts.model).reduce((a,b)=>a+b,0),890);
  assert.equal(counts.model['gpt-5.6-luna'],221);
  assert.equal(counts.seed[44004],110,'112 slots minus 2 failures');
  assert.equal(Object.keys(counts.setting).length,7);
});

test('search matches whole words over model, method, country, language, seed and run_id',()=>{
  assert.equal(applyFilters(data.runs,F({search:'gpt-4.1'})).length,223,'224 slots minus 1 failure');
  assert.equal(applyFilters(data.runs,F({search:'gpt-4'})).length,0,'partial words do not match');
  assert.equal(applyFilters(data.runs,F({search:'hindi'})).length,applyFilters(data.runs,F({setting:['us-hindi']})).length);
  assert.equal(applyFilters(data.runs,F({search:'United States'})).length,applyFilters(data.runs,F({setting:['us-english','us-hindi','us-japanese','us-portuguese']})).length);
  assert.equal(applyFilters(data.runs,F({search:'brazil 44003  LOCAL'})).length,4);
  const id=data.runs[17].run_id;
  assert.deepEqual(applyFilters(data.runs,F({search:id})).map(run=>run.run_id),[id]);
  assert.equal(applyFilters(data.runs,F({search:'44000'})).length,applyFilters(data.runs,F({seed:[44000]})).length);
});

test('chips list only narrowed values, in display order, and remove themselves',()=>{
  assert.deepEqual(activeChips(data,EMPTY_FILTERS),[]);
  assert.equal(filtersActive(EMPTY_FILTERS),false);
  const filters=F({method:['global','local'],model:['gpt-4.1'],setting:['us-hindi'],seed:[44001],graph:'nonempty',search:'x'});
  const chips=activeChips(data,filters);
  assert.deepEqual(chips.map(chip=>chip.label),['Model: GPT-4.1','Method: Global','Method: Local','Setting: United States / Hindi','Seed: 44001','Graph: With ties only','Search: “x”']);
  let next=filters;for(const chip of chips)next=withoutChip(next,chip);
  assert.equal(filtersActive(next),false);
  assert.deepEqual(toggleValue(toggleValue(EMPTY_FILTERS,'model','gpt-6-sol',data),'model','gpt-6-sol',data).model,[]);
  assert.equal(toggleValue(F({graph:'empty'}),'graph','empty').graph,'any');
  assert.equal(valueLabel('setting','india-english'),'India / English');
});

test('active summary counts empties and failures in scope',()=>{
  const summary=scopeSummary(data,F({model:['gpt-5.6-luna'],method:['global']}));
  const runs=applyFilters(data.runs,F({model:['gpt-5.6-luna'],method:['global']}));
  const empty=runs.filter(run=>!run.edges.length).length;
  assert.equal(summary.text,[`Showing ${runs.length} of 890 valid graphs`,...(empty?[`${empty} empty`]:[]),'3 failed attempts in scope (not rendered)'].join(' · '));
  assert.equal(scopeSummary(data,F({model:['gpt-4.1'],method:['iterative'],setting:['us-japanese']})).text,'Showing 7 of 890 valid graphs · 1 failed attempt in scope (not rendered)');
  assert.equal(scopeSummary(data,F({graph:'empty'})).text,'Showing 12 of 890 valid graphs · 12 empty');
});

test('slot mask marks matching graphs and failed attempts in scope',()=>{
  const mask=slotMask(data,EMPTY_FILTERS);
  assert.equal(mask.length,896);
  assert.equal(mask.filter(v=>v===1).length,890);
  assert.equal(mask.filter(v=>v===2).length,6);
  const narrow=slotMask(data,F({graph:'empty'}));
  assert.equal(narrow.filter(v=>v===1).length,12);assert.equal(narrow.filter(v=>v===2).length,0);
});

test('result order: one row per run_id, design order by default, ties and density sorts',()=>{
  const doubled=[...data.runs,...data.runs.slice(0,50)];
  const rows=sortRuns(applyFilters(doubled,EMPTY_FILTERS));
  assert.equal(rows.length,890);
  assert.deepEqual(rows.map(run=>run.slot),[...rows.map(run=>run.slot)].sort((a,b)=>a-b));
  const byTies=sortRuns(data.runs,'ties-desc');
  assert.ok(byTies[0].edges.length>=byTies[1].edges.length&&byTies.at(-1).edges.length===0);
  const byDensity=sortRuns(data.runs,'density-asc');
  assert.ok(byDensity[0].metrics.density<=byDensity[500].metrics.density);
});

test('URL grammar: short, readable, round trip',()=>{
  const layers=[0,8,16,24].map(slot=>data.slots[slot].run_id);
  const state={...INITIAL_STATE,view:'stack',filters:F({model:['gpt-4.1'],method:['local','global'],setting:['us-english'],seed:[44000],graph:'nonempty',search:'united states'}),
    layers,active:layers[1],persona:'12',colorBy:'religion',lens:'same',groups:['Muslim'],frame:0};
  const query=stateQuery(state,data);
  assert.equal(query,'v=stack&m=gpt-4.1&p=global,local&s=us-english&b=44000&g=nonempty&q=united+states&l=0.8.16.24&a=8&n=12&c=religion&x=same&i=Muslim');
  const decoded=decodeState(new URLSearchParams(query),data);
  for(const key of ['view','filters','layers','active','persona','colorBy','lens','groups'])assert.deepEqual(decoded[key],state[key],key);
  assert.deepEqual(decodeState(encodeState(state,data),data).filters,state.filters);
  const replay=decodeState(stateQuery({...state,view:'replay',frame:37},data),data);
  assert.equal(replay.view,'replay');assert.equal(replay.frame,37);
  assert.equal(encodeState({...state,view:'replay',frame:Number.MAX_SAFE_INTEGER},data).has('f'),false,'final frame is the default');
  assert.equal(stateQuery({...INITIAL_STATE,layers:[]},data),'v=stack');
  assert.deepEqual(decodeState('v=atlas',data).filters,normalizeFilters(EMPTY_FILTERS),'missing facets mean unfiltered');
  assert.deepEqual(decodeState('',data),{});
  const single=decodeState('a=8',data);
  assert.deepEqual(single.layers,[data.slots[8].run_id]);
  assert.deepEqual(decodeState('l=8.8.0',data).layers,[data.slots[8].run_id,data.slots[0].run_id]);
});

test('URL grammar: unknown or failed slots and bad values fail clearly',()=>{
  const failed=data.slots.find(slot=>slot.status==='failed');
  assert.throws(()=>decodeState(`l=${failed.slot}`,data),new RegExp(`Slot ${failed.slot} is a failed attempt`));
  assert.throws(()=>decodeState(`a=${failed.slot}`,data),/failed attempt/);
  assert.throws(()=>decodeState('l=896',data),/outside the 896 planned attempts/);
  assert.throws(()=>decodeState('l=abc',data),/not a slot number/);
  assert.throws(()=>decodeState('l=0.1&a=2',data),/not in the linked stack/);
  assert.throws(()=>decodeState(`l=${Array.from({length:MAX_STACK_LAYERS+1},(_,i)=>i*8).join('.')}`,data),/at most 8/);
  assert.throws(()=>decodeState('m=gpt-9',data),/unknown model “gpt-9”/);
  assert.throws(()=>decodeState('s=us-french',data),/unknown setting/);
  assert.throws(()=>decodeState('b=1',data),/unknown seed/);
  assert.throws(()=>decodeState('g=some',data),/unknown graph state/);
  assert.throws(()=>decodeState('v=movie',data),/unknown view/);
  assert.throws(()=>decodeState('n=77',data),/unknown persona/);
  assert.throws(()=>decodeState('c=height',data),/unknown attribute/);
  assert.throws(()=>decodeState('x=both',data),/unknown lens/);
  assert.throws(()=>decodeState('v=replay&f=-1',data),/recorded update number/);
});

test('legacy ?view= links still open; archived links throw today\'s message',()=>{
  const runs=data.design.methods.map(method=>data.runs.find(r=>r.model==='gpt-4.1'&&r.method===method&&r.culture==='us'&&r.language==='english'&&r.seed===44000).run_id);
  const legacy={runs,filters:{collection:'primary',method:'',culture:'us',language:'english',model:'gpt-4.1',seed:'44000'},mode:'compare',active:runs[2],frame:5,journey:false,person:'7',color:'gender',separation:'55',ties:'all',opacity:'50',incident:false,metric:'density','research-question':'method','node-labels':'none'};
  const decoded=decodeState(new URLSearchParams({dataset:'primary',view:JSON.stringify(legacy)}),data);
  assert.deepEqual(decoded.layers,runs);assert.equal(decoded.active,runs[2]);assert.equal(decoded.view,'stack');
  assert.equal(decoded.persona,'7');assert.equal(decoded.colorBy,'gender');assert.equal(decoded.frame,5);
  assert.deepEqual(decoded.filters,F({model:['gpt-4.1'],setting:['us-english'],seed:[44000]}));
  assert.equal(decodeState(new URLSearchParams({view:JSON.stringify({...legacy,mode:'formation'})}),data).view,'replay');
  assert.deepEqual(decodeState(new URLSearchParams({view:JSON.stringify({runs:[runs[0]],filters:{culture:'us'}})}),data).filters.setting,['us-english','us-hindi','us-japanese','us-portuguese']);
  assert.deepEqual(decodeState(new URLSearchParams({view:'{not json'}),data),{});
  for(const query of ['?dataset=revised_calibration','?dataset=calibration','?dataset=legacy','?view='+encodeURIComponent(JSON.stringify({runs:['calibration_v6_old']}))])
    assert.throws(()=>decodeState(new URLSearchParams(query),data),error=>error.message===ARCHIVED_LINK_MESSAGE);
  assert.throws(()=>decodeState(new URLSearchParams({view:JSON.stringify({runs:['final_060dcd3a86d5_main_global_gpt-9_us_english_s44000']})}),data),error=>error.message===ABSENT_RUN_MESSAGE);
});

test('mountFilters waits for data and is a no-op without a DOM root',async()=>{
  let resolve;const app={data:null,ready:new Promise(r=>{resolve=r;})};
  const handle=mountFilters(app,{});
  assert.equal(typeof handle.destroy,'function');
  handle.destroy();resolve(data);await app.ready;
});

test('replayRun ends in replay of a stacked run, in one update',()=>{
  const app=createStore(),ids=data.runs.slice(0,3).map(run=>run.run_id),seen=[];
  app.update({layers:ids,active:ids[0],view:'stack'});
  app.subscribe((state,changed)=>seen.push([...changed]));
  replayRun(app,ids[2]);
  assert.deepEqual(seen.map(keys=>keys.sort()),[['active','view']]);
  assert.equal(app.getState().active,ids[2]);assert.equal(app.getState().view,'replay');
  assert.throws(()=>replayRun(app,data.runs[5].run_id),/stacked run/);
});

test('syncUrl writes the short grammar once the store settles',async()=>{
  const app=createStore();app.init(data,{layers:[data.slots[0].run_id],active:data.slots[0].run_id,view:'stack'});
  const location={href:'https://example.test/layers.html?dataset=primary&view=%7B%7D#stage'},writes=[];
  const history={state:null,replaceState:(state,title,url)=>{writes.push(url);location.href=url;}};
  const stop=syncUrl(app,{history,location,delay:0});
  app.update({filters:normalizeFilters({model:['gpt-6-sol'],search:'hindi'},data)});
  await new Promise(resolve=>setTimeout(resolve,10));
  stop();
  assert.equal(writes.length,1);
  assert.equal(writes[0],'https://example.test/layers.html?v=stack&m=gpt-6-sol&q=hindi&l=0&a=0#stage');
  app.update({filters:EMPTY_FILTERS});await new Promise(resolve=>setTimeout(resolve,10));
  assert.equal(writes.length,1,'stopped writers stay silent');
});

test('aggregate plane setting round-trips through the short link and rejects nonsense',()=>{
  const ids=data.runs.slice(0,2).map(run=>run.run_id);
  const state={...INITIAL_STATE,view:'stack',layers:ids,active:ids[0],consensus:{scope:'filters',min:.5}};
  const params=encodeState(state,data);
  assert.equal(params.get('k'),'f50');
  assert.deepEqual(decodeState(params,data).consensus,{scope:'filters',min:.5});
  assert.equal(encodeState({...state,consensus:{scope:'stack',min:1}},data).get('k'),'s100');
  assert.equal(encodeState({...state,consensus:null},data).has('k'),false);
  assert.equal(encodeState({...state,view:'atlas'},data).has('k'),false,'the plane exists only in the Stack');
  assert.equal(decodeState(new URLSearchParams('v=stack&l=0'),data).consensus,undefined);
  for(const bad of ['x50','f0','f101','supplements'])assert.throws(()=>decodeState(new URLSearchParams('v=stack&l=0&k='+bad),data),/aggregate plane setting/);
});
