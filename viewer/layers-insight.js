// WORKSTREAM 5: attribute lens, layer comparison, consensus, study ledger, provenance.
// Contract: docs/qa/viewer-3d-2026-11-25/SPEC.md section C.7. The pure functions at the top are DOM-free so
// node:test and the 3D engine (layers-3d-*.js) can call them; mountInsight() at the bottom owns the DOM inside
// #insight-root, #study-ledger and #legend and talks to the rest of the page only through the store.
//
// Honesty rules encoded here (SPEC.md section 0): personas are fictional; every share describes ties the model
// generated, not human preferences; aggregates are descriptive, not significance tests; the consensus is an
// aggregate of graphs, never a run; supplements are never part of any figure computed here (app.data.runs holds the
// 890 primary graphs only). Saved Coleman / assortativity values are read from the frozen analysis, never recomputed.
import {LAYER_COLORS,LENSES,MAX_STACK_LAYERS} from './layers-app.js';
import * as dataModule from './layers-data-all.js';
import {applyFilters} from './layers-filters.js';
import {runLabel,layerLabel,runParts,plural,LABEL_SEPARATOR} from './formation-charts.js';

// ---------------------------------------------------------------------------------------------------------------
// Attribute palettes. Categorical hues follow Okabe & Ito (2008), chosen to stay distinguishable under
// deuteranopia; every colour keeps at least 3:1 contrast against --panel (#11171a). Age uses six ordered bands on a
// perceptually even viridis segment (light end = older), trimmed so the darkest band still meets 3:1.
// layers-insight.test.js checks contrast and simulated-deuteranopia separation for every palette.
export const PANEL_COLOR='#11171a';
export const NEUTRAL_COLOR='#6f828b'; // --empty / --faint: persona with no recorded value for the attribute
export const TIE_CLASS_COLORS=Object.freeze({same:'#80e2c4',mixed:'#f2c078'}); // mirrors --same / --mixed
export const AGE_BANDS=Object.freeze([
  {key:'18-24',label:'18–24',min:18,max:24,color:'#2c7e8e'},
  {key:'25-34',label:'25–34',min:25,max:34,color:'#21998b'},
  {key:'35-44',label:'35–44',min:35,max:44,color:'#38b479'},
  {key:'45-54',label:'45–54',min:45,max:54,color:'#6ccd5a'},
  {key:'55-64',label:'55–64',min:55,max:64,color:'#addc30'},
  {key:'65+',label:'65+',min:65,max:Infinity,color:'#fde725'}
]);
export const ATTRIBUTES=Object.freeze([
  {key:'gender',label:'Gender',short:'Gender',saved:'coleman_gender',savedLabel:'Saved Coleman index',
    groups:[['Woman','#e69f00'],['Man','#56b4e9'],['Nonbinary','#cc79a7']]},
  {key:'age',label:'Age band',short:'Age',saved:'age_assortativity',savedLabel:'Saved age assortativity',ordered:true,
    groups:AGE_BANDS.map(band=>[band.key,band.color,band.label])},
  {key:'religion',label:'Religion',short:'Religion',saved:'coleman_religion',savedLabel:'Saved Coleman index',
    groups:[['Buddhist','#e69f00'],['Christian','#56b4e9'],['Hindu','#2fbf8f'],['Muslim','#f0e442'],['Nonreligious','#cc79a7']]},
  {key:'political orientation',label:'Political orientation',short:'Politics',saved:'coleman_political orientation',savedLabel:'Saved Coleman index',ordered:true,
    groups:[['Left-leaning','#56b4e9'],['Centrist','#d7d2c8'],['Right-leaning','#e8743b']]}
]);
const ATTRIBUTE_BY_KEY=new Map(ATTRIBUTES.map(attribute=>[attribute.key,attribute]));
// Colours for values a future roster might add; never needed by the frozen 50-person roster (tested).
const SPARE_COLORS=['#0fa3b1','#f7a1c4','#b8b8ff','#ffd6a5'];
export const attributeInfo=key=>ATTRIBUTE_BY_KEY.get(key)||null;

export function ageBand(age) {
  const value=Number(age);
  if(!Number.isFinite(value))return null;
  return AGE_BANDS.find(band=>value>=band.min&&value<=band.max)?.key??null;
}
// Group key of one persona for an attribute ('age' is banded); null when the roster has no value.
export function groupOf(person,key) {
  const value=person?.attributes?.[key];
  if(value===undefined||value===null||value==='')return null;
  return key==='age'?ageBand(value):String(value);
}
// Ordered groups of an attribute as present in the roster: [{key,label,color,count}].
export function attributeGroups(personas,key) {
  const attribute=attributeInfo(key);
  if(!attribute)return [];
  const counts=new Map();
  for(const person of personas){const group=groupOf(person,key);if(group!==null)counts.set(group,(counts.get(group)||0)+1);}
  const known=attribute.groups.filter(([group])=>counts.has(group)).map(([group,color,label])=>({key:group,label:label||group,color,count:counts.get(group)}));
  const extra=[...counts.keys()].filter(group=>!attribute.groups.some(([g])=>g===group)).sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
  return [...known,...extra.map((group,i)=>({key:group,label:group,color:SPARE_COLORS[i%SPARE_COLORS.length],count:counts.get(group)}))];
}

const runIndex=new WeakMap();
function runsById(dataset) {
  let index=runIndex.get(dataset);
  if(!index){index=new Map((dataset?.runs||[]).map(run=>[run.run_id,run]));runIndex.set(dataset,index);}
  return index;
}
const stackRuns=(dataset,state)=>(state?.layers||[]).map(id=>runsById(dataset).get(id)).filter(Boolean);
const personaIndex=new WeakMap();
function personasById(dataset) {
  let index=personaIndex.get(dataset);
  if(!index){index=new Map((dataset?.personas||[]).map(person=>[String(person.id),person]));personaIndex.set(dataset,index);}
  return index;
}

// Node colour, isolation and legend for the current colorBy + groups (SPEC.md C.7).
// colorBy 'layer': colour by plane identity (LAYER_COLORS[layerIndex]); legend lists the stacked layers.
// colorBy <attribute>: colour by the persona's group, identical on every plane; legend lists the roster groups.
export function nodeEncoding(dataset,state) {
  const key=state?.colorBy,attribute=attributeInfo(key);
  if(!attribute){
    const runs=stackRuns(dataset,state);
    return {
      mode:'layer',attribute:null,
      colorOf:(personId,layerIndex=0)=>LAYER_COLORS[((layerIndex%LAYER_COLORS.length)+LAYER_COLORS.length)%LAYER_COLORS.length],
      groupOf:()=>null,
      dimmed:()=>false,
      legend:runs.map((run,i)=>({key:run.run_id,label:layerLabel(runs,i),color:LAYER_COLORS[i%LAYER_COLORS.length],count:run.edges.length}))
    };
  }
  const people=personasById(dataset),groups=attributeGroups(dataset?.personas||[],key),colors=new Map(groups.map(group=>[group.key,group.color]));
  const isolated=new Set((state?.groups||[]).filter(group=>colors.has(group)));
  const group=id=>groupOf(people.get(String(id)),key);
  return {
    mode:'attribute',attribute:key,
    colorOf:personId=>colors.get(group(personId))||NEUTRAL_COLOR,
    groupOf:group,
    // Isolation dims every persona outside the chosen groups; nothing is dimmed while no group is chosen.
    dimmed:personId=>isolated.size>0&&!isolated.has(group(personId)),
    legend:groups
  };
}

// Tie classes for the attribute lens: 'same' (both personas in one group), 'mixed', or null when the lens is off,
// colour is by layer, or a persona has no value. dimmed(edge) says whether the 3D view should fade a tie: under the
// 'same' / 'mixed' lens the other class fades; with isolated groups a tie fades when neither end is isolated.
export function edgeEncoding(dataset,state) {
  const lens=LENSES.includes(state?.lens)?state.lens:'none',nodes=nodeEncoding(dataset,state);
  if(nodes.mode!=='attribute')return {lens:'none',classOf:()=>null,colorOf:()=>null,dimmed:()=>false};
  const classify=([a,b])=>{const ga=nodes.groupOf(a),gb=nodes.groupOf(b);return ga===null||gb===null?null:ga===gb?'same':'mixed';};
  const isolating=(state?.groups||[]).length>0;
  return {
    lens,
    classOf:edge=>lens==='none'?null:classify(edge),
    colorOf:edge=>lens==='none'?null:TIE_CLASS_COLORS[classify(edge)]||null,
    dimmed:edge=>(lens!=='none'&&classify(edge)!==lens)||(isolating&&nodes.dimmed(edge[0])&&nodes.dimmed(edge[1]))
  };
}

// Same-group tie share of one or more generated graphs, with the roster-pair reference: the share of all possible
// persona pairs (both with a known group) that are same-group. Edges with an unknown end are not counted.
// Descriptive only: it says how the generated ties are distributed, not why, and not how people would connect.
export function tieMixing(personas,edgeLists,key) {
  const groupById=new Map(personas.map(person=>[String(person.id),groupOf(person,key)]));
  const sizes=new Map();
  for(const group of groupById.values())if(group!==null)sizes.set(group,(sizes.get(group)||0)+1);
  const known=[...sizes.values()].reduce((sum,n)=>sum+n,0),pairs=known*(known-1)/2;
  const samePairs=[...sizes.values()].reduce((sum,n)=>sum+n*(n-1)/2,0);
  let ties=0,same=0;
  for(const edges of edgeLists)for(const [a,b] of edges){
    const ga=groupById.get(String(a)),gb=groupById.get(String(b));
    if(ga===null||gb===null||ga===undefined||gb===undefined)continue;
    ties++;if(ga===gb)same++;
  }
  return {graphs:edgeLists.length,ties,same,mixed:ties-same,share:ties?same/ties:null,
    reference:{pairs,same:samePairs,share:pairs?samePairs/pairs:null}};
}

// Canonical undirected tie key with numeric ordering ('2','10' -> 2 before 10). Ids are strings in the dataset.
const order=(a,b)=>{const x=Number(a),y=Number(b);return Number.isFinite(x)&&Number.isFinite(y)&&x!==y?x-y:String(a).localeCompare(String(b));};
const normal=([a,b])=>order(a,b)<=0?[String(a),String(b)]:[String(b),String(a)];
const keyOf=edge=>{const [a,b]=normal(edge);return a+'|'+b;};

// Tie frequency across any number of graphs: [{edge:[a,b],count,total}] sorted by count desc, then edge.
// A graph listing the same tie twice counts once. total = number of graphs, including empty ones.
export function consensus(runs) {
  const counts=new Map();
  for(const run of runs){
    const seen=new Set();
    for(const edge of run.edges||[]){const key=keyOf(edge);if(seen.has(key))continue;seen.add(key);counts.set(key,(counts.get(key)||0)+1);}
  }
  const total=runs.length;
  return [...counts].map(([key,count])=>({edge:key.split('|'),count,total}))
    .sort((x,y)=>y.count-x.count||order(x.edge[0],y.edge[0])||order(x.edge[1],y.edge[1]));
}
// Summary of a consensus list: distinct ties, ties in every graph, ties at or above a share threshold.
export function consensusStats(list,total,minShare=.5) {
  const need=Math.max(1,Math.ceil(minShare*total-1e-9));
  return {graphs:total,distinct:list.length,inAll:total?list.filter(row=>row.count===total).length:0,
    atLeast:total?list.filter(row=>row.count>=need).length:0,minCount:need,single:list.filter(row=>row.count===1).length};
}

// Pairwise tie overlap: matrix[i][j] = {shared, union, jaccard}; jaccard is null when both graphs are empty.
export function overlapMatrix(runs) {
  const sets=runs.map(run=>new Set((run.edges||[]).map(keyOf)));
  const matrix=sets.map(a=>sets.map(b=>{
    let shared=0;for(const key of a)if(b.has(key))shared++;
    const union=a.size+b.size-shared;
    return {shared,union,jaccard:union?shared/union:null};
  }));
  return {ids:runs.map(run=>run.run_id),matrix};
}

// The one population statement. Counts are recomputed from the loaded slots and asserted against COUNTS; any
// mismatch throws so the page shows an error instead of a number it cannot back.
export function ledgerFacts(dataset,counts=dataModule.COUNTS) {
  const slots=dataset?.slots||[],status=name=>slots.filter(slot=>slot?.status===name).length;
  const facts={attempts:slots.length,valid:slots.length-status('failed'),empty:status('empty'),failures:status('failed'),
    supplements_excluded:dataset?.counts?.supplements_excluded,personas:(dataset?.personas||[]).length,
    models:dataset?.design?.models?.length,methods:dataset?.design?.methods?.length,settings:dataset?.design?.settings?.length,seeds:dataset?.design?.seeds?.length};
  const runs=dataset?.runs||[];
  const problems=[];
  for(const key of ['attempts','valid','empty','failures','supplements_excluded','personas'])if(facts[key]!==counts[key])problems.push(key);
  if(runs.length!==facts.valid)problems.push('runs');
  if(runs.filter(run=>!run.edges?.length).length!==facts.empty)problems.push('empty runs');
  if(facts.models*facts.methods*facts.settings*facts.seeds!==facts.attempts)problems.push('design');
  if(problems.length)throw new Error(`Study ledger mismatch (${problems.join(', ')}). No counts are shown rather than unverified ones.`);
  return facts;
}

// ---------------------------------------------------------------------------------------------------------------
// Analysis sections below the stage (trajectory, decision ledger, research question, metric bars, tie comparison).
// Pure: every number these sections display is computed here from the loaded runs and covered by unit tests.
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const DESIGN_KEYS=['model','method','culture','language','seed'];
const FIELD_NOUN={model:'model',method:'generation method',culture:'country framing',language:'instruction language',setting:'setting',seed:'seed'};
const AXIS_NAME={model:'Model',method:'Method',culture:'Country',language:'Language',setting:'Setting',seed:'Seed'};
// Which study fields vary across the stacked runs. Country and language form one "setting" when both vary.
export function comparisonDesign(runs) {
  const list=runs||[],varying=DESIGN_KEYS.filter(key=>new Set(list.map(run=>run[key])).size>1);
  const fields=varying.includes('culture')&&varying.includes('language')?varying.filter(key=>key!=='language').map(key=>key==='culture'?'setting':key):varying;
  const kind=list.length<2?'single':!fields.length?'identical':fields.length===1?fields[0]:'mixed';
  const first=list[0],parts=first?runParts(first):null;
  const fixedParts=!parts?[]:[
    !varying.includes('model')&&parts.model,
    !varying.includes('method')&&parts.method,
    !varying.includes('culture')&&!varying.includes('language')?parts.setting:!varying.includes('culture')?countryOnly(first):!varying.includes('language')?`${languageOnly(first)} instructions`:null,
    !varying.includes('seed')&&parts.seed].filter(Boolean);
  const short=run=>{
    const p=runParts(run),out=[];
    if(varying.includes('model'))out.push(p.model);
    if(varying.includes('method'))out.push(p.method);
    if(fields.includes('setting'))out.push(p.setting);else if(varying.includes('culture'))out.push(countryOnly(run));else if(varying.includes('language'))out.push(languageOnly(run));
    if(varying.includes('seed'))out.push(fields.length===1?String(run.seed):p.seed);
    return out.join(LABEL_SEPARATOR)||'Run';
  };
  return {kind,fields,varying,runs:list.length,shared:fixedParts.join(LABEL_SEPARATOR),
    axis:fields.length===1?AXIS_NAME[fields[0]]:'Run',labels:list.map(short)};
}
const countryOnly=run=>runParts(run).setting.split(' / ')[0];
const languageOnly=run=>runParts(run).setting.split(' / ')[1]||'';
const listWords=words=>words.length<2?words.join(''):`${words.slice(0,-1).join(', ')} and ${words[words.length-1]}`;
// The research question each kind of stack can answer. RQ2 (attribute mixing) applies to any stack.
export const QUESTION_FOR={seed:'seed',method:'method',model:'rq3',culture:'rq1',language:'rq4'};
const QUESTION_FIELD={seed:'seed',method:'method',rq3:'model',rq1:'culture',rq4:'language'};
const MATCH_BUTTON={method:'Method',rq3:'Model',rq1:'Country',rq4:'Language'};
// One sentence for what the stack shows and at most one caveat. Design problems outrank a question mismatch,
// which outranks the caveat that belongs to the comparison itself.
export function comparisonReading(runs,question='') {
  const design=comparisonDesign(runs),n=design.runs,first=runs?.[0];
  const p=first?runParts(first):{};
  const sentence={
    single:'One run is stacked. Add runs that differ in one field to compare them.',
    identical:`These ${n} runs record the same condition and seed.`,
    seed:`These ${n} runs differ only in seed, so they show run-to-run variation at a fixed condition.`,
    method:`These ${n} runs differ only in generation method, at one model, setting and seed.`,
    model:`These ${n} runs differ only in model, at one method, setting and seed.`,
    culture:`These ${n} runs differ only in country framing, with ${languageOnly(first||{})} instructions throughout.`,
    language:`These ${n} runs differ only in instruction language, at ${countryOnly(first||{})} framing.`,
    setting:`These ${n} runs differ in country framing and language together.`,
    mixed:`These ${n} runs differ in ${listWords(design.fields.map(f=>FIELD_NOUN[f]))}, so no single factor is isolated.`
  }[design.kind];
  const own={
    method:'Methods differ in call budget and interaction order, so differences are descriptive.',
    model:'Model configurations support different decoding settings; this compares configurations, not architectures.',
    culture:first&&first.language!=='english'?'RQ1 is defined with English instructions; this framing comparison uses another language.':null,
    language:'Translations have not been independently reviewed for equivalence.',
    setting:'Country and language change together, so their effects cannot be separated.',
    mixed:'Use a matched comparison to vary one field at a time.'
  }[design.kind]||null;
  let issue=null;
  if(runs?.length){
    if(new Set(runs.map(r=>r.roster_id)).size>1)issue='The runs use different persona rosters.';
    else if(new Set(runs.map(r=>r.study)).size>1)issue='The runs come from different study collections.';
    else if(new Set(runs.map(r=>r.prompt_variant||'unrecorded')).size>1)issue='The runs use different prompt or source versions.';
    else if(runs.some(r=>r.language_confound))issue='Historical language prompts also changed the persona language.';
  }
  const wanted=QUESTION_FIELD[question];
  const fits=!wanted||n<2||design.kind===wanted;
  const mismatch=fits?null:`This stack does not vary ${FIELD_NOUN[wanted]} alone${MATCH_BUTTON[question]?`; use the ${MATCH_BUTTON[question]} matched comparison for it`:''}.`;
  const caveat=issue||mismatch||own;
  return {design,sentence,caveat,matched:!issue&&!mismatch&&n>1&&design.kind!=='mixed'&&design.kind!=='setting',suggested:QUESTION_FOR[design.kind]||null};
}
// Grammatical overlap line: "1 tie shared by all 8 runs · 780 distinct ties".
export function overlapLine(shared,union,runs) {
  if(runs<2)return `${plural(union,'distinct tie')} in one run.`;
  return `${plural(shared,'tie')} shared by ${runs===2?'both':'all '+runs} runs · ${plural(union,'distinct tie')}`;
}

// Axis with round tick values covering [lo, hi]; always includes 0.
export function niceAxis(lo,hi,count=4) {
  let min=Math.min(0,Number.isFinite(lo)?lo:0),max=Math.max(0,Number.isFinite(hi)?hi:0);
  if(min===max)max=min+1;
  const raw=(max-min)/count,power=10**Math.floor(Math.log10(raw)),step=[1,2,2.5,5,10].map(f=>f*power).find(s=>s>=raw-1e-12);
  min=Math.floor(min/step+1e-9)*step;max=Math.ceil(max/step-1e-9)*step;
  const round=v=>Number(v.toFixed(10)),ticks=[];
  for(let i=0;min+i*step<=max+step/2;i++)ticks.push(round(min+i*step));
  return {min:round(min),max:round(max),step:round(step),ticks,decimals:(String(round(step)).split('.')[1]||'').length};
}
const decimalsFor=values=>values.every(v=>Number.isInteger(v))?0:3;
// Values, mean, range and signed difference from the active run for one saved metric.
export function metricSummary(runs,metric,activeId) {
  const rows=(runs||[]).map(run=>({run_id:run.run_id,value:Number.isFinite(run.metrics?.[metric])?run.metrics[metric]:null}));
  const finite=rows.map(r=>r.value).filter(v=>v!==null);
  const active=rows.find(r=>r.run_id===activeId)?.value??null;
  for(const row of rows){row.active=row.run_id===activeId;row.delta=row.value!==null&&active!==null&&!row.active?row.value-active:null;}
  const mean=finite.length?finite.reduce((a,b)=>a+b,0)/finite.length:null;
  return {rows,n:finite.length,missing:rows.length-finite.length,mean,min:finite.length?Math.min(...finite):null,max:finite.length?Math.max(...finite):null,
    decimals:decimalsFor(finite),axis:niceAxis(finite.length?Math.min(...finite):0,finite.length?Math.max(...finite):1)};
}
const fmtNumber=(value,decimals)=>value===null?'NA':value.toFixed(decimals).replace(/^-/,'−');
const fmtSigned=(value,decimals)=>value===null?'':`${value>0?'+':value<0?'−':'±'}${Math.abs(value).toFixed(decimals)}`;
// Aligned horizontal bars on one labelled axis. Rows are labelled only by what differs between the runs.
export function metricChartHtml(runs,metric,{activeId,title='Value'}={}) {
  const s=metricSummary(runs,metric,activeId),design=comparisonDesign(runs),{min,max}=s.axis,span=max-min;
  const pos=v=>100*(v-min)/span,zero=pos(0),d=s.decimals;
  const many=runs.length>1;
  const grid=s.axis.ticks.map(t=>`<span class="mchart-grid" style="left:${pos(t).toFixed(2)}%" aria-hidden="true"></span>`).join('');
  const meanMark=s.mean!==null&&many?`<span class="mchart-mean" style="left:${pos(s.mean).toFixed(2)}%" aria-hidden="true"></span>`:'';
  const head=`<div class="mchart-row mchart-head" aria-hidden="true"><span>${escapeHtml(design.axis)}</span><span class="mchart-axis">${s.axis.ticks.map(t=>`<span style="left:${pos(t).toFixed(2)}%">${escapeHtml(fmtNumber(t,s.axis.decimals))}</span>`).join('')}</span><span>${many?'vs active':''}</span></div>`;
  const body=s.rows.map((row,i)=>{
    const v=row.value,left=v===null?zero:Math.min(zero,pos(v)),width=v===null?0:Math.abs(pos(v)-zero);
    return `<div class="mchart-row${row.active?' is-active':''}" role="row"><span role="rowheader" class="mchart-label"><i class="mchart-swatch" style="background:var(--layer-${i%8+1})" aria-hidden="true"></i>${escapeHtml(design.labels[i])}${row.active?'<em class="mchart-active">active</em>':''}</span>`+
      `<span role="cell" class="mchart-track">${grid}${meanMark}${v===null?'<span class="mchart-na">NA</span>':`<span class="mchart-bar" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"></span><span class="mchart-value" style="left:${(v>=0?left+width:left).toFixed(2)}%" data-side="${v>=0?'end':'start'}">${escapeHtml(fmtNumber(v,d))}</span>`}</span>`+
      `<span role="cell" class="mchart-delta${row.delta>0?' up':row.delta<0?' down':''}">${many?escapeHtml(row.active?'—':fmtSigned(row.delta,d)||'NA'):''}</span></div>`;
  }).join('');
  const summary=s.n?(many?`Mean ${fmtNumber(s.mean,d)} · range ${fmtNumber(s.min,d)} to ${fmtNumber(s.max,d)} · ${plural(s.n,'run')}${s.missing?` (${s.missing} NA)`:''}`:`${plural(1,'run')}: ${fmtNumber(s.min,d)}`):'No defined values for this measure.';
  return `<div class="mchart" role="table" aria-label="${escapeHtml(title)} by ${escapeHtml(design.axis.toLowerCase())}">${head}<div class="mchart-body" role="rowgroup">${body}</div><p class="mchart-summary">${escapeHtml(summary)}${many&&s.mean!==null?' <span class="mchart-mean-key" aria-hidden="true"></span> dashed line = mean':''}</p></div>`;
}

// Persona pairs written lower id first with an en dash: "7 – 46".
export const pairLabel=edge=>normal(edge).join(' – ');
// Tie presence across runs, sorted by count (desc) then numeric pair; optionally only ties of one persona.
export function tieRows(runs,selected='') {
  const list=runs||[],sets=list.map(run=>new Set((run.edges||[]).map(keyOf)));
  return consensus(list).filter(row=>!selected||row.edge.includes(String(selected))).map(row=>{
    const key=row.edge.join('|');
    return {edge:row.edge,label:pairLabel(row.edge),count:row.count,total:row.total,present:sets.map(set=>set.has(key))};
  });
}
export const TIE_ROW_LIMIT=12;
export function tieTableHtml(runs,{selected='',showAll=false,limit=TIE_ROW_LIMIT}={}) {
  const rows=tieRows(runs,selected),design=comparisonDesign(runs),n=runs.length,shown=showAll?rows:rows.slice(0,limit);
  if(!rows.length)return `<p class="empty-note">${selected?`Persona ${escapeHtml(selected)} has no ties in these runs.`:'These runs have no ties.'}</p>`;
  const cols=design.labels.map((text,i)=>`<th scope="col" class="tie-run"><span style="--swatch:var(--layer-${i%8+1})">${escapeHtml(text)}</span></th>`).join('');
  const body=shown.map(row=>`<tr><th scope="row">${escapeHtml(row.label)}</th><td class="tie-count"><span class="tie-meter" aria-hidden="true"><span style="width:${(100*row.count/row.total).toFixed(1)}%"></span></span>${row.count} / ${row.total}</td>${row.present.map((on,i)=>`<td class="tie-dot"><span class="dot${on?' on':''}" aria-hidden="true"></span><span class="vh">${on?'present':'absent'} in ${escapeHtml(design.labels[i])}</span></td>`).join('')}</tr>`).join('');
  const more=rows.length>limit?`<button type="button" class="tie-more" data-action="toggle-ties" aria-expanded="${showAll}">${showAll?`Show first ${limit}`:`Show all ${rows.length}`}</button>`:'';
  return `<div class="tie-table"><table><caption class="vh">Ties${selected?` of persona ${escapeHtml(selected)}`:''} by run. Filled dot: the tie is in that run's saved final graph.</caption><thead><tr><th scope="col" rowspan="2">Personas</th><th scope="col" rowspan="2">Runs with tie</th><th scope="colgroup" colspan="${n}" class="tie-group">${escapeHtml(design.axis)}</th></tr><tr>${cols}</tr></thead><tbody>${body}</tbody></table></div><div class="tie-foot"><p class="hint"><span class="dot on" aria-hidden="true"></span> in the saved final graph <span class="dot" aria-hidden="true"></span> absent · showing ${shown.length} of ${plural(rows.length,'tie')}</p>${more}</div>`;
}

// Decision ledger rows: one per recorded update (only those touching the persona, when one is selected).
const PHASES={local:'Local','iterative-add':'Iterative add','iterative-drop':'Iterative drop',sequential:'Sequential',global:'Global batch',iterative:'Iterative'};
export const phaseName=method=>PHASES[method]||capitalWords(method);
const capitalWords=value=>String(value??'').replace(/(^|[\s-])([a-z])/g,(m,lead,c)=>(lead==='-'?' ':lead)+c.toUpperCase());
export function ledgerRows(events,selected='') {
  const person=selected?String(selected):'';
  const rows=[];
  (events||[]).forEach((event,i)=>{
    const touches=edge=>!person||edge.map(String).includes(person);
    const added=(event.added||[]).filter(touches).length,removed=(event.removed||[]).filter(touches).length;
    if(person&&!added&&!removed&&String(event.persona)!==person)return;
    rows.push({step:i+1,persona:event.persona==null?null:String(event.persona),phase:event.persona==null?'Global batch':phaseName(event.method),
      method:event.method||'',added,removed,batchAdded:(event.added||[]).length,batchRemoved:(event.removed||[]).length,pair:null});
  });
  for(let i=1;i<rows.length;i++){const a=rows[i-1],b=rows[i];
    if(!a.pair&&b.step===a.step+1&&a.persona!==null&&a.persona===b.persona&&/add$/.test(a.method)&&/drop$/.test(b.method)){a.pair='start';b.pair='end';}}
  return rows;
}

// Connection count by recorded event, plus the stats printed beside the chart.
export function trajectorySeries(frames,person='') {
  if(!frames?.length)return null;
  const values=frames.map(frame=>person?neighbors(frame.edges,person):frame.edges.length);
  return {values,events:values.length-1,final:values[values.length-1],peak:Math.max(...values)};
}
function neighbors(edges,person){const p=String(person),seen=new Set();for(const [a,b] of edges){if(String(a)===p)seen.add(String(b));else if(String(b)===p)seen.add(String(a));}return seen.size;}
export function trajectoryChartHtml(frames,person,step,{width=680}={}) {
  const series=trajectorySeries(frames,person);
  if(!series)return '<p class="empty-note">No recorded history for this run; only its final graph was saved.</p>';
  // One recorded response (Global): there is no trajectory, only "nothing" and "everything". Say so; draw no line.
  if(series.events===1&&frames[1].event?.persona==null)return `<p class="empty-note">${person?`Persona ${escapeHtml(person)}’s ${plural(series.final,'tie')} `:`All ${plural(series.final,'tie')} `}arrived in one recorded response, so there is no step-by-step trajectory to plot.</p>`;
  const at=Math.max(0,Math.min(series.events,step|0)),values=series.values,w=Math.max(300,Math.round(width)),h=200,left=48,right=16,top=14,bottom=44;
  const yAxis=niceAxis(0,series.peak,4),xAxis=niceAxis(0,Math.max(1,series.events),w<480?3:5);
  const x=i=>left+i*(w-left-right)/Math.max(1,series.events),y=v=>h-bottom-(v-yAxis.min)*(h-top-bottom)/(yAxis.max-yAxis.min);
  const path=values.map((v,i)=>i?`H${x(i).toFixed(1)}V${y(v).toFixed(1)}`:`M${x(0).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const what=person?`Ties of persona ${escapeHtml(person)}`:'Ties in the network';
  const yTicks=yAxis.ticks.map(v=>`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${left-8}" y="${y(v)+4}" text-anchor="end">${v}</text>`).join('');
  const xTicks=xAxis.ticks.filter(v=>v<=series.events&&Number.isInteger(v)).map(v=>`<text class="tick" x="${x(v)}" y="${h-bottom+16}" text-anchor="middle">${v}</text>`).join('');
  const cx=x(at),labelRight=cx>w-150;
  const stats=[['Cursor',`event ${at} of ${series.events}`],['Ties at cursor',plural(values[at],'tie')],['Final',plural(series.final,'tie')],['Peak',plural(series.peak,'tie')]];
  return `<dl class="traj-stats">${stats.map(([k,v])=>`<div><dt>${k}</dt><dd>${escapeHtml(v)}</dd></div>`).join('')}</dl>`+
    `<svg class="traj-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${what} by recorded event: ${values[0]} at event 0, ${values[at]} at event ${at}, ${series.final} at event ${series.events}."><title>${what} by recorded event</title>`+
    `${yTicks}${xTicks}<path class="traj-line" d="${path}"/>`+
    `<line class="traj-cursor" x1="${cx}" x2="${cx}" y1="${top}" y2="${h-bottom}"/><circle class="traj-dot" cx="${cx}" cy="${y(values[at])}" r="4.5"/>`+
    `<text class="traj-cursor-label" x="${labelRight?cx-8:cx+8}" y="${top+10}" text-anchor="${labelRight?'end':'start'}">Event ${at} · ${plural(values[at],'tie')}</text>`+
    `<text class="axis-title" x="${left+(w-left-right)/2}" y="${h-6}" text-anchor="middle">Recorded event (generation order)</text>`+
    `<text class="axis-title" transform="translate(13 ${top+(h-top-bottom)/2}) rotate(-90)" text-anchor="middle">Ties</text></svg>`;
}

// ---------------------------------------------------------------------------------------------------------------
// DOM. Everything below runs only in the browser.
const SOURCE='layers-insight';
const REPOSITORY='https://github.com/hemu77/UA_Capstone/blob/main/';
const percent=(value,digits=1)=>value===null||value===undefined||!Number.isFinite(value)?'NA':`${(value*100).toFixed(digits)}%`;
const fixed=value=>Number.isFinite(value)?value.toFixed(3):'NA';
function h(tag,props={},...children) {
  const node=document.createElement(tag);
  for(const [key,value] of Object.entries(props||{})){
    if(value===undefined||value===null||value===false)continue;
    if(key==='class')node.className=value;
    else if(key==='text')node.textContent=value;
    else if(key==='style')for(const [name,v] of Object.entries(value))node.style.setProperty(name,v);
    else if(key.startsWith('on'))node.addEventListener(key.slice(2),value);
    else if(key in node&&typeof value!=='string')node[key]=value;
    else node.setAttribute(key,value===true?'':String(value));
  }
  for(const child of children.flat())if(child!==null&&child!==undefined&&child!==false)node.append(child instanceof Node?child:String(child));
  return node;
}
// Re-render a container without losing keyboard focus on an element carrying data-focus.
function keepFocus(container,render) {
  const active=document.activeElement,key=container.contains(active)?active?.dataset?.focus:null;
  render();
  if(key)[...container.querySelectorAll('[data-focus]')].find(node=>node.dataset.focus===key&&!node.disabled)?.focus({preventScroll:true});
}
const replayText={actual_recorded_graph_changes:'Recorded graph changes, replayed in generation order (not elapsed time)'};

export function mountInsight(app,{root,ledger,legend}={}) {
  if(typeof document==='undefined')return {destroy(){}};
  const cleanups=[];
  let data=null,state=app.getState(),scheduled=0,pending=new Set(),metricsState='unknown',destroyed=false;
  let consensusScope='stack',threshold=.5;
  const supportsConsensusPlane=Object.hasOwn(state,'consensus');

  // --- Study ledger: the single statement of 896 / 890 / 12 / 6 ------------------------------------------------
  const ledgerBox=ledger?h('div',{class:'ins-ledger','data-state':'loading'},h('p',{class:'ins-ledger-line',text:'Verifying the study population…'})):null;
  if(ledger)ledger.replaceChildren(ledgerBox);
  function renderLedger() {
    if(!ledgerBox||!data)return;
    let facts;
    try{facts=ledgerFacts(data);}catch(error){ledgerBox.dataset.state='error';ledgerBox.replaceChildren(h('p',{class:'ins-ledger-line',role:'alert',text:error.message}));return;}
    ledgerBox.dataset.state='ready';
    const n=value=>h('strong',{text:value.toLocaleString('en-US')});
    const line=h('p',{class:'ins-ledger-line'},
      h('span',{class:'ins-ledger-fact'},n(facts.attempts),' attempts'),
      h('span',{class:'ins-ledger-fact'},n(facts.valid),' valid graphs (',n(facts.empty),' empty)'),
      h('span',{class:'ins-ledger-fact'},n(facts.failures),' failures'),
      h('span',{class:'ins-ledger-fact ins-ledger-excluded'},n(facts.supplements_excluded),' supplemental graphs excluded'));
    const row=(term,...detail)=>[h('dt',{text:term}),h('dd',{},...detail)];
    const notes=data.notes||data;
    const panel=h('div',{class:'ins-ledger-panel'},
      h('dl',{},
        row('Design',`${facts.models} models × ${facts.methods} generation methods × ${facts.settings} country / language settings × ${facts.seeds} seeds = ${facts.attempts.toLocaleString('en-US')} planned attempts, one per slot.`),
        row('Outcome',`${facts.valid.toLocaleString('en-US')} attempts produced a valid graph; ${facts.empty} of those graphs have no ties. ${facts.failures} attempts failed and are shown as hollow slots, never as graphs.`),
        row('Roster',`${facts.personas} fictional adult personas, the same roster in every attempt. Attributes: gender, age, religion, political orientation.`),
        row('Supplements',`${facts.supplements_excluded} supplemental graphs from later runs are excluded from every count, filter, comparison and aggregate. They appear only in the `,h('a',{href:'#saved-plots',text:'supplement panel'}),'.'),
        row('Reading the view','A plane is one saved run, not a time slice or a country. Positions are an algorithmic layout, not geography or social distance. Replay follows recorded generation order, not elapsed time. Country framing is a prompt condition. Every figure shown is descriptive, not a significance test.'),
        notes.limitations?row('Limitations',String(notes.limitations)):null,
        notes.verification?row('Verification',String(notes.verification)):null));
    const more=h('details',{class:'ins-ledger-more'},h('summary',{class:'ins-ledger-toggle'},'Study design'),panel);
    // The popover is fixed to the viewport so no scrolling dock or sheet can clip it; it is placed under its toggle.
    const place=()=>{
      if(!more.open)return;
      const r=more.firstElementChild.getBoundingClientRect(),gap=8,edge=16,width=Math.min(560,innerWidth-2*edge);
      // In the phone sheet the fixed tab bar covers the bottom of the viewport.
      const tabbar=document.body.dataset.shell==='sheet'?parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--tabbar-h'))||56:0;
      const below=innerHeight-tabbar-r.bottom-gap-edge,above=r.top-gap-edge,down=below>=220||below>=above;
      panel.style.width=width+'px';panel.style.left=Math.max(edge,Math.min(r.left,innerWidth-edge-width))+'px';
      panel.style.maxHeight=Math.max(160,Math.min(520,down?below:above))+'px';
      panel.style.top=down?r.bottom+gap+'px':'auto';panel.style.bottom=down?'auto':innerHeight-r.top+gap+'px';
    };
    const dismiss=event=>{if(event.type==='keydown'?event.key==='Escape':!more.contains(event.target)){const hadFocus=more.contains(document.activeElement);more.open=false;if(event.type==='keydown'){event.stopPropagation();if(hadFocus)more.firstElementChild.focus();}}};
    more.addEventListener('toggle',()=>{
      place();
      for(const [type,fn,options] of [['resize',place],['scroll',place,true],['pointerdown',dismiss,true],['keydown',dismiss,true]])(more.open?addEventListener:removeEventListener)(type,fn,options);
    });
    ledgerBox.replaceChildren(h('div',{class:'ins-ledger-facts'},line),more);
  }

  // --- Static controls inside #insight-root -------------------------------------------------------------------
  const colorButtons=new Map(),lensButtons=new Map();
  const colorGroup=h('div',{class:'ins-segment',role:'group','aria-label':'Colour personas by'});
  for(const [key,label] of [['layer','Run'],...ATTRIBUTES.map(a=>[a.key,a.short])]){
    const button=h('button',{type:'button',class:'ins-toggle','data-focus':'color:'+key,'aria-pressed':'false',text:label,
      title:key==='layer'?'Colour each plane by its run':`Colour personas by ${attributeInfo(key).label.toLowerCase()}`,
      onclick:()=>{if(app.getState().colorBy===key)return;app.update({colorBy:key,groups:[],...(key==='layer'?{lens:'none'}:{})},SOURCE);
        app.announce(key==='layer'?'Colouring planes by run.':`Colouring personas by ${attributeInfo(key).label.toLowerCase()}.`);}});
    colorButtons.set(key,button);colorGroup.append(button);
  }
  const lensGroup=h('div',{class:'ins-segment',role:'group','aria-label':'Tie lens'});
  for(const [key,label] of [['none','All ties'],['same','Same-group'],['mixed','Mixed-group']]){
    const button=h('button',{type:'button',class:'ins-toggle','data-focus':'lens:'+key,'aria-pressed':'false',text:label,
      onclick:()=>{if(app.getState().lens===key)return;app.update({lens:key},SOURCE);
        app.announce(key==='none'?'Tie lens off.':`Highlighting ${key}-group ties.`);}});
    lensButtons.set(key,button);lensGroup.append(button);
  }
  const lensHint=h('p',{class:'ins-hint'});
  const readout=h('div',{class:'ins-readout'});
  const lensSection=h('details',{class:'ins-section',open:true},
    h('summary',{class:'ins-summary'},h('span',{class:'ins-title',text:'Attribute lens'}),h('span',{class:'ins-tag',text:'Fictional personas'})),
    h('div',{class:'ins-body'},
      h('div',{class:'ins-control'},h('span',{class:'ins-control-label',text:'Colour'}),colorGroup),
      h('div',{class:'ins-control'},h('span',{class:'ins-control-label',text:'Ties'}),lensGroup),
      lensHint,readout));

  const comparisonBody=h('div',{class:'ins-body'});
  const comparisonSection=h('details',{class:'ins-section',open:true},
    h('summary',{class:'ins-summary'},h('span',{class:'ins-title',text:'Layer overlap'}),h('span',{class:'ins-tag',text:'Descriptive'})),comparisonBody);

  const scopeButtons=new Map();
  const scopeGroup=h('div',{class:'ins-segment',role:'group','aria-label':'Graphs in the aggregate'});
  for(const key of ['stack','filters']){
    const button=h('button',{type:'button',class:'ins-toggle','data-focus':'scope:'+key,'aria-pressed':'false',
      onclick:()=>{if(consensusScope===key)return;consensusScope=key;writeConsensusPlane();queue('consensus');}});
    scopeButtons.set(key,button);scopeGroup.append(button);
  }
  const thresholdInput=h('input',{type:'range',min:'5',max:'100',step:'5',value:'50',class:'ins-range',id:'ins-consensus-threshold',
    'aria-describedby':'ins-consensus-threshold-value',oninput:()=>{threshold=Number(thresholdInput.value)/100;writeConsensusPlane();queue('consensus');}});
  const thresholdValue=h('output',{id:'ins-consensus-threshold-value',for:'ins-consensus-threshold',class:'ins-range-value'});
  const planeToggle=supportsConsensusPlane?h('button',{type:'button',class:'ins-toggle','data-focus':'plane','aria-pressed':'false',text:'Show as extra plane',
    onclick:()=>{const on=!app.getState().consensus;app.update({consensus:on?{scope:consensusScope,min:threshold}:null},SOURCE);app.announce(on?'Aggregate plane shown. It is not a run.':'Aggregate plane hidden.');}}):null;
  const consensusBody=h('div',{class:'ins-consensus-result'});
  const consensusSection=h('details',{class:'ins-section',open:true},
    h('summary',{class:'ins-summary'},h('span',{class:'ins-title',text:'Consensus ties'}),h('span',{class:'ins-tag ins-tag-warn',text:'Aggregate, not a run'})),
    h('div',{class:'ins-body'},
      h('div',{class:'ins-control'},h('span',{class:'ins-control-label',text:'Graphs'}),scopeGroup),
      h('div',{class:'ins-control'},h('label',{class:'ins-control-label',for:'ins-consensus-threshold',text:'Share'}),h('div',{class:'ins-range-row'},thresholdInput,thresholdValue)),
      planeToggle?h('div',{class:'ins-control'},h('span',{class:'ins-control-label',text:'3D'}),planeToggle):null,
      consensusBody));

  const provenanceBody=h('div',{class:'ins-body'});
  const provenanceSection=h('details',{class:'ins-section ins-provenance'},
    h('summary',{class:'ins-summary'},h('span',{class:'ins-title',text:'Provenance'}),h('span',{class:'ins-tag',text:'Active run'})),provenanceBody);

  const shell=h('div',{class:'ins-root'},lensSection,comparisonSection,consensusSection,provenanceSection);
  if(root)root.replaceChildren(shell);
  const legendBox=h('div',{class:'ins-legend'});
  if(legend)legend.replaceChildren(legendBox);

  // --- Renderers ------------------------------------------------------------------------------------------------
  const currentRuns=()=>stackRuns(data,state);
  let scopeCache={filters:null,runs:[]};
  function scopeRuns() {
    if(scopeCache.filters!==state.filters){scopeCache={filters:state.filters,runs:applyFilters(data.runs,state.filters)};}
    return scopeCache.runs;
  }

  function renderControls() {
    const attribute=attributeInfo(state.colorBy);
    for(const [key,button] of colorButtons)button.setAttribute('aria-pressed',String(key===state.colorBy||(key==='layer'&&!attribute)));
    for(const [key,button] of lensButtons){button.setAttribute('aria-pressed',String(attribute?key===state.lens:key==='none'));button.disabled=!attribute&&key!=='none';}
    lensHint.textContent=attribute
      ?`Same-group: both personas share a ${attribute.key==='age'?'age band (18–24, ten-year bands to 64, then 65+)':attribute.label.toLowerCase()}. Shares describe ties the model generated for fictional personas, not human preferences.`
      :'Choose an attribute to colour personas and to split ties into same-group and mixed-group.';
  }

  function legendItem(group,attribute,isolated) {
    const pressed=isolated.has(group.key);
    return h('button',{type:'button',class:'ins-legend-item','data-focus':'group:'+group.key,'aria-pressed':String(pressed),
      title:`${pressed?'Show':'Isolate'} ${group.label} (${plural(group.count,'persona')})`,
      onclick:()=>{const now=new Set(app.getState().groups);now.has(group.key)?now.delete(group.key):now.add(group.key);
        const next=attribute.groups.map(([key])=>key).filter(key=>now.has(key)).concat([...now].filter(key=>!attribute.groups.some(([g])=>g===key)));
        app.update({groups:next},SOURCE);app.announce(next.length?`Isolating ${next.length===1?group.label:plural(next.length,'group')}.`:'All groups shown.');}},
      h('i',{class:'ins-swatch',style:{'--swatch':group.color},'aria-hidden':'true'}),
      h('b',{text:group.label}),h('small',{text:String(group.count)}));
  }
  function renderLegend() {
    const encoding=nodeEncoding(data,state);
    keepFocus(legendBox,()=>{
      if(encoding.mode==='layer'){
        legendBox.dataset.mode='layer';
        legendBox.setAttribute('role','list');legendBox.setAttribute('aria-label','Plane colours');
        legendBox.replaceChildren(...(encoding.legend.length?encoding.legend.map(item=>h('div',{class:'ins-legend-layer',role:'listitem',title:runLabel(runsById(data).get(item.key))},
          h('i',{class:'ins-swatch',style:{'--swatch':item.color},'aria-hidden':'true'}),h('b',{text:item.label}))):[h('em',{class:'ins-legend-note',text:'Stack runs to colour planes by run.'})]));
        return;
      }
      const attribute=attributeInfo(encoding.attribute),isolated=new Set(state.groups);
      legendBox.dataset.mode='attribute';
      legendBox.setAttribute('role','group');legendBox.setAttribute('aria-label',`${attribute.label}: select groups to isolate them`);
      const reset=h('button',{type:'button',class:'ins-legend-reset','data-focus':'group-reset',text:'Show all',disabled:!isolated.size,
        onclick:()=>{app.update({groups:[]},SOURCE);app.announce('All groups shown.');}});
      legendBox.replaceChildren(h('em',{class:'ins-legend-title',text:attribute.label}),...encoding.legend.map(group=>legendItem(group,attribute,isolated)),reset);
    });
  }

  function shareCell(result) {
    const share=result.share,ref=result.reference.share;
    if(share===null)return h('td',{class:'ins-share ins-na',text:'No ties'});
    return h('td',{class:'ins-share'},
      h('span',{class:'ins-bar',style:{'--share':String(share),'--ref':String(ref??0)},'aria-hidden':'true'}),
      h('span',{class:'ins-share-value',text:percent(share)}));
  }
  function savedValue(run,attribute) {
    const value=run.metrics?.[attribute.saved];
    if(Number.isFinite(value))return h('td',{class:'ins-num',text:fixed(value)});
    if(value===null||(run.metrics&&Object.hasOwn(run.metrics,attribute.saved)))return h('td',{class:'ins-num ins-na',text:'NA',title:'Undefined for this graph in the saved analysis (for example no ties).'});
    return h('td',{class:'ins-num ins-na',text:metricsState==='loading'?'…':'NA',title:metricsState==='loading'?'Loading saved metrics':'Saved metric unavailable'});
  }
  function renderReadout() {
    const attribute=attributeInfo(state.colorBy),runs=currentRuns();
    if(!attribute){
      readout.replaceChildren();return;
    }
    const personas=data.personas,reference=tieMixing(personas,[],attribute.key).reference;
    const intro=h('p',{class:'ins-reference'},'Roster-pair reference: ',h('strong',{text:percent(reference.share)}),
      ` of the ${reference.pairs.toLocaleString('en-US')} possible persona pairs are same-group. A share near the reference means ties mix groups about as often as the roster allows.`);
    const rows=runs.map((run,i)=>{const result=tieMixing(personas,[run.edges],attribute.key);
      return h('tr',{'data-active':String(run.run_id===state.active)},
        h('th',{scope:'row'},h('i',{class:'ins-swatch',style:{'--swatch':LAYER_COLORS[i%LAYER_COLORS.length]},'aria-hidden':'true'}),h('span',{text:layerLabel(runs,i),title:runLabel(run)})),
        h('td',{class:'ins-num',text:String(result.ties)}),shareCell(result),savedValue(run,attribute));});
    const table=runs.length?h('table',{class:'ins-table'},
      h('caption',{text:`Same-group share of generated ties per stacked layer. ${attribute.savedLabel} read from the frozen analysis, not recomputed here.`}),
      h('thead',{},h('tr',{},h('th',{scope:'col',text:'Layer'}),h('th',{scope:'col',class:'ins-num',text:'Ties'}),h('th',{scope:'col',text:'Same-group share'}),h('th',{scope:'col',class:'ins-num',text:attribute.key==='age'?'Assort.':'Coleman'}))),
      h('tbody',{},rows)):h('p',{class:'ins-empty',text:'Stack runs to read the share layer by layer.'});
    const scope=scopeRuns(),pooled=tieMixing(personas,scope.map(run=>run.edges),attribute.key);
    const pooledLine=h('p',{class:'ins-pooled'},
      scope.length?[`Pooled over ${plural(scope.length,'graph')} in the current filter: `,h('strong',{text:percent(pooled.share)}),` of ${plural(pooled.ties,'generated tie')} are same-group (reference ${percent(reference.share)}).`]
        :'No graphs match the current filters.');
    readout.replaceChildren(intro,table,pooledLine);
  }

  function renderComparison() {
    const runs=currentRuns();
    if(runs.length<2){
      comparisonBody.replaceChildren(h('p',{class:'ins-empty',text:runs.length?'Stack at least two runs to compare their ties.':`Stack 2 to ${MAX_STACK_LAYERS} runs to compare their ties.`}));
      return;
    }
    const {matrix}=overlapMatrix(runs),list=consensus(runs),stats=consensusStats(list,runs.length,1);
    // Shade is relative to this matrix (lowest to highest off-diagonal value) so small differences stay visible;
    // the printed number is always the absolute Jaccard value.
    const off=matrix.flatMap((row,i)=>row.filter((cell,j)=>i!==j&&cell.jaccard!==null).map(cell=>cell.jaccard));
    const low=Math.min(...off),high=Math.max(...off),shade=value=>value===null||!off.length?0:high>low?.15+.85*(value-low)/(high-low):.5;
    const head=h('tr',{},h('th',{scope:'col'},h('span',{class:'ins-visually-hidden',text:'Layer'})),...runs.map((run,i)=>h('th',{scope:'col',title:runLabel(run)},
      h('i',{class:'ins-swatch',style:{'--swatch':LAYER_COLORS[i%LAYER_COLORS.length]},'aria-hidden':'true'}),String(i+1))));
    const body=runs.map((run,i)=>h('tr',{},h('th',{scope:'row',title:runLabel(run)},h('i',{class:'ins-swatch',style:{'--swatch':LAYER_COLORS[i%LAYER_COLORS.length]},'aria-hidden':'true'}),String(i+1)),
      ...matrix[i].map((cell,j)=>{
        if(i===j)return h('td',{class:'ins-cell ins-diagonal',title:`Layer ${i+1}: ${plural(cell.union,'tie')}`,text:String(cell.union)});
        const value=cell.jaccard;
        return h('td',{class:'ins-cell',style:{'--j':String(shade(value))},'data-dark':String(shade(value)>.6),
          title:`Layers ${i+1} and ${j+1}: ${cell.shared} shared of ${plural(cell.union,'distinct tie')} (Jaccard ${value===null?'undefined':value.toFixed(3)})`,
          text:value===null?'NA':value.toFixed(2).replace(/^0/,'')});
      })));
    comparisonBody.replaceChildren(
      h('div',{class:'ins-matrix-wrap'},h('table',{class:'ins-matrix'},
        h('caption',{text:`Tie overlap (Jaccard: shared ÷ distinct ties) between stacked layers, ${off.length?`from ${low.toFixed(2)} to ${high.toFixed(2)}; shading spans that range`:'undefined for empty layers'}. Diagonal: ties in that layer.`}),
        h('thead',{},head),h('tbody',{},body))),
      h('p',{class:'ins-stat-line'},
        h('span',{},h('strong',{text:String(stats.inAll)}),` shared by all ${runs.length}`),
        h('span',{},h('strong',{text:String(stats.single)}),' in one layer only'),
        h('span',{},h('strong',{text:String(stats.distinct)}),' distinct')),
      h('p',{class:'ins-hint',text:'Overlap is descriptive: it compares generated tie lists, not significance or similarity of underlying preferences.'}));
  }

  function renderConsensus() {
    const runs=consensusScope==='stack'?currentRuns():scopeRuns();
    scopeButtons.get('stack').textContent=`Stacked (${currentRuns().length})`;
    scopeButtons.get('filters').textContent=`Filtered (${scopeRuns().length.toLocaleString('en-US')})`;
    for(const [key,button] of scopeButtons)button.setAttribute('aria-pressed',String(key===consensusScope));
    thresholdInput.setAttribute('aria-valuetext',`${Math.round(threshold*100)}% of graphs`);
    thresholdValue.textContent=`≥ ${Math.round(threshold*100)}% of graphs`;
    if(planeToggle)planeToggle.setAttribute('aria-pressed',String(Boolean(app.getState().consensus)));
    if(!runs.length){consensusBody.replaceChildren(h('p',{class:'ins-empty',text:consensusScope==='stack'?'No runs are stacked.':'No graphs match the current filters.'}));return;}
    const list=consensus(runs),stats=consensusStats(list,runs.length,threshold),edgeClass=edgeEncoding(data,{...state,lens:'same'});
    const attribute=attributeInfo(state.colorBy);
    const kept=list.filter(row=>row.count>=stats.minCount);
    const shown=kept.slice(0,10);
    const items=shown.map(row=>{
      const [a,b]=row.edge,kind=attribute?edgeClass.classOf(row.edge):null;
      const persona=id=>h('button',{type:'button',class:'ins-persona','data-focus':'persona:'+id,'aria-pressed':String(state.persona===id),
        title:`Select persona ${id}`,text:id,onclick:()=>{app.update({persona:app.getState().persona===id?'':id},SOURCE);}});
      return h('li',{class:'ins-tie'},
        h('span',{class:'ins-tie-pair'},persona(a),h('span',{class:'ins-tie-dash','aria-hidden':'true',text:'–'}),persona(b)),
        kind?h('span',{class:`ins-kind ins-kind-${kind}`,text:kind==='same'?'same':'mixed'}):null,
        h('span',{class:'ins-tie-count',text:`${row.count} of ${row.total}`}),
        h('span',{class:'ins-bar ins-bar-plain',style:{'--share':String(row.count/row.total)},'aria-hidden':'true'}));
    });
    const mixing=attribute&&kept.length?tieMixing(data.personas,[kept.map(row=>row.edge)],attribute.key):null;
    consensusBody.replaceChildren(
      h('p',{class:'ins-aggregate-label'},h('strong',{text:`Aggregate of ${plural(runs.length,consensusScope==='stack'?'stacked graph':'filtered graph')}, not a run.`}),
        ' Supplemental graphs are never included.'),
      h('p',{class:'ins-stat-line'},
        h('span',{},h('strong',{text:stats.atLeast.toLocaleString('en-US')}),` ties in ≥ ${stats.minCount} of ${runs.length}`),
        h('span',{},h('strong',{text:stats.inAll.toLocaleString('en-US')}),' in all'),
        h('span',{},h('strong',{text:stats.distinct.toLocaleString('en-US')}),' distinct')),
      mixing?h('p',{class:'ins-hint',text:`${percent(mixing.share)} of these consensus ties are same-${attribute.short.toLowerCase()} (roster reference ${percent(mixing.reference.share)}).`}):null,
      shown.length?h('ol',{class:'ins-ties','aria-label':`Most frequent ties, ${shown.length} of ${kept.length}`},items):h('p',{class:'ins-empty',text:'No tie reaches this share. Lower the threshold.'}),
      kept.length>shown.length?h('p',{class:'ins-hint',text:`Showing the 10 most frequent of ${plural(kept.length,'tie')} at or above the threshold.`}):null,
      h('p',{class:'ins-hint',text:'Counts are the number of generated graphs containing each tie. Descriptive, not a probability or a significance test.'}));
  }

  function renderProvenance() {
    const run=runsById(data).get(state.active)||null;
    if(!run){provenanceBody.replaceChildren(h('p',{class:'ins-empty',text:'Select a run to see where its graph comes from.'}));return;}
    const row=(term,value,mono=false)=>[h('dt',{text:term}),h('dd',{class:mono?'ins-mono':undefined},value)];
    const slot=Number.isInteger(run.slot)?run.slot:null;
    provenanceBody.replaceChildren(
      h('p',{class:'ins-run-label',text:runLabel(run)}),
      h('dl',{class:'ins-facts'},
        row('Status',run.edges.length?`Valid graph · ${plural(run.edges.length,'tie')}`:'Valid graph with no ties'),
        slot!==null?row('Slot',`${slot} (design order, 0–${(data.slots?.length||896)-1})`):null,
        row('Run ID',run.run_id,true),
        run.source?row('Source',h('a',{href:REPOSITORY+run.source,target:'_blank',rel:'noopener',text:run.source.split('/').pop(),title:run.source})):null,
        run.source_sha256?row('Adjacency SHA-256',run.source_sha256,true):null,
        run.record_sha256?row('Record SHA-256',run.record_sha256,true):null,
        run.protocol_sha256?row('Protocol SHA-256',run.protocol_sha256,true):null,
        run.prompt_variant?row('Prompt variant',run.prompt_variant,true):null,
        row('Replay',replayText[run.replay_status]||(run.events_url?'Recorded events, generation order':'No recorded history')),
        run.events_sha256?row('Events SHA-256',run.events_sha256,true):null,
        run.analysis_version?row('Analysis',String(run.analysis_version)):null));
  }

  const renderers={controls:renderControls,legend:renderLegend,readout:renderReadout,comparison:renderComparison,consensus:renderConsensus,provenance:renderProvenance};
  function flush() {
    scheduled=0;if(destroyed||!data)return;
    const work=pending;pending=new Set();
    for(const name of Object.keys(renderers))if(work.has(name)){try{renderers[name]();}catch(error){console.error(`layers-insight ${name} failed`,error);}}
  }
  function queue(...names) {
    for(const name of names)pending.add(name);
    if(!scheduled)scheduled=requestAnimationFrame(flush);
  }
  function writeConsensusPlane() {
    if(supportsConsensusPlane&&app.getState().consensus)app.update({consensus:{scope:consensusScope,min:threshold}},SOURCE);
  }
  function needsMetrics() {return data.runs.some(run=>!run.metrics||!Object.hasOwn(run.metrics,'age_assortativity'));}
  async function ensureMetrics() {
    if(metricsState!=='unknown'||!attributeInfo(state.colorBy))return;
    if(!needsMetrics()){metricsState='ready';return;}
    if(typeof dataModule.loadMetrics!=='function'){metricsState='unavailable';return;}
    metricsState='loading';
    try{await dataModule.loadMetrics(data);metricsState='ready';}
    catch(error){metricsState='unavailable';console.error('Saved metrics could not be loaded',error);}
    if(!destroyed)queue('readout');
  }

  const unsubscribe=app.subscribe((next,changed)=>{
    state=next;if(!data)return;
    if(changed.has('colorBy')||changed.has('lens')||changed.has('groups'))queue('controls','legend','readout','consensus');
    if(changed.has('colorBy'))ensureMetrics();
    if(changed.has('layers'))queue('legend','readout','comparison','consensus','provenance');
    if(changed.has('active'))queue('readout','provenance');
    if(changed.has('filters'))queue('readout','consensus');
    if(changed.has('consensus')&&next.consensus){consensusScope=next.consensus.scope;threshold=next.consensus.min;thresholdInput.value=String(Math.round(threshold*100));}
    if(changed.has('persona')||changed.has('consensus'))queue('consensus');
  });
  cleanups.push(unsubscribe);
  app.ready.then(dataset=>{
    if(destroyed)return;
    data=dataset;state=app.getState();
    if(state.consensus){consensusScope=state.consensus.scope;threshold=state.consensus.min;thresholdInput.value=String(Math.round(threshold*100));} // a link may carry the aggregate plane
    renderLedger();queue(...Object.keys(renderers));ensureMetrics();
    import('./plot-gallery.js').then(({mountSupplementPanel})=>{
      if(destroyed)return;
      const section=document.getElementById('saved-plots');
      if(section&&typeof mountSupplementPanel==='function'){const panel=mountSupplementPanel(section,{dataset,announce:message=>app.announce(message)});cleanups.push(()=>panel.destroy());}
    }).catch(error=>console.error('Supplement panel unavailable',error));
  });
  return {destroy(){destroyed=true;if(scheduled)cancelAnimationFrame(scheduled);for(const fn of cleanups.splice(0))fn();root?.replaceChildren();legend?.replaceChildren();ledger?.replaceChildren();}};
}
