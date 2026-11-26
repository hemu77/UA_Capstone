import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ATTRIBUTES,AGE_BANDS,PANEL_COLOR,NEUTRAL_COLOR,TIE_CLASS_COLORS,ageBand,groupOf,attributeGroups,nodeEncoding,edgeEncoding,
  tieMixing,consensus,consensusStats,overlapMatrix,ledgerFacts,mountInsight} from './layers-insight.js';
import {runLabel,layerLabel,modelName,settingName,plural,runParts} from './formation-charts.js';
import {researchEvidence} from './formation-charts.js';
import {loadPrimaryDataset,COUNTS} from './layers-data-all.js';
import {LAYER_COLORS,INITIAL_STATE,EMPTY_FILTERS} from './layers-app.js';

const fetchImpl=async url=>{const name=String(url).replace(/^\.\/data\//,'').split('?')[0];try{const bytes=readFileSync(new URL('./public/data/'+name,import.meta.url));return {ok:true,status:200,json:async()=>JSON.parse(bytes),arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};}catch{return {ok:false,status:404};}};
const data=await loadPrimaryDataset({fetchImpl});
const state=patch=>({...INITIAL_STATE,...patch});
const brute=(runs)=>{ // independent reference: count each unordered pair by string sort of the numeric pair
  const counts={};for(const run of runs){const seen=new Set();for(const [a,b] of run.edges){const key=[Number(a),Number(b)].sort((x,y)=>x-y).join('-');if(!seen.has(key)){seen.add(key);counts[key]=(counts[key]||0)+1;}}}
  return counts;
};

// --- colour science helpers (sRGB -> linear -> WCAG luminance; Machado 2009 deuteranopia; CIE Lab dE76) ---------
const channel=c=>{c/=255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4;};
const linear=hex=>[1,3,5].map(i=>channel(parseInt(hex.slice(i,i+2),16)));
const luminance=hex=>{const [r,g,b]=linear(hex);return .2126*r+.7152*g+.0722*b;};
const contrast=(a,b)=>{const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);};
const DEUTERANOPIA=[[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.01182,.04294,.968881]];
const lab=([r,g,b])=>{const X=.4124*r+.3576*g+.1805*b,Y=.2126*r+.7152*g+.0722*b,Z=.0193*r+.1192*g+.9505*b;const f=t=>t>216/24389?Math.cbrt(t):(24389/27*t+16)/116;const fx=f(X/.95047),fy=f(Y),fz=f(Z/1.08883);return [116*fy-16,500*(fx-fy),200*(fy-fz)];};
const deuteranope=hex=>{const c=linear(hex);return lab(DEUTERANOPIA.map(row=>Math.min(1,Math.max(0,row[0]*c[0]+row[1]*c[1]+row[2]*c[2]))));};
const distance=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2]);

test('one run-label format: display names, fixed order, no raw keys',()=>{
  const run={model:'gpt-5.6-luna',method:'local',culture:'us',language:'english',seed:44000};
  assert.equal(runLabel(run),'GPT-5.6 Luna · Local · United States / English · seed 44000');
  assert.equal(runLabel({...run,model:'gpt-4.1',culture:'brazil',language:'portuguese'}),'GPT-4.1 · Local · Brazil / Portuguese · seed 44000');
  assert.deepEqual(data.models.map(modelName),['GPT-4.1','GPT-5.6 Luna','GPT-6 Luna','GPT-6 Sol']);
  assert.deepEqual(data.settings.map(([c,l])=>settingName(c,l)),['United States / English','India / English','Japan / English','Brazil / English','United States / Hindi','United States / Japanese','United States / Portuguese']);
  assert.equal(modelName('claude-x'),'claude-x');
  for(const r of data.runs){const label=runLabel(r);assert.ok(!/\b(us|english|hindi|global|local)\b/.test(label),label);assert.equal(label.split(' · ').length,4);}
  assert.deepEqual(runParts(run),{model:'GPT-5.6 Luna',method:'Local',setting:'United States / English',seed:'seed 44000'});
  assert.equal(layerLabel([run,{...run,culture:'japan'}],1),'2. Japan');
  assert.equal(layerLabel([run,{...run,language:'portuguese'}],1),'2. Portuguese');
  assert.equal(layerLabel([run,{...run,culture:'india',language:'hindi'}],1),'2. India / Hindi');
  assert.equal(layerLabel([run,{...run,seed:44001,method:'global'}],0),'1. Local · seed 44000');
  assert.equal(layerLabel([run],0),'1. GPT-5.6 Luna · Local');
  assert.equal(plural(1,'tie'),'1 tie');assert.equal(plural(0,'tie'),'0 ties');assert.equal(plural(156730,'tie'),'156,730 ties');assert.equal(plural(2,'empty graph'),'2 empty graphs');
});

test('age bands are six ordered, gap-free adult bands',()=>{
  assert.equal(AGE_BANDS.length,6);
  assert.deepEqual([17,18,24,25,34,35,64,65,99,'x'].map(ageBand),[null,'18-24','18-24','25-34','25-34','35-44','55-64','65+','65+',null]);
  const counts=attributeGroups(data.personas,'age').map(g=>[g.key,g.count]);
  assert.deepEqual(counts,[['18-24',7],['25-34',10],['35-44',10],['45-54',10],['55-64',10],['65+',3]]);
});

test('every roster value has a defined colour and the groups cover all 50 personas',()=>{
  for(const attribute of ATTRIBUTES){
    const groups=attributeGroups(data.personas,attribute.key);
    assert.equal(groups.reduce((n,g)=>n+g.count,0),COUNTS.personas,attribute.key);
    assert.deepEqual(groups.map(g=>g.key),attribute.groups.map(([key])=>key),`${attribute.key}: no spare colours, defined order`);
    for(const person of data.personas)assert.notEqual(groupOf(person,attribute.key),null);
  }
  assert.deepEqual(attributeGroups(data.personas,'gender').map(g=>[g.key,g.count]),[['Woman',20],['Man',20],['Nonbinary',10]]);
  assert.deepEqual(attributeGroups(data.personas,'political orientation').map(g=>g.count),[17,16,17]);
  assert.deepEqual(attributeGroups(data.personas,'religion').map(g=>g.count),[10,10,10,10,10]);
  assert.deepEqual(attributeGroups(data.personas,'nope'),[]);
});

test('palettes: >= 3:1 against the panel, separable under deuteranopia, age ramp evenly ordered',()=>{
  for(const attribute of ATTRIBUTES){
    const colors=attribute.groups.map(([,color])=>color);
    assert.equal(new Set(colors).size,colors.length);
    for(const color of colors)assert.ok(contrast(color,PANEL_COLOR)>=3,`${attribute.key} ${color} contrast ${contrast(color,PANEL_COLOR).toFixed(2)}`);
    for(let i=0;i<colors.length;i++)for(let j=i+1;j<colors.length;j++){
      const d=distance(deuteranope(colors[i]),deuteranope(colors[j]));
      assert.ok(d>=15,`${attribute.key}: ${colors[i]} vs ${colors[j]} only dE ${d.toFixed(1)} for deuteranopes`);
    }
  }
  const lightness=AGE_BANDS.map(band=>lab(linear(band.color))[0]),steps=lightness.slice(1).map((v,i)=>v-lightness[i]);
  assert.ok(steps.every(step=>step>0),'age ramp must increase monotonically in lightness');
  assert.ok(Math.max(...steps)/Math.min(...steps)<1.4,`age ramp steps uneven: ${steps.map(s=>s.toFixed(1))}`);
  for(const color of [...Object.values(TIE_CLASS_COLORS),...LAYER_COLORS])assert.ok(contrast(color,PANEL_COLOR)>=3);
  assert.ok(contrast(NEUTRAL_COLOR,PANEL_COLOR)>=3);
});

test('node encoding: layer identity, attribute colour, isolation and legend',()=>{
  const layers=data.runs.slice(0,3).map(r=>r.run_id);
  const byLayer=nodeEncoding(data,state({colorBy:'layer',layers}));
  assert.equal(byLayer.mode,'layer');
  assert.equal(byLayer.colorOf('0',2),LAYER_COLORS[2]);assert.equal(byLayer.colorOf('0',9),LAYER_COLORS[1]);
  assert.equal(byLayer.dimmed('0'),false);
  assert.deepEqual(byLayer.legend.map(item=>[item.key,item.color,item.count]),data.runs.slice(0,3).map((r,i)=>[r.run_id,LAYER_COLORS[i],r.edges.length]));
  const religion=nodeEncoding(data,state({colorBy:'religion'}));
  const person=data.personas.find(p=>p.attributes.religion==='Hindu');
  assert.equal(religion.colorOf(person.id),'#2fbf8f');
  assert.equal(religion.colorOf('no-such-person'),NEUTRAL_COLOR);
  assert.equal(religion.legend.length,5);assert.ok(religion.legend.every(g=>g.label&&g.color&&g.count===10));
  assert.equal(data.personas.filter(p=>religion.dimmed(p.id)).length,0);
  const isolated=nodeEncoding(data,state({colorBy:'religion',groups:['Hindu','Muslim','not-a-group']}));
  assert.equal(data.personas.filter(p=>isolated.dimmed(p.id)).length,30);
  assert.equal(isolated.dimmed(person.id),false);
  const age=nodeEncoding(data,state({colorBy:'age'}));
  assert.equal(age.colorOf(data.personas.find(p=>p.attributes.age===67).id),'#fde725');
  assert.equal(age.legend[0].label,'18–24');
});

test('edge encoding: same / mixed classes, lens and isolation',()=>{
  const id=(attr,value)=>data.personas.find(p=>p.attributes[attr]===value).id;
  const woman1=data.personas.filter(p=>p.attributes.gender==='Woman')[0].id,woman2=data.personas.filter(p=>p.attributes.gender==='Woman')[1].id,man=id('gender','Man');
  assert.equal(edgeEncoding(data,state({colorBy:'gender',lens:'none'})).classOf([woman1,man]),null);
  assert.equal(edgeEncoding(data,state({colorBy:'layer',lens:'same'})).classOf([woman1,woman2]),null);
  const same=edgeEncoding(data,state({colorBy:'gender',lens:'same'}));
  assert.equal(same.classOf([woman1,woman2]),'same');assert.equal(same.classOf([man,woman1]),'mixed');
  assert.equal(same.colorOf([woman1,woman2]),TIE_CLASS_COLORS.same);
  assert.equal(same.dimmed([woman1,woman2]),false);assert.equal(same.dimmed([woman1,man]),true);
  const mixed=edgeEncoding(data,state({colorBy:'gender',lens:'mixed'}));
  assert.equal(mixed.dimmed([woman1,man]),false);assert.equal(mixed.dimmed([woman1,woman2]),true);
  const isolate=edgeEncoding(data,state({colorBy:'gender',lens:'none',groups:['Man']}));
  assert.equal(isolate.dimmed([woman1,man]),false,'a tie touching an isolated group stays visible');
  assert.equal(isolate.dimmed([woman1,woman2]),true);
  assert.equal(edgeEncoding(data,state({colorBy:'gender',lens:'bogus'})).lens,'none');
});

test('same-group tie share and roster-pair reference are exact',()=>{
  const people=[{id:'1',attributes:{g:'A'}},{id:'2',attributes:{g:'A'}},{id:'3',attributes:{g:'B'}},{id:'4',attributes:{g:'B'}},{id:'5',attributes:{g:'B'}},{id:'6',attributes:{}}];
  const result=tieMixing(people,[[['1','2'],['1','3'],['3','4']],[['4','5'],['2','6']]],'g');
  // pairs among the 5 known personas: 10; same-group: A 1 + B 3 = 4. Tie 2-6 has an unknown end and is skipped.
  assert.deepEqual(result,{graphs:2,ties:4,same:3,mixed:1,share:.75,reference:{pairs:10,same:4,share:.4}});
  assert.equal(tieMixing(people,[[]],'g').share,null);
  const gender=tieMixing(data.personas,[],'gender').reference;
  assert.deepEqual([gender.pairs,gender.same],[1225,190+190+45]);
  assert.equal(tieMixing(data.personas,[],'religion').reference.same,5*45);
  assert.equal(tieMixing(data.personas,[],'political orientation').reference.same,136+120+136);
  assert.equal(tieMixing(data.personas,[],'age').reference.same,21+45*4+3);
  // Pooling is the sum over graphs, and the whole primary set has every tie once per graph.
  const all=tieMixing(data.personas,data.runs.map(r=>r.edges),'gender');
  assert.equal(all.ties,156730);assert.equal(all.graphs,890);
  const parts=data.runs.map(r=>tieMixing(data.personas,[r.edges],'gender'));
  assert.equal(parts.reduce((n,p)=>n+p.same,0),all.same);
  // Cross-check one run against a direct count.
  const run=data.runs.find(r=>r.edges.length>100),gender_of=new Map(data.personas.map(p=>[p.id,p.attributes.gender]));
  assert.equal(tieMixing(data.personas,[run.edges],'gender').same,run.edges.filter(([a,b])=>gender_of.get(a)===gender_of.get(b)).length);
});

test('consensus counts each tie once per graph, sorts by count then numeric edge, and matches a brute-force count',()=>{
  const list=consensus([{edges:[['2','10'],['1','2'],['2','10']]},{edges:[['10','2']]},{edges:[]}]);
  assert.deepEqual(list,[{edge:['2','10'],count:2,total:3},{edge:['1','2'],count:1,total:3}]);
  assert.deepEqual(consensus([]),[]);
  const condition=data.runs.filter(r=>r.model==='gpt-4.1'&&r.method==='local'&&r.setting==='us-english');
  assert.equal(condition.length,8);
  const ours=consensus(condition),reference=brute(condition);
  assert.equal(ours.length,Object.keys(reference).length);
  for(const row of ours)assert.equal(row.count,reference[`${row.edge[0]}-${row.edge[1]}`]);
  for(let i=1;i<ours.length;i++)assert.ok(ours[i-1].count>ours[i].count||(ours[i-1].count===ours[i].count&&(Number(ours[i-1].edge[0])<Number(ours[i].edge[0])||(ours[i-1].edge[0]===ours[i].edge[0]&&Number(ours[i-1].edge[1])<Number(ours[i].edge[1])))));
  const everything=consensus(data.runs);
  assert.equal(everything.reduce((n,row)=>n+row.count,0),156730);
  assert.ok(everything.every(row=>row.total===890&&row.count<=890&&Number(row.edge[0])<Number(row.edge[1])));
  assert.ok(everything.length<=1225);
});

test('consensus stats use a ceiling threshold and count ties in every graph',()=>{
  const list=[{edge:['1','2'],count:4,total:4},{edge:['1','3'],count:2,total:4},{edge:['2','3'],count:1,total:4}];
  assert.deepEqual(consensusStats(list,4,.5),{graphs:4,distinct:3,inAll:1,atLeast:2,minCount:2,single:1});
  assert.equal(consensusStats(list,4,.55).minCount,3);
  assert.equal(consensusStats(list,4,1).atLeast,1);
  assert.equal(consensusStats(list,4,.05).minCount,1);
  assert.equal(consensusStats([],0,.5).inAll,0);
});

test('overlap matrix is symmetric, exact, and undefined only for two empty graphs',()=>{
  const runs=[{run_id:'a',edges:[['1','2'],['2','3']]},{run_id:'b',edges:[['3','2'],['4','5']]},{run_id:'c',edges:[]},{run_id:'d',edges:[]}];
  const {ids,matrix}=overlapMatrix(runs);
  assert.deepEqual(ids,['a','b','c','d']);
  assert.deepEqual(matrix[0][1],{shared:1,union:3,jaccard:1/3});
  assert.deepEqual(matrix[1][0],matrix[0][1]);
  assert.deepEqual(matrix[0][0],{shared:2,union:2,jaccard:1});
  assert.deepEqual(matrix[0][2],{shared:0,union:2,jaccard:0});
  assert.deepEqual(matrix[2][3],{shared:0,union:0,jaccard:null});
  const condition=data.runs.filter(r=>r.model==='gpt-6-sol'&&r.method==='iterative'&&r.setting==='japan-english');
  const ours=overlapMatrix(condition).matrix,theirs=researchEvidence(condition,'').pairs;
  for(const pair of theirs)assert.equal(ours[pair.a-1][pair.b-1].jaccard,pair.jaccard);
});

test('study ledger states 896 / 890 / 12 / 6 from the loaded slots and refuses a mismatch',()=>{
  const facts=ledgerFacts(data);
  assert.deepEqual([facts.attempts,facts.valid,facts.empty,facts.failures,facts.supplements_excluded,facts.personas],[896,890,12,6,6,50]);
  assert.deepEqual([facts.models,facts.methods,facts.settings,facts.seeds],[4,4,7,8]);
  assert.throws(()=>ledgerFacts({...data,runs:data.runs.slice(1)}),/runs/);
  const slots=data.slots.map(s=>s.status==='failed'?{...s,status:'valid'}:s);
  assert.throws(()=>ledgerFacts({...data,slots}),/failures/);
  assert.throws(()=>ledgerFacts({...data,counts:{...data.counts,supplements_excluded:0}}),/supplements/);
  assert.throws(()=>ledgerFacts(data,{...COUNTS,valid:896}),/valid/);
});

test('the primary set used for every aggregate contains no supplement',()=>{
  assert.ok(data.runs.every(r=>r.run_id.startsWith('final_060dcd3a86d5_main_')));
  assert.ok(!data.runs.some(r=>/^supp/.test(r.run_id)));
});

test('mountInsight is a no-op without a DOM',()=>{
  assert.equal(typeof mountInsight({getState:()=>INITIAL_STATE,subscribe:()=>()=>{},ready:Promise.resolve(data)},{}).destroy,'function');
  assert.deepEqual(EMPTY_FILTERS.model,[]);
});
