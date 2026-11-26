// Supplement-only panel (SPEC.md C.7). The six supplemental graphs live here and nowhere else: never in the
// atlas, the stack, filters, counts, overlaps or the consensus. Primary graphs are browsed in the layered 3D view.
// validatePlotCatalog / filterPlots still guard the pinned saved-plots.json fixture used by the build.
import * as dataModule from './layers-data-all.js?v=d068babb8c4c';
import {runLabel,runParts,plural,LABEL_SEPARATOR} from './formation-charts.js?v=d068babb8c4c';
const folders={primary:'outputs/final_study/060dcd3a86d5/main/runs',supplement_initial:'outputs/supplement_20261105_v1/runs',supplement_completion:'outputs/supplement_completion_20261105_v1/runs'};
const sameKeys=(value,keys)=>Object.keys(value).sort().join(',')===[...keys].sort().join(',');
export function validatePlotCatalog(data){
  if(data?.catalog_sha256!=='b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775'||!Array.isArray(data.plots)||data.plots.length!==896)throw Error('Unexpected saved-plot catalog');
  if(!sameKeys(data,['catalog_sha256','plots']))throw Error('Unexpected public catalog fields');
  const ids=new Set(),counts={primary:0,supplement_initial:0,supplement_completion:0};
  for(const p of data.plots){
    if(!p||!Object.hasOwn(folders,p.cohort)||typeof p.run_id!=='string'||!/^[a-zA-Z0-9_.-]+$/.test(p.run_id)||ids.has(p.run_id))throw Error('Invalid plot identity');
    if(!sameKeys(p,['run_id','cohort','model','method','culture','language','seed','edges','empty_graph','source_path','png_url','png_sha256']))throw Error('Unexpected public plot fields');
    ids.add(p.run_id);counts[p.cohort]++;
    if(p.source_path!==folders[p.cohort]+'/'+p.run_id+'.png'||p.png_url!=='./data/saved-plots/'+p.run_id+'.png'||!/^[a-f0-9]{64}$/.test(p.png_sha256))throw Error('Invalid plot path or hash');
    if(!['global','local','sequential','iterative'].includes(p.method)||!['model','culture','language'].every(k=>typeof p[k]==='string'&&p[k].length>0&&p[k].length<80)||!Number.isInteger(p.seed)||!Number.isInteger(p.edges)||p.edges<0||typeof p.empty_graph!=='boolean'||p.empty_graph!==(p.edges===0))throw Error('Invalid plot metadata');
  }
  if(counts.primary!==890||counts.supplement_initial!==4||counts.supplement_completion!==2||data.plots.filter(p=>p.cohort==='primary'&&p.empty_graph).length!==12)throw Error('Saved-plot cohort mismatch');
  return data.plots;
}
export function filterPlots(plots,cohort,method,query){
  const words=query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return plots.filter(p=>(cohort==='all'||(cohort==='primary'?p.cohort==='primary':p.cohort!=='primary'))&&(!method||p.method===method)&&words.every(w=>[p.run_id,p.model,p.culture,p.language,String(p.seed)].join(' ').toLowerCase().includes(w)));
}

export const SUPPLEMENT_COHORTS=Object.freeze({supplement_initial:'Initial supplement run',supplement_completion:'Completion supplement run'});
const SUPPLEMENT_FOLDERS={supplement_initial:'outputs/supplement_20261105_v1/runs/',supplement_completion:'outputs/supplement_completion_20261105_v1/runs/'};
// Accepts the loadSupplements() result ({graphs:[...]} or an array) and refuses anything that could leak into, or
// be confused with, the primary study: wrong count, a primary or failed-slot run_id, unknown cohort, foreign
// personas, self or duplicate ties. Returns normalised records with ties as sorted [a,b] string pairs.
export function validateSupplements(value,dataset) {
  const graphs=Array.isArray(value)?value:value?.graphs;
  const expected=dataModule.COUNTS.supplements_excluded;
  if(!Array.isArray(graphs)||graphs.length!==expected)throw Error(`Expected exactly ${expected} supplemental graphs.`);
  const primary=new Set([...(dataset.runs||[]).map(run=>run.run_id),...(dataset.failures||[]).map(run=>run.run_id),...(dataset.slots||[]).map(slot=>slot?.run_id)]);
  const people=new Set((dataset.personas||[]).map(person=>String(person.id))),seen=new Set();
  return graphs.map(graph=>{
    if(!graph||typeof graph.run_id!=='string'||!/^[a-zA-Z0-9_.-]+$/.test(graph.run_id)||seen.has(graph.run_id))throw Error('Invalid supplemental graph identity.');
    if(primary.has(graph.run_id))throw Error('A primary run was offered as a supplement.');
    if(!Object.hasOwn(SUPPLEMENT_COHORTS,graph.cohort))throw Error('Unknown supplement cohort.');
    if(typeof graph.source!=='string'||graph.source!==SUPPLEMENT_FOLDERS[graph.cohort]+graph.run_id+'.adj')throw Error('Invalid supplement source path.');
    if(graph.source_sha256!==undefined&&!/^[a-f0-9]{64}$/.test(graph.source_sha256))throw Error('Invalid supplement source hash.');
    if(!['model','method','culture','language'].every(key=>typeof graph[key]==='string'&&graph[key].length>0&&graph[key].length<80)||!Number.isInteger(Number(graph.seed)))throw Error('Invalid supplement metadata.');
    if(!Array.isArray(graph.edges))throw Error('Invalid supplement ties.');
    const ties=new Set(),edges=graph.edges.map(edge=>{
      if(!Array.isArray(edge)||edge.length!==2)throw Error('Invalid supplement tie.');
      const [a,b]=edge.map(String).sort((x,y)=>Number(x)-Number(y));
      if(a===b||!people.has(a)||!people.has(b))throw Error('Supplement tie outside the roster.');
      const key=a+'|'+b;if(ties.has(key))throw Error('Duplicate supplement tie.');ties.add(key);return [a,b];
    }).sort((x,y)=>Number(x[0])-Number(y[0])||Number(x[1])-Number(y[1]));
    seen.add(graph.run_id);
    return {run_id:graph.run_id,cohort:graph.cohort,model:graph.model,method:graph.method,culture:graph.culture,language:graph.language,seed:Number(graph.seed),
      edges,source:graph.source,source_sha256:graph.source_sha256??null};
  });
}
// How a supplement relates to the primary design, computed from the loaded slots. It never fills or replaces a slot.
export function supplementContext(graph,dataset) {
  let slot=null;
  try{slot=dataModule.slotOf(dataset.design,graph);}catch{return {slot:null,status:'outside',text:'Condition outside the primary design.'};}
  const record=dataset.slots?.[slot],status=record?.status||'unknown';
  const text=status==='failed'?`Same condition as failed primary slot ${slot}. It does not fill that slot; the primary attempt stays a failure.`
    :status==='valid'||status==='empty'?`Same condition as primary slot ${slot}, which has its own graph. This one is not counted.`
    :`Primary slot ${slot}.`;
  return {slot,status,text};
}

// Builds the panel inside the #saved-plots section. Data loads only when the panel is opened.
export function mountSupplementPanel(section,{dataset,announce=()=>{},loadSupplements=dataModule.loadSupplements,stageModule=()=>import('./network-stage.js?v=d068babb8c4c')}={}) {
  const h=(tag,props={},...children)=>{const node=document.createElement(tag);
    for(const [key,value] of Object.entries(props)){if(value===undefined||value===null||value===false)continue;if(key==='class')node.className=value;else if(key==='text')node.textContent=value;else if(key.startsWith('on'))node.addEventListener(key.slice(2),value);else node.setAttribute(key,value===true?'':String(value));}
    for(const child of children.flat())if(child!==null&&child!==undefined&&child!==false)node.append(child);return node;};
  // The legacy primary gallery in this section is retired; its controls would otherwise sit there dead.
  for(const id of ['plot-drawer','plot-status','plot-retry'])section.querySelector('#'+id)?.setAttribute('hidden','');
  const count=dataModule.COUNTS.supplements_excluded;
  const list=h('ol',{class:'ins-supp-list','aria-label':'Supplemental graphs'});
  const stageHost=h('div',{class:'ins-supp-stage'});
  const status=h('p',{class:'ins-supp-status',role:'status'});
  const meta=h('div',{class:'ins-supp-meta'});
  const tools=h('div',{class:'ins-supp-tools',role:'group','aria-label':'Supplement camera'});
  const viewer=h('div',{class:'ins-supp-viewer',hidden:true},h('div',{class:'ins-supp-frame'},stageHost,h('span',{class:'ins-supp-watermark',text:'Supplement · not primary'})),status,tools,meta);
  const message=h('p',{class:'ins-supp-message',role:'status',text:'Loading the supplemental graphs…'});
  const retry=h('button',{type:'button',class:'ins-supp-retry',hidden:true,text:'Retry loading supplements'});
  const drawer=h('details',{class:'ins-supp-drawer'},
    h('summary',{class:'ins-supp-summary'},h('span',{class:'ins-supp-summary-title',text:'Show supplemental graphs'}),h('span',{class:'ins-supp-summary-hint',text:'Separate canvas · loads only when opened'})),
    h('div',{class:'ins-supp-body'},message,retry,h('div',{class:'ins-supp-layout'},list,viewer)));
  const panel=h('div',{class:'ins-supp'},
    h('p',{class:'ins-supp-eyebrow',text:'Supplements'}),
    h('h3',{class:'ins-supp-title',text:`${count} supplemental graphs, excluded from the primary study`}),
    h('p',{class:'ins-supp-lede',text:'Saved by two later runs on the same fictional roster. They never enter the atlas, stack, filters, counts or comparisons above.'}),
    drawer);
  const anchor=section.querySelector('#supplement-root');
  (anchor||section).append(panel);
  let graphs=null,stage=null,current=-1,loading=false,destroyed=false;
  const buttons=[];
  const describe=graph=>{const parts=runParts(graph);return {title:[parts.model,parts.method].join(LABEL_SEPARATOR),detail:[parts.setting,parts.seed].join(LABEL_SEPARATOR)};};
  function setStatus(selected='',peers=[]){
    const graph=graphs?.[current];if(!graph)return;
    status.textContent=selected?`Persona ${selected}: ${plural(peers.length,'tie')} in this supplement${peers.length?` (${peers.join(', ')})`:''}.`
      :graph.edges.length?`${plural(graph.edges.length,'generated tie')} among 50 fictional personas. Select a persona to see its ties.`:'Valid graph with no ties: 50 unlinked personas.';
  }
  async function show(index){
    current=index;const graph=graphs[index],context=supplementContext(graph,dataset),text=describe(graph);
    buttons.forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
    viewer.hidden=false;
    meta.replaceChildren(
      h('p',{class:'ins-supp-label',text:runLabel(graph)}),
      h('p',{class:'ins-supp-context',text:context.text}),
      h('dl',{},h('dt',{text:'Run ID'}),h('dd',{class:'ins-mono',text:graph.run_id}),
        h('dt',{text:'Source'}),h('dd',{},h('a',{href:'https://github.com/hemu77/UA_Capstone/blob/main/'+graph.source,target:'_blank',rel:'noopener',text:graph.source.split('/').pop()})),
        graph.source_sha256?h('dt',{text:'Adjacency SHA-256'}):null,graph.source_sha256?h('dd',{class:'ins-mono',text:graph.source_sha256}):null));
    try{
      if(!stage){
        const {createNetworkStage}=await stageModule();
        if(destroyed||current!==index)return;
        stage=createNetworkStage(stageHost,dataset.personas,setStatus);
        tools.replaceChildren(
          h('button',{type:'button',text:'Reset view',onclick:()=>stage.reset()}),
          h('button',{type:'button',text:'Zoom in','aria-label':'Zoom in',onclick:()=>stage.zoom(.8)}),
          h('button',{type:'button',text:'Zoom out','aria-label':'Zoom out',onclick:()=>stage.zoom(1.25)}),
          h('button',{type:'button',text:'Clear persona',onclick:()=>stage.select('')}));
      }
      stage.show(graph);setStatus();
      announce(`Showing supplemental graph ${index+1} of ${graphs.length}: ${text.title}, ${text.detail}. Excluded from the primary study.`);
    }catch{
      stageHost.replaceChildren(h('p',{class:'ins-supp-fallback',text:'3D is unavailable in this browser. The tie list under this box is the complete graph.'}));
      tools.replaceChildren();
      status.textContent=graph.edges.length?`${plural(graph.edges.length,'generated tie')}: ${graph.edges.map(([a,b])=>a+'–'+b).join(', ')}.`:'Valid graph with no ties.';
    }
  }
  async function load(){
    if(graphs||loading)return;
    loading=true;retry.hidden=true;message.textContent='Loading and checking the supplemental graphs…';
    try{
      if(typeof loadSupplements!=='function')throw Error('Supplement loader is not available in this build.');
      graphs=validateSupplements(await loadSupplements(),dataset);
      if(destroyed)return;
      list.replaceChildren(...graphs.map((graph,i)=>{const text=describe(graph);
        const button=h('button',{type:'button',class:'ins-supp-item','aria-pressed':'false',onclick:()=>show(i)},
          h('span',{class:'ins-supp-item-title',text:text.title}),h('span',{class:'ins-supp-item-detail',text:text.detail}),
          h('span',{class:'ins-supp-item-meta',text:`${SUPPLEMENT_COHORTS[graph.cohort]} · ${graph.edges.length?plural(graph.edges.length,'tie'):'no ties'}`}));
        buttons.push(button);return h('li',{},button);}));
      message.textContent=`${plural(graphs.length,'supplemental graph')}, checked against the primary roster and run list. Select one to view it.`;
    }catch(error){
      graphs=null;message.textContent=`Supplemental graphs could not be loaded: ${error.message} The primary study is unaffected.`;retry.hidden=false;
    }finally{loading=false;}
  }
  drawer.addEventListener('toggle',()=>{if(drawer.open)load();});
  retry.addEventListener('click',load);
  const openFromLink=()=>{if(['#saved-plots','#saved-networks','#supplements'].includes(location.hash)){drawer.open=true;load();}};
  window.addEventListener('hashchange',openFromLink);openFromLink();
  const pagehide=event=>{if(!event.persisted){stage?.dispose();stage=null;}};
  window.addEventListener('pagehide',pagehide);
  return {destroy(){destroyed=true;window.removeEventListener('hashchange',openFromLink);window.removeEventListener('pagehide',pagehide);stage?.dispose();stage=null;panel.remove();}};
}
