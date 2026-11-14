// This final-graph browser is isolated from primary replay and statistical comparisons.
import {SAVED_NETWORKS_SHA256,validateNetworkCatalog} from './network-catalog.js';
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
if(typeof document!=='undefined')initGallery();
function initGallery(){
  const $=id=>document.getElementById(id),section=$('saved-plots');
  if(!section)return;
  const drawer=$('plot-drawer'),grid=$('plot-grid'),dialog=$('plot-dialog'),status=$('plot-status'),pageSize=8;
  let plots,graphs,primary,stage,filtered=[],page=0,current=0,loading=false,viewToken=0;
  const label=p=>p.cohort==='primary'?'Primary study':'Supplement / excluded from primary analysis';
  const title=p=>p.method+' / '+p.model+' / '+p.culture+' / '+p.language+' / seed '+p.seed;
  const element=(tag,text,className)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
  const source=p=>'https://github.com/hemu77/UA_Capstone/blob/main/'+p.source_path;
  async function show(index){
    current=index;const p=filtered[current],graph=graphs.get(p.run_id),token=++viewToken;
    $('plot-dialog-title').textContent=title(p);$('plot-dialog-cohort').dataset.cohort=p.cohort;$('plot-dialog-cohort').textContent=label(p)+(p.empty_graph?' / valid empty graph':'');
    $('plot-record').textContent=p.run_id;$('plot-source').href=source(p).replace(/\.png$/,'.adj');
    $('plot-original').href=source(p);
    $('plot-replay').hidden=p.cohort!=='primary';
    $('plot-replay').href='./layers.html?dataset=primary&view='+encodeURIComponent(JSON.stringify({runs:[p.run_id],active:p.run_id,mode:'formation',frame:0}));
    $('plot-dialog-position').textContent=(current+1)+' / '+filtered.length+' matching networks';
    $('plot-back').disabled=current===0;$('plot-forward').disabled=current===filtered.length-1;
    if(!dialog.open)dialog.showModal();
    $('network-status').textContent='Loading final 3D graph...';
    try{
      if(!stage){
        const {createNetworkStage}=await import('./network-stage.js');
        if(token!==viewToken||!dialog.open)return;
        stage=createNetworkStage($('network-stage'),primary.personas,(id,peers)=>{
          $('network-person').value=id;
          $('network-status').textContent=id?'Persona '+id+' / '+peers.length+' neighbors: '+(peers.join(', ')||'none'):'50 fictional personas / '+graphs.get(filtered[current].run_id).edges.length+' recorded ties. Positions are a layout, not social distance.';
        });
      }
      if(token!==viewToken||!dialog.open)return;
      stage.show(graph,p.method);
    }catch{if(token===viewToken){$('network-status').textContent='3D unavailable in this browser. Original adjacency and PNG remain available below. Try reopening the graph.';}}
  }
  function render(){
    filtered=filterPlots(plots,$('plot-cohort').value,$('plot-method').value,$('plot-search').value);
    const start=page*pageSize;grid.replaceChildren();
    for(const [offset,p] of filtered.slice(start,start+pageSize).entries()){
      const card=element('article',undefined,'plot-card'),badge=element('p',label(p),'plot-badge'),button=element('button','Explore in 3D','network-open');
      badge.dataset.cohort=p.cohort;card.dataset.method=p.method;button.type='button';button.setAttribute('aria-label','Explore in 3D '+title(p));button.addEventListener('click',()=>show(start+offset));
      card.append(badge,element('h3',title(p)),element('p',p.empty_graph?'50 personas / valid empty graph':'50 personas / '+p.edges+' recorded ties','hint'),button);grid.append(card);
    }
    status.textContent=filtered.length?('Showing '+(start+1)+'-'+Math.min(start+pageSize,filtered.length)+' of '+filtered.length+' matching networks. 896 saved networks total.'):'No matching networks. Clear the search or change the filters.';
    $('plot-page-back').disabled=page===0;$('plot-page-next').disabled=start+pageSize>=filtered.length;
  }
  async function load(){
    if(plots||loading)return;
    loading=true;$('plot-retry').hidden=true;status.textContent='Verifying saved network catalog...';
    try{
      const get=async name=>{const r=await fetch('./data/'+name);if(!r.ok)throw Error('Catalog unavailable');return r;};
      const metadata=validatePlotCatalog(await (await get('saved-plots.json')).json());
      const raw=await (await get('saved-networks.json')).arrayBuffer();
      const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(b=>b.toString(16).padStart(2,'0')).join('');
      if(hash!==SAVED_NETWORKS_SHA256)throw Error('Final graph checksum mismatch');
      primary=await (await get('primary-viewer.json')).json();
      graphs=validateNetworkCatalog(JSON.parse(new TextDecoder().decode(raw)),metadata,primary);
      plots=metadata;
      $('network-person').replaceChildren(new Option('Whole network',''));
      for(const person of primary.personas)$('network-person').add(new Option('Persona '+person.id,person.id));
      render();$('plot-filters').disabled=false;
    }catch{status.textContent='Saved networks could not be verified. Primary replay above is unaffected.';$('plot-retry').hidden=false;}
    finally{loading=false;}
  }
  drawer.addEventListener('toggle',()=>{if(drawer.open)load();});
  function openFromLink(){if(location.hash==='#saved-plots'||location.hash==='#saved-networks'){drawer.open=true;load();}}
  window.addEventListener('hashchange',openFromLink);openFromLink();
  $('plot-retry').addEventListener('click',load);
  $('plot-filters').addEventListener('input',()=>{if(plots){page=0;render();}});
  $('plot-clear').addEventListener('click',()=>{$('plot-search').value='';$('plot-cohort').value='all';$('plot-method').value='';page=0;render();});
  for(const [id,delta] of [['plot-page-back',-1],['plot-page-next',1]])$(id).addEventListener('click',()=>{page+=delta;render();status.scrollIntoView({block:'nearest'});});
  $('plot-back').addEventListener('click',()=>show(current-1));$('plot-forward').addEventListener('click',()=>show(current+1));
  $('plot-close').addEventListener('click',()=>dialog.close());
  $('network-person').addEventListener('change',()=>stage?.select($('network-person').value));
  $('network-reset').addEventListener('click',()=>stage?.reset());
  $('network-zoom-in').addEventListener('click',()=>stage?.zoom(.8));
  $('network-zoom-out').addEventListener('click',()=>stage?.zoom(1.25));
  dialog.addEventListener('close',()=>{viewToken++;});
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
  dialog.addEventListener('keydown',e=>{
    if(e.target.tagName!=='CANVAS'&&e.target.tagName!=='SELECT'&&(e.key==='ArrowLeft'||e.key==='ArrowRight')){e.preventDefault();const next=current+(e.key==='ArrowLeft'?-1:1);if(next>=0&&next<filtered.length)show(next);}
    e.stopPropagation();
  });
  window.addEventListener('pagehide',event=>{if(!event.persisted){stage?.dispose();stage=null;}});
}
