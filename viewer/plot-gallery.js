// This gallery displays saved images only; it never feeds supplements into replay or analysis.
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
  let plots,filtered=[],page=0,current=0,loading=false;
  const label=p=>p.cohort==='primary'?'Primary study':'Supplement / excluded from primary analysis';
  const title=p=>p.method+' / '+p.model+' / '+p.culture+' / '+p.language+' / seed '+p.seed;
  const element=(tag,text,className)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
  const image=p=>{
    const img=element('img');img.alt=title(p)+(p.empty_graph?' / valid empty graph':' / '+p.edges+' edges');img.loading='lazy';img.decoding='async';img.src=p.png_url;
    img.addEventListener('error',()=>{img.hidden=true;const message=element('p','Image unavailable. Open the original file below or retry.','hint');img.after(message);},{once:true});
    return img;
  };
  function show(index){
    current=index;const p=filtered[current];
    $('plot-dialog-title').textContent=title(p);$('plot-dialog-cohort').textContent=label(p)+(p.empty_graph?' / valid empty graph':'');
    $('plot-dialog-image').replaceChildren(image(p));
    $('plot-original').href=p.png_url;$('plot-record').textContent=p.run_id;
    $('plot-source').href='https://github.com/hemu77/UA_Capstone/blob/main/'+p.source_path;
    $('plot-dialog-position').textContent=(current+1)+' / '+filtered.length+' matching plots';
    $('plot-back').disabled=current===0;$('plot-forward').disabled=current===filtered.length-1;
    if(!dialog.open)dialog.showModal();
  }
  function render(){
    filtered=filterPlots(plots,$('plot-cohort').value,$('plot-method').value,$('plot-search').value);
    const start=page*pageSize;
    grid.replaceChildren();
    for(const [offset,p] of filtered.slice(start,start+pageSize).entries()){
      const card=element('article',undefined,'plot-card'),badge=element('p',label(p),'plot-badge'),button=element('button',undefined,'plot-thumbnail');
      badge.dataset.cohort=p.cohort;button.type='button';button.setAttribute('aria-label','Enlarge '+title(p));button.append(image(p));button.addEventListener('click',()=>show(start+offset));
      const link=element('a','Original PNG');link.href=p.png_url;link.target='_blank';link.rel='noopener';
      card.append(badge,button,element('h3',title(p)),element('p',p.empty_graph?'Valid empty graph / 0 edges':p.edges+' edges','hint'),link);grid.append(card);
    }
    status.textContent=filtered.length?('Showing '+(start+1)+'-'+Math.min(start+pageSize,filtered.length)+' of '+filtered.length+' matching plots. 896 saved plots total.'):'No matching plots. Clear the search or change the filters.';
    $('plot-page-back').disabled=page===0;$('plot-page-next').disabled=start+pageSize>=filtered.length;
  }
  async function load(){
    if(plots||loading)return;
    loading=true;$('plot-retry').hidden=true;status.textContent='Checking saved plot catalog...';
    try{
      const response=await fetch('./data/saved-plots.json');if(!response.ok)throw Error('Catalog unavailable');
      plots=validatePlotCatalog(await response.json());render();$('plot-filters').disabled=false;
    }catch{status.textContent='Saved plots could not be loaded. Replay above is unaffected.';$('plot-retry').hidden=false;}
    finally{loading=false;}
  }
  drawer.addEventListener('toggle',()=>{if(drawer.open)load();});
  function openFromLink(){if(location.hash==='#saved-plots'){drawer.open=true;load();}}
  window.addEventListener('hashchange',openFromLink);openFromLink();
  $('plot-retry').addEventListener('click',load);
  $('plot-filters').addEventListener('input',()=>{if(plots){page=0;render();}});
  $('plot-clear').addEventListener('click',()=>{$('plot-search').value='';$('plot-cohort').value='all';$('plot-method').value='';page=0;render();});
  for(const [id,delta] of [['plot-page-back',-1],['plot-page-next',1]])$(id).addEventListener('click',()=>{page+=delta;render();status.scrollIntoView({block:'nearest'});});
  $('plot-back').addEventListener('click',()=>show(current-1));$('plot-forward').addEventListener('click',()=>show(current+1));
  $('plot-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
  dialog.addEventListener('keydown',e=>{
    if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const next=current+(e.key==='ArrowLeft'?-1:1);if(next>=0&&next<filtered.length)show(next);}
    // Do not let the replay's global shortcuts alter the graph behind this modal.
    e.stopPropagation();
  });
}

