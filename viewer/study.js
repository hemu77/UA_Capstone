const $=id=>document.getElementById(id);
const repo='https://github.com/hemu77/UA_Capstone/blob/main/';
const ns='http://www.w3.org/2000/svg';
const definitions={global:'Global: the model proposed the whole network in one response.',local:'Local: the model made choices for individual people.',sequential:'Sequential: people were considered in order as the network formed.',iterative:'Iterative: the network was updated over repeated steps.'};
const pts=Array.from({length:50},(_,i)=>[215+184*Math.cos(i*2*Math.PI/50-Math.PI/2),215+184*Math.sin(i*2*Math.PI/50-Math.PI/2)]);
function el(name,attrs){const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;}
function draw(r){
  const graph=$('network'),title=el('title',{id:'graph-title'});
  graph.replaceChildren(title);
  title.textContent=!r?'No matching record':r.edges===null?'Failed attempt; no graph exists':`Final network: 50 people and ${r.edges.length} friendships`;
  if(!r||r.edges===null)return;
  const involved=new Set();
  const path=r.edges.map(pair=>{const a=Math.floor(pair/50),b=pair%50;involved.add(a);involved.add(b);return 'M'+pts[a]+'L'+pts[b];}).join('');
  graph.append(el('path',{d:path,fill:'none',stroke:'#b5dbc9','stroke-width':1.2,opacity:r.edges.length>200?.27:.65}));
  pts.forEach(([cx,cy],i)=>graph.append(el('circle',{cx,cy,r:3.8,fill:involved.has(i)?'#e6c378':'#b4c6cb'})));
}
async function start(){
  // Preserve bookmarks created by the historical replay before the landing page changed.
  if(['/','/index','/index.html'].includes(location.pathname)&&[...new URLSearchParams(location.search).keys()].some(k=>['dataset','view','runs','person','mode','selection','event','incident','run'].includes(k))){
    location.replace('./layers.html'+location.search+location.hash);return;
  }
  const resp=await fetch('./data/primary-study.json');
  if(!resp.ok)throw Error('Data unavailable');
  const rows=await resp.json();if(rows.length!==896)throw Error('Incomplete catalog');
  const outcome=r=>r.edges===null?'failed':r.edges.length?'ties':'empty';
  let selected=[],i=0;
  for(const v of Object.keys(definitions))$('method').add(new Option(v[0].toUpperCase()+v.slice(1),v));
  function render(){
    const r=selected[i];draw(r);
    $('prev').disabled=i<=0;$('next').disabled=i+1>=selected.length;
    $('source').hidden=!r;
    $('run').textContent=r?`${i+1} of ${selected.length} matching attempts · ${r.model} · ${r.method} · ${r.culture.toUpperCase()} / ${r.language} · seed ${r.seed}`:'No attempts match these filters.';
    $('definition').textContent=definitions[r?.method]||'Choose another method or outcome.';
    $('graph-note').textContent=!r?'Nothing to display.':r.edges===null?'No network was saved. Recorded failure: '+r.failure.replaceAll('_',' ')+'.':r.edges.length?`${r.edges.length} saved ties. Gold dots have at least one tie; grey dots are isolated.`:'Valid empty graph. All 50 people are present, with zero saved ties.';
    if(r){$('source').href=repo+(r.edges===null?'outputs/analysis_ready/060dcd3a86d5/failures.csv':'outputs/final_study/060dcd3a86d5/main/runs/'+encodeURIComponent(r.id)+'.adj');$('source').textContent=r.edges===null?'Open failure ledger (CSV) ↗':'Open original adjacency file ↗';}
  }
  function filter(){selected=rows.filter(r=>(!$('method').value||r.method===$('method').value)&&(!$('outcome').value||outcome(r)===$('outcome').value));i=0;render();}
  for(const id of ['method','outcome']){$(id).disabled=false;$(id).addEventListener('change',filter);}
  $('prev').addEventListener('click',()=>{if(i>0){i--;render();}});$('next').addEventListener('click',()=>{if(i+1<selected.length){i++;render();}});filter();
}
start().catch(()=>{$('graph-note').textContent='The saved network catalog could not be loaded. Reload the page or use the source repository; the findings below are still available.';});
