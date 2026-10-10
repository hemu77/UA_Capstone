// Small, labeled SVGs keep the recorded values inspectable without a chart library.
import {neighbors,edgeFrequencies} from './graph-data.js';
export const countryNames={us:'United States',india:'India',japan:'Japan',brazil:'Brazil'};
export const runLabel=run=>`${run.model} | ${run.method} | ${countryNames[run.culture]||run.culture} | ${run.language} | ${Number.isInteger(run.repetition)?`rep ${run.repetition+1} / `:''}seed ${run.seed}`;
export function layerLabel(runs,index) {
  const run=runs[index],varied=['model','method','culture','language','seed'].filter(key=>new Set(runs.map(r=>r[key])).size>1);
  const fields=varied.length?varied:['model','method'];
  return `${index+1}. ${fields.map(key=>key==='culture'?countryNames[run[key]]||run[key]:key==='seed'?`${Number.isInteger(run.repetition)?`rep ${run.repetition+1} / `:''}seed ${run.seed}`:run[key]).join(' / ')}`;
}
export async function loadPresentation(fetcher) {
  try {
    const response=await fetcher('./data/presentation.json',{cache:'no-store'});
    if(!response.ok)throw new Error('Unavailable');
    const value=await response.json();
    if(!value?.runs||typeof value.runs!=='object'||Array.isArray(value.runs))throw new Error('Malformed');
    return {runs:value.runs,error:null};
  }catch{return {runs:{},error:'Presentation gallery unavailable. Saved networks and playback remain usable.'};}
}

// Matching is a design check, not a significance test or evidence of human validity.
export function researchEvidence(runs, question) {
  const varied={rq1:'culture',rq3:'model',rq4:'language'}[question];
  const issues=[];
  if(!runs.length)return {matched:false,issues:['Select saved runs first.'],pairs:[]};
  if(new Set(runs.map(r=>r.roster_id)).size>1)issues.push('Different persona rosters.');
  if(new Set(runs.map(r=>r.prompt_variant||'unknown')).size>1)issues.push('Prompt/source variants differ.');
  if(runs.some(r=>!r.prompt_variant||r.prompt_variant==='legacy_pre_source_hash'))issues.push('Some generation source versions are unrecorded.');
  if(new Set(runs.map(r=>r.study)).size>1)issues.push('Different study collections selected.');
  if(question==='rq1'&&runs.some(r=>r.language!=='english'))issues.push('RQ1 here requires English instructions throughout.');
  if(varied){
    if(new Set(runs.map(r=>r[varied])).size<2)issues.push(`Select at least two ${varied==='culture'?'country framings':varied==='language'?'instruction languages':'models'}.`);
    for(const key of ['method','model','culture','language','seed'])if(key!==varied&&new Set(runs.map(r=>r[key])).size>1)issues.push(`${key} also varies; hold it constant for this comparison.`);
  }
  if(question==='rq4'&&runs.some(r=>r.language_confound))issues.push('Historical language prompts changed participant language too.');
  const pairs=[];
  for(let i=0;i<runs.length;i++)for(let j=i+1;j<runs.length;j++){
    const key=edge=>JSON.stringify([...edge].sort()),a=new Set(runs[i].edges.map(key)),b=new Set(runs[j].edges.map(key));
    const shared=[...a].filter(e=>b.has(e)).length,union=new Set([...a,...b]).size;
    pairs.push({a:i+1,b:j+1,shared,union,jaccard:union?shared/union:null});
  }
  return {matched:issues.length===0,issues,pairs};
}

export function homophilyMatrix(runs) {
  const demos=[...new Set(runs.flatMap(run=>Object.keys(run.homophily||{})))];
  // Separate age: a numeric assortativity coefficient is not categorical Coleman homophily.
  return `<table><caption>RQ2 / Saved final networks. Coleman: 0 = random-mixing reference, positive = more within-group ties. Undefined values stay NA.</caption><thead><tr><th>Run</th>${demos.map(d=>`<th>${esc(d)}</th>`).join('')}<th>Age assortativity (different measure)</th></tr></thead><tbody>${runs.map((r,i)=>`<tr><th>${i+1}. ${esc(runLabel(r))}</th>${[...demos.map(d=>r.homophily?.[d]),r.age_assortativity].map(v=>`<td>${Number.isFinite(v)?`<span class="coefficient ${v>0?'positive':v<0?'negative':''}">${v.toFixed(3)}</span>`:'NA'}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
export function frequencySvg(personas,runs,layout,selected='') {
  const rows=edgeFrequencies(runs).filter(row=>!selected||row.edge.includes(selected));
  // A fixed-order matrix avoids projecting a dense 3D layout into a hairball.
  const ids=[...new Set(selected?[selected,...rows.flatMap(row=>row.edge)]:personas.map(p=>p.id))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
  const size=600,cell=size/Math.max(1,ids.length),at=id=>55+ids.indexOf(id)*cell;
  const labels=ids.map(id=>`<text x="${at(id)+cell/2}" y="43" text-anchor="middle" font-size="8">${esc(id)}</text><text x="43" y="${at(id)+cell/2+3}" text-anchor="end" font-size="9">${esc(id)}</text>`).join('');
  const cells=rows.flatMap(row=>[row.edge,[...row.edge].reverse()].map(([a,b])=>`<rect x="${at(a)}" y="${at(b)}" width="${cell-.5}" height="${cell-.5}" fill="#213d32" fill-opacity="${row.frequency}" tabindex="0"><title>Personas ${esc(a)} and ${esc(b)}: ${row.count}/${row.total} selected runs</title></rect>`)).join('');
  const chart=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 710 705" role="img" aria-label="Tie frequency matrix"><title>${rows.length} distinct ${selected?'persona '+esc(selected)+' incident ':''}ties across ${runs.length} saved runs. Symmetric cells represent one undirected tie.</title><rect width="710" height="705" fill="#f2f3ef"/><rect x="55" y="55" width="600" height="600" fill="#dce2dc"/>${labels}${cells}<text x="55" y="680" font-size="12">Row / column: persona ID. Darker = more runs. Blank = no recorded tie.</text></svg>`;
  const sets=runs.map(run=>new Set(run.edges.map(edge=>JSON.stringify([...edge].sort()))));
  return `<p>${rows.length} distinct ties. A count of 2 / 3 means two of the three selected runs contain that connection. It is not a probability or accuracy score.</p><div class="tie-table"><table><caption>Exact recorded connections${selected?' involving persona '+esc(selected):''}. Run columns follow the experiment list above.</caption><thead><tr><th>Personas</th><th>Runs containing tie</th>${runs.map((r,i)=>`<th>Run ${i+1}<br>${esc(r.model)}<br>${esc(r.method)}</th>`).join('')}</tr></thead><tbody>${rows.sort((a,b)=>b.count-a.count).map(row=>`<tr><th>${row.edge.map(esc).join(' - ')}</th><td>${row.count} / ${row.total}</td>${sets.map(set=>`<td>${set.has(JSON.stringify([...row.edge].sort()))?'Present':'Absent'}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${runs.length+2}">No recorded connections for this selection.</td></tr>`}</tbody></table></div><details><summary>Optional matrix view</summary>${chart}</details>`;
}
export function trajectorySvg(frames, person, step) {
  if (!frames) return '<p class="empty-note">No recorded history. A final graph cannot reveal its construction order.</p>';
  const values=frames.map(frame=>person?neighbors(frame.edges,person).size:frame.edges.length);
  const w=680,h=145,left=40,right=20,top=20,bottom=30,max=Math.max(1,...values);
  const x=i=>left+i*(w-left-right)/Math.max(1,values.length-1),y=v=>h-bottom-v*(h-top-bottom)/max;
  const path=values.map((v,i)=>i?`H${x(i)}V${y(v)}`:`M${x(0)},${y(v)}`).join(' ');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${person?'Selected persona degree':'Network edge count'} by recorded event"><title>${person?`Persona ${esc(person)} connections`:'Network edges'} by recorded event</title>${[0,max].map(v=>`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="#dfe5e6"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end" fill="#647477" font-size="10">${v}</text>`).join('')}<path d="${path}" fill="none" stroke="#237d7c" stroke-width="2"/><line x1="${x(step)}" x2="${x(step)}" y1="${top}" y2="${h-bottom}" stroke="#c15836" stroke-dasharray="3 3"/><circle cx="${x(step)}" cy="${y(values[step])}" r="4" fill="#c15836"/><text x="${left}" y="${h-10}" fill="#647477" font-size="10">0</text><text x="${w-right}" y="${h-10}" fill="#647477" text-anchor="end" font-size="10">${values.length-1} recorded events</text></svg>`;
}
export function metricBars(runs, metric, title) {
  const values=runs.map(run=>run.metrics[metric]);
  const max=Math.max(1,...values.filter(Number.isFinite)),min=Math.min(0,...values.filter(Number.isFinite)),w=740,h=42+runs.length*55;
  const x=value=>320+340*(value-min)/(max-min),zero=x(0);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)} by saved run"><title>${esc(title)} by saved run, not predictive accuracy</title>${runs.map((run,i)=>{const v=values[i],y=20+i*55;return `<text x="12" y="${y}" font-size="12" fill="#20393c">${i+1}. ${esc(run.model)} / ${esc(run.method)}</text><text x="12" y="${y+17}" font-size="10" fill="#647477">${esc(run.culture)} / ${esc(run.language)} / seed ${esc(run.seed)}</text><rect x="320" y="${y-9}" width="340" height="6" rx="3" fill="#e3e9e8"/>${Number.isFinite(v)?`<rect x="${Math.min(zero,x(v))}" y="${y-9}" width="${Math.abs(x(v)-zero)}" height="6" rx="3" fill="#237d7c"/><line x1="${zero}" x2="${zero}" y1="${y-12}" y2="${y}" stroke="#819392"/><text x="720" y="${y}" text-anchor="end" font-size="13" fill="#20393c">${v.toFixed(3)}</text>`:'<text x="720" y="'+y+'" text-anchor="end">NA</text>'}`;}).join('')}<text x="320" y="${h-7}" font-size="10" fill="#647477">${min}</text><text x="660" y="${h-7}" text-anchor="end" font-size="10" fill="#647477">${max}</text></svg>`;
}
