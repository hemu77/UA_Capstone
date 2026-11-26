// Final-only gallery data never enters the primary replay or statistical inputs.
export const SAVED_NETWORKS_SHA256='5ea36a4aa914ad05d1fe88303b0e5218689de9bc85138c359eda4fbf5918960b';
export function validateNetworkCatalog(data, plots, primary) {
  const keys=value=>Object.keys(value).sort().join(',');
  if(keys(data)!=='catalog_sha256,graphs,roster_id,schema_version'||data.schema_version!==1||data.catalog_sha256!=='b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775'||data.roster_id!==primary.roster_id||!Array.isArray(data.graphs)||data.graphs.length!==896)throw Error('Unexpected final network catalog');
  const people=new Set(primary.personas.map(p=>p.id)),meta=new Map(plots.map(p=>[p.run_id,p])),seen=new Set();
  if(people.size!==50||meta.size!==896)throw Error('Unexpected roster or metadata');
  for(const g of data.graphs){
    const p=meta.get(g.run_id);
    if(keys(g)!=='edges,run_id,source,source_sha256'||!p||seen.has(g.run_id)||g.source!==p.source_path.replace(/\.png$/,'.adj')||!/^[a-f0-9]{64}$/.test(g.source_sha256)||!Array.isArray(g.edges)||g.edges.length!==p.edges)throw Error('Invalid final network identity');
    seen.add(g.run_id);const edges=new Set();
    for(const e of g.edges){if(!Array.isArray(e)||e.length!==2||e[0]===e[1]||!e.every(id=>people.has(id)))throw Error('Invalid final network edge');const key=JSON.stringify([...e].sort());if(edges.has(key))throw Error('Duplicate final network edge');edges.add(key);}
  }
  return new Map(data.graphs.map(g=>[g.run_id,g]));
}
