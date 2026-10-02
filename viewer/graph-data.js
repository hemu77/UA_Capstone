// These helpers transform edges for display, never recompute research statistics.
export const edgeKey = ([a, b]) => JSON.stringify([a, b].sort());
export const togglePersona = (current, next) => current === next ? '' : next;
export function tapTracker() {
  let start=null;
  return {
    down(e) { if(start||e.button!==0||e.isPrimary===false){start=null;return;} start={id:e.pointerId,x:e.clientX,y:e.clientY,moved:false}; },
    move(e) { if(start&&e.pointerId===start.id&&Math.hypot(e.clientX-start.x,e.clientY-start.y)>5)start.moved=true; },
    cancel() { start=null; },
    up(e) { const s=start;start=null;return !!s&&s.id===e.pointerId&&!s.moved&&Math.hypot(e.clientX-s.x,e.clientY-s.y)<=5; }
  };
}
export function personaView(edges, selected, isolate=false) {
  // Ego view means the selected person's direct ties, not neighbors' other ties.
  const visibleEdges=selected&&isolate?edges.filter(edge=>edge.includes(selected)):edges;
  return {edges:visibleEdges, neighbors:selected?[...neighbors(edges,selected)].sort():[],
    visibleIds:selected&&isolate?new Set([selected,...visibleEdges.flat()]):null};
}
export function personaTag(id,selected,actor,step) {
  return `Persona ${id}${id===selected?' / selected':''}${id===actor?` / actor at event ${step}`:''}`;
}
export function personaComparison(runs,id) {
  layerComparison(runs);
  const sets=runs.map(run=>neighbors(run.edges,id));
  const union=new Set(sets.flatMap(set=>[...set]));
  const shared=[...union].filter(person=>sets.every(set=>set.has(person)));
  return {union:union.size,shared:shared.length,jaccard:union.size?shared.length/union.size:null};
}
export function edgeFrequencies(runs) {
  const {counts}=layerComparison(runs);
  return [...counts].map(([key,count])=>({edge:JSON.parse(key),count,total:runs.length,frequency:count/runs.length}));
}
export const comparisonModels = ['gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol'];
export function layerCandidates(runs, filters) {
  return runs.filter(run => (filters.collection === 'calibration' ? run.study === 'calibration' : filters.collection === 'historical' ? ['cultural','method','language'].includes(run.study) : run.study === 'engineering_pilot') &&
    (filters.collection !== 'revised' || comparisonModels.includes(run.model)) &&
    ['method', 'culture', 'language', 'model', 'seed'].every(key => !filters[key] || String(run[key]) === String(filters[key])));
}
export function applyLayerSelection(candidates) {
  // Never silently truncate a query: that would bias the visible comparison.
  layerComparison(candidates);
  return [...candidates];
}
export function layerComparison(runs) {
  // Identity links only make sense when the same roster appears in every layer.
  if (!runs.length || runs.length > 6) throw new Error('Choose between one and six layers.');
  if (new Set(runs.map(r => r.run_id)).size !== runs.length) throw new Error('Duplicate layer.');
  if (runs.some(r => !r.roster_id || r.roster_id !== runs[0].roster_id)) throw new Error('Layers require the same roster.');
  const counts = new Map();
  for (const run of runs) for (const key of new Set(run.edges.map(edgeKey))) counts.set(key, (counts.get(key) || 0) + 1);
  return {counts, shared: [...counts.values()].filter(n => n === runs.length).length, union: counts.size};
}
export function comparisonLayout(personas, leftEdges, rightEdges) {
  // Both panes use the same union layout. Differences cannot be layout drift.
  // ponytail: O(150*n^2), bounded to this 50-person viewer; use a worker for larger rosters.
  const ids = personas.map(p => p.id).sort(), points = new Map();
  ids.forEach((id, i) => points.set(id, [55*Math.cos(2*Math.PI*i/ids.length), 55*Math.sin(2*Math.PI*i/ids.length), (i%5-2)*8]));
  const edges = [...new Map([...leftEdges, ...rightEdges].map(e => [edgeKey(e), e])).values()];
  for (const edge of edges) if (edge[0] === edge[1] || !edge.every(id => points.has(id))) throw new Error('Invalid layout edge.');
  for (let step=0; step<150; step++) {
    const force = new Map(ids.map(id => [id, [0,0,0]]));
    for (let i=0;i<ids.length;i++) for (let j=i+1;j<ids.length;j++) {
      const a=points.get(ids[i]), b=points.get(ids[j]), delta=a.map((v,k)=>v-b[k]);
      const distance=Math.max(1,Math.hypot(...delta));
      delta.forEach((v,k)=>{const f=500*v/(distance*distance);force.get(ids[i])[k]+=f;force.get(ids[j])[k]-=f;});
    }
    for (const [u,v] of edges) {
      const a=points.get(u),b=points.get(v),delta=a.map((x,k)=>b[k]-x),distance=Math.max(1,Math.hypot(...delta));
      delta.forEach((x,k)=>{const f=.025*(distance-23)*x/distance;force.get(u)[k]+=f;force.get(v)[k]-=f;});
    }
    const temperature=3*(1-step/150)+.05;
    for (const id of ids) {
      const point=points.get(id),f=force.get(id).map((v,k)=>v-.02*point[k]),length=Math.max(1,Math.hypot(...f));
      point.forEach((v,k)=>point[k]=v+f[k]*Math.min(temperature,length)/length);
    }
  }
  const scale=75/Math.max(1,...[...points.values()].map(p=>Math.hypot(...p)));
  for (const point of points.values()) point.forEach((v,k)=>point[k]=v*scale);
  return points;
}
export function validateData(data) {
  if (data.schema_version !== 1 || !Array.isArray(data.personas) || !data.personas.length || !Array.isArray(data.runs) || !data.runs.length) throw new Error('Unsupported or empty export.');
  const ids = new Set(data.personas.map(p => p.id));
  if (ids.size !== data.personas.length || data.personas.some(p => !Array.isArray(p.position) || p.position.length !== 3 || !p.position.every(Number.isFinite))) throw new Error('Invalid roster or coordinates.');
  const runIds = new Set();
  for (const run of data.runs) {
    if (runIds.has(run.run_id) || !run.roster_id) throw new Error('Duplicate run or missing roster identity.');
    if (run.roster_id !== (data.roster_id || data.runs[0].roster_id)) throw new Error('Graph roster identity differs from displayed personas.');
    runIds.add(run.run_id);
    const edges = new Set();
    for (const edge of run.edges) {
      if (edge.length !== 2 || edge[0] === edge[1] || !edge.every(id => ids.has(id)) || edges.has(edgeKey(edge))) throw new Error(`Invalid edge in ${run.run_id}.`);
      edges.add(edgeKey(edge));
    }
  }
  return data;
}

// Hold every other recorded field fixed; never fill missing conditions with a pilot.
export function matchedRuns(all, reference, dimension) {
  if (!['model','culture','language','method'].includes(dimension)) throw new Error('Unknown comparison dimension.');
  const keys=['study','roster_id','prompt_variant','model','method','culture','language','seed'];
  return all.filter(run=>keys.every(key=>key===dimension||run[key]===reference[key]));
}

export function filterOptions(all, filters, key) {
  return new Set(layerCandidates(all,{...filters,[key]:''}).map(run=>String(run[key])));
}
export function differences(left, right) {
  if (left.roster_id !== right.roster_id) return null;
  const a = new Set(left.edges.map(edgeKey)), b = new Set(right.edges.map(edgeKey));
  return {shared: new Set([...a].filter(e => b.has(e))), onlyA: new Set([...a].filter(e => !b.has(e))), onlyB: new Set([...b].filter(e => !a.has(e)))};
}
export function neighbors(edges, id) {
  return new Set(edges.filter(e => e.includes(id)).map(([a, b]) => a === id ? b : a));
}
export function replayEdges(events, step) {
  if (!Array.isArray(events) || !Number.isInteger(step) || step < 0 || step > events.length) throw new Error('Replay requires recorded events and a valid step.');
  const edges = new Map();
  for (const event of events.slice(0, step)) {
    for (const edge of event.removed) {
      if (!edges.delete(edgeKey(edge))) throw new Error('Replay attempted to remove a missing edge.');
    }
    for (const edge of event.added) {
      if (edge[0] === edge[1] || edges.has(edgeKey(edge))) throw new Error('Invalid replay addition.');
      edges.set(edgeKey(edge), edge);
    }
  }
  return [...edges.values()];
}
export function personaSteps(events, id) {
  if (!Array.isArray(events) || !id) return [];
  // Include decisions with no net change and ties initiated by somebody else.
  return events.flatMap((event, i) => event.persona === id || [...event.added, ...event.removed].some(edge => edge.includes(id)) ? [i + 1] : []);
}
export function replayFrames(run, ids) {
  if (!Array.isArray(run.events) || !run.events.length) return null;
  const known = new Set(ids);
  for (const event of run.events) {
    if (event.persona != null && !known.has(event.persona)) throw new Error('Unknown replay actor.');
    for (const edge of [...event.added, ...event.removed]) if (!edge.every(id => known.has(id))) throw new Error('Unknown replay endpoint.');
  }
  // ponytail: cache prefixes for logs of at most 350 events here; use incremental
  // checkpointing if future datasets grow beyond this small research viewer.
  const frames = [{edges: [], event: null}, ...run.events.map((event, i) => ({edges: replayEdges(run.events, i + 1), event}))];
  const actual = new Set(frames.at(-1).edges.map(edgeKey)), expected = new Set(run.edges.map(edgeKey));
  if (actual.size !== expected.size || [...actual].some(key => !expected.has(key))) throw new Error('Replay does not reconstruct final network.');
  return frames;
}
