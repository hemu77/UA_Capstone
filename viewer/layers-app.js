// Shared state contract for the layers page (see docs/qa/viewer-3d-2026-11-25/SPEC.md, section C.2).
// FROZEN during the parallel upgrade: only the integrator edits this file.
// Every module talks to the others through this store, never through another module's DOM.
export const MAX_STACK_LAYERS=8; // one condition = 8 schedule blocks (seeds 44000-44007)
export const VIEWS=Object.freeze(['atlas','stack','replay']);
export const GRAPH_STATES=Object.freeze(['any','nonempty','empty']);
export const LENSES=Object.freeze(['none','same','mixed']);
// Plane / run identity hues. Mirrored as --layer-1 .. --layer-8 in layers.css.
export const LAYER_COLORS=Object.freeze(['#6fd6ba','#f0b56c','#8fb6ff','#e995c4','#c5d86d','#b7a6ff','#5cc8e8','#e8a08a']);
export const EMPTY_FILTERS=Object.freeze({model:Object.freeze([]),method:Object.freeze([]),setting:Object.freeze([]),seed:Object.freeze([]),graph:'any',search:''});
export const INITIAL_STATE=Object.freeze({
  filters:EMPTY_FILTERS, // which of the 890 valid graphs are "in scope" (atlas emphasis, result list, counts)
  view:'stack',          // 'atlas' = all 896 slots; 'stack' = up to 8 chosen planes; 'replay' = one run's recorded order
  layers:Object.freeze([]), // run_ids shown as planes, in order, 1..MAX_STACK_LAYERS
  active:'',             // run_id inspected / replayed; must be in layers
  frame:0,               // replay cursor (recorded update index); generation order, not time
  persona:'',            // selected roster id or ''
  colorBy:'age',         // 'layer' | roster attribute key
  lens:'none',           // tie emphasis for colorBy attribute: same-group / mixed-group / none
  groups:Object.freeze([]), // isolated categories of colorBy; [] = all
  focusSlot:null,        // atlas slot (0..895) under focus, or null
  consensus:null         // optional aggregate plane in the Stack: null | {scope:'stack'|'filters', min: share in (0,1]}; never a run
});
const same=(a,b)=>Object.is(a,b)||(Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>Object.is(v,b[i])))||
  (a&&b&&typeof a==='object'&&typeof b==='object'&&!Array.isArray(a)&&!Array.isArray(b)&&Object.keys(a).length===Object.keys(b).length&&Object.keys(a).every(k=>same(a[k],b[k])));
const freeze=value=>{if(value&&typeof value==='object'&&!Object.isFrozen(value)){for(const v of Object.values(value))freeze(v);Object.freeze(value);}return value;};
export const CONSENSUS_SCOPES=Object.freeze(['stack','filters']);
const validConsensus=value=>value===null||(value&&typeof value==='object'&&Object.keys(value).length===2&&CONSENSUS_SCOPES.includes(value.scope)&&typeof value.min==='number'&&value.min>0&&value.min<=1);
export function createStore(initial={}) {
  let state=INITIAL_STATE,data=null,resolveReady,announcer=()=>{};
  const listeners=new Set(),ready=new Promise(resolve=>{resolveReady=resolve;});
  const emit=(changed,source)=>{for(const fn of [...listeners]){try{fn(state,changed,source);}catch(error){console.error('layers-app listener failed',error);}}};
  const store={
    ready,
    get data(){return data;},
    getState:()=>state,
    // Merge a partial state. Returns the Set of keys that really changed (empty Set = no-op, no emit).
    update(patch,source='') {
      const changed=new Set();
      for(const [key,value] of Object.entries(patch)){
        if(!Object.hasOwn(INITIAL_STATE,key))throw new Error(`Unknown state key: ${key}`);
        if(key==='consensus'&&!validConsensus(value))throw new Error('consensus must be null or {scope:\'stack\'|\'filters\', min in (0,1]}');
        if(!same(state[key],value))changed.add(key);
      }
      if(!changed.size)return changed;
      state=Object.freeze({...state,...Object.fromEntries([...changed].map(key=>[key,freeze(structuredClone(patch[key]))]))});
      emit(changed,source);return changed;
    },
    // fn(state, changedKeys:Set<string>, source:string). Returns unsubscribe.
    subscribe(fn){listeners.add(fn);return()=>listeners.delete(fn);},
    // Called once by layers.js when the dataset has loaded and validated.
    init(dataset,patch={}) {if(data)throw new Error('Store already initialised');data=dataset;store.update(patch,'init');resolveReady(dataset);emit(new Set(Object.keys(INITIAL_STATE)),'init');},
    // One polite live region for the whole page (#action-status); layers.js installs the sink.
    announce(message){announcer(String(message));},
    setAnnouncer(fn){announcer=fn;}
  };
  if(Object.keys(initial).length)store.update(initial,'initial');
  return store;
}
