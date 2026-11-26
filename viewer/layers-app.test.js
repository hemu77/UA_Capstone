import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createStore,INITIAL_STATE,EMPTY_FILTERS,LAYER_COLORS,MAX_STACK_LAYERS} from './layers-app.js';
test('store emits only real changes, freezes state and rejects unknown keys',async()=>{
  const store=createStore(),seen=[];
  store.subscribe((state,changed,source)=>seen.push([[...changed].sort().join(','),source]));
  assert.equal(store.update({view:'stack'}).size,0);
  assert.deepEqual([...store.update({view:'atlas',layers:['a','b']},'test')].sort(),['layers','view']);
  assert.equal(store.update({layers:['a','b'],filters:{...EMPTY_FILTERS}}).size,0);
  assert.throws(()=>store.update({nope:1}),/Unknown state key/);
  assert.throws(()=>{store.getState().layers.push('c');});
  assert.deepEqual(seen,[['layers,view','test']]);
  store.init({runs:[]},{persona:'3'});
  assert.equal((await store.ready).runs.length,0);
  assert.equal(seen.at(-1)[1],'init');assert.equal(seen.at(-1)[0],Object.keys(INITIAL_STATE).sort().join(','));
  assert.throws(()=>store.init({}),/already/);
});
test('a failing listener cannot starve the others',()=>{
  const store=createStore(),error=console.error;let reached=false;console.error=()=>{};
  try{store.subscribe(()=>{throw Error('boom');});store.subscribe(()=>{reached=true;});store.update({persona:'1'});}finally{console.error=error;}
  assert.equal(reached,true);
});
test('eight distinct plane hues for eight schedule blocks',()=>{assert.equal(MAX_STACK_LAYERS,8);assert.equal(new Set(LAYER_COLORS).size,8);});
test('consensus plane is an optional, validated store key that starts off',()=>{
  const store=createStore(),seen=[];
  assert.equal(INITIAL_STATE.consensus,null);
  store.subscribe((state,changed,source)=>seen.push([[...changed].join(','),source]));
  assert.deepEqual([...store.update({consensus:{scope:'filters',min:.5}},'hud')],['consensus']);
  assert.deepEqual(store.getState().consensus,{scope:'filters',min:.5});
  assert.equal(store.update({consensus:{scope:'filters',min:.5}}).size,0,'same value is a no-op');
  assert.throws(()=>{store.getState().consensus.min=1;});
  for(const bad of [{scope:'supplements',min:.5},{scope:'stack',min:0},{scope:'stack',min:1.5},{scope:'stack'},true,'filters',{scope:'stack',min:.5,runs:[]}])assert.throws(()=>store.update({consensus:bad}),/consensus must be/);
  assert.deepEqual([...store.update({consensus:null},'hud')],['consensus']);
  assert.deepEqual(seen,[['consensus','hud'],['consensus','hud']]);
});
