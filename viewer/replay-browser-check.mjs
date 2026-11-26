// Replay browser check (docs/qa/viewer-polish-2): event counts per method, opening at event 0, stepping changes the
// tie count exactly as recorded, the default run of a mixed stack, the one-response (Global) state, keyboard,
// phone layout, and the Share view button (label always returns).
// Run against a static build:  BROWSER_CHANNEL=chromium VIEWER_URL=http://127.0.0.1:8781 node replay-browser-check.mjs
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge',args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const store=(page,fn,arg)=>page.evaluate(async([source,arg])=>{const {app}=await import(document.querySelector('script[src*="layers.js"]').src);return new Function('app','arg',`return (${source})(app,arg);`)(app,arg);},[fn.toString(),arg]);
const SINGLE='Global generation returned the whole network in one response — there is no step-by-step record to replay.';
const watch=page=>{const errors=[];page.on('pageerror',e=>errors.push('pageerror: '+e.message));page.on('console',m=>{if(['error','warning'].includes(m.type())&&!/GPU stall|GL Driver Message|Automatic fallback to software WebGL/.test(m.text()))errors.push(m.type()+': '+m.text());});return errors;};
const ready=async page=>{await page.waitForSelector('body[data-ready=true]');await page.waitForFunction(()=>Number(document.querySelector('#viewport canvas')?.dataset.frames)>0);};
const loaded=page=>page.waitForFunction(()=>!document.querySelector('#play').disabled);
const ties=async page=>Number((await page.locator('#status').textContent()).match(/(\d+) current ties?/)[1]);
const frame=page=>store(page,app=>app.getState().frame);
// The recorded events of the active run, as the page verified them.
const record=page=>store(page,app=>{const s=app.getState(),run=app.data.runs.find(r=>r.run_id===s.active);return {method:run.method,final:run.edges.length,events:(run.events||[]).map(e=>[e.persona,e.added.length,e.removed.length])};});
try{
  const page=await browser.newPage({viewport:{width:1440,height:900},deviceScaleFactor:1});
  const errors=watch(page);
  await page.goto(base+'/layers.html');await ready(page);

  // --- Default: the stack is Global, Local, Sequential, Iterative with the Global run active. Replay picks the
  // first run that has steps, says so, and opens at event 0 with no ties drawn.
  const stack=await store(page,app=>app.getState().layers.map(id=>app.data.runs.find(r=>r.run_id===id).method));
  assert.deepEqual(stack,['global','local','sequential','iterative']);
  await page.locator('#formation-mode').click();await loaded(page);
  let state=await store(page,app=>({...app.getState()}));
  assert.equal(state.view,'replay');assert.equal(state.active,state.layers[1],'Replay prefers a run with a multi-event record');assert.equal(state.frame,0);
  assert.match(await page.locator('#event-delta').innerText(),/Replaying run 2 \(Local\): run 1 \(Global\) was returned in one response/);
  assert.match(await page.locator('#active-badge').innerText(),/Replaying run 2 of 4/i);
  assert.match(await page.locator('#stage-title').innerText(),/Local/);
  assert.equal(await page.locator('#step-label').innerText(),'Event 0 of 50');
  assert.equal(await ties(page),0,'Replay opens at the start: no ties yet');
  assert.equal(await page.locator('#timeline').inputValue(),'0');
  for(const id of ['rewind','previous','play','next','show-final','speed','timeline'])assert.equal(await page.locator('#'+id).isVisible(),true,id+' is visible');
  assert.equal(await page.locator('#play').isEnabled(),true);assert.equal(await page.locator('#next').isEnabled(),true);
  assert.equal(await page.locator('#previous').isDisabled(),true);assert.equal(await page.locator('#rewind').isDisabled(),true);
  const playBox=await page.locator('#play').boundingBox(),nextBox=await page.locator('#next').boundingBox();
  assert.ok(playBox.width>=nextBox.width*2,'Play is the primary, largest transport control');

  // --- Event counts per method and stepping: every opening route lands on event 0; each step changes the tie
  // count by exactly the recorded additions minus removals; ticks = recorded events.
  const expected={local:n=>n===50,sequential:n=>n===50,iterative:n=>n>50&&n<=350};
  for(const [key,method] of [['2','local'],['3','sequential'],['4','iterative']]){
    await page.keyboard.press(key);
    await page.waitForFunction(m=>document.querySelector('#stage-title').textContent.toLowerCase().includes(m),method);await loaded(page);
    const rec=await record(page),total=rec.events.length;
    assert.equal(rec.method,method);assert.ok(expected[method](total),`${method}: ${total} events`);
    assert.equal(await frame(page),0,`${method} opens at event 0`);
    assert.equal(await page.locator('#step-label').innerText(),`Event 0 of ${total}`);
    assert.equal(await page.locator('#timeline').getAttribute('max'),String(total));
    assert.equal(await ties(page),0);
    assert.equal(await page.locator('#timeline-ticks svg').first().evaluate(svg=>svg.getAttribute('viewBox')),`0 0 ${total} 20`);
    assert.equal(await page.locator('#timeline-ticks svg').first().locator('.tick-add,.tick-remove,.tick-none').count(),rec.events.reduce((n,e)=>n+(e[1]?1:0)+(e[2]?1:0)+(!e[1]&&!e[2]?1:0),0),'one tick mark per recorded change');
    let running=0;
    const steps=method==='iterative'?64:12;
    for(let k=1;k<=steps;k++){
      await page.locator('#next').click();running+=rec.events[k-1][1]-rec.events[k-1][2];
      assert.equal(await frame(page),k);assert.equal(await ties(page),running,`${method} event ${k}: tie count as recorded`);
    }
    const [persona,added,removed]=rec.events[steps-1],card=await page.locator('#event-delta').innerText();
    assert.match(card,new RegExp(`Event ${steps} of ${total}`,'i'));assert.match(card,new RegExp(`Persona ${persona}\\b`));
    assert.match(card,new RegExp(`\\+${added} added`));assert.match(card,new RegExp(`−${removed} removed`));assert.match(card,new RegExp(`${running} ties? now`));
    if(method==='iterative')assert.ok(rec.events.slice(50,steps).some(e=>e[2]>0),'the iterative walk crossed recorded removals');
    await page.locator('#previous').click();assert.equal(await ties(page),running-(rec.events[steps-1][1]-rec.events[steps-1][2]),'stepping back undoes exactly that event');
    await page.locator('#show-final').click();assert.equal(await frame(page),total);assert.equal(await ties(page),rec.final,'the end is the saved final graph');
    assert.match(await page.locator('#play').innerText(),/Play again/);
    await page.locator('#rewind').click();assert.equal(await frame(page),0);assert.equal(await ties(page),0);
  }

  // --- Play / pause and keyboard: Space plays from the start, events advance in order, Space pauses.
  await page.locator('#speed').selectOption('160');
  await page.locator('#stage').focus();
  await page.keyboard.press('Space');
  await page.waitForFunction(()=>document.querySelector('#play').dataset.state==='playing');
  assert.match(await page.locator('#play').innerText(),/Pause/);
  await page.waitForFunction(()=>Number(document.querySelector('#timeline').value)>=3);
  await page.keyboard.press('Space');
  assert.match(await page.locator('#play').innerText(),/^Play$/);
  const paused=await frame(page);assert.ok(paused>=3);await page.waitForTimeout(500);assert.equal(await frame(page),paused,'paused playback does not advance');
  await page.keyboard.press('ArrowRight');assert.equal(await frame(page),paused+1);
  await page.keyboard.press('ArrowLeft');assert.equal(await frame(page),paused);
  await page.keyboard.press('End');assert.equal(Number(await page.locator('#timeline').inputValue()),Number(await page.locator('#timeline').getAttribute('max')));
  await page.keyboard.press('Home');assert.equal(await frame(page),0);
  // Other routes into Replay also open at the start: the stacked-run list and the plane label on the stage.
  await page.keyboard.press('End');
  await page.locator('#compare-mode').click();
  await page.locator('.g3-plane').nth(2).click();await loaded(page);
  assert.equal(await store(page,app=>app.getState().view),'replay');assert.equal(await frame(page),0,'a plane label opens Replay at event 0');assert.equal(await ties(page),0);

  // --- A Global run: one response. The statement, the alternatives, and no dead play button.
  await page.keyboard.press('1');
  await page.waitForFunction(()=>document.body.dataset.replayRecord==='single');
  assert.equal(await page.locator('#replay-single-text').innerText(),SINGLE);
  for(const id of ['play','next','previous','rewind','show-final','timeline'])assert.equal(await page.locator('#'+id).isVisible(),false,id+' is not offered for a one-response record');
  await page.waitForFunction(()=>(document.querySelector('#status').textContent.match(/(\d+) current/)||[])[1]>0);
  const globalRecord=await record(page);
  assert.equal(globalRecord.events.length,1,'Global: exactly one recorded event');assert.equal(globalRecord.events[0][0],null);
  assert.equal(await ties(page),globalRecord.final,'the whole saved network is on stage, not an empty start');
  assert.match(await page.locator('#event-delta').innerText(),new RegExp(`One response[\\s\\S]*${globalRecord.final} ties returned together`,'i'));
  assert.equal(await page.locator('#timeline-ticks svg').count(),0,'no invented ticks');
  await page.keyboard.press('Space');await page.keyboard.press('ArrowRight');
  assert.equal(await ties(page),globalRecord.final,'keys do nothing to a one-response record');
  assert.deepEqual(await page.locator('#replay-single [data-replay-alt]').allInnerTexts(),['Replay the Local run','Replay the Sequential run','Replay the Iterative run']);
  assert.match(await page.locator('#trajectory').innerText(),/arrived in one recorded response/);
  // "View ties by persona" selects a persona and isolates its ties.
  await page.locator('#single-by-persona').click();
  const persona=await store(page,app=>app.getState().persona);assert.ok(persona,'a persona is selected');
  assert.equal(await page.locator('#incident').isChecked(),true);
  await store(page,app=>app.update({persona:''},'check'));
  // The alternative is a real saved run of the same model, setting and seed, replayed from event 0.
  const before=await store(page,app=>{const s=app.getState();return app.data.runs.find(r=>r.run_id===s.active);});
  await page.locator('#replay-single [data-replay-alt=sequential]').click();await loaded(page);
  const after=await store(page,app=>{const s=app.getState();return {...app.data.runs.find(r=>r.run_id===s.active),events:undefined,frame:s.frame,view:s.view};});
  assert.equal(after.method,'sequential');for(const key of ['model','culture','language','seed'])assert.equal(after[key],before[key]);
  assert.equal(after.view,'replay');assert.equal(after.frame,0);assert.equal(await ties(page),0);
  assert.equal(await page.locator('#play').isVisible(),true);

  // An all-Global stack (one condition's seeds): Replay stays on the active run and states the single response.
  await store(page,app=>{const ids=app.data.runs.filter(r=>r.model==='gpt-6-sol'&&r.method==='global'&&r.culture==='india'&&r.language==='english').map(r=>r.run_id);app.update({layers:ids,active:ids[0],view:'stack'},'check');});
  await page.locator('#formation-mode').click();
  await page.waitForFunction(()=>document.body.dataset.replayRecord==='single');
  assert.equal(await page.locator('#replay-single-text').innerText(),SINGLE);
  // A result-row Replay of a person-level run opens at event 0 as well.
  await store(page,app=>{const s=app.getState();app.update({layers:s.layers.slice(0,2),active:s.layers[0]},'check');});
  await store(page,app=>app.update({view:'atlas',filters:{model:['gpt-5.6-luna'],method:['iterative'],setting:['brazil-english'],seed:[44003],graph:'any',search:''}},'check'));
  await page.locator('#tab-results').click();
  await page.locator('#run-results .flt-row .flt-replay').first().click();await loaded(page);
  assert.equal(await frame(page),0);assert.equal(await ties(page),0);assert.match(await page.locator('#stage-title').innerText(),/GPT-5\.6 Luna · Iterative · Brazil \/ English · seed 44003/);
  // Links: a shared replay position is kept; the same link without a position opens at the start.
  await page.locator('#next').click();await page.locator('#next').click();await page.waitForTimeout(700);
  const link=new URL(page.url());assert.equal(link.searchParams.get('f'),'2');
  await page.goto(link.href);await ready(page);await loaded(page);
  assert.equal(await frame(page),2);
  link.searchParams.delete('f');await page.goto(link.href);await ready(page);await loaded(page);
  assert.equal(await frame(page),0,'a replay link without a position opens at event 0');assert.equal(await ties(page),0);
  // --- Share view: the label always returns to "Share view" (success and failure), and the width never changes.
  const share=page.locator('#share'),width=(await share.boundingBox()).width;
  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.__copied=text;}}});});
  await share.click();
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Link copied');
  assert.equal(await page.evaluate(()=>window.__copied),page.url());
  assert.match(await page.locator('#action-status').textContent(),/copied/i);
  assert.equal((await share.boundingBox()).width,width,'the header does not jump');
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Share view',null,{timeout:4000});
  // Failure: the clipboard API rejects and the textarea fallback is refused.
  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw new Error('denied');}}});document.execCommand=()=>false;});
  await share.click();
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Not copied');
  assert.match(await page.locator('#action-status').textContent(),/Copy the URL from your address bar/);
  assert.equal((await share.boundingBox()).width,width);
  await share.click();await page.waitForTimeout(1200);await share.click(); // repeated clicks restart the timer
  await page.waitForTimeout(1200);assert.equal(await share.textContent(),'Not copied','an earlier timer does not cut a later message short');
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Share view',null,{timeout:4000});
  assert.equal(await page.locator('textarea[aria-hidden=true]').count(),0,'the fallback textarea is removed');
  // Fallback path: no clipboard API, execCommand succeeds.
  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});document.execCommand=command=>command==='copy';});
  await share.click();
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Link copied');
  await page.waitForFunction(()=>document.querySelector('#share').textContent==='Share view',null,{timeout:4000});
  assert.deepEqual(errors,[],'no console errors or warnings');

  // --- Phone: the transport is on the first screen and nothing overflows sideways.
  const phone=await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1,hasTouch:true,isMobile:true});
  const phoneErrors=watch(phone);
  await phone.goto(base+'/layers.html');await ready(phone);
  await phone.locator('#formation-mode').tap();await loaded(phone);
  for(const id of ['rewind','previous','play','next','show-final','timeline']){const box=await phone.locator('#'+id).boundingBox();assert.ok(box&&box.y>=0&&box.y+box.height<=844-56&&box.x>=0&&box.x+box.width<=390,`${id} is on the first phone screen`);}
  assert.equal(await phone.locator('#step-label').innerText(),'Event 0 of 50');
  await phone.locator('#next').tap();assert.equal(await phone.locator('#timeline').inputValue(),'1');
  assert.equal(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'no sideways overflow');
  await store(phone,app=>{const s=app.getState();app.update({active:s.layers[0],frame:Number.MAX_SAFE_INTEGER},'check');});
  await phone.waitForFunction(()=>document.body.dataset.replayRecord==='single');
  assert.equal(await phone.locator('#replay-single-text').innerText(),SINGLE);
  assert.equal(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(phoneErrors,[]);
  console.log('PASS: Replay prefers a multi-event run from the mixed default stack and names it; Local 50, Sequential 50, Iterative >50 events open at event 0 with 0 ties; each step changes the tie count exactly as recorded (and back); end = saved final graph; play/pause, arrows, Home/End; plane label, result row and bare links open at the start; Global shows the one-response statement with real alternatives and no transport; Share view label reverts on success, fallback and failure; phone transport on the first screen; zero console errors.');
}finally{await browser.close();}
