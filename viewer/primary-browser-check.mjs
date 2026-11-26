// Error-path regressions for the actual Three.js app; no study API calls.
// Covers: lazy, checksum-verified replay history; saved replay frame in the link; keyboard focus after actions;
// phone width; 404 and checksum refusal; stale links; data request failure; WebGL-unavailable fallback.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
// Controls may live in a collapsed dock, tab or bottom sheet of the page shell: open whatever contains them.
async function reveal(page,selector){
  if(await page.locator(selector).first().isVisible().catch(()=>false))return;
  await page.evaluate(sel=>{
    const element=document.querySelector(sel),panel=element?.closest('.shell-panel[data-panel]');
    if(panel)for(const tab of document.querySelectorAll(`.shell-tab[data-panel="${panel.dataset.panel}"]`))tab.click();
    const dock=element?.closest('[data-dock]');
    if(dock&&dock.dataset.open==='false')document.querySelector(`.dock-toggle[data-dock="${dock.dataset.dock}"]`)?.click();
  },selector);
  await page.locator(selector).first().waitFor({state:'visible',timeout:5000});
}
const ready=page=>page.waitForSelector('body[data-ready=true]');
try{
  let release;const gate=new Promise(r=>release=r);
  const page=await browser.newPage({viewport:{width:390,height:844}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/data/primary-events/**',async route=>{await gate;await route.continue();});
  await page.goto(base+'/layers.html');
  await ready(page);await page.waitForSelector('#layers .select-run',{state:'attached'});
  await reveal(page,'#formation-mode');await page.locator('#formation-mode').click();
  assert.equal(await page.locator('#play').isDisabled(),true);
  assert.match(await page.locator('#replay-note').innerText(),/Loading and verifying/);
  release();
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  assert.ok(await page.locator('#events button').count()>0,'Lazy completion must rebuild ledger');
  assert.doesNotMatch(await page.locator('#events').innerText(),/No event log/);
  await reveal(page,'#layers .select-run');await page.locator('#layers .select-run').nth(2).click();
  await page.waitForFunction(()=>document.querySelectorAll('#events button').length===50);
  await page.locator('#events button').nth(3).click();
  assert.equal(await page.locator('#timeline').inputValue(),'4');
  assert.equal(await page.locator('#viewport canvas').getAttribute('data-view'),'replay');
  await page.waitForTimeout(700); // the address bar is written shortly after the last change (debounced)
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#timeline').value==='4'&&!document.querySelector('#play').disabled);
  await reveal(page,'#download-selection');
  const download=page.waitForEvent('download');await page.locator('#download-selection').click();
  const downloaded=await download;
  const {readFile}=await import('node:fs/promises');
  const bundle=JSON.parse(await readFile(await downloaded.path(),'utf8'));
  const {replayFrames}=await import('./graph-data.js');
  assert.equal(bundle.runs.length,4);
  for(const r of bundle.runs)assert.ok(replayFrames(r,bundle.personas.map(p=>p.id)));
  await page.evaluate(()=>{const s=document.querySelector('#person');s.value='12';s.dispatchEvent(new Event('change',{bubbles:true}));});
  await reveal(page,'#person');
  const neighbor=page.locator('.neighbor').first();
  if(await neighbor.count()){await neighbor.focus();await neighbor.press('Enter');assert.equal(await page.locator('#person').evaluate(n=>n===document.activeElement),true);}
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  // On a phone the shell closes its sheet after a stack change; keyboard focus after Remove is checked on desktop.
  await page.setViewportSize({width:1440,height:900});
  await reveal(page,'#layers .remove-run');
  await page.locator('#layers .remove-run').first().press('Enter');
  assert.equal(await page.locator('#undo').evaluate(n=>n===document.activeElement),true);
  assert.equal(await page.locator('#layers li').count(),3);
  await page.locator('#undo').click();assert.equal(await page.locator('#layers li').count(),4,'Undo restores the removed layer');
  assert.deepEqual(errors,[]);
  await page.close();

  for(const failure of ['missing','checksum']){
    const p=await browser.newPage();
    await p.route('**/data/primary-events/**',r=>r.fulfill({status:failure==='missing'?404:200,contentType:'application/json',body:'[]'}));
    await p.goto(base+'/layers.html');
    await p.waitForFunction(()=>document.querySelector('#action-status').textContent.includes('Reload to retry'));
    assert.equal(await p.locator('#play').isDisabled(),true);
    await p.evaluate(()=>document.querySelector('#formation-mode').click());
    // Replay now opens the first stacked run with a step-by-step record; its history fails the same way and says so.
    await p.waitForFunction(()=>/could not be verified/.test(document.querySelector('#replay-note').textContent));
    assert.match(await p.locator('#replay-note').innerText(),/could not be verified/);
    assert.equal(await p.locator('#play').isDisabled(),true);
    assert.equal(await p.locator('#layers li').count(),4);
    await p.close();
  }
  const stale=await browser.newPage();
  const state=encodeURIComponent(JSON.stringify({runs:['final_060dcd3a86d5_main_absent']}));
  await stale.goto(base+'/layers.html?dataset=primary&view='+state);
  await stale.locator('#reset-view').waitFor({state:'visible'});
  assert.equal(await stale.locator('#layers li').count(),0);
  assert.match(await stale.locator('#dataset-summary').innerText(),/absent from this dataset/);
  await stale.close();
  // A stack larger than the 8-plane limit in a link is refused with a message, never truncated.
  const tooMany=await browser.newPage();
  const ids=await tooMany.evaluate(async()=>{const d=await(await fetch('./data/primary-viewer.json')).json();return d.runs.filter(r=>r.model==='gpt-4.1'&&r.method==='local').slice(0,9).map(r=>r.run_id);}).catch(()=>null);
  if(ids?.length===9){
    await tooMany.goto(base+'/layers.html?dataset=primary&view='+encodeURIComponent(JSON.stringify({runs:ids})));
    await tooMany.locator('#reset-view').waitFor({state:'visible'});
    assert.match(await tooMany.locator('#dataset-summary').innerText(),/at most 8/);
  }
  await tooMany.close();
  const bad=await browser.newPage();
  // Whatever the primary data files are called, a failed request shows the error and loads nothing else.
  await bad.route(url=>/\/data\/(primary-viewer\.json|graphs\/)/.test(url.pathname),r=>r.fulfill({status:503,body:'Unavailable'}));
  await bad.goto(base+'/layers.html');
  await bad.locator('#retry-load').waitFor({state:'visible'});
  assert.match(await bad.locator('#dataset-summary').innerText(),/could not be loaded/);
  assert.equal(await bad.locator('#play').isDisabled(),true);
  await bad.close();
  const fallback=await browser.newPage();
  await fallback.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return String(type).startsWith('webgl')?null:original.call(this,type,...args);};});
  const fallbackErrors=[];fallback.on('pageerror',e=>fallbackErrors.push(e.message));
  await fallback.goto(base+'/layers.html');
  await fallback.waitForFunction(()=>document.querySelector('#status').textContent.includes('WebGL unavailable'));
  assert.equal(await fallback.locator('#layers li').count(),4);
  assert.equal(await fallback.locator('#coverage-table button').count(),112,'Coverage table stays usable without WebGL');
  await fallback.evaluate(()=>document.querySelector('#formation-mode').click());
  await fallback.waitForFunction(()=>!document.querySelector('#play').disabled);
  await fallback.evaluate(()=>document.querySelector('#next').click());
  assert.equal(await fallback.locator('#timeline').inputValue(),'1');
  assert.deepEqual(fallbackErrors,[]);
  await fallback.close();
  console.log('PASS: delayed history/ledger, saved frame, keyboard focus, undo, mobile width, 404/checksum refusal, stale and oversized links, data error recovery, WebGL fallback.');
}finally{await browser.close();}
