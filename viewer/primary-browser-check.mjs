// Error-path regressions for the actual Three.js app; no study API calls.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
try{
  let release;const gate=new Promise(r=>release=r);
  const page=await browser.newPage({viewport:{width:390,height:844}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/data/primary-events/**',async route=>{await gate;await route.continue();});
  await page.goto(base+'/layers.html');
  await page.waitForSelector('#layers .select-run');
  await page.locator('#formation-mode').click();
  assert.equal(await page.locator('#play').isDisabled(),true);
  assert.match(await page.locator('#replay-note').innerText(),/Loading and verifying/);
  release();
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  assert.ok(await page.locator('#events button').count()>0,'Lazy completion must rebuild ledger');
  assert.doesNotMatch(await page.locator('#events').innerText(),/No event log/);
  await page.locator('#layers .select-run').nth(2).click();
  await page.waitForFunction(()=>document.querySelectorAll('#events button').length===50);
  await page.locator('#events button').nth(3).click();
  assert.equal(await page.locator('#timeline').inputValue(),'4');
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#timeline').value==='4'&&!document.querySelector('#play').disabled);
  const download=page.waitForEvent('download');await page.locator('#download-selection').click();
  const downloaded=await download;
  const {readFile}=await import('node:fs/promises');
  const bundle=JSON.parse(await readFile(await downloaded.path(),'utf8'));
  const {replayFrames}=await import('./graph-data.js');
  assert.equal(bundle.runs.length,4);
  for(const r of bundle.runs)assert.ok(replayFrames(r,bundle.personas.map(p=>p.id)));
  await page.locator('#person').selectOption('12');
  const neighbor=page.locator('.neighbor').first();
  if(await neighbor.count()){await neighbor.focus();await neighbor.press('Enter');assert.equal(await page.locator('#person').evaluate(n=>n===document.activeElement),true);}
  await page.locator('#layers .remove-run').first().press('Enter');
  assert.equal(await page.locator('#undo').evaluate(n=>n===document.activeElement),true);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  assert.deepEqual(errors,[]);
  await page.close();

  for(const failure of ['missing','checksum']){
    const p=await browser.newPage();
    await p.route('**/data/primary-events/**',r=>r.fulfill({status:failure==='missing'?404:200,contentType:'application/json',body:'[]'}));
    await p.goto(base+'/layers.html');
    await p.waitForFunction(()=>document.querySelector('#action-status').textContent.includes('Reload to retry'));
    assert.equal(await p.locator('#play').isDisabled(),true);
    assert.match(await p.locator('#replay-note').innerText(),/could not be verified/);
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
  const bad=await browser.newPage();
  await bad.route('**/data/primary-viewer.json',r=>r.fulfill({status:503,body:'Unavailable'}));
  await bad.goto(base+'/layers.html');
  await bad.locator('#retry-load').waitFor({state:'visible'});
  assert.match(await bad.locator('#dataset-summary').innerText(),/could not be loaded/);
  assert.equal(await bad.locator('#play').isDisabled(),true);
  await bad.close();
  const fallback=await browser.newPage();
  await fallback.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return String(type).startsWith('webgl')?null:original.call(this,type,...args);};});
  await fallback.goto(base+'/layers.html');
  await fallback.waitForFunction(()=>document.querySelector('#status').textContent.includes('WebGL unavailable'));
  assert.equal(await fallback.locator('#layers li').count(),4);
  await fallback.locator('#formation-mode').click();
  await fallback.waitForFunction(()=>!document.querySelector('#play').disabled);
  await fallback.locator('#next').click();
  assert.equal(await fallback.locator('#timeline').inputValue(),'1');
  await fallback.close();
  console.log('PASS: delayed history/ledger, saved frame, keyboard focus, mobile width, 404/checksum refusal and manifest error recovery.');
}finally{await browser.close();}
