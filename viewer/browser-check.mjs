// Browser plugin absent: exercise the static build with installed Playwright.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
const errors=[],requests=[];
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.goto(base+'/');await page.waitForSelector('#layers button');
  assert.match(await page.locator('#dataset-summary').innerText(),/890 valid/);
  assert.equal(await page.locator('#dataset, #collection, #failure-list').count(),0);
  assert.equal(await page.locator('#layers li').count(),4);
  await page.locator('#coverage-panel summary').click();
  assert.equal(await page.locator('#coverage-table button').count(),112);
  await page.locator('#coverage-table button').first().click();
  assert.match(await page.locator('#action-status').innerText(),/Choose a seed/);
  await page.locator('#seed').selectOption('44000');await page.locator('#apply').click();
  assert.equal(await page.locator('#layers li').count(),1);
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await page.locator('#formation-mode').click();await page.locator('#next').click();
  assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.locator('#person').selectOption('12');
  assert.equal(await page.locator('#incident').isChecked(),true);
  await page.locator('#clear-person').click();assert.equal(await page.locator('#person').inputValue(),'');
  const empty=await page.evaluate(async()=>{const d=await(await fetch('./data/primary-viewer.json')).json();return d.runs.find(r=>!r.edges.length);});
  for(const key of ['model','method','culture','language','seed'])await page.locator('#'+key).selectOption(String(empty[key]));
  await page.locator('#apply').click();await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await page.locator('#formation-mode').click();await page.locator('#show-final').click();
  assert.match(await page.locator('#replay-note').innerText(),/valid run has no ties/);
  await page.locator('#research-question').selectOption('rq2');assert.match(await page.locator('#rq-homophily').innerText(),/NA/);
  const download=page.waitForEvent('download');await page.locator('#download-selection').click();
  assert.match((await download).suggestedFilename(),/primary/);
  for(const query of ['?dataset=revised_calibration','?dataset=calibration','?dataset=legacy','?view='+encodeURIComponent(JSON.stringify({runs:['calibration_v6_old']}))]){
    await page.goto(base+'/layers.html'+query);await page.waitForSelector('body[data-ready=true]');
    assert.match(await page.locator('#status').innerText(),/archived research/);
    assert.equal(await page.locator('#layers li').count(),0);
    await page.locator('#reset-view').click();await page.waitForSelector('#layers button');
  }
  assert.ok(!requests.some(u=>/data\/(?:calibration|revised-calibration|networks\.json|presentation)/.test(u)),'Legacy payload requested');
  for(const route of ['study.html','archive.html']){await page.goto(base+'/'+route);await page.waitForSelector('#layers button');assert.match(await page.locator('#dataset-summary').innerText(),/890 valid/);}
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);await page.locator('#formation-mode').click();await page.locator('#next').click();
  assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.screenshot({path:process.env.VIEWER_SCREENSHOT||'current-study-mobile.png'});
  assert.deepEqual(errors,[]);
  console.log('PASS: current-only routing, preserved failure counts, coverage/filter, empty graph, export, legacy rejection/reset, aliases, mobile.');
}finally{await browser.close();}
