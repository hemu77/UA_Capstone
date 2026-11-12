// Interaction regressions for Claude's presentation on the final primary study.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/layers.html');await page.waitForSelector('#layers .select-run');
  await page.waitForFunction(()=>!document.querySelector('#play').disabled);
  await page.keyboard.press('?');assert.equal(await page.locator('#help').evaluate(n=>n.open),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#help').evaluate(n=>n.open),false);
  await page.keyboard.press('3');
  await page.waitForFunction(()=>document.querySelectorAll('#events button').length===50);
  await page.keyboard.press('Home');assert.equal(await page.locator('#timeline').inputValue(),'0');
  await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.locator('#viewport canvas').focus();
  await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.locator('#person').selectOption('12');
  await page.locator('#person-next').click();assert.equal(await page.locator('#person').inputValue(),'13');
  await page.locator('#person-prev').click();assert.equal(await page.locator('#person').inputValue(),'12');
  await page.locator('#clear-person').click();
  await page.getByRole('button',{name:'Compare runs in 3D',exact:true}).click();
  const viewport=await page.locator('#viewport').boundingBox(),heading=await page.locator('.stage-heading').boundingBox();
  assert.ok(viewport.y>=heading.y+70,'Projection must reserve heading space');
  await page.locator('.filter-chip').first().click();
  await page.locator('#model').selectOption('gpt-6-sol');
  assert.match(await page.locator('#filter-chips').innerText(),/Filter query/);
  assert.match(await page.locator('#layers').innerText(),/gpt-4.1/,'Unapplied query must not change layers');
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.getByRole('link',{name:'Archive & analysis'}).isVisible(),true);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No mobile overflow');
  assert.doesNotMatch(await page.locator('.replay-banner').innerText(),/Historical calibration replay/);
  assert.deepEqual(errors,[]);
  console.log('PASS: shortcuts/modal, persona stepping, camera-arrow ownership, pending query labels, overlay-safe projection, mobile navigation.');
}finally{await browser.close();}
