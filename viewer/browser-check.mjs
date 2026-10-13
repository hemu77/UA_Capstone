// Browser plugin absent: exercise the static build with installed Playwright.
// Supply PLAYWRIGHT_MODULE only when Playwright is outside this checkout.
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.VIEWER_URL||'http://127.0.0.1:8766';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
const errors=[];
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',error=>errors.push(error.message));
  page.on('response',response=>{if(response.status()>=400)errors.push(`${response.status()} ${response.url()}`);});
  await page.goto(`${base}/`);
  await page.waitForSelector('#layers button');
  await page.getByRole('link',{name:'Archive & analysis'}).click();
  await page.waitForURL(url=>/^\/archive(?:\.html)?$/.test(url.pathname));
  await page.getByRole('link',{name:'Open the revised V6 calibration workspace'}).click();
  await page.waitForURL(url=>/^\/layers(?:\.html)?$/.test(url.pathname));
  await page.waitForSelector('#layers button');
  assert.match(await page.locator('#dataset-summary').innerText(),/104 saved/);
  assert.equal(await page.locator('#layers li').count(),4);
  assert.match(await page.locator('#evidence-notice').innerText(),/22\/26/);
  await page.locator('#coverage-panel summary').click();
  assert.equal(await page.locator('#coverage-table button').count(),112);
  assert.equal(await page.getByRole('button',{name:/gpt-4.1, global, Brazil, english: 0 saved of 0 allocated/}).isDisabled(),true);
  await page.locator('.run-drawer summary').click();
  await page.locator('#language').selectOption('portuguese');
  await page.locator('#apply').click();
  assert.equal(await page.locator('#layers li').count(),4);
  assert.match(await page.locator('#layers').innerText(),/portuguese/);
  await page.locator('#person').selectOption('12');
  assert.equal(await page.locator('#incident').isChecked(),true);
  await page.locator('#play-persona').click();
  assert.equal(await page.locator('#play-persona').getAttribute('aria-pressed'),'true');
  await page.locator('#next').click();
  assert.ok(Number(await page.locator('#timeline').inputValue())>0);
  await page.locator('#clear-person').click();
  assert.equal(await page.locator('#person').inputValue(),'');
  await page.locator('#compare-mode').click();
  await page.locator('#layers button.remove-run').first().click();
  assert.equal(await page.locator('#layers li').count(),3);
  await page.locator('#undo').click();
  assert.equal(await page.locator('#layers li').count(),4);
  // Selecting an explicitly empty graph must not replace it with a populated one.
  const graph=await page.evaluate(async()=>{
    const data=await (await fetch('./data/revised-calibration.json')).json();
    return data.runs.find(run=>run.method==='global'&&!run.edges.length);
  });
  for(const key of ['model','method','culture','language','seed'])await page.locator(`#${key}`).selectOption(String(graph[key]));
  await page.locator('#apply').click();
  assert.equal(await page.locator('#layers li').count(),1);
  await page.locator('#formation-mode').click();
  await page.locator('#show-final').click();
  assert.match(await page.locator('#replay-note').innerText(),/recorded NONE/);
  assert.equal(await page.locator('#timeline').getAttribute('max'),'1');
  await page.locator('#research-question').selectOption('rq2');
  assert.match(await page.locator('#rq-homophily').innerText(),/NA/);
  for(const link of await page.locator('#source-gallery a').evaluateAll(nodes=>nodes.map(node=>node.href))){
    assert.equal((await page.request.get(link)).status(),200);
  }
  const download=page.waitForEvent('download');await page.locator('#download-selection').click();
  assert.match((await download).suggestedFilename(),/revised_calibration/);
  await page.locator('#dataset').selectOption('calibration');
  await page.waitForFunction(()=>document.querySelector('#dataset-summary').textContent.includes('68 saved'));
  await page.locator('#dataset').selectOption('legacy');
  await page.waitForFunction(()=>document.querySelector('#dataset-summary').textContent.includes('historical/pilot'));
  await page.goto(`${base}/layers.html`);await page.waitForSelector('#layers button');
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile page has horizontal overflow');
  await page.locator('#formation-mode').click();await page.locator('#next').click();
  assert.equal(await page.locator('#timeline').inputValue(),'1');
  await page.locator('#viewport').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('.transport').evaluate(node=>getComputedStyle(node).position),'static');
  await page.screenshot({path:process.env.VIEWER_SCREENSHOT||'../outputs/qa/revised-mobile.png'});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({status:'PASS',checks:['104 default','112 coverage cells','Portuguese models','persona journey/clear','remove/undo','empty global replay','undefined homophily','PNG/adjacency links','download','V5/legacy isolation','mobile layout/playback','no HTTP/page errors']}));
}finally{await browser.close();}
