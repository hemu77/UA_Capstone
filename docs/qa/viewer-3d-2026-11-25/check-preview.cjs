// Opens preview/layers-preview.html from file:// with every network request blocked and checks that the Atlas,
// filters, a Stack, Replay, saved metrics and the supplement panel work with zero console errors.
// Run:  NODE_PATH=$(npm root -g) node docs/qa/viewer-3d-2026-11-25/check-preview.cjs
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('playwright');
const file=pathToFileURL(path.join(__dirname,'preview','layers-preview.html')).href;
(async()=>{
  const browser=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  try{
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1,offline:true});
    const requests=[],blocked=[],errors=[];
    // Only the preview file itself may be read; blob: and data: URLs never leave the document.
    await context.route('**/*',route=>{const url=route.request().url();if(url.split(/[?#]/)[0]===file)return route.continue();blocked.push(url);return route.abort();});
    const page=await context.newPage();
    page.on('request',request=>{const url=request.url();if(!/^(blob|data):/.test(url))requests.push(url);});
    page.on('pageerror',error=>errors.push('pageerror: '+error.message));
    page.on('console',message=>{if(['error','warning'].includes(message.type())&&!/GPU stall|GL Driver Message|Automatic fallback to software WebGL/.test(message.text()))errors.push(message.type()+': '+message.text());});
    const canvas=()=>page.evaluate(()=>({...document.querySelector('#viewport canvas').dataset}));
    const ready=async()=>{await page.waitForSelector('body[data-ready=true]',{timeout:60000});assert.notEqual(await page.evaluate(()=>document.body.dataset.state),'error',await page.locator('#status').textContent());await page.waitForFunction(()=>Number(document.querySelector('#viewport canvas')?.dataset.frames)>0);};
    const summary=()=>page.locator('#active-filters .flt-summary').innerText();

    // Atlas: all 896 slots, within the draw-call budget.
    await page.goto(file);await ready();
    let data=await canvas();
    const atlasCalls=data.drawCalls;
    assert.equal(data.view,'atlas');assert.equal(data.slots,'896');assert.ok(Number(data.drawCalls)<=12,'atlas draw calls '+data.drawCalls);
    assert.deepEqual(JSON.parse(data.atlas),{slabs:890,failed:6,nodes:44500,ties:156730});
    assert.match((await page.locator('#study-ledger').textContent()).replace(/\s+/g,' '),/896 attempts.*890 valid graphs \(12 empty\).*6 failures.*6 supplemental graphs excluded/);
    assert.equal(await summary(),'All 890 valid graphs · no filters set');

    // Filters: two facet values narrow the scope live; the chips and the result list agree.
    await page.locator('#filter-root button[aria-pressed]',{hasText:'GPT-4.1'}).first().click();
    await page.locator('#filter-root button[aria-pressed]',{hasText:'Local'}).first().click();
    assert.equal(await summary(),'Showing 56 of 890 valid graphs');
    assert.equal(await page.locator('#active-filters .flt-chip').count(),2);
    await page.locator('#tab-results').click();
    assert.equal(await page.locator('#run-results').getAttribute('data-count'),'56');
    await page.locator('#active-filters button',{hasText:'Reset filters'}).click();
    assert.equal(await summary(),'All 890 valid graphs · no filters set');

    // Stack of one condition's eight seeds, by link.
    await page.goto(file+'?v=stack&m=gpt-4.1&p=local&s=us-english&l=56.57.58.59.60.61.62.63&a=56');await ready();
    data=await canvas();
    assert.equal(data.view,'stack');assert.equal(data.layers,'8');assert.equal(data.planes,'8');assert.ok(Number(data.drawCalls)<=48);
    assert.equal(await summary(),'Showing 8 of 890 valid graphs');
    assert.equal(await page.locator('#plane-labels button.g3-plane').count(),8);

    // Replay: the recorded history is inlined and still passes its checksum.
    await page.locator('#formation-mode').click();
    await page.waitForFunction(()=>/^Event \d+ \/ \d+$/.test(document.getElementById('step-label').textContent),null,{timeout:30000});
    assert.equal((await canvas()).view,'replay');
    await page.locator('#next').click();
    assert.match(await page.locator('#step-label').textContent(),/^Event 1 \/ \d+$/);

    // Saved metrics (lazy file) and the supplement panel (lazy file, separate canvas).
    await page.locator('#metric').selectOption('modularity');
    await page.waitForFunction(()=>document.querySelector('#plot svg'),null,{timeout:15000});
    await page.locator('.ins-supp-drawer summary').click();
    await page.waitForSelector('.ins-supp-item');
    assert.equal(await page.locator('.ins-supp-item').count(),6);

    assert.deepEqual(errors,[],'console errors or warnings');
    assert.deepEqual(blocked,[],'the preview attempted a request');
    assert.deepEqual([...new Set(requests.map(url=>url.split(/[?#]/)[0]))],[file],'only the preview file is ever requested');
    const size=require('node:fs').statSync(path.join(__dirname,'preview','layers-preview.html')).size;
    console.log(`PASS: preview opens from file:// with the network blocked (${(size/1048576).toFixed(2)} MB): Atlas 896 slots in ${atlasCalls} draw calls, filters, 8-plane Stack, verified Replay, saved metrics, 6 supplements; 0 console errors, 0 external requests.`);
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
