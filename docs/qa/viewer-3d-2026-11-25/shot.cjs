const {chromium}=require('playwright');
const out=__dirname+'/'+(process.env.SHOT_DIR||'shots')+'/';
(async()=>{
  const browser=await chromium.launch({args:['--use-gl=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  for(const [name,w,h] of [['desktop',1440,900],['phone',390,844]]){
    const ctx=await browser.newContext({viewport:{width:w,height:h},deviceScaleFactor:1});
    const page=await ctx.newPage();const logs=[],reqs=[];
    page.on('console',m=>{if(['error','warning'].includes(m.type()))logs.push(m.type()+': '+m.text());});
    page.on('pageerror',e=>logs.push('pageerror: '+e.message));
    page.on('response',async r=>{try{const b=await r.body();reqs.push([r.url().replace('http://127.0.0.1:8781',''),r.status(),b.length]);}catch{}});
    const t=Date.now();
    await page.goto('http://127.0.0.1:8781/layers.html');await page.waitForSelector('#layers button');await page.waitForTimeout(1500);
    console.log(name,'ready ms',Date.now()-t,'renderer',await page.evaluate(()=>document.body.dataset.renderer||'webgl'));
    await page.screenshot({path:out+name+'-top.png'});
    await page.screenshot({path:out+name+'-full.png',fullPage:true});
    const m=await page.evaluate(()=>({sw:document.documentElement.scrollWidth,cw:document.documentElement.clientWidth,sh:document.documentElement.scrollHeight,
      overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>document.documentElement.clientWidth+1&&getComputedStyle(e).position!=='fixed'}).slice(0,15).map(e=>e.tagName+'#'+e.id+'.'+e.className+' '+Math.round(e.getBoundingClientRect().right)),
      small:[...document.querySelectorAll('button,select,input,a,summary')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&(r.height<24||r.width<24)}).map(e=>(e.id||e.textContent.slice(0,20))+' '+Math.round(e.getBoundingClientRect().width)+'x'+Math.round(e.getBoundingClientRect().height)).slice(0,30),
      vp:document.querySelector('#viewport').getBoundingClientRect().toJSON(),ids:[...document.querySelectorAll('[id]')].map(e=>e.id).filter((v,i,a)=>a.indexOf(v)!==i)}));
    console.log(JSON.stringify(m));
    // compare-mode shot of stage
    await page.locator('#viewport').scrollIntoViewIfNeeded();await page.waitForTimeout(300);
    await page.screenshot({path:out+name+'-stage-compare.png'});
    await page.locator('#formation-mode').click();await page.waitForTimeout(1500);
    await page.locator('#viewport').scrollIntoViewIfNeeded();
    await page.screenshot({path:out+name+'-stage-replay.png'});
    if(name==='desktop'){
      await page.locator('.run-drawer summary').click();await page.waitForTimeout(200);
      await page.locator('#method').selectOption('local');await page.waitForTimeout(200);
      await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:out+'desktop-filters.png'});
      console.log('candidate options',await page.locator('#candidate option').count(),'preview',await page.locator('#filter-preview').innerText(),'apply disabled',await page.locator('#apply').isDisabled());
      await page.goto('http://127.0.0.1:8781/layers.html#saved-plots');await page.waitForFunction(()=>document.querySelectorAll('#plot-grid article').length>0,null,{timeout:30000});
      await page.locator('#saved-plots').scrollIntoViewIfNeeded();await page.screenshot({path:out+'desktop-gallery.png'});
      await page.locator('#plot-grid button').first().click();await page.waitForTimeout(1500);await page.screenshot({path:out+'desktop-gallery-dialog.png'});
    }
    console.log(logs.slice(0,20));console.log(reqs.filter(r=>r[2]>50000||r[1]>=400).map(r=>r.join(' ')).join('\n'));
    console.log('url len',page.url().length);
    await ctx.close();
  }
  await browser.close();
})();
