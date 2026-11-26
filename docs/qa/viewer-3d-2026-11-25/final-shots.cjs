// Final screenshots for the PR (deviceScaleFactor 2; desktop 1440x900, phone 390x844) from a served static build.
// Run from viewer/:  NODE_PATH=$(npm root -g) VIEWER_URL=http://127.0.0.1:8781 node ../docs/qa/viewer-3d-2026-11-25/final-shots.cjs
// Then:  python3 ../docs/qa/viewer-3d-2026-11-25/compress-shots.py   (palette PNGs; keeps the folder small)
const {chromium}=require('playwright');
const fs=require('node:fs');
const out=__dirname+'/'+(process.env.SHOT_DIR||'final')+'/';fs.mkdirSync(out,{recursive:true});
const base=process.env.VIEWER_URL||'http://127.0.0.1:8781';
const condition='m=gpt-4.1&p=local&s=us-english';
const shots=[
  ['desktop-atlas','desktop','/layers.html'],
  ['desktop-atlas-filtered','desktop','/layers.html?m=gpt-5.6-luna&p=sequential,iterative'],
  ['desktop-stack-8','desktop',`/layers.html?v=stack&${condition}&l=56.57.58.59.60.61.62.63&a=56`],
  ['desktop-replay-persona','desktop',`/layers.html?v=replay&${condition}&l=56.57.58.59&a=56&n=12&f=20`,async page=>{await page.locator('#tab-persona').click();}],
  ['desktop-attribute-lens','desktop',`/layers.html?v=stack&${condition}&l=56.57.58.59&a=56&c=religion&x=same`,async page=>{await page.locator('#tab-compare').click();}],
  ['desktop-overlap-consensus','desktop',`/layers.html?v=stack&${condition}&l=56.57.58.59&a=56&k=f50`,async page=>{await page.locator('#tab-compare').click();await page.locator('#insight-root .ins-section',{hasText:'Layer overlap'}).evaluate(node=>{const panel=document.getElementById('panel-compare');panel.scrollTop+=node.getBoundingClientRect().top-panel.getBoundingClientRect().top-12;});}],
  ['phone-atlas','phone','/layers.html'],
  ['phone-stack','phone',`/layers.html?v=stack&${condition}&l=56.57.58.59&a=56`]
];
(async()=>{
  const browser=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  let problems=0;
  for(const [name,size,path,act] of shots){
    const phone=size==='phone';
    const context=await browser.newContext({viewport:phone?{width:390,height:844}:{width:1440,height:900},deviceScaleFactor:2,hasTouch:phone,isMobile:phone});
    const page=await context.newPage(),log=[];
    page.on('pageerror',error=>log.push('pageerror: '+error.message));
    page.on('console',message=>{if(['error','warning'].includes(message.type())&&!/GPU stall|GL Driver Message|Automatic fallback to software WebGL/.test(message.text()))log.push(message.type()+': '+message.text());});
    await page.goto(base+path);
    await page.waitForFunction(()=>document.body.dataset.ready==='true'&&Number(document.querySelector('#viewport canvas')?.dataset.frames)>0,null,{timeout:60000});
    await page.waitForTimeout(1800);
    if(act){await act(page);await page.waitForTimeout(1500);}
    await page.screenshot({path:out+name+'.png'});
    const data=await page.evaluate(()=>{const c=document.querySelector('#viewport canvas');return `${c.dataset.view} · ${c.dataset.drawCalls} draw calls · ${c.dataset.planes} planes · scroll width ${document.documentElement.scrollWidth}`;});
    console.log(name.padEnd(28),data,log.length?'| '+log.join(' | '):'| 0 console errors');problems+=log.length;
    await context.close();
  }
  await browser.close();process.exit(problems?1:0);
})();
