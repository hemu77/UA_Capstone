// Builds preview/layers-preview.html: the layers page as ONE self-contained file that makes no network request.
// Input is the verified static build in viewer/dist (run `npm run build` in viewer/ first). Nothing is generated or
// altered: the same modules, styles and saved data are inlined byte for byte and still pass their SHA-256 checks
// in the browser. Usage (from the repository root):  node docs/qa/viewer-3d-2026-11-25/build-preview.mjs
import {readFile, writeFile, readdir, mkdir} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const here=path.dirname(fileURLToPath(import.meta.url));
const dist=path.resolve(here,'../../../viewer/dist');
const out=path.join(here,'preview','layers-preview.html');
const read=file=>readFile(path.join(dist,file));
const text=async file=>(await read(file)).toString('utf8');

// --- Modules: every script of the page plus the vendored three.js, keyed by its path inside dist/. ------------
const modulePaths=[...(await readdir(dist)).filter(name=>name.endsWith('.js')),'vendor/three.module.js','vendor/three.core.js',
  'vendor/addons/controls/OrbitControls.js','vendor/addons/lines/LineSegments2.js','vendor/addons/lines/LineSegmentsGeometry.js','vendor/addons/lines/LineMaterial.js'];
const known=new Set(modulePaths);
const BARE={three:'vendor/three.module.js'};
// Imports become bare specifiers ("pv/<path>") that an import map, created at start-up, points at blob URLs.
function rewrite(source,file){
  const resolve=specifier=>{
    const clean=specifier.replace(/\?.*$/,'');
    let target;
    if(BARE[clean])target=BARE[clean];
    else if(clean.startsWith('three/addons/'))target='vendor/addons/'+clean.slice('three/addons/'.length);
    else if(clean.startsWith('./')||clean.startsWith('../'))target=path.posix.normalize(path.posix.join(path.posix.dirname(file),clean));
    else return null;
    if(!known.has(target))throw new Error(`${file} imports ${specifier}, which is not part of the preview`);
    return 'pv/'+target;
  };
  return source.replace(/(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"\n]+)\2/g,(match,head,quote,specifier)=>{
    const target=resolve(specifier);return target?`${head}${quote}${target}${quote}`:match;
  });
}
const modules={};
for(const file of modulePaths)modules[file]=rewrite(await text(file),file);

// --- Saved data: index, ties, metrics and supplements travel in the compressed payload; the 890 recorded
// histories stay as the gzip files the site serves (already compressed), one base64 string per run. ------------
const files={};
for(const file of ['index.json','metrics.json','supplements.json'])files['data/graphs/'+file]={text:await text('data/graphs/'+file)};
files['data/graphs/edges.bin']={base64:(await read('data/graphs/edges.bin')).toString('base64')};
const events={};
for(const name of (await readdir(path.join(dist,'data/primary-events'))).sort()){
  if(!name.endsWith('.json.gz'))throw new Error('Unexpected replay file '+name);
  events[name.slice(0,-'.json.gz'.length)]=(await read('data/primary-events/'+name)).toString('base64');
}
const payload=gzipSync(Buffer.from(JSON.stringify({modules,files})),{level:9}).toString('base64');

// --- Page: the built HTML with its stylesheets inlined and its script/link tags replaced by the loader. --------
let html=await text('layers.html');
const styles=[];
for(const [tag,href] of [...html.matchAll(/<link rel="stylesheet" href="\.\/([\w-]+\.css)[^"]*">/g)].map(match=>[match[0],match[1]])){
  const css=await text(href);
  if(/@import|url\(/.test(css))throw new Error(href+' references an external resource');
  styles.push(`/* ${href} */\n${css}`);html=html.replace(tag,'');
}
if(styles.length!==4)throw new Error('Expected four stylesheets, found '+styles.length);
html=html.replace(/<link rel="modulepreload"[^>]*>/g,'').replace(/<script type="importmap">[\s\S]*?<\/script>/,'').replace(/<script type="module"[\s\S]*?<\/script>/g,'');
if(/<script|<link rel="(?:stylesheet|modulepreload)"/.test(html))throw new Error('A script or stylesheet tag survived');
for(const [tag,src] of [...html.matchAll(/<img src="\.\/(data\/study-figures\/[\w.-]+\.png)"/g)].map(match=>[match[0],match[1]]))
  html=html.replace(tag,`<img src="data:image/png;base64,${(await read(src)).toString('base64')}"`);
html=html.replace(/href="\.\/layers\.html"/g,'href="#"');
html=html.replace('<span class="brand-tag" aria-hidden="true">Recorded network lab</span>','<span class="brand-tag" title="Self-contained preview file: no network requests, not deployed">Offline preview</span>');
if(/(?:src|href)="\.\//.test(html))throw new Error('A relative resource reference survived: '+html.match(/(?:src|href)="\.\/[^"]*"/)[0]);
html=html.replace('</head>',`<style>\n${styles.join('\n')}\n</style></head>`);

const loader=`
(async()=>{
  const fail=message=>{document.body.dataset.state='error';document.body.dataset.ready='true';const box=document.getElementById('status')||document.body;box.textContent='Preview could not start: '+message;};
  try{
    if(typeof DecompressionStream==='undefined')throw new Error('this browser lacks DecompressionStream; use a current Chrome, Edge, Firefox or Safari.');
    const bytesOf=base64=>Uint8Array.from(atob(base64),c=>c.charCodeAt(0));
    const packed=bytesOf(document.getElementById('pv-payload').textContent);
    const payload=JSON.parse(await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
    // Saved files are answered from this document; anything else is refused, so the preview never touches a network.
    let events=null;
    const saved=name=>{
      const file=payload.files[name];
      if(file)return file.text!==undefined?new TextEncoder().encode(file.text):bytesOf(file.base64);
      const replay=/^data\\/primary-events\\/([A-Za-z0-9_.-]+)\\.json\\.gz$/.exec(name);
      if(replay){events=events||JSON.parse(document.getElementById('pv-events').textContent);if(events[replay[1]])return bytesOf(events[replay[1]]);}
      return null;
    };
    window.fetch=async input=>{
      const url=String(typeof input==='string'?input:input&&input.url||input).replace(/[?#].*$/,''),at=url.lastIndexOf('data/');
      const body=at>=0&&(at===0||url[at-1]==='/')?saved(url.slice(at)):null;
      return body?new Response(body,{status:200}):new Response('',{status:404,statusText:'Not available in the preview'});
    };
    const imports={};
    for(const [file,source] of Object.entries(payload.modules))imports['pv/'+file]=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
    const map=document.createElement('script');map.type='importmap';map.textContent=JSON.stringify({imports});document.head.append(map);
    const load=new Function('specifier','return import(specifier)');
    const [{app}, ,{mountShell}]=await Promise.all([load('pv/layers.js'),load('pv/layers-ui.js'),load('pv/layers-shell.js')]);
    mountShell(app);
  }catch(error){fail(error&&error.message||String(error));}
})();`;
html=html.replace('</body>',`<script type="application/octet-stream" id="pv-payload">${payload}</script>\n<script type="application/json" id="pv-events">${JSON.stringify(events)}</script>\n<script>${loader}\n</script></body>`);
html=html.replace('<!doctype html>','<!doctype html>\n<!-- Self-contained preview of viewer/layers.html, built by docs/qa/viewer-3d-2026-11-25/build-preview.mjs from viewer/dist.\n     No network requests: scripts (including three.js, MIT licence), styles and the saved study data are inlined. Not deployed. -->');
await mkdir(path.dirname(out),{recursive:true});
await writeFile(out,html);
console.log(`${path.relative(process.cwd(),out)}: ${(Buffer.byteLength(html)/1048576).toFixed(2)} MB (${Buffer.byteLength(html)} bytes); ${modulePaths.length} modules, ${Object.keys(files).length} data files, ${Object.keys(events).length} recorded histories`);
