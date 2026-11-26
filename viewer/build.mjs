// A static build needs no application server, bundler, credentials, or remote CDN.
import {mkdir, copyFile, readFile, writeFile, rm, readdir, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import {resolve,sep} from 'node:path';
import {verifiedPlotProjection} from './plot-catalog-build.mjs';
import {SAVED_NETWORKS_SHA256,validateNetworkCatalog} from './network-catalog.js';
import {generateGraphFiles,OUTPUT_FILES as GRAPH_FILES} from './build-primary-graphs.mjs';
import {loadPrimaryDataset} from './layers-data-all.js';
// Only this script's generated dist may be replaced, never a caller's directory.
const root=fileURLToPath(new URL('.',import.meta.url));
if(resolve(process.cwd())!==resolve(root))throw new Error('Run the build from viewer/.');
const destination=resolve(root,'dist');
if(destination!==resolve(process.cwd(),'dist'))throw new Error('Unsafe build destination.');
await rm(destination,{recursive:true,force:true});
await mkdir('dist/vendor/addons/controls', {recursive: true});
const files=['graph-data.js', 'research-plots.js', 'layers.html', 'layers.css', 'layers.js', 'formation-charts.js', 'navigation.js', 'layers-ui.js', 'plot-gallery.js', 'verified-events.js', 'network-catalog.js', 'network-stage.js',
  // 3D upgrade modules (docs/qa/viewer-3d-2026-11-25/SPEC.md). One owner per file.
  'layers-app.js', 'layers-data-all.js', 'layers-3d-scene.js', 'layers-3d-atlas.js', 'layers-3d-stack.js', 'layers-3d-labels.js', 'layers-filters.js', 'layers-insight.js', 'layers-shell.js',
  'layers-3d.css', 'layers-filters.css', 'layers-insight.css'];
const sources=[];
for(const file of files)sources.push([file,await readFile(file,'utf8')]);
// An updated HTML document must not load an older cached controller or stylesheet.
const version=createHash('sha256').update(JSON.stringify(sources)).digest('hex').slice(0,12);
for(const [file,source] of sources)await writeFile(`dist/${file}`,source.replace(/(\.\/[\w-]+\.(?:js|css))(['"])/g,`$1?v=${version}$2`));
// Old entry URLs open the current study; historical sources stay repo-only.
await copyFile('dist/layers.html','dist/archive.html');
await copyFile('dist/layers.html','dist/index.html');
// Retire the mistaken landing page while keeping its shared URLs usable.
await copyFile('dist/layers.html','dist/study.html');
await mkdir('dist/data/study-figures',{recursive:true});
// Pinned fixtures stay in public/data as cross-checks only; none of them is deployed.
const sourceCatalog=await readFile(resolve(root,'../outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv'));
if(createHash('sha256').update(sourceCatalog).digest('hex')!=='b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775')throw Error('Source plot catalog changed');
const plots=verifiedPlotProjection(await readFile('public/data/saved-plots.json'));
for(const p of plots){
  const source=resolve(root,'..',p.source_path),repo=resolve(root,'..');
  if(!source.startsWith(repo+sep))throw Error('Plot escaped repository');
  const bytes=await readFile(source);
  if(createHash('sha256').update(bytes).digest('hex')!==p.png_sha256)throw Error('Saved PNG hash mismatch: '+p.run_id);
}
const networkBytes=await readFile('public/data/saved-networks.json');
if(createHash('sha256').update(networkBytes).digest('hex')!==SAVED_NETWORKS_SHA256)throw Error('Final-network projection changed');
validateNetworkCatalog(JSON.parse(networkBytes),plots,JSON.parse(await readFile('public/data/primary-viewer.json','utf8')));

// Compact primary data: the committed files must equal a fresh generation from the frozen sources.
const generated=await generateGraphFiles();
await mkdir('dist/data/graphs',{recursive:true});
for(const name of GRAPH_FILES){
  const committed=await readFile(`public/data/graphs/${name}`);
  if(!committed.equals(generated.get(name)))throw Error(`public/data/graphs/${name} differs from its sources; run node build-primary-graphs.mjs`);
  await writeFile(`dist/data/graphs/${name}`,committed);
}
const readPublic=async url=>{const bytes=await readFile(resolve(root,'public',url.replace(/^\.\//,'')));return {ok:true,status:200,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};};
const primary=await loadPrimaryDataset({fetchImpl:readPublic});
// Recorded event logs travel gzip-compressed; the hash covers the uncompressed recorded bytes.
await mkdir('dist/data/primary-events',{recursive:true});
for(const run of primary.runs){
  if(!/^[a-zA-Z0-9_.-]+$/.test(run.run_id)||run.events_url!==`./data/primary-events/${run.run_id}.json.gz`)throw Error('Invalid public event path');
  const bytes=await readFile('public/data/primary-events/'+run.run_id+'.json');
  if(createHash('sha256').update(bytes).digest('hex')!==run.events_sha256)throw Error('Recorded event hash mismatch');
  await writeFile('dist/data/primary-events/'+run.run_id+'.json.gz',gzipSync(bytes));
}
for(const name of ['15_efficiency_density','14_demographic_mixing'])await copyFile(`public/data/study-figures/${name}.png`,`dist/data/study-figures/${name}.png`);
for (const file of ['three.module.js', 'three.core.js']) await copyFile(`node_modules/three/build/${file}`, `dist/vendor/${file}`);
await copyFile('node_modules/three/examples/jsm/controls/OrbitControls.js', 'dist/vendor/addons/controls/OrbitControls.js');
await mkdir('dist/vendor/addons/lines', {recursive: true});
for (const file of ['LineSegments2.js','LineSegmentsGeometry.js','LineMaterial.js']) await copyFile(`node_modules/three/examples/jsm/lines/${file}`, `dist/vendor/addons/lines/${file}`);
await copyFile('node_modules/three/LICENSE', 'dist/vendor/THREE-LICENSE.txt');
// Leave conservative per-file archive overhead below the host's expanded limit.
let expanded=0;
for(const file of await readdir('dist',{recursive:true})){const info=await stat(resolve(destination,file));if(info.isFile())expanded+=info.size+2048;}
if(expanded>256*1024*1024)throw Error('Static package exceeds the 256 MiB host limit');
console.log('Static viewer built in viewer/dist. Serve this directory on localhost.');
