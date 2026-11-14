// A static build needs no application server, bundler, credentials, or remote CDN.
import {mkdir, copyFile, cp, readFile, writeFile, rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve,sep} from 'node:path';
import {verifiedPlotProjection} from './plot-catalog-build.mjs';
// Only this script's generated dist may be replaced, never a caller's directory.
const root=fileURLToPath(new URL('.',import.meta.url));
if(resolve(process.cwd())!==resolve(root))throw new Error('Run the build from viewer/.');
const destination=resolve(root,'dist');
if(destination!==resolve(process.cwd(),'dist'))throw new Error('Unsafe build destination.');
await rm(destination,{recursive:true,force:true});
await mkdir('dist/vendor/addons/controls', {recursive: true});
const files=['index.html', 'style.css', 'main.js', 'graph-data.js', 'research-plots.js', 'layers.html', 'layers.css', 'layers.js', 'formation-charts.js', 'navigation.js', 'layers-ui.js', 'plot-gallery.js'];
const sources=[];
for(const file of files)sources.push([file,await readFile(file,'utf8')]);
// An updated HTML document must not load an older cached controller or stylesheet.
const version=createHash('sha256').update(JSON.stringify(sources)).digest('hex').slice(0,12);
for(const [file,source] of sources)await writeFile(`dist/${file}`,source.replace(/(\.\/[\w-]+\.(?:js|css))(['"])/g,`$1?v=${version}$2`));
// The public entry opens current evidence; retain the original archive separately.
await copyFile('dist/index.html','dist/archive.html');
await copyFile('dist/layers.html','dist/index.html');
// Retire the mistaken landing page while keeping its shared URLs usable.
await copyFile('dist/layers.html','dist/study.html');
await mkdir('dist/data/study-figures',{recursive:true});
await mkdir('dist/data/saved-plots',{recursive:true});
const sourceCatalog=await readFile(resolve(root,'../outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv'));
if(createHash('sha256').update(sourceCatalog).digest('hex')!=='b5c9ce9ed55d7432f7fcb3847bdb7316bbe2ab7d4ea7041379c88a7cb02fd775')throw Error('Source plot catalog changed');
const plots=verifiedPlotProjection(await readFile('public/data/saved-plots.json'));
for(const p of plots){
  const source=resolve(root,'..',p.source_path),repo=resolve(root,'..');
  if(!source.startsWith(repo+sep))throw Error('Plot escaped repository');
  const bytes=await readFile(source);
  if(createHash('sha256').update(bytes).digest('hex')!==p.png_sha256)throw Error('Saved PNG hash mismatch: '+p.run_id);
  await writeFile(resolve(destination,p.png_url),bytes);
}
await copyFile('public/data/saved-plots.json','dist/data/saved-plots.json');
await copyFile('public/data/primary-viewer.json','dist/data/primary-viewer.json');
await mkdir('dist/data/primary-events',{recursive:true});
const primary=JSON.parse(await readFile('public/data/primary-viewer.json','utf8'));
for(const run of primary.runs){
  if(!/^[a-zA-Z0-9_.-]+$/.test(run.run_id)||run.events_url!==`./data/primary-events/${run.run_id}.json`)throw Error('Invalid public event path');
  await copyFile('public/data/primary-events/'+run.run_id+'.json','dist/data/primary-events/'+run.run_id+'.json');
}
for(const name of ['15_efficiency_density','14_demographic_mixing'])await copyFile(`public/data/study-figures/${name}.png`,`dist/data/study-figures/${name}.png`);
for (const file of ['three.module.js', 'three.core.js']) await copyFile(`node_modules/three/build/${file}`, `dist/vendor/${file}`);
await copyFile('node_modules/three/examples/jsm/controls/OrbitControls.js', 'dist/vendor/addons/controls/OrbitControls.js');
await mkdir('dist/vendor/addons/lines', {recursive: true});
for (const file of ['LineSegments2.js','LineSegmentsGeometry.js','LineMaterial.js']) await copyFile(`node_modules/three/examples/jsm/lines/${file}`, `dist/vendor/addons/lines/${file}`);
await copyFile('node_modules/three/LICENSE', 'dist/vendor/THREE-LICENSE.txt');
// Only exported public manifests/artifacts belong on the static host.
for(const name of ['networks.json','calibration.json','revised-calibration.json','presentation.json','artifacts','calibration','revised-calibration','presentation']){
  await cp(`public/data/${name}`,`dist/data/${name}`,{recursive:true});
}
console.log('Static viewer built in viewer/dist. Serve this directory on localhost.');
