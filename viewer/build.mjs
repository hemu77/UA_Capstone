// A static build needs no application server, bundler, credentials, or remote CDN.
import {mkdir, copyFile, cp, readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
await mkdir('dist/vendor/addons/controls', {recursive: true});
const files=['index.html', 'style.css', 'main.js', 'graph-data.js', 'research-plots.js', 'layers.html', 'layers.css', 'layers.js', 'formation-charts.js'];
const sources=[];
for(const file of files)sources.push([file,await readFile(file,'utf8')]);
// An updated HTML document must not load an older cached controller or stylesheet.
const version=createHash('sha256').update(JSON.stringify(sources)).digest('hex').slice(0,12);
for(const [file,source] of sources)await writeFile(`dist/${file}`,source.replace(/(\.\/[\w-]+\.(?:js|css))(['"])/g,`$1?v=${version}$2`));
for (const file of ['three.module.js', 'three.core.js']) await copyFile(`node_modules/three/build/${file}`, `dist/vendor/${file}`);
await copyFile('node_modules/three/examples/jsm/controls/OrbitControls.js', 'dist/vendor/addons/controls/OrbitControls.js');
await mkdir('dist/vendor/addons/lines', {recursive: true});
for (const file of ['LineSegments2.js','LineSegmentsGeometry.js','LineMaterial.js']) await copyFile(`node_modules/three/examples/jsm/lines/${file}`, `dist/vendor/addons/lines/${file}`);
await copyFile('node_modules/three/LICENSE', 'dist/vendor/THREE-LICENSE.txt');
await cp('public/data', 'dist/data', {recursive: true});
console.log('Static viewer built in viewer/dist. Serve this directory on localhost.');
