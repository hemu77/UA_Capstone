// A static build needs no application server, bundler, credentials, or remote CDN.
import {mkdir, copyFile, cp} from 'node:fs/promises';
await mkdir('dist/vendor/addons/controls', {recursive: true});
for (const file of ['index.html', 'style.css', 'main.js', 'graph-data.js', 'research-plots.js', 'layers.html', 'layers.css', 'layers.js', 'formation-charts.js']) await copyFile(file, `dist/${file}`);
for (const file of ['three.module.js', 'three.core.js']) await copyFile(`node_modules/three/build/${file}`, `dist/vendor/${file}`);
await copyFile('node_modules/three/examples/jsm/controls/OrbitControls.js', 'dist/vendor/addons/controls/OrbitControls.js');
await mkdir('dist/vendor/addons/lines', {recursive: true});
for (const file of ['LineSegments2.js','LineSegmentsGeometry.js','LineMaterial.js']) await copyFile(`node_modules/three/examples/jsm/lines/${file}`, `dist/vendor/addons/lines/${file}`);
await copyFile('node_modules/three/LICENSE', 'dist/vendor/THREE-LICENSE.txt');
await cp('public/data', 'dist/data', {recursive: true});
console.log('Static viewer built in viewer/dist. Serve this directory on localhost.');
