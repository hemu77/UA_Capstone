import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

test('current project surfaces retain the established research name',()=>{
  const name='Cross-Cultural and Linguistic LLM Network Study';
  const readme=readFileSync(new URL('../README.md',import.meta.url),'utf8');
  const html=readFileSync(new URL('./layers.html',import.meta.url),'utf8');
  const deployment=readFileSync(new URL('../docs/SIMULATOR_DEPLOYMENT.md',import.meta.url),'utf8');
  assert.equal(readme.split(/\r?\n/)[0],`# ${name}`);
  assert.ok(html.includes(`<title>${name} / Recorded network lab</title>`));
  assert.ok(html.includes(`aria-label="${name}"`));
  assert.ok(html.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').includes(name));
  for(const text of [readme,html,deployment])assert.doesNotMatch(text,/Network Observatory/i);
});

test('public workspace offers current study only and links the preserved audit',()=>{
  const html=readFileSync(new URL('./layers.html',import.meta.url),'utf8');
  assert.doesNotMatch(html,/id="(?:dataset|collection|failure-panel|failure-list)"/);
  assert.doesNotMatch(html,/revised_calibration|Historical V5|engineering pilots/);
  assert.match(html,/NETWORKS\.md/);
  assert.match(html,/id="saved-plots"/);
});
