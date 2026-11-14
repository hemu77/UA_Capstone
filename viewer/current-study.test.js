import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

test('public workspace offers current study only and links the preserved audit',()=>{
  const html=readFileSync(new URL('./layers.html',import.meta.url),'utf8');
  assert.doesNotMatch(html,/id="(?:dataset|collection|failure-panel|failure-list)"/);
  assert.doesNotMatch(html,/revised_calibration|Historical V5|engineering pilots/);
  assert.match(html,/NETWORKS\.md/);
  assert.match(html,/id="saved-plots"/);
});
