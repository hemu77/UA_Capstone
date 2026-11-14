import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {decodeVerifiedEvents} from './verified-events.js';
const bytes=Buffer.from('[{"step":1,"added":[["1","2"]],"removed":[]}]');
const hash=createHash('sha256').update(bytes).digest('hex');
test('Raw and gzip transport reproduce identical verified recorded events',async()=>{
  assert.deepEqual(await decodeVerifiedEvents(bytes,hash),JSON.parse(bytes));
  assert.deepEqual(await decodeVerifiedEvents(gzipSync(bytes),hash),JSON.parse(bytes));
});
test('Compressed replay rejects tampered data, truncated gzip, and expansion beyond limit',async()=>{
  await assert.rejects(decodeVerifiedEvents(gzipSync(Buffer.from('[]')),hash),/integrity/);
  await assert.rejects(decodeVerifiedEvents(gzipSync(bytes).subarray(0,12),hash));
  await assert.rejects(decodeVerifiedEvents(gzipSync(Buffer.alloc(2_000_001)),hash),/size limit/);
  await assert.rejects(decodeVerifiedEvents(Buffer.alloc(2_000_001),hash),/size limit/);
});
