import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {configureNavigation,panCamera} from './navigation.js';

test('navigation allows close zoom, pointer zoom and reversible gesture modes',()=>{
  const controls={mouseButtons:{},touches:{}};
  configureNavigation(controls,true,'orbit');
  assert.equal(controls.minDistance,4);assert.equal(controls.zoomToCursor,true);
  assert.equal(controls.mouseButtons.LEFT,THREE.MOUSE.ROTATE);
  assert.equal(controls.minPolarAngle,0);assert.equal(controls.maxPolarAngle,Math.PI);
  configureNavigation(controls,true,'pan');
  assert.equal(controls.mouseButtons.LEFT,THREE.MOUSE.PAN);
  assert.equal(controls.touches.TWO,THREE.TOUCH.DOLLY_PAN);
  configureNavigation(controls,false,'orbit');
  assert.equal(controls.enabled,false);assert.equal(controls.enableZoom,false);
});

test('diagonal panning translates camera and target together without changing zoom',()=>{
  const camera=new THREE.PerspectiveCamera(42,1,.1,3000);
  camera.position.set(0,0,500);camera.lookAt(0,0,0);
  const controls={target:new THREE.Vector3(),update(){}};
  panCamera(camera,controls,45,30,500);
  assert.ok(camera.position.x>0);assert.ok(camera.position.y<0);
  assert.ok(Math.abs(camera.position.distanceTo(controls.target)-500)<1e-9);
  assert.deepEqual(camera.position.clone().sub(controls.target).toArray(),[0,0,500]);
});
