import * as THREE from 'three';

// Camera-only operations: never move personas or modify recorded graph edges.
export function configureNavigation(controls, enabled, dragMode) {
  controls.enabled=enabled;
  controls.enableZoom=enabled;controls.enablePan=enabled;controls.enableRotate=enabled;
  controls.minDistance=4;controls.maxDistance=4000;
  controls.minPolarAngle=0;controls.maxPolarAngle=Math.PI;
  controls.minAzimuthAngle=-Infinity;controls.maxAzimuthAngle=Infinity;
  controls.zoomToCursor=true;controls.screenSpacePanning=true;
  controls.zoomSpeed=.8;controls.panSpeed=1;controls.keyPanSpeed=18;
  controls.mouseButtons.LEFT=dragMode==='pan'?THREE.MOUSE.PAN:THREE.MOUSE.ROTATE;
  controls.touches.ONE=dragMode==='pan'?THREE.TOUCH.PAN:THREE.TOUCH.ROTATE;
  controls.touches.TWO=THREE.TOUCH.DOLLY_PAN;
}

export function panCamera(camera,controls,dx,dy,height) {
  const scale=2*camera.position.distanceTo(controls.target)*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))/Math.max(1,height);
  camera.updateMatrixWorld();
  const offset=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0).multiplyScalar(dx*scale)
    .add(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1).multiplyScalar(-dy*scale));
  camera.position.add(offset);controls.target.add(offset);controls.update();
}
