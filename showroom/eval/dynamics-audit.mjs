import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import fs from 'node:fs';
import { DANCES } from '../src/choreography.js';
import { DanceEngine } from '../src/danceEngine.js';

const buf = fs.readFileSync('public/models/humanoid.glb');
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const JOINTS = ['head','handL','handR','chest','hips','footL','footR','kneeL','kneeR'];

async function load() {
  const gltf = await new Promise((res,rej)=>new GLTFLoader().parse(ab.slice(0), '', res, rej));
  return gltf;
}

for (const dance of DANCES) {
  const gltf = await load();               // instance bersih per dance
  const scene = new THREE.Scene();
  scene.add(gltf.scene);
  const eng = new DanceEngine(gltf.scene, gltf.animations);
  eng.playDance(dance); eng.blend = 1; eng.update(0);

  const dt = 1/60; const loopSec = (dance.beats*60)/dance.bpm; const frames = Math.round(loopSec*60);
  const speeds=[], hipY=[], beats=[], headY=[];
  let prev=null;
  for (let i=0;i<frames;i++){
    eng.update(dt);
    const j = eng.sampleJoints;
    if (prev){ let s=0; for(const g of JOINTS) s+=j[g].distanceTo(prev[g]); speeds.push(s/JOINTS.length/dt); }
    prev={}; for(const g of JOINTS) prev[g]=j[g].clone();
    hipY.push(j.hips.y); headY.push(j.head.y); beats.push(eng.beatPos);
  }
  const mean=speeds.reduce((a,b)=>a+b,0)/speeds.length;
  const sorted=[...speeds].sort((a,b)=>a-b);
  const p95=sorted[Math.floor(sorted.length*0.95)];
  const perBeat=[];
  for (let b=0;b<dance.beats;b++){
    const ys=hipY.filter((_,i)=>Math.floor(beats[i])===b);
    if(ys.length) perBeat.push(Math.max(...ys)-Math.min(...ys));
  }
  const bounce = perBeat.reduce((an,b)=>an+b,0)/perBeat.length;
  console.log(
    dance.id.padEnd(15),
    'mean', mean.toFixed(2).padStart(5),
    'p95', p95.toFixed(2).padStart(5),
    'p95/mean', (p95/mean).toFixed(2).padStart(5),
    '| hipAmp', (Math.max(...hipY)-Math.min(...hipY)).toFixed(3),
    'bounce/beat', bounce.toFixed(3),
    '| headAmp', (Math.max(...headY)-Math.min(...headY)).toFixed(3),
  );
  eng.mixer.stopAllAction();
}
