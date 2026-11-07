/**
 * evaluate.js — evaluasi iteratif koreografi tanpa browser.
 *
 *   node eval/evaluate.js            # tabel metrik + sheets
 *   node eval/evaluate.js --json     # keluaran JSON saja
 *   node eval/evaluate.js --only=neon-groove
 *
 * Menjalankan loop penuh tiap dance pada 60 fps memakai engine yang sama
 * dengan yang dipakai di browser, lalu menghitung metrik MotionQA dan
 * merender contact-sheet stick figure (front & side) ke eval/out/.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import fs from 'node:fs';
import path from 'node:path';
import { DANCES } from '../src/choreography.js';
import { DanceEngine } from '../src/danceEngine.js';
import { MotionQA } from '../src/motionQA.js';
import { renderSheetPNG } from './stickfig.js';

const ROOT = path.resolve(import.meta.dirname, '..');
const FPS = 60;
const args = process.argv.slice(2);
const only = args.find((a) => a.startsWith('--only='))?.split('=')[1];

const buf = fs.readFileSync(path.join(ROOT, 'public/models/humanoid.glb'));
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const gltf = await new Promise((res, rej) => new GLTFLoader().parse(ab, '', res, rej));

function makeEngine() {
  const scene = new THREE.Scene();
  const root = gltf.scene.clone(true);
  // SkeletonUtils tidak tersedia lewat import murni di Node → cukup satu instance per run.
  scene.add(root);
  root.updateMatrixWorld(true);
  return { scene, root, engine: new DanceEngine(root, gltf.animations) };
}

const outDir = path.join(ROOT, 'eval/out');
fs.mkdirSync(outDir, { recursive: true });

const results = [];

for (const dance of DANCES) {
  if (only && dance.id !== only) continue;
  const { engine } = makeEngine();
  engine.playDance(dance, { restart: true });
  engine.blend = 1;
  engine.time = 0;

  const loopSeconds = (dance.beats * 60) / dance.bpm;
  const frames = Math.round(loopSeconds * FPS);
  const qa = new MotionQA({ fps: FPS, beats: dance.beats, bpm: dance.bpm, danceId: dance.id });
  qa.contactOffset = engine.rig.contactFloorOffset;

  const dt = 1 / FPS;
  // warm-up: frame pertama setelah ganti dance masih berupa crossfade/IK yang
  // belum mengendap — kalau ikut diukur, metrik loop & dinamika jadi melenceng.
  for (let i = 0; i < 12; i++) engine.update(dt);
  engine.time = 0;
  engine.update(0); // isi pose & sampel sendi awal (loop 0, state sudah stabil)
  const sheetFrames = [];
  for (let i = 0; i <= frames; i++) {
    if (i > 0) engine.update(dt);
    if (!engine.sampleJoints.hips) continue;
    const t = i / FPS;
    qa.push(t, engine.beat, engine.sampleJoints);
    if (i % Math.round(frames / 8) === 0 && sheetFrames.length < 8) {
      const copy = {};
      for (const [g, p] of Object.entries(engine.sampleJoints)) copy[g] = p.clone();
      // ujung kepala kira-kira 0.12 m di atas bone head, untuk proporsi gambar
      copy.headTop = copy.head.clone().add(new THREE.Vector3(0, 0.16, 0));
      sheetFrames.push({
        joints: copy,
        beat: (engine.beatPos ?? 0).toFixed(1),
        label: (engine.pose?.keyLabel || '').slice(0, 24),
      });
    }
  }

  const summary = qa.summary();
  results.push(summary);

  for (const view of ['front', 'side']) {
    renderSheetPNG({
      frames: sheetFrames,
      view,
      title: `${dance.name} / ${view}`,
      subtitle: `${dance.tagline} · ${dance.beats} ketukan · skor ${summary.overall}/100`,
      color: dance.accentColor,
      outPath: path.join(outDir, `${dance.id}-${view}.png`),
    });
  }
  engine.mixer.stopAllAction();
}

if (args.includes('--json')) {
  console.log(JSON.stringify(results, null, 2));
} else {
  const pad = (s, n) => String(s).padEnd(n);
  console.log('\n=== MotION QA — evaluasi koreografi ===\n');
  console.log(pad('dance', 16), pad('bpm', 5), pad('energi', 7), pad('slip', 7), pad('lantai', 8), pad('jerk', 7), pad('seam', 7), pad('balance', 8), pad('ritme', 7), 'skor');
  for (const r of results) {
    console.log(
      pad(r.danceId, 16),
      pad(r.bpm, 5),
      pad(r.energy.toFixed(2), 7),
      pad(r.footSlip.toFixed(2), 7),
      pad(r.floorMin.toFixed(3), 8),
      pad(Math.round(r.jerk), 7),
      pad(r.loopSeam.toFixed(3), 7),
      pad(r.balance.toFixed(2), 8),
      pad(r.beatSync.toFixed(2), 7),
      r.overall,
    );
  }
  console.log('\nskor per aspek (0-100):');
  for (const r of results) {
    console.log(' ', pad(r.danceId, 16), JSON.stringify(r.scores));
  }
  console.log('\nROM vertikal sendi utama (m):');
  for (const r of results) {
    console.log(' ', pad(r.danceId, 16), Object.entries(r.rom).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' '));
  }
  fs.writeFileSync(path.join(outDir, 'report.json'), JSON.stringify(results, null, 2));
  console.log('\nsheet → eval/out/*.png, data → eval/out/report.json');
}
