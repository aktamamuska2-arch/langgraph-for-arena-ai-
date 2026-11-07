/**
 * main.js — perakitan: stage + model + engine + UI.
 */
import * as THREE from 'three';
import './style.css';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createShowroom } from './showroom.js';
import { DanceEngine } from './danceEngine.js';
import { DANCES, CLIP_PREVIEWS } from './choreography.js';
import { MotionQA } from './motionQA.js';
import {
  buildBars,
  buildBeats,
  buildClipChips,
  buildPlaylist,
  renderQA,
  updateBeats,
} from './ui.js';

const canvas = document.getElementById('stage');
const showroom = createShowroom(canvas);
const { scene, camera, renderer } = showroom;

const dom = {
  loader: document.getElementById('loader'),
  loaderProgress: document.getElementById('loader-progress'),
  playlist: document.getElementById('playlist'),
  clipChips: document.getElementById('clip-chips'),
  cameraChips: document.getElementById('camera-chips'),
  playingTag: document.getElementById('playing-tag'),
  fps: document.getElementById('stat-fps'),
  bones: document.getElementById('stat-bones'),
  clips: document.getElementById('stat-clips'),
  scoreValue: document.getElementById('score-value'),
  scoreRing: document.getElementById('score-ring'),
  qaDance: document.getElementById('qa-dance'),
  qaPose: document.getElementById('qa-pose'),
  qaNotes: document.getElementById('qa-notes'),
  qaBars: document.getElementById('qa-bars'),
  beats: document.getElementById('beats'),
  btnPlay: document.getElementById('btn-play'),
  btnPrev: document.getElementById('btn-prev'),
  btnNext: document.getElementById('btn-next'),
  btnRestart: document.getElementById('btn-restart'),
  btnOrbit: document.getElementById('btn-orbit'),
  btnSkeleton: document.getElementById('btn-skeleton'),
  btnMedley: document.getElementById('btn-medley'),
  intensity: document.getElementById('slider-intensity'),
  outIntensity: document.getElementById('out-intensity'),
  dynamics: document.getElementById('slider-dynamics'),
  outDynamics: document.getElementById('out-dynamics'),
  speed: document.getElementById('slider-speed'),
  outSpeed: document.getElementById('out-speed'),
  panel: document.getElementById('qa-panel'),
};

const qaRefs = buildBars(dom.qaBars);
const beatDots = buildBeats(dom.beats, 16);

const state = {
  engine: null,
  playing: true,
  selected: 0,
  mode: 'dance', // 'dance' | 'clip'
  qa: null,
  qaClock: 0,
  fpsAcc: 0,
  fpsFrames: 0,
  loopGuard: 0,
  medley: false,
};

function makeQA(dance) {
  return new MotionQA({
    fps: 60,
    beats: dance.beats,
    bpm: dance.bpm,
    danceId: dance.id,
  });
}

function selectDance(index, { restart = true } = {}) {
  const dance = DANCES[(index + DANCES.length) % DANCES.length];
  state.selected = (index + DANCES.length) % DANCES.length;
  state.mode = 'dance';
  state.engine.playDance(dance, { restart });
  state.qa = makeQA(dance);
  if (state.engine) state.qa.contactOffset = state.engine.rig.contactFloorOffset;
  showroom.setAccent(dance.accentColor);
  document.documentElement.style.setProperty('--accent', dance.accentColor);
  dom.playingTag.textContent = dance.name;
  dom.qaDance.textContent = dance.name;
  dom.scoreRing.style.setProperty('--p', 0);
  dom.scoreValue.textContent = '–';

  for (const el of dom.playlist.children) {
    el.classList.toggle('is-active', el.dataset.id === dance.id);
  }
  for (const el of dom.clipChips.children) el.classList.remove('is-on');
  if (!state.playing) setPlaying(true);
}

function selectClip(clip) {
  state.mode = 'clip';
  state.engine.playClip(clip.clip);
  state.qa = null;
  showroom.setAccent('#6f8ba8');
  dom.playingTag.textContent = `clip · ${clip.name}`;
  dom.qaDance.textContent = 'clip asli';
  dom.qaPose.textContent = clip.clip;
  for (const el of dom.playlist.children) el.classList.remove('is-active');
  for (const el of dom.clipChips.children) el.classList.toggle('is-on', el.dataset.clip === clip.id);
  if (!state.playing) setPlaying(true);
}

function setPlaying(playing) {
  state.playing = playing;
  dom.btnPlay.textContent = playing ? '❚❚' : '▶';
  dom.playlist.classList.toggle('is-paused', !playing);
}

// ── muat model ──────────────────────────────────────────────────────────────
const loader = new GLTFLoader();
loader.load(
  `${import.meta.env.BASE_URL}models/humanoid.glb`,
  (gltf) => {
    try {
      boot(gltf);
    } catch (err) {
      console.error(err);
      dom.loaderProgress.textContent = `error: ${err.message}`;
    }
  },
  (ev) => {
    if (ev.total) {
      const pct = Math.round((ev.loaded / ev.total) * 100);
      dom.loaderProgress.textContent = `memuat model… ${pct}%`;
    }
  },
  (err) => {
    console.error(err);
    dom.loaderProgress.textContent = 'gagal memuat model — cek console';
  },
);

function boot(gltf) {
  const root = gltf.scene;

  // Normalisasi tinggi supaya proporsional dengan panggung (~1.88 m).
  const box = new THREE.Box3().setFromObject(root);
  const height = box.max.y - box.min.y;
  const scale = 1.88 / height;
  root.scale.setScalar(scale);
  root.position.y = -box.min.y * scale;
  root.updateMatrixWorld(true);

  showroom.attachModel(root);

  const engine = new DanceEngine(root, gltf.animations);
  state.engine = engine;

  // auto medley: tiap satu putaran selesai, lanjut ke dance berikutnya
  engine.onLoop = () => {
    if (!state.medley) return;
    queueMicrotask(() => selectDance(state.selected + 1));
  };
  engine.intensity = Number(dom.intensity.value);
  engine.dynScale = Number(dom.dynamics.value);
  engine.speed = Number(dom.speed.value);

  dom.bones.textContent = Object.keys(engine.rig.bones).length;
  dom.clips.textContent = gltf.animations.length;

  buildPlaylist(dom.playlist, DANCES, (i) => selectDance(i));
  buildClipChips(dom.clipChips, CLIP_PREVIEWS, selectClip);
  selectDance(0);

  dom.loaderProgress.textContent = 'siap';
  dom.loader.classList.add('is-hidden');
}

// ── kontrol UI ──────────────────────────────────────────────────────────────
dom.btnPlay.addEventListener('click', () => setPlaying(!state.playing));
dom.btnRestart.addEventListener('click', () => {
  if (state.mode === 'dance' && state.engine.dance) {
    selectDance(state.selected);
  }
});
dom.btnNext.addEventListener('click', () => selectDance(state.selected + 1));
dom.btnPrev.addEventListener('click', () => selectDance(state.selected - 1));

dom.intensity.addEventListener('input', () => {
  const v = Number(dom.intensity.value);
  state.engine && (state.engine.intensity = v);
  dom.outIntensity.textContent = v.toFixed(2);
});

dom.dynamics.addEventListener('input', () => {
  const v = Number(dom.dynamics.value);
  state.engine && (state.engine.dynScale = v);
  dom.outDynamics.textContent = `${v.toFixed(2)}×`;
});

dom.speed.addEventListener('input', () => {
  const v = Number(dom.speed.value);
  state.engine && (state.engine.speed = v);
  dom.outSpeed.textContent = `${v.toFixed(2)}×`;
});

dom.btnOrbit.addEventListener('click', () => {
  showroom.controls.autoRotate = !showroom.controls.autoRotate;
  dom.btnOrbit.classList.toggle('is-on', showroom.controls.autoRotate);
});

dom.btnMedley.addEventListener('click', () => {
  state.medley = !state.medley;
  dom.btnMedley.classList.toggle('is-on', state.medley);
});

dom.btnSkeleton.addEventListener('click', () => {
  const next = !(showroom.skeletonHelper?.visible ?? false);
  showroom.setSkeletonVisible(next);
  dom.btnSkeleton.classList.toggle('is-on', next);
});

for (const btn of dom.cameraChips.querySelectorAll('[data-cam]')) {
  btn.addEventListener('click', () => {
    showroom.setPreset(btn.dataset.cam);
    for (const b of dom.cameraChips.querySelectorAll('[data-cam]')) {
      b.classList.toggle('is-on', b === btn);
    }
  });
}

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const key = e.key.toLowerCase();
  if (e.code === 'Space') {
    e.preventDefault();
    setPlaying(!state.playing);
  } else if (key === 'r') {
    if (state.mode === 'dance') selectDance(state.selected);
  } else if (key === 'c') {
    dom.btnOrbit.click();
  } else if (key === 's') {
    dom.btnSkeleton.click();
  } else if (key === 'm') {
    dom.btnMedley.click();
  } else if (key === 'arrowright') {
    selectDance(state.selected + 1);
  } else if (key === 'arrowleft') {
    selectDance(state.selected - 1);
  } else if (/^[1-4]$/.test(key)) {
    selectDance(Number(key) - 1);
  }
});

window.addEventListener('resize', () => showroom.resize());

// ── loop render ─────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
let lastQaUpdate = 0;

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05);
  const engine = state.engine;

  if (engine) {
    engine.update(state.playing ? dt : 0);

    // telemetri QA live
    if (state.mode === 'dance' && state.qa && state.playing && engine.sampleJoints.hips) {
      state.qa.push(engine.time, engine.beat ?? 0, engine.sampleJoints);
      state.qaClock += dt;
      if (state.qaClock > 0.45) {
        state.qaClock = 0;
        const summary = state.qa.summary();
        if (summary) {
          renderQA(qaRefs, summary, {
            scoreValue: dom.scoreValue,
            scoreRing: dom.scoreRing,
            danceName: dom.qaDance,
            poseName: dom.qaPose,
            notes: dom.qaNotes,
          });
          dom.qaDance.textContent = engine.dance?.name ?? '–';
          dom.qaPose.textContent = engine.pose ? `${engine.pose.keyLabel} → ${engine.pose.nextLabel}` : '';
        }
      }
    }

    showroom.pulse(engine.beatPos ?? 0, 1);
    updateBeats(beatDots, engine.beatPos ?? 0, state.playing);
  }

  showroom.tick(dt, clock.elapsedTime);
  renderer.render(scene, camera);

  // fps
  state.fpsAcc += dt;
  state.fpsFrames += 1;
  if (state.fpsAcc >= 0.5) {
    dom.fps.textContent = Math.round(state.fpsFrames / state.fpsAcc);
    state.fpsAcc = 0;
    state.fpsFrames = 0;
  }
}

frame();

// ekspos untuk debugging di console
window.__arena = { state, showroom, DANCES };
