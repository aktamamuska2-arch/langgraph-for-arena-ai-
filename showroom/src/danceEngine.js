/**
 * danceEngine.js — penggerak showroom.
 *
 * Urutan tiap frame:
 *   1. reset bone ke bind pose (supaya bone yang tidak dianimasikan clip tidak menumpuk offset)
 *   2. AnimationMixer menulis clip dasar (idle/jog/…)
 *   3. Rig menangkap pose dasar → menerapkan layer koreografi aditif (world-space)
 *   4. ground lock + kompensasi telapak → dijamin tidak menembus lantai
 */
import * as THREE from 'three';
import { Rig } from './rig.js';
import { samplePose } from './choreography.js';

const FADE = 0.45;

export class DanceEngine {
  constructor(root, animations) {
    this.root = root;
    this.rig = new Rig(root);
    this.mixer = new THREE.AnimationMixer(root);
    this.clips = new Map(animations.map((c) => [c.name, c]));

    this.actions = new Map();
    for (const clip of animations) {
      const action = this.mixer.clipAction(clip);
      action.enabled = true;
      action.setEffectiveWeight(0);
      this.actions.set(clip.name, action);
    }

    this.dance = null;
    this.currentAction = null;
    this.time = 0;
    this.beat = 0;
    this.blend = 0;
    this.targetBlend = 0;
    this.intensity = 1;
    this.dynScale = 1;
    this.oscScale = 1;
    this.speed = 1;
    this.pose = null;
    this.loopCount = 0;
    this.onLoop = null;
    this.calibratedClip = null;

    this.sampleJoints = {};
    this.prevJoints = {};
    this.jointVelocity = {};

    this.missingBones = this.rig.missing;
  }

  /** Putar clip asli tanpa koreografi. */
  playClip(name, fade = FADE) {
    if (!this.clips.has(name)) throw new Error(`clip tidak ada: ${name}`);
    this.dance = null;
    this.targetBlend = 0;
    this.blend = 0;
    this.time = 0;
    this.beatPos = 0;
    this.pose = null;
    this._useAction(name, fade);
  }

  /** Mulai sebuah dance dari definisi koreografi. */
  playDance(dance, { restart = true, fade = FADE } = {}) {
    const changed = this.dance?.id !== dance.id;
    this.dance = dance;
    this.rig.stanceOverride = dance.stance || null;
    this.targetBlend = 1;
    if (restart) {
      this.time = 0;
      this.loopCount = 0;
      this.beat = 0;
    }
    if (changed) this.blend = 0;
    this._useAction(dance.base?.clip || 'idle', fade);
    // Kalibrasi stance sekali saja, saat clip dasar masih bersih (belum ada
    // crossfade) — kalau tidak, orientasi telapak "rata" bisa ikut miring.
    if (this.calibratedClip !== dance.base?.clip) {
      this.mixer.update(0);
      this.rig.calibrateStance();
      this.calibratedClip = dance.base?.clip;
    }
  }

  _useAction(name, fade = FADE) {
    const next = this.actions.get(name);
    if (!next || next === this.currentAction) return;
    next.reset();
    next.setLoop(THREE.LoopRepeat, Infinity);
    next.setEffectiveTimeScale(this.dance?.base?.timeScale ?? 1);
    next.setEffectiveWeight(1);
    next.play();
    if (this.currentAction) next.crossFadeFrom(this.currentAction, fade, true);
    this.currentAction = next;
    // Berapa kali clip dasar harus berputar dalam satu loop koreografi, supaya
    // fasenya kembali tepat di titik loop (lihat penguncian fase di update()).
    const dur = next.getClip().duration;
    const loopSec = this.dance ? (this.dance.beats * 60) / this.dance.bpm : 0;
    this.clipCycles = dur > 0 && loopSec ? Math.max(1, Math.round(loopSec / dur)) : 1;
  }

  /** dt dalam detik; kirim 0 untuk pause tanpa kehilangan pose. */
  update(dt) {
    const step = dt * this.speed;
    this.time += step;

    this.beat = this.dance ? (this.time * this.dance.bpm) / 60 : 0;
    const loopPos = this.dance ? ((this.beat % this.dance.beats) + this.dance.beats) % this.dance.beats : 0;
    const loops = this.dance ? Math.floor(this.beat / this.dance.beats) : 0;
    if (this.dance && loops > this.loopCount) {
      this.loopCount = loops;
      this.onLoop?.(this.dance, loops);
    }

    // 1+2: bind pose → clip dasar (selalu, supaya offset frame lalu tidak menumpuk)
    this.rig.resetToRest();
    if (this.currentAction) this.currentAction.setEffectiveTimeScale(this.dance?.base?.timeScale ?? 1);
    this.mixer.update(step);

    // Fase clip dasar dikunci ke posisi loop koreografi. Tanpa ini, clip terus
    // berjalan dengan panjangnya sendiri sehingga pose di akhir loop tidak sama
    // dengan pose di awal (loop "pop" puluhan sentimeter).
    if (this.dance && this.currentAction) {
      const act = this.currentAction;
      const dur = act.getClip().duration;
      if (dur > 0) {
        const lp = ((this.beat % this.dance.beats) + this.dance.beats) % this.dance.beats;
        act.time = ((lp / this.dance.beats) * this.clipCycles * dur) % dur;
        this.mixer.update(0);
      }
    }

    if (!this.dance) {
      this.pose = null;
      this.rig.root.updateMatrixWorld(true);
      this._trackJoints(step);
      return;
    }

    // 3: blending masuk/keluar layer koreografi
    const dir = Math.sign(this.targetBlend - this.blend);
    if (dir !== 0) {
      const advance = step === 0 ? 0 : Math.abs(step) / FADE;
      this.blend = THREE.MathUtils.clamp(this.blend + dir * advance, 0, 1);
      if (Math.abs(this.targetBlend - this.blend) < 0.01) this.blend = this.targetBlend;
    }

    // 4: layer prosedural
    this.rig.captureBase();
    const pose = samplePose(this.dance, loopPos, { oscScale: this.oscScale, dynScale: this.dynScale });
    this.pose = pose;
    this.rig.applyPose(pose, this.blend * this.intensity);
    this.rig.root.updateMatrixWorld(true);

    this._trackJoints(step);
    this.beatPos = loopPos;
    this.loop = loops;
  }

  _trackJoints(dt) {
    const prev = this.prevJoints;
    this.prevJoints = this.sampleJoints;
    this.sampleJoints = prev;
    this.rig.sampleJoints(this.sampleJoints);
    if (dt > 0) {
      for (const [g, p] of Object.entries(this.sampleJoints)) {
        const q = this.prevJoints[g];
        if (!q) continue;
        (this.jointVelocity[g] ||= new THREE.Vector3()).copy(p).sub(q).divideScalar(dt);
      }
    }
  }

  get poseLabel() {
    if (!this.dance) return 'Clip asli';
    return this.pose?.section || this.dance.name;
  }
}
