/**
 * choreography.js — partitur tari prosedural.
 *
 * Model authoring (dipakai bersama browser & evaluator):
 *   torso  : offset rotasi per grup, DERAJAT, sumbu dunia
 *            X = pitch (+ ujung atas ke belakang), Y = yaw (+ ke kiri),
 *            Z = roll (+ ujung atas ke kiri)
 *            Grup engsel (elbow*, knee*, hand*) memakai sumbu bone.
 *   root   : geser panggul [x, y, z] meter (dunia). y negatif = turun/tekuk lutut.
 *   feet   : target telapak [dx, dy, dz] meter relatif ke stance dance.
 *            dy > 0 = angkat kaki (tap/kick).
 *   footAngle : rotasi telapak, X negatif = jinjit (toe-off).
 *   yaw    : putaran badan penuh (derajat) pada bone root.
 *   stance : posisi jangkar telapak absolut (meter, dunia) — default dari clip idle.
 *
 * Karena tungkai memakai IK, berpindah berat badan TIDAK membuat kaki meluncur:
 * kaki tetap di titik jangkar, lutut yang menyesuaikan.
 */

const ARM = {
  down: { armL: [0, 0, 0], armR: [0, 0, 0], elbowL: [0, 0, 0], elbowR: [0, 0, 0] },
  guard: { armL: [26, 0, -9], armR: [26, 0, 9], elbowL: [66, 0, 0], elbowR: [66, 0, 0] },
  guardLow: { armL: [15, 0, -6], armR: [15, 0, 6], elbowL: [42, 0, 0], elbowR: [42, 0, 0] },
  out: { armL: [4, 0, -74], armR: [4, 0, 74], elbowL: [12, 0, 0], elbowR: [12, 0, 0] },
  up: { armL: [-18, 0, -140], armR: [-18, 0, 140], elbowL: [10, 0, 0], elbowR: [10, 0, 0] },
  upDiagL: { armL: [-10, 0, -134], elbowL: [8, 0, 0], armR: [24, 0, 10], elbowR: [70, 0, 0] },
  upDiagR: { armR: [-10, 0, 134], elbowR: [8, 0, 0], armL: [24, 0, -10], elbowL: [70, 0, 0] },
  punchFwdL: { armL: [82, 0, -6], elbowL: [8, 0, 0], armR: [18, 0, 10], elbowR: [78, 0, 0] },
  punchFwdR: { armR: [82, 0, 6], elbowR: [8, 0, 0], armL: [18, 0, -10], elbowL: [78, 0, 0] },
  goalpost: { armL: [42, 0, -60], armR: [42, 0, 60], elbowL: [95, 0, 0], elbowR: [95, 0, 0] },
  robotL: { armL: [8, 0, -94], elbowL: [92, 0, 0], armR: [8, 0, 8], elbowR: [0, 0, 0] },
  robotR: { armR: [8, 0, 94], elbowR: [92, 0, 0], armL: [8, 0, -8], elbowL: [0, 0, 0] },
  lowReachL: { armL: [56, 0, -30], elbowL: [22, 0, 0], armR: [10, 0, 10], elbowR: [30, 0, 0] },
  lowReachR: { armR: [56, 0, 30], elbowR: [22, 0, 0], armL: [10, 0, -10], elbowL: [30, 0, 0] },
  waveHighL: { armL: [-38, 0, -128], elbowL: [64, 0, 0], armR: [30, 0, 14], elbowR: [80, 0, 0] },
  waveHighR: { armR: [-38, 0, 128], elbowR: [64, 0, 0], armL: [30, 0, -14], elbowL: [80, 0, 0] },
};

const TORSO = {
  none: { hips: [0, 0, 0], spine1: [0, 0, 0], chest: [0, 0, 0], head: [0, 0, 0] },
  swayL: { hips: [0, 5, 7], spine1: [0, -2, -2], chest: [-3, -7, -6], head: [-2, 8, 5] },
  swayR: { hips: [0, -5, -7], spine1: [0, 2, 2], chest: [-3, 7, 6], head: [-2, -8, -5] },
  twistL: { hips: [-2, 12, 3], chest: [0, -16, -3], head: [-2, 14, 0] },
  twistR: { hips: [-2, -12, -3], chest: [0, 16, 3], head: [-2, -14, 0] },
  dip: { hips: [-6, 0, 0], chest: [-9, 0, 0], head: [5, 0, 0] },
  lift: { hips: [4, 0, 0], chest: [7, 0, 0], head: [7, 0, 0] },
  look: { chest: [-2, 0, 0], head: [-8, 0, 0] },
};

const F = {
  neutral: { L: [0, 0, 0], R: [0, 0, 0] },
  wide: { L: [-0.07, 0, 0], R: [0.07, 0, 0] },
  stepL: { L: [0, 0, -0.18], R: [0, 0, 0.06] },
  stepR: { L: [0, 0, 0.06], R: [0, 0, -0.18] },
  tapL: { L: [-0.12, 0.03, -0.05], R: [0, 0, 0] },
  tapR: { R: [0.12, 0.03, -0.05], L: [0, 0, 0] },
  liftL: { L: [-0.02, 0.16, 0.1], R: [0, 0, 0] },
  liftR: { R: [0.02, 0.16, 0.1], L: [0, 0, 0] },
  tuckL: { L: [0.02, 0.22, 0.02], R: [0, 0, 0] },
  tuckR: { R: [-0.02, 0.22, 0.02], L: [0, 0, 0] },
  pivotL: { L: [0.15, 0.03, -0.03], R: [-0.02, 0.24, 0.06] },
  kickL: { L: [-0.05, 0.3, -0.24], R: [0, 0, 0] },
  kickR: { R: [0.05, 0.3, -0.24], L: [0, 0, 0] },
};

const merge = (...parts) => Object.assign({}, ...parts);

const K = (b, ease, label, torso, opts = {}) => ({
  b,
  ease,
  label,
  torso,
  root: opts.root || [0, 0, 0],
  feet: opts.feet || F.neutral,
  yaw: opts.yaw || 0,
  footAngle: opts.footAngle || {},
});

const STANCE = {
  L: [-0.15, 0.079, 0.03],
  R: [0.15, 0.079, 0.03],
};

// ── 1. NEON GROOVE ──────────────────────────────────────────────────────────
const NEON_GROOVE = {
  id: 'neon-groove',
  name: 'Neon Groove',
  tagline: 'step-touch · 110 BPM',
  bpm: 110,
  beats: 32,
  base: { clip: 'idle', timeScale: 1 },
  accentColor: '#4df0ff',
  ik: true,
  stance: STANCE,
  sections: [
    { b: 0, label: 'Bait A · Step-touch' },
    { b: 8, label: 'Bait B · Raise the roof' },
    { b: 16, label: 'Bait C · Twist' },
    { b: 24, label: 'Bait D · Breakdown' },
  ],
  keys: [
    K(0, 'smooth', 'Tap kanan', merge(ARM.upDiagR, TORSO.swayL), {
      root: [-0.05, -0.03, 0], feet: F.tapR, footAngle: { R: [0, 0, 0], L: [-6, 0, 0] },
    }),
    K(1.4, 'snap', 'Tutup', merge(ARM.guard, TORSO.swayL), { root: [-0.07, -0.07, 0] }),
    K(2, 'smooth', 'Tap kiri', merge(ARM.upDiagL, TORSO.swayR), {
      root: [0.05, -0.03, 0], feet: F.tapL, footAngle: { L: [0, 0, 0], R: [-6, 0, 0] },
    }),
    K(3.4, 'snap', 'Tutup', merge(ARM.guard, TORSO.swayR), { root: [0.07, -0.07, 0] }),
    K(4, 'smooth', 'Sway kiri', merge(ARM.guardLow, TORSO.swayL), { root: [-0.08, -0.05, 0], feet: F.wide }),
    K(5, 'pop', 'Bounce', merge(ARM.guard, TORSO.dip), { root: [0, -0.16, 0], feet: F.wide }),
    K(6, 'smooth', 'Sway kanan', merge(ARM.guardLow, TORSO.swayR), { root: [0.08, -0.05, 0], feet: F.wide }),
    K(7, 'pop', 'Open', merge(ARM.out, TORSO.dip), { root: [0, -0.15, 0], feet: F.wide }),
    K(8, 'smooth', 'Naik', merge(ARM.up, TORSO.lift), { root: [0, 0.02, 0], feet: F.wide }),
    K(9, 'snap', 'Press', merge(ARM.goalpost, TORSO.dip), { root: [0, -0.17, 0], feet: F.wide }),
    K(10, 'smooth', 'Naik lagi', merge(ARM.up, TORSO.lift), { root: [0, 0.02, 0], feet: F.wide }),
    K(11, 'snap', 'Goalpost', merge(ARM.goalpost, TORSO.dip), { root: [0, -0.14, 0], feet: F.wide }),
    K(12, 'smooth', 'Punch kiri', merge(ARM.punchFwdL, TORSO.twistR), {
      root: [0.06, -0.06, 0], feet: F.stepR,
    }),
    K(13.4, 'snap', 'Retract', merge(ARM.guard, TORSO.swayR), { root: [0.05, -0.08, 0], feet: F.stepR }),
    K(14, 'smooth', 'Punch kanan', merge(ARM.punchFwdR, TORSO.twistL), {
      root: [-0.06, -0.06, 0], feet: F.stepL,
    }),
    K(15.4, 'snap', 'Retract', merge(ARM.guard, TORSO.swayL), { root: [-0.05, -0.08, 0], feet: F.stepL }),
    K(16, 'smooth', 'Twist kiri', merge(ARM.out, TORSO.twistL), { root: [-0.03, -0.05, 0], yaw: -14, feet: F.wide }),
    K(17, 'smooth', 'Drop', merge(ARM.guardLow, TORSO.dip), { root: [0, -0.15, 0], yaw: -4, feet: F.wide }),
    K(18, 'smooth', 'Twist kanan', merge(ARM.out, TORSO.twistR), { root: [0.03, -0.05, 0], yaw: 14, feet: F.wide }),
    K(19, 'smooth', 'Drop', merge(ARM.guardLow, TORSO.dip), { root: [0, -0.15, 0], yaw: 4, feet: F.wide }),
    K(20, 'smooth', 'Tap kiri tinggi', merge(ARM.upDiagL, TORSO.swayR), {
      root: [0.05, -0.02, 0], feet: F.liftL,
    }),
    K(21, 'pop', 'Robot kiri', merge(ARM.robotL, TORSO.dip), { root: [-0.04, -0.13, 0], feet: F.wide }),
    K(22, 'smooth', 'Tap kanan tinggi', merge(ARM.upDiagR, TORSO.swayL), {
      root: [-0.05, -0.02, 0], feet: F.liftR,
    }),
    K(23, 'pop', 'Robot kanan', merge(ARM.robotR, TORSO.dip), { root: [0.04, -0.13, 0], feet: F.wide }),
    K(24, 'snap', 'Breakdown', merge(ARM.goalpost, TORSO.dip), { root: [0, -0.26, 0], feet: F.wide }),
    K(25, 'smooth', 'Crouch sway', merge(ARM.guard, TORSO.swayL), { root: [-0.05, -0.24, 0], feet: F.wide }),
    K(26, 'smooth', 'Crouch sway', merge(ARM.guard, TORSO.swayR), { root: [0.05, -0.24, 0], feet: F.wide }),
    K(27, 'pop', 'Burst', merge(ARM.out, TORSO.lift), { root: [0, -0.02, 0], feet: F.wide, footAngle: { L: [-8, 0, 0], R: [-8, 0, 0] } }),
    K(28, 'smooth', 'Tap kanan', merge(ARM.upDiagR, TORSO.swayL), { root: [-0.05, -0.03, 0], feet: F.tapR }),
    K(29, 'smooth', 'Tap kiri', merge(ARM.upDiagL, TORSO.swayR), { root: [0.05, -0.03, 0], feet: F.tapL }),
    K(30, 'smooth', 'Goalpost kiri', merge(ARM.goalpost, TORSO.swayL), { root: [-0.05, -0.06, 0], feet: F.wide }),
    K(31, 'smooth', 'Goalpost kanan', merge(ARM.goalpost, TORSO.swayR), { root: [0.05, -0.06, 0], feet: F.wide }),
  ],
  osc: [
    { group: 'head', axis: 0, amp: 4.5, freq: 1, wave: 'sin', phase: 0.05 },
    { group: 'chest', axis: 1, amp: 3, freq: 0.5, wave: 'sin', phase: 0.25 },
    { group: 'handL', axis: 0, amp: 8, freq: 1, wave: 'sin', phase: 0.5 },
    { group: 'handR', axis: 0, amp: 8, freq: 1, wave: 'sin', phase: 0.5 },
    { group: 'spine2', axis: 2, amp: 2, freq: 0.5, wave: 'sin', phase: -0.1 },
  ],
};

// ── 2. WAVE RIDER ───────────────────────────────────────────────────────────
const WAVE_RIDER = {
  id: 'wave-rider',
  name: 'Wave Rider',
  tagline: 'body roll · 96 BPM',
  bpm: 96,
  beats: 24,
  base: { clip: 'idle', timeScale: 0.9 },
  accentColor: '#b06bff',
  ik: true,
  stance: STANCE,
  sections: [
    { b: 0, label: 'Bait A · Body roll' },
    { b: 8, label: 'Bait B · Arm wave' },
    { b: 16, label: 'Bait C · Slow turn' },
  ],
  keys: [
    K(0, 'smooth', 'Gelombang turun', merge(ARM.guardLow, TORSO.swayL), { root: [-0.05, -0.02, 0], feet: F.wide }),
    K(3, 'smooth', 'Gelombang naik', merge(ARM.guard, TORSO.lift), { root: [0.03, 0.02, 0], feet: F.wide }),
    K(6, 'smooth', 'Buka lebar', merge(ARM.out, { hips: [-3, 0, -4], chest: [-4, 0, 4], head: [-8, 0, 3] }), {
      root: [0.05, -0.04, 0], feet: F.wide, yaw: -10,
    }),
    K(9, 'smooth', 'Angkat', merge(ARM.waveHighL, { hips: [0, 0, 6], chest: [-2, 0, -6], head: [-4, 0, -4] }), {
      root: [-0.05, 0.01, 0], feet: F.wide, yaw: -4,
    }),
    K(12, 'smooth', 'Silang', merge(ARM.guard, TORSO.twistR), { root: [0.04, -0.05, 0], feet: F.wide, yaw: 12 }),
    K(15, 'smooth', 'Gelombang dalam', merge(ARM.goalpost, TORSO.dip), {
      root: [0, -0.14, 0], feet: F.wide, yaw: 4,
    }),
    K(18, 'smooth', 'Putar', merge(ARM.waveHighR, TORSO.swayL), { root: [-0.04, -0.03, 0], feet: F.wide, yaw: 22 }),
    K(21, 'smooth', 'Kembali', merge(ARM.guardLow, TORSO.swayR), { root: [0.03, -0.06, 0], feet: F.wide, yaw: 8 }),
  ],
  osc: [
    // aksen halus tiap ketukan (menjaga "legato" tapi tetap terasa pulse-nya)
    { group: 'chest', axis: 0, amp: 5.2, freq: 1, wave: 'pulse', phase: 0.25 },
    { group: 'hips', axis: 0, amp: 3.2, freq: 1, wave: 'pulse', phase: 0.25 },
    { group: 'head', axis: 0, amp: 4.5, freq: 1, wave: 'pulse', phase: 0.35 },
    { group: 'hips', axis: 2, amp: 5, freq: 0.5, wave: 'sin', phase: 0.0 },
    { group: 'spine1', axis: 2, amp: 5, freq: 0.5, wave: 'sin', phase: -0.09 },
    { group: 'spine2', axis: 2, amp: 6, freq: 0.5, wave: 'sin', phase: -0.18 },
    { group: 'chest', axis: 2, amp: 6.5, freq: 0.5, wave: 'sin', phase: -0.27 },
    { group: 'chest2', axis: 2, amp: 5, freq: 0.5, wave: 'sin', phase: -0.36 },
    { group: 'neck', axis: 0, amp: 4.5, freq: 0.5, wave: 'sin', phase: -0.45 },
    { group: 'head', axis: 0, amp: 5.5, freq: 0.5, wave: 'sin', phase: -0.54 },
    { group: 'armL', axis: 0, amp: 14, freq: 0.5, wave: 'sin', phase: 0.1 },
    { group: 'elbowL', axis: 0, amp: 20, freq: 0.5, wave: 'sin', phase: -0.05 },
    { group: 'handL', axis: 0, amp: 18, freq: 0.5, wave: 'sin', phase: -0.2 },
    { group: 'armR', axis: 0, amp: 14, freq: 0.5, wave: 'sin', phase: 0.35 },
    { group: 'elbowR', axis: 0, amp: 20, freq: 0.5, wave: 'sin', phase: 0.2 },
    { group: 'handR', axis: 0, amp: 18, freq: 0.5, wave: 'sin', phase: 0.05 },
  ],
};

// ── 3. BREAK CIRCUIT ────────────────────────────────────────────────────────
const BREAK_CIRCUIT = {
  id: 'break-circuit',
  name: 'Break Circuit',
  tagline: 'footwork · spin · 128 BPM',
  bpm: 128,
  beats: 32,
  base: { clip: 'idle', timeScale: 1.05 },
  accentColor: '#ff5c7a',
  ik: true,
  stance: STANCE,
  sections: [
    { b: 0, label: 'Top rock' },
    { b: 4, label: 'Drop & footwork' },
    { b: 12, label: 'Spin' },
    { b: 16, label: 'Rise & pump' },
    { b: 20, label: 'Freeze' },
    { b: 24, label: 'Recover' },
  ],
  keys: [
    K(0, 'snap', 'Rock kiri', merge(ARM.guard, TORSO.twistL), { root: [-0.06, -0.08, 0], feet: F.stepL }),
    K(1, 'snap', 'Rock kanan', merge(ARM.out, TORSO.twistR), { root: [0.06, -0.06, 0], feet: F.stepR }),
    K(2, 'snap', 'Rock kiri', merge(ARM.guard, TORSO.twistL), { root: [-0.06, -0.08, 0], feet: F.stepL }),
    K(3, 'snap', 'Rock kanan', merge(ARM.out, TORSO.twistR), { root: [0.06, -0.06, 0], feet: F.stepR }),
    K(4, 'smooth', 'Turun', merge(ARM.guard, TORSO.dip), {
      root: [0, -0.3, 0], feet: F.wide, footAngle: { L: [-10, 0, 0], R: [-10, 0, 0] },
    }),
    K(5.5, 'snap', 'Low rock', merge(ARM.lowReachL, TORSO.dip), {
      root: [-0.08, -0.32, 0], feet: { L: [0, 0, 0], R: [0.16, 0.05, 0.02] },
    }),
    K(6.5, 'snap', 'Tendang kiri', merge(ARM.lowReachR, TORSO.dip), {
      root: [0.08, -0.3, 0], feet: F.kickL,
    }),
    K(7.5, 'snap', 'Tendang kanan', merge(ARM.lowReachL, TORSO.dip), {
      root: [-0.08, -0.3, 0], feet: F.kickR,
    }),
    K(8.5, 'snap', 'Tendang kiri', merge(ARM.lowReachR, TORSO.dip), {
      root: [0.08, -0.3, 0], feet: F.kickL,
    }),
    K(9.5, 'snap', 'Tendang kanan', merge(ARM.lowReachL, TORSO.dip), {
      root: [-0.08, -0.3, 0], feet: F.kickR,
    }),
    K(11, 'smooth', 'Kumpul', merge(ARM.guard, TORSO.dip), { root: [0, -0.28, 0], feet: F.wide }),
    // Pivot: telapak kiri dibawa ke titik pusat putaran supaya badan berputar
    // MENGELILINGI kaki, bukan kaki ikut berputar mengelilingi badan (anti-slip).
    K(12, 'smooth', 'Mulai putar', merge(ARM.robotR, TORSO.dip), {
      root: [0, -0.24, 0], feet: F.pivotL, yaw: 45, footAngle: { L: [-24, 0, 0] },
    }),
    K(14, 'smooth', 'Putar', merge(ARM.robotR, TORSO.dip), {
      root: [0, -0.22, 0], feet: F.pivotL, yaw: 200, footAngle: { L: [-24, 0, 0] },
    }),
    K(16, 'smooth', 'Selesai putar', merge(ARM.goalpost, TORSO.lift), {
      root: [0, -0.06, 0], feet: F.wide, yaw: 360,
    }),
    K(17, 'snap', 'Ledak', merge(ARM.up, TORSO.lift), { root: [0, 0.01, 0], feet: F.wide, yaw: 360 }),
    K(18, 'smooth', 'Pump kiri', merge(ARM.punchFwdL, TORSO.dip), { root: [0.05, -0.12, 0], feet: F.wide, yaw: 360 }),
    K(19, 'smooth', 'Pump kanan', merge(ARM.punchFwdR, TORSO.dip), { root: [-0.05, -0.12, 0], feet: F.wide, yaw: 360 }),
    K(20, 'snap', 'Freeze', merge(ARM.waveHighL, { hips: [4, 0, 8], chest: [-4, 0, -8], head: [-10, 0, -5] }), {
      root: [0.06, -0.2, 0], feet: F.tuckR, yaw: 360,
    }),
    K(22, 'hold', 'Tahan', merge(ARM.waveHighL, { hips: [4, 0, 8], chest: [-4, 0, -8], head: [-10, 0, -5] }), {
      root: [0.06, -0.2, 0], feet: F.tuckR, yaw: 360,
    }),
    K(24, 'smooth', 'Lepas', merge(ARM.guard, TORSO.dip), { root: [0, -0.24, 0], feet: F.wide, yaw: 360 }),
    K(25, 'snap', 'Buka', merge(ARM.out, TORSO.twistL), { root: [-0.07, -0.1, 0], feet: F.wide, yaw: 360 }),
    K(26, 'snap', 'Buka', merge(ARM.out, TORSO.twistR), { root: [0.07, -0.1, 0], feet: F.wide, yaw: 360 }),
    K(27, 'smooth', 'Netral', merge(ARM.guard, TORSO.none), { root: [0, -0.06, 0], feet: F.wide, yaw: 360 }),
    K(28, 'snap', 'Rock kiri', merge(ARM.guard, TORSO.twistL), { root: [-0.06, -0.08, 0], feet: F.stepL, yaw: 360 }),
    K(29, 'snap', 'Rock kanan', merge(ARM.out, TORSO.twistR), { root: [0.06, -0.06, 0], feet: F.stepR, yaw: 360 }),
    K(30, 'snap', 'Rock kiri', merge(ARM.guard, TORSO.twistL), { root: [-0.06, -0.08, 0], feet: F.stepL, yaw: 360 }),
    K(31, 'snap', 'Rock kanan', merge(ARM.out, TORSO.twistR), { root: [0.06, -0.06, 0], feet: F.stepR, yaw: 360 }),
  ],
  osc: [
    { group: 'head', axis: 0, amp: 4, freq: 1, wave: 'sin', phase: 0.1 },
    { group: 'handL', axis: 1, amp: 12, freq: 1, wave: 'sin', phase: 0 },
    { group: 'handR', axis: 1, amp: 12, freq: 1, wave: 'sin', phase: 0.5 },
  ],
};

// ── 4. ROBOT POP ────────────────────────────────────────────────────────────
const ROBOT_POP = {
  id: 'robot-pop',
  name: 'Robot Pop',
  tagline: 'staccato · 120 BPM',
  bpm: 120,
  beats: 16,
  base: { clip: 'idle', timeScale: 1 },
  accentColor: '#ffd166',
  ik: true,
  posture: { hips: 0, chest: 0, head: 0 },
  stance: STANCE,
  sections: [
    { b: 0, label: 'Pose 1 · Sudut' },
    { b: 4, label: 'Pose 2 · Kotak' },
    { b: 8, label: 'Pose 3 · Tunjuk' },
    { b: 12, label: 'Pose 4 · Rendah' },
  ],
  keys: [
    K(0, 'snap', 'Sudut kiri', merge(ARM.robotL, TORSO.swayR), { root: [0.03, -0.05, 0], feet: F.wide }),
    K(2, 'snap', 'Sudut kanan', merge(ARM.robotR, TORSO.swayL), { root: [-0.03, -0.05, 0], feet: F.wide }),
    K(4, 'snap', 'Kotak', merge(ARM.goalpost, TORSO.dip), { root: [0, -0.12, 0], feet: F.wide }),
    K(6, 'snap', 'Buka', merge(ARM.out, TORSO.lift), { root: [0, -0.02, 0], feet: F.wide }),
    K(8, 'snap', 'Tunjuk kiri', merge(ARM.punchFwdL, TORSO.twistR), { root: [0.05, -0.07, 0], feet: F.stepR }),
    K(10, 'snap', 'Tunjuk kanan', merge(ARM.punchFwdR, TORSO.twistL), { root: [-0.05, -0.07, 0], feet: F.stepL }),
    K(12, 'snap', 'Rendah kiri', merge(ARM.robotL, TORSO.dip), { root: [-0.06, -0.24, 0], feet: F.wide }),
    K(14, 'snap', 'Rendah kanan', merge(ARM.robotR, TORSO.dip), { root: [0.06, -0.24, 0], feet: F.wide }),
  ],
  osc: [
    { group: 'head', axis: 0, amp: 2, freq: 2, wave: 'square', phase: 0 },
    { group: 'handL', axis: 2, amp: 5, freq: 2, wave: 'square', phase: 0.25 },
    { group: 'handR', axis: 2, amp: -5, freq: 2, wave: 'square', phase: 0.25 },
  ],
};

export const DANCES = [NEON_GROOVE, WAVE_RIDER, BREAK_CIRCUIT, ROBOT_POP];

/** Clip asli yang bisa diputar apa adanya (tanpa koreografi). */
export const CLIP_PREVIEWS = [
  { id: 'idle', name: 'Idle', clip: 'idle' },
  { id: 'walk', name: 'Walk', clip: 'walk' },
  { id: 'run', name: 'Run', clip: 'run' },
  { id: 'slide', name: 'Slide', clip: 'slide_loop' },
];

export const getDance = (id) => DANCES.find((d) => d.id === id) || null;

// ── easing ──────────────────────────────────────────────────────────────────
const EASE = {
  linear: (t) => t,
  smooth: (t) => t * t * (3 - 2 * t),
  snap: (t) => {
    if (t < 0.78) return 0;
    const u = (t - 0.78) / 0.22;
    return u * u * (3 - 2 * u);
  },
  pop: (t) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    const u = t - 1;
    return 1 + c3 * u * u * u + c1 * u * u;
  },
  hold: () => 0,
  out: (t) => 1 - (1 - t) * (1 - t),
  in: (t) => t * t,
};

const WAVES = {
  sin: (a) => Math.sin(a),
  cos: (a) => Math.cos(a),
  square: (a) => (Math.sin(a) >= 0 ? 1 : -1),
  tri: (a) => (2 / Math.PI) * Math.asin(Math.sin(a)),
  pulse: (a) => Math.max(0, Math.sin(a)) ** 2,
};

const TAU = Math.PI * 2;

function bracket(keys, beat, beats) {
  const n = keys.length;
  let lo = n - 1;
  for (let i = 0; i < n; i++) if (keys[i].b <= beat + 1e-9) lo = i;
  const hi = (lo + 1) % n;
  const b0 = keys[lo].b;
  const b1 = hi === 0 ? beats : keys[hi].b;
  return { k0: keys[lo], k1: keys[hi], b0, b1 };
}

const lerp3 = (a, b, t, out = [0, 0, 0]) => {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
};

const IDENTITY3 = [0, 0, 0];

/**
 * Sampling pose pada fase `beat`.
 * → { groups, root, feet, footAngle, yaw, ik, groundLock, section, keyLabel }
 */
export function samplePose(dance, beat, opts = {}) {
  const beats = dance.beats;
  const b = ((beat % beats) + beats) % beats;
  const { k0, k1, b0, b1 } = bracket(dance.keys, b, beats);
  const raw = b1 > b0 ? (b - b0) / (b1 - b0) : 0;
  const t = (EASE[k1.ease] || EASE.smooth)(Math.min(1, Math.max(0, raw)));

  // torso
  const groups = {};
  const all = new Set([...Object.keys(k0.torso), ...Object.keys(k1.torso)]);
  for (const g of all) {
    groups[g] = lerp3(k0.torso[g] || IDENTITY3, k1.torso[g] || IDENTITY3, t);
  }

  const root = lerp3(k0.root, k1.root, t);
  // kalau panggul turun, geser sedikit ke depan supaya berat badan tetap
  // di atas kaki (tanpa ini karakter terlihat "duduk" ke belakang)
  root[2] -= Math.max(0, -root[1]) * 0.34;
  // unwrap yaw: 360° → 0° harus dianggap "tidak berputar", bukan putar balik
  let yaw1 = k1.yaw;
  while (yaw1 - k0.yaw > 180) yaw1 -= 360;
  while (yaw1 - k0.yaw < -180) yaw1 += 360;
  const yaw = k0.yaw + (yaw1 - k0.yaw) * t;
  const yawSweep = Math.abs(((yaw1 - k0.yaw) * Math.PI) / 180);

  // kaki — dengan "auto step": kalau telapak berpindah tempat (atau tersapu
  // putaran badan), ia diangkat mengikuti busur supaya tidak menyeret lantai.
  const feet = {};
  const stance = dance.stance || { L: [-0.15, 0.079, 0.03], R: [0.15, 0.079, 0.03] };
  for (const side of ['L', 'R']) {
    const a = k0.feet?.[side] || IDENTITY3;
    const c = k1.feet?.[side] || IDENTITY3;
    feet[side] = lerp3(a, c, t);
    if (!opts.noAutoStep) {
      const arcT = Math.min(1, Math.max(0, t)); // easing 'pop' boleh overshoot, busur tidak
      const radius = Math.hypot(stance[side][0] + c[0], stance[side][2] + c[2]);
      // (arc putaran badan ikut dihitung supaya kaki terangkat saat pivot)
      const travel = Math.hypot(c[0] - a[0], c[2] - a[2]) + yawSweep * radius;
      if (travel > 0.02) {
        feet[side][1] += Math.sin(Math.PI * arcT) * Math.min(0.26, 0.03 + travel * 0.7);
      }
    }
    // lantai adalah batas keras: tumit/telapak tidak boleh tenggelam
    feet[side][1] = Math.max(0, feet[side][1]);
  }
  const footAngle = {};
  for (const side of ['L', 'R']) {
    footAngle[side] = lerp3(k0.footAngle?.[side] || IDENTITY3, k1.footAngle?.[side] || IDENTITY3, t);
  }

  // postur dasar: lawan "sit-back" alami saat lutut sedikit tertekuk
  const posture = dance.posture ?? { hips: -1.5, spine2: -1.5, chest: -3, head: 2.5 };
  for (const [g, v] of Object.entries(posture)) {
    groups[g] ||= [0, 0, 0];
    groups[g][0] += v;
  }

  // osilator sekunder
  if (dance.osc && !opts.muteOsc) {
    for (const o of dance.osc) {
      const wave = WAVES[o.wave] || WAVES.sin;
      const angle = TAU * (o.freq * b + (o.phase || 0));
      const g = groups[o.group] || (groups[o.group] = [0, 0, 0]);
      g[o.axis] += wave(angle) * o.amp * (opts.oscScale ?? 1);
    }
  }

  let section = dance.sections?.[0]?.label || '';
  if (dance.sections) for (const s of dance.sections) if (s.b <= b) section = s.label;

  return {
    groups,
    root,
    feet,
    footAngle,
    yaw,
    ik: !!dance.ik,
    groundLock: !dance.ik && dance.groundLock !== false,
    section,
    keyLabel: k0.label || '',
    nextLabel: k1.label || '',
    progress: t,
    beat: b,
  };
}
