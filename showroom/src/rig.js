/**
 * rig.js — lapisan "additive world-space" di atas AnimationMixer.
 *
 * Model humanoid_basic_animation_set.glb memakai rig Rigify (bone DEF-*).
 * Orientasi rest-nya tidak ramah untuk animasi tangan (T-pose dengan sumbu
 * bone Blender), jadi seluruh offset koreografi di sini dinyatakan dalam
 * SUMBU DUNIA (scene): X = kanan-kiri, Y = atas-bawah, Z = depan-belakang.
 *
 * Rumus aditif yang dipakai:
 *   Lb' = Bp⁻¹ · O · Bp · Lb
 * dengan Bp = world quaternion parent saat pose dasar (hasil mixer),
 * Lb = local quaternion bone saat pose dasar, O = rotasi offset di ruang dunia.
 * Turunannya: world baru bone = Δ · world dasar, dan anak-anaknya otomatis
 * mewarisi Δ tanpa perlu ditelusuri top-down.
 */
import * as THREE from 'three';

/** Nama grup anatomis → nama bone (sudah disanitasi oleh GLTFLoader). */
export const BONE_GROUPS = {
  root: 'root_01',
  hips: 'DEF-spine_02',
  spine1: 'DEF-spine001_03',
  spine2: 'DEF-spine002_04',
  chest: 'DEF-spine003_05',
  chest2: 'DEF-spine004_06',
  neck: 'DEF-neck_07',
  head: 'DEF-head_08',
  shoulderL: 'DEF-shoulderL_09',
  armL: 'DEF-upper_armL_010',
  elbowL: 'DEF-forearmL_011',
  handL: 'DEF-handL_012',
  shoulderR: 'DEF-shoulderR_032',
  armR: 'DEF-upper_armR_033',
  elbowR: 'DEF-forearmR_034',
  handR: 'DEF-handR_035',
  thighL: 'DEF-thighL_055',
  kneeL: 'DEF-shinL_056',
  footL: 'DEF-footL_057',
  toeL: 'DEF-toeL_058',
  thighR: 'DEF-thighR_059',
  kneeR: 'DEF-shinR_060',
  footR: 'DEF-footR_061',
  toeR: 'DEF-toeR_00',
};

export const LEG_GROUPS = ['thighL', 'kneeL', 'footL', 'thighR', 'kneeR', 'footR'];
export const CONTACT_GROUPS = ['footL', 'toeL', 'footR', 'toeR'];

/**
 * Grup yang porosnya lebih intuitif di ruang BONE (engsel anatomis):
 * siku & lutut hanya punya satu derajat kebebasan, dan porosnya ikut
 * berputar bersama lengan/tungkai induknya.
 */
export const LOCAL_SPACE_GROUPS = new Set(['elbowL', 'elbowR', 'kneeL', 'kneeR', 'handL', 'handR']);

const sanitize = (s) => String(s).replace(/[^a-zA-Z0-9_]/g, '').toLowerCase();

const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _v1 = new THREE.Vector3();
const _s1 = new THREE.Vector3();
const _s2 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _tmpQ = new THREE.Quaternion();
const _m2 = new THREE.Matrix4();
const IDENTITY = new THREE.Quaternion();
const UP = new THREE.Vector3(0, 1, 0);

/**
 * Ubah derajat [x,y,z] (world) menjadi quaternion: yaw · pitch · roll.
 * Memakai scratch internal, jadi `out` boleh sama dengan variabel pemanggil.
 */
const _AXIS_Y = new THREE.Vector3(0, 1, 0);
const _AXIS_X = new THREE.Vector3(1, 0, 0);
const _AXIS_Z = new THREE.Vector3(0, 0, 1);
const _qs = new THREE.Quaternion();

function quatFromDegrees(rot, out = new THREE.Quaternion()) {
  const [x, y, z] = rot || [0, 0, 0];
  out.identity();
  if (y) out.multiply(_qs.setFromAxisAngle(_AXIS_Y, THREE.MathUtils.degToRad(y)));
  if (x) out.multiply(_qs.setFromAxisAngle(_AXIS_X, THREE.MathUtils.degToRad(x)));
  if (z) out.multiply(_qs.setFromAxisAngle(_AXIS_Z, THREE.MathUtils.degToRad(z)));
  return out;
}

export class Rig {
  constructor(root) {
    this.root = root;
    this.bones = {};
    this.entries = [];
    this.missing = [];

    const bySanitized = new Map();
    root.traverse((o) => {
      if (o.isBone) bySanitized.set(sanitize(o.name), o);
    });

    for (const [group, name] of Object.entries(BONE_GROUPS)) {
      const bone = bySanitized.get(sanitize(name)) || root.getObjectByName(name);
      if (!bone) {
        this.missing.push(`${group}:${name}`);
        continue;
      }
      this.bones[group] = bone;
      this.entries.push({
        group,
        bone,
        parent: bone.parent,
        restQ: bone.quaternion.clone(),
        restPos: bone.position.clone(),
        baseQ: bone.quaternion.clone(),
        basePos: bone.position.clone(),
        baseWorldQ: new THREE.Quaternion(),
        parentWorldQ: new THREE.Quaternion(),
      });
    }
    this.byGroup = new Map(this.entries.map((e) => [e.group, e]));

    // Kalibrasi: jarak vertikal titik kontak kaki ke lantai saat bind pose.
    root.updateMatrixWorld(true);
    this.contactFloorOffset = this.lowestContactY();
  }

  /** Kembalikan semua bone grup ke pose referensi (bind). */
  resetToRest() {
    for (const e of this.entries) {
      e.bone.quaternion.copy(e.restQ);
      e.bone.position.copy(e.restPos);
    }
  }

  /** Ambil pose dasar (hasil mixer) + world quaternion parent-nya. */
  captureBase() {
    this.root.updateMatrixWorld(true);
    for (const e of this.entries) {
      e.baseQ.copy(e.bone.quaternion);
      e.basePos.copy(e.bone.position);
      e.bone.matrixWorld.decompose(_v2, e.baseWorldQ, _s1);
      e.baseWorldQ.normalize();
      // decompose (bukan setFromRotationMatrix) karena bone punya skala non-uniform
      e.parent.matrixWorld.decompose(_v1, e.parentWorldQ, _s2);
      e.parentWorldQ.normalize();
    }
  }

  /**
   * Terapkan offset rotasi [x,y,z] derajat pada satu grup.
   * space 'world' → poros mengikuti sumbu scene (cocok untuk torso/lengan).
   * space 'local' → poros mengikuti sumbu bone (cocok untuk engsel siku/lutut).
   */
  applyRotation(entry, rot, weight = 1, space = 'world') {
    if (!entry) return;
    const q = quatFromDegrees(rot, _q2);
    if (weight < 1) q.slerp(IDENTITY, 1 - weight);
    if (space === 'local') {
      entry.bone.quaternion.copy(entry.baseQ).multiply(q).normalize();
      return;
    }
    entry.bone.quaternion
      .copy(entry.parentWorldQ)
      .invert()
      .multiply(q)
      .multiply(entry.parentWorldQ)
      .multiply(entry.baseQ)
      .normalize();
  }

  /**
   * Auto "telapak kaki rata": setelah tungkai diputar, hitung delta dunia
   * telapak lalu balikkan tepat sebesar comp (0..1). Karena rotasi offset
   * bersifat world-space, O = D⁻¹ membuat telapak kembali ke orientasi dasar.
   */
  compensateFeet(comp = 0.9, weight = 1) {
    const q = new THREE.Quaternion();
    for (const side of ['L', 'R']) {
      const entry = this.byGroup.get('foot' + side);
      if (!entry) continue;
      entry.bone.getWorldQuaternion(q);
      // D = Qsekarang · Qdasar⁻¹
      const D = q.clone().multiply(entry.baseWorldQ.clone().invert()).invert();
      if (weight < 1) D.slerp(IDENTITY, 1 - weight);
      else D.slerp(IDENTITY, 1 - comp);
      const local = entry.parentWorldQ
        .clone()
        .invert()
        .multiply(D)
        .multiply(entry.parentWorldQ)
        .multiply(entry.baseQ)
        .normalize();
      entry.bone.quaternion.copy(local);
    }
  }

  /**
   * Terapkan translasi world (meter) pada satu grup.
   * Parent bone dipakai sebagai ruang lokal, jadi skala/rotasi parent
   * (termasuk scale 0.01 milik model) otomatis ikut diperhitungkan.
   */
  applyTranslation(entry, worldVec, weight = 1) {
    if (!entry) return;
    _v1.copy(entry.basePos).applyMatrix4(entry.parent.matrixWorld); // posisi dunia dasar
    _v1.addScaledVector(worldVec, weight);
    _m2.copy(entry.parent.matrixWorld).invert();
    entry.bone.position.copy(_v1.applyMatrix4(_m2));
  }

  /**
   * Kalibrasi stance: posisi jangkar telapak, panjang tulang tungkai, dan
   * orientasi telapak "rata". Dipanggil sekali setelah mixer memutar frame
   * pertama clip dasar (bukan bind pose yang aneh).
   */
  calibrateStance() {
    this.root.updateMatrixWorld(true);
    this.stance = {};
    this.limb = {};
    for (const side of ['L', 'R']) {
      const thigh = this.bones['thigh' + side];
      const shin = this.bones['knee' + side];
      const foot = this.bones['foot' + side];
      const hip = thigh.getWorldPosition(new THREE.Vector3());
      const knee = shin.getWorldPosition(new THREE.Vector3());
      const ankle = foot.getWorldPosition(new THREE.Vector3());
      const toe = this.bones['toe' + side]?.getWorldPosition(new THREE.Vector3()) ?? ankle;
      this.limb[side] = {
        L1: hip.distanceTo(knee),
        L2: knee.distanceTo(ankle),
        footLen: Math.max(0.06, ankle.distanceTo(toe)),
        pole: new THREE.Vector3(side === 'L' ? -0.28 : 0.28, 0, -1).normalize(),
      };
      const toeBone = this.bones['toe' + side];
      this.stance[side] = {
        pos: ankle.clone(),
        quat: foot.getWorldQuaternion(new THREE.Quaternion()),
        toeQuat: toeBone ? toeBone.getWorldQuaternion(new THREE.Quaternion()) : null,
      };
    }
  }

  /** Titik jangkar telapak (dunia) setelah diputar yaw badan. */
  _footTarget(side, offset, yawDeg) {
    const base = this.stanceOverride?.[side];
    if (base) _a.set(base[0], base[1], base[2]);
    else _a.copy(this.stance[side].pos);
    _a.add(_b.set(offset?.[0] ?? 0, offset?.[1] ?? 0, offset?.[2] ?? 0));
    if (yawDeg) _a.applyAxisAngle(UP, THREE.MathUtils.degToRad(yawDeg));
    return _a;
  }

  /**
   * IK dua-tulang untuk kedua tungkai. Telapak ditaruh persis di target,
   * lutut diarahkan ke pole (depan + sedikit keluar), lalu telapak dibuat
   * rata (atau miring sesuai `footAngle` untuk toe-off/heel-lift).
   */
  applyLegIK(feet = {}, yawDeg = 0, weight = 1, footAngle = {}) {
    if (!this.stance) this.calibrateStance();
    const vHip = new THREE.Vector3();
    const vKnee = new THREE.Vector3();
    const vAnkle = new THREE.Vector3();
    const vTarget = new THREE.Vector3();
    const qTmp = new THREE.Quaternion();

    for (const side of ['L', 'R']) {
      const thighE = this.byGroup.get('thigh' + side);
      const shinE = this.byGroup.get('knee' + side);
      const footE = this.byGroup.get('foot' + side);
      const toeE = this.byGroup.get('toe' + side);
      if (!thighE || !shinE || !footE) continue;
      const pole = this.limb[side].pole;
      const baseTarget = this._footTarget(side, feet[side], yawDeg).clone();
      const goalY = (this.stanceOverride?.[side]?.[1] ?? this.stance[side].pos.y) - 0.002;
      if (baseTarget.y < goalY) baseTarget.y = goalY;

      const angX = THREE.MathUtils.degToRad(footAngle[side]?.[0] ?? 0);
      const footLen = this.limb[side].footLen;
      baseTarget.y += Math.max(0, -angX) * footLen + Math.max(0, angX) * footLen * 0.45;

      // Skala bone non-uniform membuat rotasi anak menghasilkan sedikit shear,
      // jadi solusi diiterasi 2x memakai error posisi pergelangan yang terukur.
      let correction = new THREE.Vector3();
      for (let iter = 0; iter < 2; iter++) {
        vTarget.copy(baseTarget).add(correction);

        thighE.bone.getWorldPosition(vHip);
        shinE.bone.getWorldPosition(vKnee);
        footE.bone.getWorldPosition(vAnkle);
        const L1 = vHip.distanceTo(vKnee);
        const L2 = vKnee.distanceTo(vAnkle);

        // geometri lutut (law of cosines + pole vector)
        const toTarget = vTarget.clone().sub(vHip);
        let dist = toTarget.length();
        dist = THREE.MathUtils.clamp(dist, Math.abs(L1 - L2) + 0.04, L1 + L2 - 0.004);
        const axis = toTarget.normalize();
        const aLen = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist);
        const hLen = Math.sqrt(Math.max(0, L1 * L1 - aLen * aLen));
        const perp = pole.clone().addScaledVector(axis, -pole.dot(axis));
        if (perp.lengthSq() < 1e-8) perp.set(0, 0, -1);
        perp.normalize();
        const kneeAt = vHip.clone().addScaledVector(axis, aLen).addScaledVector(perp, hLen);

        // arahkan tulang paha
        const thighDirNow = vKnee.clone().sub(vHip).normalize();
        const q1 = new THREE.Quaternion().setFromUnitVectors(thighDirNow, kneeAt.clone().sub(vHip).normalize());
        const thighWorld = q1.clone().multiply(thighE.bone.getWorldQuaternion(new THREE.Quaternion()));

        // arahkan tulang kering
        const shinDirAfter = vAnkle.clone().sub(vKnee).normalize().applyQuaternion(q1);
        const q2 = new THREE.Quaternion().setFromUnitVectors(shinDirAfter, vTarget.clone().sub(kneeAt).normalize());
        const shinWorld = q2.multiply(q1).multiply(shinE.bone.getWorldQuaternion(new THREE.Quaternion()));

        // telapak rata (orientasi stance) + sudut toe-off, ikut yaw badan
        const footWorld = this.stance[side].quat
          .clone()
          .multiply(quatFromDegrees(footAngle[side] || [0, 0, 0], _q1))
          .premultiply(qTmp.setFromAxisAngle(UP, THREE.MathUtils.degToRad(yawDeg)));

        const parentQ = new THREE.Quaternion();
        thighE.parent.matrixWorld.decompose(new THREE.Vector3(), parentQ, new THREE.Vector3());
        const thighLocal = parentQ.invert().multiply(thighWorld).normalize();
        const shinLocal = thighWorld.clone().invert().multiply(shinWorld).normalize();
        const footLocal = shinWorld.clone().invert().multiply(footWorld).normalize();

        thighE.bone.quaternion.copy(thighE.bone.quaternion).slerp(thighLocal, weight);
        shinE.bone.quaternion.copy(shinE.bone.quaternion).slerp(shinLocal, weight);
        footE.bone.quaternion.copy(footE.bone.quaternion).slerp(footLocal, weight);
        if (toeE) {
          const toeWorld = this.stance[side].toeQuat
            ? this.stance[side].toeQuat.clone().premultiply(
                qTmp.setFromAxisAngle(UP, THREE.MathUtils.degToRad(yawDeg)),
              )
            : null;
          if (toeWorld) {
            const toeLocal = footWorld.clone().invert().multiply(toeWorld).normalize();
            toeE.bone.quaternion.copy(toeE.bone.quaternion).slerp(toeLocal, weight);
          }
        }
        thighE.bone.updateWorldMatrix(true, true);

        // error posisi → koreksi untuk iterasi berikutnya
        footE.bone.getWorldPosition(vAnkle);
        correction.copy(baseTarget).sub(vAnkle);
        if (correction.length() < 0.0015) break;
      }
    }
  }

  /**
   * Terapkan satu pose hasil sampling koreografi.
   * pose = { groups: {name: [x,y,z]}, root: [dx,dy,dz], lift, yaw }
   * weight dipakai untuk crossfade masuk/keluar koreografi.
   */
  applyPose(pose, weight = 1) {
    const groups = pose.groups || {};
    this.resetToReference();

    // 1. rotasi per grup (world-space atau bone-space)
    for (const [group, rot] of Object.entries(groups)) {
      if (group === 'footL' || group === 'footR') continue; // ditangani compensator
      this.applyRotation(
        this.byGroup.get(group),
        rot,
        weight,
        LOCAL_SPACE_GROUPS.has(group) ? 'local' : 'world',
      );
    }

    // 2. yaw seluruh tubuh pada bone root
    const rootEntry = this.byGroup.get('root');
    if (rootEntry && pose.yaw) {
      this.applyRotation(rootEntry, [0, pose.yaw * weight, 0], 1, 'world');
    }
    this.root.updateMatrixWorld(true);

    // 3. translasi panggul (world meter)
    const hips = this.byGroup.get('hips');
    if (hips && pose.root) {
      this.applyTranslation(hips, new THREE.Vector3(...pose.root), weight);
      this.root.updateMatrixWorld(true);
    }

    // 4. tungkai: IK bila mode-nya IK, kalau tidak cukup kompensasi telapak
    if (pose.ik) {
      this.applyLegIK(pose.feet || {}, pose.yaw || 0, weight, pose.footAngle || {});
      this.root.updateMatrixWorld(true);
    } else if (pose.footComp !== 0) {
      this.compensateFeet(pose.footComp ?? 0.9, weight);
      this.root.updateMatrixWorld(true);
    }

    // 5. ground lock — titik kontak terendah selalu di lantai (plus lift untuk lompatan)
    if (pose.groundLock !== false) {
      const delta = (pose.groundY ?? 0.005) - (this.lowestContactY() - this.contactFloorOffset);
      if (rootEntry) {
        this.applyTranslation(rootEntry, new THREE.Vector3(0, delta + (pose.lift || 0) * weight, 0), 1);
        this.root.updateMatrixWorld(true);
      }
    } else if (rootEntry && pose.lift) {
      this.applyTranslation(rootEntry, new THREE.Vector3(0, pose.lift * weight, 0), 1);
      this.root.updateMatrixWorld(true);
    }
  }

  /** Restore ke pose dasar frame ini (hasil mixer) — bukan bind pose. */
  resetToReference() {
    for (const e of this.entries) {
      e.bone.quaternion.copy(e.baseQ);
      e.bone.position.copy(e.basePos);
    }
  }

  lowestContactY() {
    let min = Infinity;
    for (const g of CONTACT_GROUPS) {
      const b = this.bones[g];
      if (!b) continue;
      b.getWorldPosition(_v1);
      if (_v1.y < min) min = _v1.y;
    }
    return min;
  }

  /** Posisi dunia semua grup (untuk QA metrik & debug skeleton). */
  sampleJoints(out = {}) {
    for (const [group, bone] of Object.entries(this.bones)) {
      bone.getWorldPosition(_v1);
      (out[group] ||= new THREE.Vector3()).copy(_v1);
    }
    return out;
  }
}
