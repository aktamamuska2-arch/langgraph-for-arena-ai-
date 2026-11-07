/**
 * motionQA.js — evaluasi kualitas gerak.
 *
 * Dipakai dua tempat:
 *  - di browser (panel "Motion QA" live, window bergulir)
 *  - di Node (eval/evaluate.js, satu loop penuh per dance → laporan iteratif)
 *
 * Metrik sengaja dipilih supaya setiap masalah punya angka yang bisa ditindak:
 *   energy      → tarian terlalu kaku / terlalu liar
 *   footSlip    → kaki meluncur di lantai (kontak tapi bergerak horizontal)
 *   floor       → kaki menembus lantai
 *   loopSeam    → loop tidak nyambung (pose akhir ≠ pose awal)
 *   balance     → berat badan keluar dari bidang tumpuan
 *   beatSync    → aksen gerak jatuh di ketukan atau tidak
 *   jerk        → gerak patah / jitter
 *   rom         → rentang gerak sendi utama (terlalu kecil = kaku)
 */

const KEY_JOINTS = [
  'head', 'handL', 'handR', 'footL', 'footR', 'toeL', 'toeR', 'hips', 'chest',
  'elbowL', 'elbowR', 'kneeL', 'kneeR', 'thighL', 'thighR', 'spine1',
];

const CONTACT = 0.055; // tinggi maksimum ujung kaki untuk dianggap menapak (meter)

const clamp01 = (v) => Math.max(0, Math.min(1, v));

export class MotionQA {
  constructor({ fps = 60, beats = 1, bpm = 120, danceId = null } = {}) {
    this.fps = fps;
    this.beats = beats;
    this.bpm = bpm;
    this.danceId = danceId;
    this.contactOffset = 0.026; // tinggi bone ujung kaki saat telapak menempel lantai
    this.frames = [];
    this.reset();
  }

  reset() {
    this.frames = [];
    this.stats = {
      speedSum: 0,
      speedMax: 0,
      n: 0,
      jerkSum: 0,
      jerkN: 0,
      slipSum: 0,
      slipN: 0,
      minContact: Infinity,
      maxHipY: -Infinity,
      minHipY: Infinity,
      balanceSum: 0,
      balanceN: 0,
      ranges: {},
      beatSpeedSamples: [],
    };
    this._prevVel = null;
  }

  push(t, beat, joints) {
    const keys = {};
    for (const g of KEY_JOINTS) if (joints[g]) keys[g] = joints[g];

    let speed = 0;
    if (this.last) {
      const dt = Math.max(1e-4, t - this.last.t);
      for (const g of KEY_JOINTS) {
        const a = this.last.joints[g];
        const b = keys[g];
        if (!a || !b) continue;
        speed += a.distanceTo(b) / dt;
      }
      speed /= KEY_JOINTS.length;
    }

    const s = this.stats;
    s.speedSum += speed;
    s.speedMax = Math.max(s.speedMax, speed);
    s.n += 1;

    // slip: PERGELANGAN kaki bergeser horizontal sementara telapak masih menapak.
    // (Pakai pergelangan, bukan ujung jari, supaya pivot/putar-di-tempat tidak
    //  dihitung sebagai slaid.)
    if (this.last) {
      const dt = Math.max(1e-4, t - this.last.t);
      for (const side of ['L', 'R']) {
        const foot = keys['foot' + side];
        const toe = keys['toe' + side];
        const prevFoot = this.last.joints['foot' + side];
        if (!foot || !toe || !prevFoot) continue;
        if (toe.y < 0.055) {
          s.slipSum += Math.hypot(foot.x - prevFoot.x, foot.z - prevFoot.z) / dt;
          s.slipN += 1;
        }
      }
    }

    // penetrasi lantai
    const contactY = Math.min(keys.toeL?.y ?? 1, keys.toeR?.y ?? 1) - this.contactOffset;
    s.minContact = Math.min(s.minContact, contactY);

    // bounce panggul (untuk deteksi irama)
    if (keys.hips) {
      s.maxHipY = Math.max(s.maxHipY, keys.hips.y);
      s.minHipY = Math.min(s.minHipY, keys.hips.y);
    }

    // keseimbangan: panggul vs titik tengah kaki
    if (keys.hips && keys.footL && keys.footR) {
      const midX = (keys.footL.x + keys.footR.x) / 2;
      const midZ = (keys.footL.z + keys.footR.z) / 2;
      s.balanceSum += Math.hypot(keys.hips.x - midX, keys.hips.z - midZ);
      s.balanceN += 1;
    }

    // jerk (turunan percepatan) untuk sendi paling terlihat
    if (this.last && this._prevVel) {
      const dt = Math.max(1e-4, t - this.last.t);
      for (const g of ['head', 'handL', 'handR']) {
        const a = this.last.joints[g];
        const b = keys[g];
        if (!a || !b) continue;
        const vx = (b.x - a.x) / dt;
        const vy = (b.y - a.y) / dt;
        const vz = (b.z - a.z) / dt;
        const pv = this._prevVel[g];
        if (pv) {
          s.jerkSum += Math.hypot(vx - pv[0], vy - pv[1], vz - pv[2]) / dt;
          s.jerkN += 1;
        }
      }
    }

    // rentang gerak
    for (const g of ['handL', 'handR', 'head', 'footL', 'footR', 'kneeL', 'kneeR']) {
      const r = (s.ranges[g] ||= { min: Infinity, max: -Infinity });
      const y = keys[g]?.y;
      if (y !== undefined) {
        r.min = Math.min(r.min, y);
        r.max = Math.max(r.max, y);
      }
    }

    if (this.last) {
      const dt = Math.max(1e-4, t - this.last.t);
      const vel = {};
      for (const g of ['head', 'handL', 'handR']) {
        const a = this.last.joints[g];
        const b = keys[g];
        vel[g] = a && b ? [(b.x - a.x) / dt, (b.y - a.y) / dt, (b.z - a.z) / dt] : null;
      }
      this._prevVel = vel;
      s.beatSpeedSamples.push({ t, beat, speed });
    }

    this.last = { t, beat, joints: keys, speed };
    // PENTING: rig memakai buffer ganda (dua objek Vector3 dipakai ulang tiap
    // frame), jadi frame historis WAJIB di-snapshot — kalau menyimpan referensi,
    // seluruh riwayat menunjuk pose terakhir dan metrik loop/dinamika jadi palsu.
    const snap = {};
    for (const g in keys) {
      const v = keys[g];
      snap[g] = { x: v.x, y: v.y, z: v.z };
    }
    this.frames.push({ t, beat, joints: snap, speed });
    if (this.frames.length > 3000) this.frames.shift();
  }

  /** Ringkasan + skor 0-100 per aspek. */
  summary() {
    const s = this.stats;
    if (!s.n) return null;
    const energy = s.speedSum / s.n;
    const jerk = s.jerkN ? s.jerkSum / s.jerkN : 0;
    const slip = s.slipN ? s.slipSum / s.slipN : 0;
    const balance = s.balanceN ? s.balanceSum / s.balanceN : 0;
    const bounce = s.maxHipY - s.minHipY;
    const rom = {};
    for (const [g, r] of Object.entries(s.ranges)) rom[g] = r.max - r.min;

    const seam = this._loopSeam();
    const { rate: beatSync, offsetMs } = this._beatSync();

    // dinamika: seberapa jauh panggul memantul DI DALAM satu ketukan, dan
    // seberapa kontras puncak kecepatan terhadap rata-rata (aksen vs mengalir)
    const perBeat = new Map();
    let speedSum = 0;
    const speeds = [];
    for (const fr of this.frames) {
      const y = fr.joints.hips.y;
      const k = Math.floor(fr.beat);
      const cur = perBeat.get(k);
      if (!cur) perBeat.set(k, { min: y, max: y });
      else {
        if (y < cur.min) cur.min = y;
        if (y > cur.max) cur.max = y;
      }
      speedSum += fr.speed;
      speeds.push(fr.speed);
    }
    let bouncePerBeat = 0;
    for (const v of perBeat.values()) bouncePerBeat += v.max - v.min;
    if (perBeat.size) bouncePerBeat /= perBeat.size;
    speeds.sort((a, b) => a - b);
    const meanSpeed = speeds.length ? speedSum / speeds.length : 0;
    const p95 = speeds.length ? speeds[Math.min(speeds.length - 1, Math.floor(speeds.length * 0.95))] : 0;
    const contrast = meanSpeed > 1e-6 ? p95 / meanSpeed : 0;
    const dynamism =
      clamp01((bouncePerBeat - 0.015) / 0.06) * 0.65 + clamp01((contrast - 1.3) / 2.0) * 0.35;

    // amplitudo: seberapa besar lengan & kepala benar-benar bergerak
    const armRange = (rom.handL ?? 0) + (rom.handR ?? 0);
    const headRange = rom.head ?? 0;
    const amplitude =
      clamp01((armRange - 0.35) / 1.1) * 0.7 + clamp01((headRange - 0.04) / 0.28) * 0.3;

    const scores = {
      activity: Math.round(clamp01(energy / 0.42) * 100),
      amplitude: Math.round(amplitude * 100),
      smoothness: Math.round(clamp01(1 - jerk / 900) * 100),
      footwork: Math.round(clamp01(1 - slip / 0.55) * 100),
      balance: Math.round(clamp01(1 - balance / 0.3) * 100),
      loop: Math.round(clamp01(1 - seam / 0.12) * 100),
      rhythm: Math.round(beatSync * 100),
      dynamics: Math.round(dynamism * 100),
    };
    const vals = Object.values(scores).filter((v) => Number.isFinite(v));
    const overall = Math.round(vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length));

    return {
      danceId: this.danceId,
      bpm: this.bpm,
      beats: this.beats,
      frames: s.n,
      energy: +energy.toFixed(3),
      jerk: +jerk.toFixed(1),
      footSlip: +slip.toFixed(3),
      floorMin: +s.minContact.toFixed(4),
      balance: +balance.toFixed(3),
      bounce: +bounce.toFixed(3),
      loopSeam: +seam.toFixed(4),
      beatSync: +beatSync.toFixed(2),
      offsetMs: Math.round(offsetMs),
      bouncePerBeat: +bouncePerBeat.toFixed(3),
      contrast: +contrast.toFixed(2),
      rom,
      scores,
      overall,
    };
  }

  _loopSeam() {
    const f = this.frames;
    if (f.length < 4) return 0;
    const target = f[0].beat;
    const span = this.beats;
    // Cari frame yang fasenya sama dengan frame pertama, TAPI satu (atau lebih)
    // putaran kemudian: beat ≈ target + k·span. Membandingkan dengan frame
    // terakhir apa adanya justru mengukur fase yang berbeda.
    let best = null;
    let bestD = Infinity;
    for (let k = 1; k <= 4; k++) {
      const want = target + k * span;
      for (const fr of f) {
        const d = Math.abs(fr.beat - want);
        if (d < bestD) {
          bestD = d;
          best = fr;
        }
      }
      if (bestD < 0.5) break;
    }
    if (!best || bestD > 0.5) return 0; // belum ada putaran penuh terekam
    let sum = 0;
    let n = 0;
    for (const g of KEY_JOINTS) {
      const a = f[0].joints[g];
      const b = best.joints[g];
      if (!a || !b) continue;
      sum += Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      n++;
    }
    return n ? sum / n : 0;
  }

  /**
   * Sinkronisasi ketukan: cari puncak percepatan (aksen gerak) di sekitar
   * tiap ketukan. Skor = fraksi ketukan yang punya aksen; offsetMs =
   * rata-rata jarak aksen ke ketukan (negatif = gerak mendahului musik).
   */
  _beatSync() {
    const f = this.frames;
    if (f.length < 12) return { rate: 0, offsetMs: 0 };
    const accel = [];
    let sum = 0;
    for (let i = 1; i < f.length; i++) {
      const dt = Math.max(1e-4, f[i].t - f[i - 1].t);
      const a = Math.abs((f[i].speed ?? 0) - (f[i - 1].speed ?? 0)) / dt;
      accel.push({ t: f[i].t, beat: f[i].beat, a });
      sum += a;
    }
    const mean = sum / Math.max(1, accel.length);
    if (mean <= 0) return { rate: 0, offsetMs: 0 };

    const beatDur = 60 / this.bpm;
    const startBeat = f[0].beat;
    const t0 = f[0].t;
    let hits = 0;
    let count = 0;
    let offsetSum = 0;
    for (let n = 0; n < this.beats; n++) {
      const targetBeat = startBeat + n;
      const center = t0 + (targetBeat - startBeat) * beatDur;
      if (center > f[f.length - 1].t) break;
      let best = { a: 0, t: center };
      for (const s of accel) {
        if (Math.abs(s.t - center) <= beatDur * 0.28 && s.a > best.a) best = s;
      }
      if (best.a > mean * 1.55) {
        hits += 1;
        offsetSum += (best.t - center) * 1000;
      }
      count += 1;
    }
    return { rate: count ? hits / count : 0, offsetMs: hits ? offsetSum / hits : 0 };
  }
}
