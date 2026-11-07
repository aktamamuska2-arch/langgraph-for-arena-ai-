/**
 * raster.js — rasterizer + PNG writer minimal (tanpa dependensi).
 * Dipakai hanya oleh evaluator untuk membuat contact-sheet pose.
 */
import zlib from 'node:zlib';

export class Canvas {
  constructor(w, h, bg = [10, 13, 20]) {
    this.w = w;
    this.h = h;
    this.buf = new Uint8Array(w * h * 3);
    this.fill(bg);
  }

  fill([r, g, b]) {
    for (let i = 0; i < this.w * this.h; i++) {
      this.buf[i * 3] = r;
      this.buf[i * 3 + 1] = g;
      this.buf[i * 3 + 2] = b;
    }
  }

  px(x, y, [r, g, b], alpha = 1) {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    const i = (y * this.w + x) * 3;
    if (alpha >= 1) {
      this.buf[i] = r;
      this.buf[i + 1] = g;
      this.buf[i + 2] = b;
      return;
    }
    this.buf[i] += (r - this.buf[i]) * alpha;
    this.buf[i + 1] += (g - this.buf[i + 1]) * alpha;
    this.buf[i + 2] += (b - this.buf[i + 2]) * alpha;
  }

  /** Garis tebal (rasterisasi berbasis sampling titik terdekat). */
  line(x0, y0, x1, y1, color, width = 2, alpha = 1) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) * 1.5));
    const r = width / 2;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const cx = x0 + dx * t;
      const cy = y0 + dy * t;
      for (let oy = -Math.ceil(r); oy <= Math.ceil(r); oy++) {
        for (let ox = -Math.ceil(r); ox <= Math.ceil(r); ox++) {
          const d = Math.hypot(ox, oy);
          if (d > r) continue;
          this.px(cx + ox, cy + oy, color, alpha * Math.min(1, (r - d + 0.6) / 1.2));
        }
      }
    }
  }

  circle(cx, cy, radius, color, alpha = 1) {
    for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
        const d = Math.hypot(x, y);
        if (d > radius) continue;
        this.px(cx + x, cy + y, color, alpha * Math.min(1, radius - d + 0.5));
      }
    }
  }

  rect(x, y, w, h, color, alpha = 1) {
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) this.px(x + i, y + j, color, alpha);
  }

  /** Kotak teks sederhana: bar sebagai "teks" agar label tetap terbaca tanpa font. */
  tint(x, y, w, h, color, alpha = 0.25) {
    this.rect(x, y, w, h, color, alpha);
  }

  toPNG() {
    const raw = Buffer.alloc((this.w * 3 + 1) * this.h);
    for (let y = 0; y < this.h; y++) {
      raw[y * (this.w * 3 + 1)] = 0;
      Buffer.from(this.buf.subarray(y * this.w * 3, (y + 1) * this.w * 3)).copy(
        raw,
        y * (this.w * 3 + 1) + 1,
      );
    }
    const idat = zlib.deflateSync(raw, { level: 9 });
    const chunk = (type, data) => {
      const len = Buffer.alloc(4);
      len.writeUInt32BE(data.length);
      const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
      const crc = Buffer.alloc(4);
      crc.writeUInt32BE(crc32(body) >>> 0);
      return Buffer.concat([len, body, crc]);
    };
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(this.w, 0);
    ihdr.writeUInt32BE(this.h, 4);
    ihdr[8] = 8;
    ihdr[9] = 2;
    return Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', idat),
      chunk('IEND', Buffer.alloc(0)),
    ]);
  }
}

let CRC_TABLE = null;
function crc32(buf) {
  if (!CRC_TABLE) {
    CRC_TABLE = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return c ^ 0xffffffff;
}
