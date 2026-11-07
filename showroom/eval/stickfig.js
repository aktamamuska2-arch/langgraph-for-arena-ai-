/**
 * stickfig.js — contact sheet pose sebagai PNG.
 * Render kerangka ke raster sendiri, lalu ImageMagick hanya dipakai untuk teks.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { Canvas } from './raster.js';

const CELL_W = 176;
const CELL_H = 236;
const BAR_H = 52;
const SCALE = 92; // piksel per meter

const CHAINS = [
  ['hips', 'spine1', 'spine2', 'chest', 'chest2', 'neck', 'head', 'headTop'],
  ['chest2', 'shoulderL', 'armL', 'elbowL', 'handL'],
  ['chest2', 'shoulderR', 'armR', 'elbowR', 'handR'],
  ['hips', 'thighL', 'kneeL', 'footL', 'toeL'],
  ['hips', 'thighR', 'kneeR', 'footR', 'toeR'],
];

const hexToRgb = (hex) => [
  parseInt(hex.slice(1, 3), 16),
  parseInt(hex.slice(3, 5), 16),
  parseInt(hex.slice(5, 7), 16),
];

function point(p, view, cx, cy) {
  const [a] = [view === 'front' ? p.x : -p.z];
  return [cx + a * SCALE, cy - p.y * SCALE];
}

function drawFigure(canvas, joints, { view, cx, cy, color }) {
  const rgb = hexToRgb(color);
  const soft = rgb.map((c) => Math.round(c * 0.55 + 20));
  for (const chain of CHAINS) {
    const pts = chain.map((g) => joints[g]).filter(Boolean).map((p) => point(p, view, cx, cy));
    if (pts.length < 2) continue;
    const isTorso = chain[0] === 'hips';
    for (let i = 0; i < pts.length - 1; i++) {
      canvas.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], isTorso ? rgb : soft, isTorso ? 5 : 3.4, 0.95);
    }
  }
  // kepala sebagai massa (biar orientasi terbaca)
  if (joints.head) {
    const [hx, hy] = point(joints.head, view, cx, cy);
    const [nx, ny] = joints.neck ? point(joints.neck, view, cx, cy) : [hx, hy + 12];
    const dx = hx - nx;
    const dy = hy - ny;
    const len = Math.hypot(dx, dy) || 1;
    canvas.circle(hx + (dx / len) * 8, hy + (dy / len) * 8, 9.5, rgb, 0.95);
    canvas.circle(hx + (dx / len) * 8, hy + (dy / len) * 8, 9.5, [235, 245, 255], 0.18);
  }
  for (const g of ['handL', 'handR', 'footL', 'footR']) {
    if (!joints[g]) continue;
    const [x, y] = point(joints[g], view, cx, cy);
    canvas.circle(x, y, 3.6, [245, 250, 255], 0.95);
  }
}

/**
 * frames: [{joints, label}]
 * Menulis PNG ke `outPath`; objek label diteruskan ke ImageMagick.
 */
export function renderSheetPNG({ frames, view, title, subtitle, color, outPath, cols = 8 }) {
  const rows = Math.ceil(frames.length / cols);
  const w = cols * CELL_W;
  const h = rows * CELL_H + BAR_H;
  const canvas = new Canvas(w, h, [10, 13, 20]);

  // header bar & grid
  canvas.tint(0, 0, w, BAR_H, hexToRgb(color).map((c) => Math.round(c * 0.22)), 1);
  canvas.line(0, BAR_H, w, BAR_H, [40, 50, 68], 1.5, 1);

  const labels = [];
  frames.forEach((f, i) => {
    const x0 = (i % cols) * CELL_W;
    const y0 = Math.floor(i / cols) * CELL_H + BAR_H;
    canvas.line(x0, y0, x0, y0 + CELL_H, [26, 34, 46], 1, 1);
    const cx = x0 + CELL_W / 2;
    const cy = y0 + CELL_H - 26;
    // lantai
    canvas.line(x0 + 10, cy, x0 + CELL_W - 10, cy, [52, 64, 84], 1.4, 1);
    // bayangan sederhana
    canvas.tint(cx - 26, cy - 1, 52, 3, [30, 40, 54], 1);
    drawFigure(canvas, f.joints, { view, cx, cy, color });
    labels.push({ text: `b${f.beat ?? i}`, x: x0 + 8, y: y0 + 14, size: 11, fill: '#8ea3bd' });
    if (f.label) labels.push({ text: f.label, x: x0 + 8, y: y0 + 28, size: 10, fill: '#cfe0f5' });
  });

  const tmp = `${outPath}.ppm`;
  fs.writeFileSync(tmp, ppm(canvas));

  const args = [tmp, '-font', 'DejaVu-Sans'];
  args.push('-fill', '#eaf4ff', '-pointsize', '15', '-annotate', `+10+21`, title);
  args.push('-fill', '#9db0c8', '-pointsize', '10', '-annotate', `+10+38`, subtitle);
  for (const l of labels) {
    args.push('-fill', l.fill, '-pointsize', String(l.size), '-annotate', `+${l.x}+${l.y}`, l.text);
  }
  args.push(outPath);
  execFileSync('convert', args, { stdio: 'ignore' });
  fs.unlinkSync(tmp);
  return outPath;
}

function ppm(canvas) {
  const header = Buffer.from(`P6\n${canvas.w} ${canvas.h}\n255\n`, 'ascii');
  return Buffer.concat([header, Buffer.from(canvas.buf)]);
}
