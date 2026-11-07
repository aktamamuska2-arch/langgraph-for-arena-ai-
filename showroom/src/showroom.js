/**
 * showroom.js — lingkungan panggung: lantai reflektif, cincin neon,
 * cyc backdrop, pencahayaan studio, kontrol kamera bebas.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

export const CAMERA_PRESETS = {
  hero: { pos: [2.35, 1.72, 3.15], target: [0, 1.02, 0] },
  front: { pos: [0.05, 1.5, 4.05], target: [0, 1.0, 0] },
  side: { pos: [4.15, 1.45, 0.15], target: [0, 1.0, 0] },
  close: { pos: [1.15, 1.55, 1.65], target: [0, 1.35, 0] },
  top: { pos: [0.35, 4.6, 2.1], target: [0, 0.9, 0] },
};

function gradientTexture(stops, size = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = 16;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, size);
  for (const [stop, color] of stops) grad.addColorStop(stop, color);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 16, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function floorTexture() {
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#080b12';
  ctx.fillRect(0, 0, size, size);

  // radial glow di tengah panggung
  const glow = ctx.createRadialGradient(size / 2, size / 2, 20, size / 2, size / 2, size / 2);
  glow.addColorStop(0, 'rgba(70,110,150,0.55)');
  glow.addColorStop(0.45, 'rgba(24,42,64,0.35)');
  glow.addColorStop(1, 'rgba(4,6,10,0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, size, size);

  // garis grid halus
  ctx.strokeStyle = 'rgba(120,190,230,0.10)';
  ctx.lineWidth = 1.2;
  const step = size / 32;
  for (let i = 0; i <= 32; i++) {
    ctx.beginPath();
    ctx.moveTo(i * step, 0);
    ctx.lineTo(i * step, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * step);
    ctx.lineTo(size, i * step);
    ctx.stroke();
  }

  // cincin penanda
  ctx.strokeStyle = 'rgba(110,240,255,0.30)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.315, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(180,110,255,0.16)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.44, 0, Math.PI * 2);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export function createShowroom(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#05070c');
  scene.fog = new THREE.FogExp2('#05070c', 0.055);

  const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 120);
  camera.position.set(...CAMERA_PRESETS.hero.pos);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...CAMERA_PRESETS.hero.target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.rotateSpeed = 0.85;
  controls.zoomSpeed = 0.9;
  controls.panSpeed = 0.7;
  controls.minDistance = 0.9;
  controls.maxDistance = 9;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minPolarAngle = 0.12;
  controls.autoRotateSpeed = 0.55;
  controls.update();

  // ── environment map untuk refleksi PBR ───────────────────────────────────
  try {
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envScene = new RoomEnvironment();
    scene.environment = pmrem.fromScene(envScene, 0.04).texture;
    scene.environmentIntensity = 0.35;
    envScene.dispose?.();
    pmrem.dispose();
  } catch (err) {
    console.warn('environment map dilewati:', err);
  }

  // ── lantai ───────────────────────────────────────────────────────────────
  const floorMat = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    map: floorTexture(),
    metalness: 0.62,
    roughness: 0.27,
    envMapIntensity: 1.1,
  });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(7.5, 96), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // panggung bundar rendah
  const dais = new THREE.Mesh(
    new THREE.CylinderGeometry(2.05, 2.16, 0.1, 96),
    new THREE.MeshStandardMaterial({ color: '#0d131d', metalness: 0.9, roughness: 0.28 }),
  );
  dais.position.y = -0.052;
  dais.receiveShadow = true;
  scene.add(dais);

  // cincin neon
  const ringMat = new THREE.MeshBasicMaterial({ color: '#4df0ff', toneMapped: false });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(2.06, 0.012, 12, 160), ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.008;
  scene.add(ring);

  const ringOuter = new THREE.Mesh(
    new THREE.TorusGeometry(2.55, 0.008, 10, 160),
    new THREE.MeshBasicMaterial({ color: '#b06bff', transparent: true, opacity: 0.55, toneMapped: false }),
  );
  ringOuter.rotation.x = -Math.PI / 2;
  ringOuter.position.y = 0.006;
  scene.add(ringOuter);

  const ringLight = new THREE.PointLight('#4df0ff', 6, 5.5, 2.2);
  ringLight.position.set(0, 0.5, 0);
  scene.add(ringLight);

  // ── cyc / backdrop ───────────────────────────────────────────────────────
  const backdrop = new THREE.Mesh(
    new THREE.CylinderGeometry(9.5, 9.5, 9, 64, 1, true),
    new THREE.MeshBasicMaterial({
      side: THREE.BackSide,
      map: gradientTexture([
        [0, '#04060a'],
        [0.55, '#070c15'],
        [0.82, '#0d1725'],
        [1, '#05070c'],
      ]),
      fog: false,
    }),
  );
  backdrop.position.y = 3.6;
  scene.add(backdrop);

  // ── pencahayaan ──────────────────────────────────────────────────────────
  scene.add(new THREE.HemisphereLight('#26405c', '#05070a', 1.1));

  // lampu utama: directional → intensitas tidak bergantung jarak, aman di semua skala
  const key = new THREE.DirectionalLight('#eaf4ff', 2.6);
  key.position.set(3.2, 6.4, 3.4);
  key.target.position.set(0, 1.0, 0);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0009;
  key.shadow.normalBias = 0.02;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 20;
  key.shadow.camera.left = -4;
  key.shadow.camera.right = 4;
  key.shadow.camera.top = 4.5;
  key.shadow.camera.bottom = -2;
  scene.add(key, key.target);

  const topSpot = new THREE.SpotLight('#cfe4ff', 110, 12, 0.62, 0.9, 2);
  topSpot.position.set(-1.6, 4.6, -1.4);
  topSpot.target.position.set(0, 1.0, 0);
  scene.add(topSpot, topSpot.target);

  const frontSpot = new THREE.SpotLight('#a9d8ff', 90, 14, 0.55, 0.85, 2);
  frontSpot.position.set(0.8, 4.2, 4.2);
  frontSpot.target.position.set(0, 1.1, 0);
  scene.add(frontSpot, frontSpot.target);

  const rimCyan = new THREE.PointLight('#37e6ff', 26, 7, 2);
  rimCyan.position.set(-2.5, 1.9, -1.9);
  scene.add(rimCyan);

  const rimMagenta = new THREE.PointLight('#c163ff', 24, 7, 2);
  rimMagenta.position.set(2.4, 2.2, -2.1);
  scene.add(rimMagenta);

  const fillWarm = new THREE.PointLight('#ffb37a', 10, 6, 2);
  fillWarm.position.set(0.4, 1.2, 3.2);
  scene.add(fillWarm);

  // ── partikel debu panggung ───────────────────────────────────────────────
  const dustCount = 420;
  const dustGeo = new THREE.BufferGeometry();
  const dustPos = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    const r = 1.4 + Math.random() * 4.4;
    const a = Math.random() * Math.PI * 2;
    dustPos[i * 3] = Math.cos(a) * r;
    dustPos[i * 3 + 1] = Math.random() * 4.2;
    dustPos[i * 3 + 2] = Math.sin(a) * r;
  }
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(
    dustGeo,
    new THREE.PointsMaterial({
      color: '#9fd8ff',
      size: 0.018,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  scene.add(dust);

  const state = {
    scene,
    camera,
    renderer,
    controls,
    key,
    ring,
    ringOuter,
    ringLight,
    rimCyan,
    rimMagenta,
    accent: new THREE.Color('#4df0ff'),
    accent2: new THREE.Color('#b06bff'),
    skeletonHelper: null,
    modelRoot: null,
  };

  /** Pasang model + siapkan material & shadow. */
  state.attachModel = (root) => {
    state.modelRoot = root;
    root.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.receiveShadow = true;
      o.frustumCulled = false;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        if (!m) continue;
        m.envMapIntensity = 1.25;
        if (m.metalness !== undefined) m.metalness = 0.45;
        if (m.roughness !== undefined) m.roughness = 0.3;
        if (m.color) m.color.set('#d3dcea');
        m.needsUpdate = true;
      }
    });
    scene.add(root);
  };

  state.setSkeletonVisible = (visible) => {
    if (!state.modelRoot) return;
    if (!state.skeletonHelper) {
      const helper = new THREE.SkeletonHelper(state.modelRoot);
      helper.material.depthTest = false;
      helper.material.transparent = true;
      helper.material.opacity = 0.72;
      helper.material.linewidth = 2;
      helper.material.color = new THREE.Color('#4df0ff');
      helper.visible = false;
      scene.add(helper);
      state.skeletonHelper = helper;
    }
    state.skeletonHelper.visible = visible;
  };

  state.setAccent = (hex) => {
    state.accent.set(hex);
    ringMat.color.copy(state.accent);
    ringLight.color.copy(state.accent);
    if (state.skeletonHelper) state.skeletonHelper.material.color.copy(state.accent);
  };

  state.setPreset = (name, instant = false) => {
    const preset = CAMERA_PRESETS[name];
    if (!preset) return;
    state.cameraGoal = {
      pos: new THREE.Vector3(...preset.pos),
      target: new THREE.Vector3(...preset.target),
    };
    if (instant) {
      state.camera.position.copy(state.cameraGoal.pos);
      controls.target.copy(state.cameraGoal.target);
      state.cameraGoal = null;
    }
  };

  /** Animasi halus menuju preset kamera. */
  state.updateCamera = (dt) => {
    if (!state.cameraGoal) return;
    const t = 1 - Math.pow(0.001, dt); // ~0.55 s
    state.camera.position.lerp(state.cameraGoal.pos, t);
    controls.target.lerp(state.cameraGoal.target, t);
    if (state.camera.position.distanceTo(state.cameraGoal.pos) < 0.01) state.cameraGoal = null;
  };

  state.resize = () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };

  /** Denyut mengikuti ketukan musik. */
  state.pulse = (beatPos, bass = 1) => {
    const frac = beatPos - Math.floor(beatPos);
    const p = Math.pow(1 - Math.min(1, frac * 1.35), 2.6);
    ringMat.color.copy(state.accent).multiplyScalar(0.42 + p * 0.95);
    ringOuter.material.opacity = 0.22 + p * 0.5;
    ringLight.intensity = 4 + p * 12 * bass;
    state.key.intensity = 2.4 + p * 0.7;
  };

  state.tick = (dt, elapsed) => {
    dust.rotation.y += dt * 0.02;
    const pos = dustGeo.attributes.position;
    for (let i = 0; i < dustCount; i += 1) {
      const y = pos.getY(i) + dt * (0.045 + (i % 7) * 0.006);
      pos.setY(i, y > 4.4 ? 0 : y);
    }
    pos.needsUpdate = true;
    state.updateCamera(dt);
    controls.update();
    if (state.skeletonHelper?.visible) state.skeletonHelper.update();
    void elapsed;
  };

  return state;
}
