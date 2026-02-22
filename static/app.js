/**
 * KBilliards — Three.js Web Client
 *
 * Features: ball rendering with quaternion rotation, RGB axis visualization,
 * angular velocity arrows, trail lines, cue stick, hit-point mouse drag,
 * physics params editor, advanced command panel, Web Audio sounds.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Constants (overwritten by server init message) ─────────────────────────

let TABLE_WIDTH = 1.27;
let TABLE_LENGTH = 2.54;
let BALL_RADIUS = 0.03275;
let SIM_DT = 0.001;
let SIM_SUBSTEPS = 4;
let MAX_HIT_OFFSET = 0.025;

const BALL_COLORS = {
  white:  0xf0f0f0,
  yellow: 0xf0dc50,
  red1:   0xdc2828,
  red2:   0xc83232,
};

const TRAIL_COLORS = {
  white:  0xf0f0f0,
  yellow: 0xf0dc50,
  red1:   0xdc2828,
  red2:   0x993333,
};

// Spin arrow constants (matching Ursina main.py)
const SPIN_ARROW_SCALE = 0.003;
const SPIN_ARROW_MIN = 2.0;

// Trail constants
const MAX_TRAIL_POINTS = 200;

// ── Three.js scene setup ───────────────────────────────────────────────────

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(
  60, window.innerWidth / window.innerHeight, 0.01, 50
);
camera.position.set(0, 2.5, 0.01);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(2, 5, 3);
scene.add(dirLight);

// ── Table ──────────────────────────────────────────────────────────────────

let tableGroup = null;

function buildTable() {
  if (tableGroup) scene.remove(tableGroup);
  tableGroup = new THREE.Group();

  const surfaceGeo = new THREE.PlaneGeometry(TABLE_WIDTH, TABLE_LENGTH);
  const surfaceMat = new THREE.MeshStandardMaterial({ color: 0x2d6b3f });
  const surface = new THREE.Mesh(surfaceGeo, surfaceMat);
  surface.rotation.x = -Math.PI / 2;
  surface.position.y = -0.001;
  tableGroup.add(surface);

  const cushionH = 0.03, cushionT = 0.04;
  const cushionMat = new THREE.MeshStandardMaterial({ color: 0x6b3a1f });
  const hw = TABLE_WIDTH / 2, hl = TABLE_LENGTH / 2;

  const cRight = new THREE.Mesh(
    new THREE.BoxGeometry(cushionT, cushionH, TABLE_LENGTH), cushionMat
  );
  cRight.position.set(hw + cushionT / 2, cushionH / 2, 0);
  tableGroup.add(cRight);

  const cLeft = new THREE.Mesh(
    new THREE.BoxGeometry(cushionT, cushionH, TABLE_LENGTH), cushionMat
  );
  cLeft.position.set(-hw - cushionT / 2, cushionH / 2, 0);
  tableGroup.add(cLeft);

  const cFar = new THREE.Mesh(
    new THREE.BoxGeometry(TABLE_WIDTH + cushionT * 2, cushionH, cushionT), cushionMat
  );
  cFar.position.set(0, cushionH / 2, hl + cushionT / 2);
  tableGroup.add(cFar);

  const cNear = new THREE.Mesh(
    new THREE.BoxGeometry(TABLE_WIDTH + cushionT * 2, cushionH, cushionT), cushionMat
  );
  cNear.position.set(0, cushionH / 2, -hl - cushionT / 2);
  tableGroup.add(cNear);

  scene.add(tableGroup);
}

// ── Ball management ────────────────────────────────────────────────────────

const ballObjects = {};  // name -> { mesh, axes[] }

function createBallTexture(color) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');

  const hex = '#' + color.toString(16).padStart(6, '0');
  ctx.fillStyle = hex;
  ctx.fillRect(0, 0, 128, 128);

  // Dot pattern for rotation visibility
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  for (let i = 0; i < 8; i++) {
    ctx.beginPath();
    ctx.arc(16 + i * 14, 64, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.beginPath(); ctx.arc(64, 12, 8, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(64, 116, 8, 0, Math.PI * 2); ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createBall(name, pos) {
  if (ballObjects[name]) {
    ballObjects[name].mesh.position.set(pos[0], pos[1] + BALL_RADIUS, pos[2]);
    return;
  }

  const color = BALL_COLORS[name] || 0xcccccc;
  const geo = new THREE.SphereGeometry(BALL_RADIUS, 32, 32);
  const mat = new THREE.MeshStandardMaterial({
    map: createBallTexture(color),
    roughness: 0.3,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(pos[0], pos[1] + BALL_RADIUS, pos[2]);
  scene.add(mesh);

  // Add RGB axis visualization as children of mesh (rotate with ball)
  const axes = addAxes(mesh);

  ballObjects[name] = { mesh, axes };

  // Initialize trail for this ball
  initTrail(name);
}

function clearBalls() {
  for (const name of Object.keys(ballObjects)) {
    const obj = ballObjects[name];
    scene.remove(obj.mesh);
    obj.mesh.geometry.dispose();
    obj.mesh.material.map?.dispose();
    obj.mesh.material.dispose();
    // Axes are children of mesh, removed with it, but dispose geometry
    for (const ax of obj.axes) {
      ax.geometry.dispose();
      ax.material.dispose();
    }
    delete ballObjects[name];
  }
  clearSpinArrows();
  clearAllTrails();
}

// ── RGB axis visualization (children of ball mesh) ─────────────────────────

function addAxes(parentMesh) {
  // Scale factor: Ursina parent scale = BALL_RADIUS*2
  const S = BALL_RADIUS * 2;
  const AX_L = 1.6 * S;   // shaft length
  const AX_T = 0.14 * S;  // shaft thickness
  const AX_H = 0.32 * S;  // tip cube size

  const axisData = [
    { pos: [AX_L / 2, 0, 0], scale: [AX_L, AX_T, AX_T], tip: [AX_L, 0, 0], color: 0xff0000 },  // X red
    { pos: [0, AX_L / 2, 0], scale: [AX_T, AX_L, AX_T], tip: [0, AX_L, 0], color: 0x00ff00 },  // Y green
    { pos: [0, 0, AX_L / 2], scale: [AX_T, AX_T, AX_L], tip: [0, 0, AX_L], color: 0x00ccff },  // Z azure
  ];

  const meshes = [];
  for (const ax of axisData) {
    const shaftGeo = new THREE.BoxGeometry(ax.scale[0], ax.scale[1], ax.scale[2]);
    const mat = new THREE.MeshBasicMaterial({ color: ax.color });
    const shaft = new THREE.Mesh(shaftGeo, mat);
    shaft.position.set(ax.pos[0], ax.pos[1], ax.pos[2]);
    parentMesh.add(shaft);
    meshes.push(shaft);

    const tipGeo = new THREE.BoxGeometry(AX_H, AX_H, AX_H);
    const tipMat = new THREE.MeshBasicMaterial({ color: ax.color });
    const tip = new THREE.Mesh(tipGeo, tipMat);
    tip.position.set(ax.tip[0], ax.tip[1], ax.tip[2]);
    parentMesh.add(tip);
    meshes.push(tip);
  }
  return meshes;
}

// ── Angular velocity (spin) arrow ──────────────────────────────────────────

const spinArrowObjects = {};  // name -> THREE.ArrowHelper

function updateSpinArrows(ballsData) {
  clearSpinArrows();

  for (const bd of ballsData) {
    const wx = bd.w[0], wy = bd.w[1], wz = bd.w[2];
    const wMag = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (wMag < SPIN_ARROW_MIN) continue;

    const obj = ballObjects[bd.name];
    if (!obj) continue;

    const origin = new THREE.Vector3(
      bd.pos[0], BALL_RADIUS, bd.pos[2]
    );
    const dir = new THREE.Vector3(wx / wMag, wy / wMag, wz / wMag);
    const arrowLen = wMag * SPIN_ARROW_SCALE;
    const headLen = arrowLen * 0.3;
    const headW = BALL_RADIUS * 0.3;

    const arrow = new THREE.ArrowHelper(dir, origin, arrowLen, 0x000000, headLen, headW);
    scene.add(arrow);
    spinArrowObjects[bd.name] = arrow;
  }
}

function clearSpinArrows() {
  for (const name of Object.keys(spinArrowObjects)) {
    const arrow = spinArrowObjects[name];
    scene.remove(arrow);
    arrow.dispose();
    delete spinArrowObjects[name];
  }
}

// ── Trail system (client-side tracking) ────────────────────────────────────

const trailData = {};  // name -> { points: [[x,z],...], line, geometry }

function initTrail(name) {
  if (trailData[name]) return;

  const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setDrawRange(0, 0);

  const color = TRAIL_COLORS[name] || 0xffffff;
  const material = new THREE.LineBasicMaterial({ color });
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  trailData[name] = { points: [], line, geometry };
}

function updateTrails(ballsData) {
  for (const bd of ballsData) {
    if (bd.state === 'STATIONARY') continue;

    const td = trailData[bd.name];
    if (!td) continue;

    const x = bd.pos[0], z = bd.pos[2];
    const pts = td.points;

    // Only add if moved enough from last point
    if (pts.length > 0) {
      const last = pts[pts.length - 1];
      if (Math.abs(last[0] - x) < 0.001 && Math.abs(last[1] - z) < 0.001) continue;
    }

    pts.push([x, z]);
    if (pts.length > MAX_TRAIL_POINTS) pts.shift();

    // Update geometry
    const posAttr = td.geometry.getAttribute('position');
    for (let i = 0; i < pts.length; i++) {
      posAttr.setXYZ(i, pts[i][0], 0.001, pts[i][1]);
    }
    posAttr.needsUpdate = true;
    td.geometry.setDrawRange(0, pts.length);
  }
}

function clearTrail(name) {
  const td = trailData[name];
  if (td) {
    td.points = [];
    td.geometry.setDrawRange(0, 0);
  }
}

function clearAllTrails() {
  for (const name in trailData) {
    const td = trailData[name];
    scene.remove(td.line);
    td.line.geometry.dispose();
    td.line.material.dispose();
    delete trailData[name];
  }
}

// ── Cue stick visualization ────────────────────────────────────────────────

let cueMesh = null;
let aimLine = null;

function createCue() {
  if (!cueMesh) {
    const geo = new THREE.BoxGeometry(0.008, 0.008, 0.5);
    geo.translate(0, 0, -0.25 - BALL_RADIUS - 0.02);
    const mat = new THREE.MeshStandardMaterial({ color: 0xd4a76a });
    cueMesh = new THREE.Mesh(geo, mat);
    scene.add(cueMesh);
  }
  if (!aimLine) {
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 2),
    ]);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xffffff, transparent: true, opacity: 0.3,
    });
    aimLine = new THREE.Line(lineGeo, lineMat);
    scene.add(aimLine);
  }
}

function updateCue(aimData, mode) {
  const showCue = (mode === 'idle') && aimData;
  if (!showCue) {
    if (cueMesh) cueMesh.visible = false;
    if (aimLine) aimLine.visible = false;
    return;
  }

  createCue();

  const whiteBall = ballObjects['white'];
  if (!whiteBall) {
    cueMesh.visible = false;
    aimLine.visible = false;
    return;
  }

  const ballPos = whiteBall.mesh.position;
  const angle = THREE.MathUtils.degToRad(aimData.angle);
  const elev = THREE.MathUtils.degToRad(aimData.elevation);
  const cosE = Math.cos(elev);

  const dir = new THREE.Vector3(
    Math.sin(angle) * cosE,
    Math.sin(elev),
    Math.cos(angle) * cosE,
  );

  const backDist = 0.02 + aimData.power * 0.15;
  cueMesh.position.copy(ballPos).addScaledVector(dir, -backDist);
  cueMesh.lookAt(ballPos);
  cueMesh.visible = true;

  const lineEnd = ballPos.clone().addScaledVector(dir, 2.0);
  aimLine.geometry.setFromPoints([ballPos.clone(), lineEnd]);
  aimLine.geometry.attributes.position.needsUpdate = true;
  aimLine.visible = true;
}

// ── Ball rotation (quaternion accumulation) ────────────────────────────────

let frameCount = 0;

function updateBallTransforms(ballData) {
  const dt = SIM_DT * SIM_SUBSTEPS;
  frameCount++;

  for (const bd of ballData) {
    const obj = ballObjects[bd.name];
    if (!obj) continue;

    obj.mesh.position.set(bd.pos[0], bd.pos[1] + BALL_RADIUS, bd.pos[2]);

    const wx = bd.w[0], wy = bd.w[1], wz = bd.w[2];
    const wMag = Math.sqrt(wx * wx + wy * wy + wz * wz);

    if (wMag > 1e-6) {
      const angle = wMag * dt;
      const axis = new THREE.Vector3(wx / wMag, wy / wMag, wz / wMag);
      const dq = new THREE.Quaternion().setFromAxisAngle(axis, angle);
      obj.mesh.quaternion.premultiply(dq);  // dq * q = world-space rotation (Three.js convention)
    }

    if (frameCount % 60 === 0) {
      obj.mesh.quaternion.normalize();
    }
  }
}

// ── Hit-point indicator (mouse drag) ───────────────────────────────────────

const hitpointCircle = document.getElementById('hitpoint-circle');
const hitpointDot = document.getElementById('hitpoint-dot');
let hitDragging = false;

function updateHitpointDot(offsetX, offsetY) {
  // offset in meters, normalize to [-1, 1]
  const nx = offsetX / MAX_HIT_OFFSET;
  const ny = offsetY / MAX_HIT_OFFSET;
  const radius = 55;  // pixels (120/2 - dot_size/2)
  const cx = 55, cy = 55;  // center in hitpoint-circle coords
  hitpointDot.style.left = (cx + nx * radius) + 'px';
  hitpointDot.style.top = (cy - ny * radius) + 'px';  // Y inverted for screen
}

function hitpointFromMouse(e) {
  const rect = hitpointCircle.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  let dx = e.clientX - rect.left - cx;
  let dy = e.clientY - rect.top - cy;
  const radius = rect.width / 2;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist > radius) {
    dx = (dx / dist) * radius;
    dy = (dy / dist) * radius;
  }
  const nx = dx / radius;         // -1 to 1 (right = +)
  const ny = -dy / radius;        // -1 to 1 (top = +, screen Y inverted)
  return { nx, ny };
}

hitpointCircle.addEventListener('mousedown', (e) => {
  hitDragging = true;
  const { nx, ny } = hitpointFromMouse(e);
  sendHitOffset(nx, ny);
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!hitDragging) return;
  const { nx, ny } = hitpointFromMouse(e);
  sendHitOffset(nx, ny);
});

document.addEventListener('mouseup', () => {
  hitDragging = false;
});

function sendHitOffset(nx, ny) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'set_hit_offset', x: nx, y: ny }));
  }
}

// ── Sound (Web Audio API) ──────────────────────────────────────────────────

let audioCtx = null;

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function playSound(type, speed) {
  ensureAudio();
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const vol = Math.min(1.0, speed * 0.5);
  if (vol < 0.01) return;

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);

  if (type === 'ball_ball') {
    osc.frequency.value = 800;
    gain.gain.setValueAtTime(vol * 0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.08);
  } else if (type === 'cushion') {
    osc.frequency.value = 400;
    gain.gain.setValueAtTime(vol * 0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.1);
  }
}

// ── UI updates ─────────────────────────────────────────────────────────────

const infoBar = document.getElementById('info-bar');
const statusBar = document.getElementById('status-bar');
const gameUI = document.getElementById('game-ui');
const playerScoreEl = document.getElementById('player-score');
const aiScoreEl = document.getElementById('ai-score');
const turnDisplay = document.getElementById('turn-display');
const powerBarContainer = document.getElementById('power-bar-container');
const powerBarFill = document.getElementById('power-bar-fill');
const resultText = document.getElementById('result-text');
const hitpointPanel = document.getElementById('hitpoint-panel');
const cueInfoEl = document.getElementById('cue-info');
const advPanel = document.getElementById('adv-panel');
const advInput = document.getElementById('adv-input');

let resultTimeout = null;

function showResult(msg, colorName) {
  const colors = {
    red: '#f44336', green: '#4caf50', orange: '#ff9800',
    yellow: '#ffd700', cyan: '#00bcd4', gray: '#999',
  };
  resultText.style.color = colors[colorName] || '#fff';
  resultText.textContent = msg;
  resultText.style.display = 'block';
  resultText.style.animation = 'none';
  void resultText.offsetHeight;
  resultText.style.animation = 'fadeOut 2s forwards';

  if (resultTimeout) clearTimeout(resultTimeout);
  resultTimeout = setTimeout(() => { resultText.style.display = 'none'; }, 2000);
}

function hideAimVisuals() {
  if (cueMesh) cueMesh.visible = false;
  if (aimLine) aimLine.visible = false;
  clearSpinArrows();
  cueInfoEl.style.display = 'none';
}

function updateCueInfo(aim) {
  if (!aim) { cueInfoEl.style.display = 'none'; return; }
  const ox = aim.hit_offset[0], oy = aim.hit_offset[1];
  const tipX = (ox / MAX_HIT_OFFSET).toFixed(2);
  const tipY = (oy / MAX_HIT_OFFSET).toFixed(2);
  cueInfoEl.textContent =
    `Az: ${aim.angle.toFixed(1)}\u00B0  El: ${aim.elevation.toFixed(1)}\u00B0  ` +
    `Tip: (${tipX}, ${tipY})  Power: ${(aim.power * 100).toFixed(0)}%`;
  cueInfoEl.style.display = 'block';
}

function updateGameUI(game) {
  if (!game) return;
  playerScoreEl.textContent = `Player: ${game.player_score}`;
  aiScoreEl.textContent = `AI: ${game.ai_score}`;
  if (game.practice) {
    turnDisplay.textContent = 'Practice Mode';
  } else if (game.active) {
    turnDisplay.textContent = game.player_turn ? 'Your Turn' : 'AI Turn';
  }
}

// ── Physics params panel ───────────────────────────────────────────────────

const paramsPanel = document.getElementById('params-panel');
const paramsRows = document.getElementById('params-rows');
let paramsData = [];       // [{attr, label, value, min, max, step}, ...]
let selectedParam = 0;
let paramsVisible = false;

function buildParamsPanel(data) {
  paramsData = data;
  paramsRows.innerHTML = '';
  data.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'param-row' + (i === selectedParam ? ' selected' : '');
    row.innerHTML = `<span class="param-label">${p.label}</span>` +
                    `<span class="param-value" id="pv-${i}">${fmtVal(p.value)}</span>`;
    row.addEventListener('click', () => selectParam(i));
    paramsRows.appendChild(row);
  });
}

function selectParam(idx) {
  selectedParam = idx;
  const rows = paramsRows.querySelectorAll('.param-row');
  rows.forEach((r, i) => r.classList.toggle('selected', i === idx));
}

function fmtVal(v) {
  if (v === 0) return '0';
  return v.toPrecision(4);
}

function updateParamValue(idx, value) {
  if (paramsData[idx]) paramsData[idx].value = value;
  const el = document.getElementById(`pv-${idx}`);
  if (el) el.textContent = fmtVal(value);
}

function adjustParam(direction, fine) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      cmd: 'adjust_param',
      index: selectedParam,
      direction: direction,
      fine: fine,
    }));
  }
}

function resetParams() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'reset_params' }));
  }
}

function toggleParams() {
  paramsVisible = !paramsVisible;
  paramsPanel.classList.toggle('hidden', !paramsVisible);
  if (paramsVisible && paramsData.length === 0) {
    // Request params from server
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ cmd: 'get_params' }));
    }
  }
}

// Prevent scroll wheel on params panel from zooming the camera
paramsPanel.addEventListener('wheel', (e) => e.stopPropagation());

// ── Advanced Command Panel ─────────────────────────────────────────────────

let advOpen = false;

function openAdvPanel() {
  advOpen = true;
  advPanel.style.display = 'block';
  advInput.value = '';
  advInput.focus();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'get_state' }));
  }
}

function closeAdvPanel() {
  advOpen = false;
  advPanel.style.display = 'none';
}

function executeAdvCommand() {
  const text = advInput.value.trim();
  if (!text) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'execute', text }));
  }
  closeAdvPanel();
}

// ── WebSocket connection ───────────────────────────────────────────────────

let ws = null;

function connect() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    statusBar.textContent = 'Connected.';
    // Request params on connect
    ws.send(JSON.stringify({ cmd: 'get_params' }));
  };

  ws.onclose = () => {
    statusBar.textContent = 'Disconnected. Reconnecting...';
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    statusBar.textContent = 'Connection error.';
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };
}

let spinUpdateCounter = 0;
let prevMode = '';

function handleMessage(msg) {
  if (msg.type === 'init') {
    TABLE_WIDTH = msg.table_width;
    TABLE_LENGTH = msg.table_length;
    BALL_RADIUS = msg.ball_radius;
    SIM_DT = msg.sim_dt;
    SIM_SUBSTEPS = msg.sim_substeps;
    if (msg.max_hit_offset) MAX_HIT_OFFSET = msg.max_hit_offset;
    buildTable();
    return;
  }

  if (msg.type === 'state_json') {
    if (advOpen) {
      advInput.value = msg.data;
      advInput.select();
    }
    return;
  }

  if (msg.type === 'params') {
    buildParamsPanel(msg.data);
    return;
  }

  if (msg.type === 'param_update') {
    updateParamValue(msg.index, msg.value);
    return;
  }

  if (msg.type !== 'frame') return;

  // Process events
  if (msg.events) {
    for (const ev of msg.events) handleEvent(ev);
  }

  // Update ball transforms
  if (msg.balls && msg.balls.length > 0) {
    updateBallTransforms(msg.balls);

    // Update trails (client-side tracking)
    updateTrails(msg.balls);

    // Update spin arrows (every 3 frames for performance)
    spinUpdateCounter++;
    if (spinUpdateCounter % 3 === 0) {
      updateSpinArrows(msg.balls);
    }
  }

  // Sounds
  if (msg.sounds) {
    for (const s of msg.sounds) playSound(s.type, s.speed);
  }

  // UI text
  if (msg.info !== undefined) infoBar.textContent = msg.info;
  if (msg.status !== undefined) statusBar.textContent = msg.status;

  // Game UI
  if (msg.game) updateGameUI(msg.game);

  // Aim / power / hitpoint / cue info
  if (msg.aim) {
    powerBarFill.style.width = (msg.aim.power * 100) + '%';
    updateHitpointDot(msg.aim.hit_offset[0], msg.aim.hit_offset[1]);
    if (msg.mode === 'idle') {
      updateCueInfo(msg.aim);
    } else {
      cueInfoEl.style.display = 'none';
    }
  } else {
    cueInfoEl.style.display = 'none';
  }

  // Cue stick
  updateCue(msg.aim, msg.mode);

  // Mode transition: running → idle = session ended → clear trails & spin arrows
  if (prevMode === 'running' && msg.mode === 'idle') {
    for (const name in trailData) clearTrail(name);
    clearSpinArrows();
  }
  prevMode = msg.mode;
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'spawn_ball':
      createBall(ev.ball.name, ev.ball.pos);
      break;
    case 'clear_balls':
      clearBalls();
      hideAimVisuals();
      break;
    case 'show_result':
      showResult(ev.msg, ev.color_name);
      break;
    case 'init_game_ui':
      gameUI.style.display = 'block';
      powerBarContainer.style.display = 'block';
      hitpointPanel.style.display = 'block';
      break;
    case 'destroy_game_ui':
      gameUI.style.display = 'none';
      powerBarContainer.style.display = 'none';
      hitpointPanel.style.display = 'none';
      break;
    case 'hide_game_score_ui':
      playerScoreEl.style.display = 'none';
      aiScoreEl.style.display = 'none';
      break;
    case 'update_game_ui':
      playerScoreEl.style.display = 'block';
      aiScoreEl.style.display = 'block';
      break;
    case 'update_power_bar':
      powerBarFill.style.width = '0%';
      break;
    case 'clear_game_visuals':
      for (const name in trailData) clearTrail(name);
      hideAimVisuals();
      break;
    case 'clear_aim':
      if (cueMesh) cueMesh.visible = false;
      if (aimLine) aimLine.visible = false;
      break;
    case 'adv_close':
      closeAdvPanel();
      break;
  }
}

// ── Input handling ─────────────────────────────────────────────────────────

const KEY_MAP = {
  'ArrowLeft':  'left',
  'ArrowRight': 'right',
  'ArrowUp':    'up',
  'ArrowDown':  'down',
  ' ':          'space',
  'Shift':      'shift',
};

const DIRECT_KEYS = new Set([
  'w', 'a', 's', 'd', 'g', 't', 'r', '1', '2', '3', '4', '5',
]);

function getKeyName(event) {
  if (KEY_MAP[event.key]) return KEY_MAP[event.key];
  const lower = event.key.toLowerCase();
  if (DIRECT_KEYS.has(lower)) return lower;
  return null;
}

document.addEventListener('keydown', (event) => {
  // Advanced panel open
  if (advOpen) {
    if (event.key === 'Escape') { closeAdvPanel(); event.preventDefault(); return; }
    if (event.key === 'Enter' && event.ctrlKey) { executeAdvCommand(); event.preventDefault(); return; }
    return;
  }

  // Shift+A → advanced panel
  if (event.key === 'A' && event.shiftKey) {
    openAdvPanel();
    event.preventDefault();
    return;
  }

  // Shift+P → reset all params
  if (event.key === 'P' && event.shiftKey) {
    resetParams();
    event.preventDefault();
    return;
  }

  // P → toggle params panel
  if (event.key === 'p' && !event.shiftKey) {
    toggleParams();
    event.preventDefault();
    return;
  }

  // ] → param up, [ → param down
  if (event.key === ']') {
    if (paramsVisible) adjustParam(1, event.shiftKey);
    event.preventDefault();
    return;
  }
  if (event.key === '[') {
    if (paramsVisible) adjustParam(-1, event.shiftKey);
    event.preventDefault();
    return;
  }

  const keyName = getKeyName(event);
  if (!keyName) return;

  event.preventDefault();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'key_down', key: keyName }));
  }
});

document.addEventListener('keyup', (event) => {
  if (advOpen) return;

  const keyName = getKeyName(event);
  if (!keyName) return;

  event.preventDefault();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'key_up', key: keyName }));
  }
});

// Button bar helper
window._sendKey = (k) => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'key_down', key: k }));
    setTimeout(() => {
      ws.send(JSON.stringify({ cmd: 'key_up', key: k }));
    }, 50);
  }
};

// Resume audio on first interaction
document.addEventListener('click', () => ensureAudio(), { once: true });
document.addEventListener('keydown', () => ensureAudio(), { once: true });

// ── Resize handling ────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Render loop ────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Start ──────────────────────────────────────────────────────────────────

buildTable();
connect();
animate();
