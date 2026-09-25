/**
 * explorer.js — 3D NMM position explorer (Three.js ES module)
 *
 * Bar height = how often humans played that move (most-played = tallest).
 * Bar color  = human win-rate gradient: green (winning) → orange → red (losing).
 * Arrows     = cylinder+cone from the piece's current square to its destination.
 * Malom overlay (toggle) = colored rings + DTW numbers on candidate squares.
 *
 * Interaction state machine:
 *   Place phase : click bar → if all variants need capture enter capture mode, else apply.
 *   Move/fly    : click own piece → piece_selected; click destination bar → apply or capture.
 *   Capture mode: click red opponent piece → complete move; click empty / Esc → cancel.
 *
 * Hint rings on board:
 *   Gold  — HumanDB best destination (highest win%)
 *   Blue  — Sentinel best destination (highest sentinel_score)
 *   Red   — Capturable opponent piece in capture mode
 *   Gold  — Sentinel's recommended capture (overrides red for that piece)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ── Sprite text helper (for DTW numbers) ─────────────────────────────────────
function makeDtwSprite(text, hexColor) {
  const W = 64, H = 32;
  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 20px monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign    = 'center';
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.lineWidth   = 4;
  ctx.strokeText(text, W / 2, H / 2);
  ctx.fillStyle = hexColor;
  ctx.fillText(text, W / 2, H / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.7, 0.35, 1);
  return sprite;
}

// ── Board geometry data ───────────────────────────────────────────────────────

const POS_COORDS = {
  a7: [-3, 0, -3], d7: [0, 0, -3], g7: [3, 0, -3],
  a4: [-3, 0,  0],                  g4: [3, 0,  0],
  a1: [-3, 0,  3], d1: [0, 0,  3], g1: [3, 0,  3],
  b6: [-2, 0, -2], d6: [0, 0, -2], f6: [2, 0, -2],
  b4: [-2, 0,  0],                  f4: [2, 0,  0],
  b2: [-2, 0,  2], d2: [0, 0,  2], f2: [2, 0,  2],
  c5: [-1, 0, -1], d5: [0, 0, -1], e5: [1, 0, -1],
  c4: [-1, 0,  0],                  e4: [1, 0,  0],
  c3: [-1, 0,  1], d3: [0, 0,  1], e3: [1, 0,  1],
};

const EDGES = [
  ['a7','d7'],['d7','g7'],['g7','g4'],['g4','g1'],['g1','d1'],['d1','a1'],['a1','a4'],['a4','a7'],
  ['b6','d6'],['d6','f6'],['f6','f4'],['f4','f2'],['f2','d2'],['d2','b2'],['b2','b4'],['b4','b6'],
  ['c5','d5'],['d5','e5'],['e5','e4'],['e4','e3'],['e3','d3'],['d3','c3'],['c3','c4'],['c4','c5'],
  ['d7','d6'],['d6','d5'],
  ['g4','f4'],['f4','e4'],
  ['d1','d2'],['d2','d3'],
  ['a4','b4'],['b4','c4'],
];

// ── Colors ────────────────────────────────────────────────────────────────────

const C = {
  board:  0x6b4f22,
  pad:    0x4a3518,
  lineWd: 0x8b6b3a,
  white:  0xf5f0dc,
  black:  0x1a1a1a,
  barHov: 0xffd700,
};

// Net definitions — field names, colors, sort direction, normalization
// isAbsNorm: true = heuristic-style (raw eval, normalize by abs max)
const NET_DEFS = {
  sentinel: { label:'Sentinel',  cssColor:'#e07030', hexColor:0xe07030, field:'sentinel_score',  isHigherBetter:true,  isAbsNorm:false },
  heuristic:{ label:'Heuristic', cssColor:'#c8a96e', hexColor:0xc8a96e, field:'heuristic_score', isHigherBetter:true,  isAbsNorm:true  },
  gapnet:   { label:'GapNet',    cssColor:'#cc5555', hexColor:0xcc5555, field:'gapnet_score',    isHigherBetter:false, isAbsNorm:false },
  value:    { label:'ValueNet',  cssColor:'#50aaaa', hexColor:0x50aaaa, field:'value_score',     isHigherBetter:true,  isAbsNorm:false },
  pref:     { label:'PrefNet',   cssColor:'#c4a020', hexColor:0xc4a020, field:'pref_score',      isHigherBetter:true,  isAbsNorm:false },
  pred:     { label:'Pred',      cssColor:'#a06fe0', hexColor:0xa06fe0, field:'pred_human_prob', isHigherBetter:true,  isAbsNorm:false },
  regret:   { label:'Regret',    cssColor:'#ff6020', hexColor:0xff6020, field:'regret_score',    isHigherBetter:true,  isAbsNorm:false },
};

// User-selected overlay nets for the 3 ring/bar slots (null = slot off)
let selectedNets = ['sentinel', null, null];
let generalistRingEnabled = false;
let _regretFen = null;  // FEN for which regret_score is currently injected into moves

// Wilson score lower bound (z=1.645 → 95% one-sided confidence)
function wilsonLower(wins, total, z = 1.645) {
  if (total === 0) return 0;
  const p  = wins / total;
  const z2 = z * z;
  return (p + z2 / (2 * total) - z * Math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / (1 + z2 / total);
}

function winPctColor(pct) {
  const t = Math.max(0, Math.min(1, pct));
  if (t < 0.5) {
    const u = t * 2;
    return new THREE.Color().setRGB(0.94, 0.24 + 0.38 * u, 0.07);
  }
  const u = (t - 0.5) * 2;
  return new THREE.Color().setRGB(0.94 - 0.8 * u, 0.62 + 0.14 * u, 0.07);
}

function sentinelColor(score) {
  const t = Math.max(0, Math.min(1, score));
  return new THREE.Color().setHSL(0.58 + t * 0.08, 0.75, 0.28 + t * 0.35);
}

function barColor(moveData) {
  if (moveData.has_db_data) return winPctColor(moveData.win_pct);
  if (moveData.pred_human_prob != null) return new THREE.Color(0x5591c7);
  return new THREE.Color(0x555555);
}

// ── Scene setup ───────────────────────────────────────────────────────────────

const canvas  = document.getElementById('board-canvas');
const wrap    = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');
const loading = document.getElementById('loading-overlay');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
wrap.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1612);
scene.fog = new THREE.Fog(0x1a1612, 18, 30);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 8, 9);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor  = 0.08;
controls.minDistance    = 4;
controls.maxDistance    = 20;
controls.maxPolarAngle  = Math.PI / 2.1;

const ambient = new THREE.AmbientLight(0xfff8e7, 0.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffe8b0, 1.2);
dirLight.position.set(5, 10, 8);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0x8fb3d4, 0.3);
fillLight.position.set(-5, 3, -5);
scene.add(fillLight);

// ── Static board geometry ─────────────────────────────────────────────────────

const padMeshList = [];

function buildStaticBoard() {
  const planeGeo = new THREE.PlaneGeometry(9, 9);
  const planeMat = new THREE.MeshLambertMaterial({ color: C.board });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI / 2;
  plane.receiveShadow = true;
  scene.add(plane);

  const padGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.06, 16);
  for (const [pos, [x,, z]] of Object.entries(POS_COORDS)) {
    const mat  = new THREE.MeshLambertMaterial({ color: C.pad });
    const mesh = new THREE.Mesh(padGeo, mat);
    mesh.position.set(x, 0.03, z);
    mesh.receiveShadow = true;
    mesh.userData.pos   = pos;
    mesh.userData.isPad = true;
    scene.add(mesh);
    padMeshList.push(mesh);
  }

  const lineMat = new THREE.MeshLambertMaterial({ color: C.lineWd });
  for (const [a, b] of EDGES) {
    const [ax,, az] = POS_COORDS[a];
    const [bx,, bz] = POS_COORDS[b];
    const mid = new THREE.Vector3((ax+bx)/2, 0.02, (az+bz)/2);
    const len = new THREE.Vector3(bx-ax, 0, bz-az).length();
    const geo = new THREE.CylinderGeometry(0.04, 0.04, len, 6);
    const mesh = new THREE.Mesh(geo, lineMat);
    mesh.position.copy(mid);
    mesh.rotation.z = Math.PI / 2;
    mesh.rotation.y = -Math.atan2(bz - az, bx - ax);
    scene.add(mesh);
  }
}

buildStaticBoard();

// ── Coordinate labels (CSS2D) ─────────────────────────────────────────────────

function buildCoordLabels() {
  ['a','b','c','d','e','f','g'].forEach((letter, i) => {
    const el = document.createElement('div');
    el.className = 'coord-label';
    el.textContent = letter;
    const obj = new CSS2DObject(el);
    obj.position.set(i - 3, 0.5, 4.5);
    scene.add(obj);
  });

  ['7','6','5','4','3','2','1'].forEach((num, i) => {
    const el = document.createElement('div');
    el.className = 'coord-label';
    el.textContent = num;
    const obj = new CSS2DObject(el);
    obj.position.set(-4.5, 0.5, i - 3);
    scene.add(obj);
  });
}

buildCoordLabels();

// ── Dynamic layers ────────────────────────────────────────────────────────────

const pieceGroup = new THREE.Group();
const barGroup     = new THREE.Group();
const arrowGroup   = new THREE.Group();
const malomGroup   = new THREE.Group();
const hintGroup    = new THREE.Group();
malomGroup.visible = false;
scene.add(pieceGroup, barGroup, arrowGroup, malomGroup, hintGroup);

const pieceGeoW = new THREE.CylinderGeometry(0.26, 0.26, 0.13, 20);
const pieceGeoB = new THREE.CylinderGeometry(0.26, 0.26, 0.13, 20);

// ── Interaction state machine ─────────────────────────────────────────────────

let selectionState      = 'idle'; // 'idle' | 'piece_selected' | 'capture'
let selectedPieceSq     = null;   // own piece square when piece_selected
let pendingCaptureMoves = [];     // move variants waiting for capture-sq pick
let captureReturnState  = 'idle'; // state to restore on capture cancel
let currentPhase        = 'place';
let currentTurn         = 'W';
let explorerEloBand     = 'middle';

// ── Piece rebuild — per-piece materials for individual highlighting ────────────

function rebuildPieces(boardDict) {
  pieceGroup.clear();
  for (const [pos, piece] of Object.entries(boardDict)) {
    if (!piece || !POS_COORDS[pos]) continue;
    const [x,, z] = POS_COORDS[pos];
    const geo  = piece === 'W' ? pieceGeoW : pieceGeoB;
    const base = new THREE.Color(piece === 'W' ? C.white : C.black);
    const mat  = new THREE.MeshLambertMaterial({ color: base.clone(), transparent: true, opacity: 1.0 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, 0.10, z);
    mesh.castShadow = true;
    mesh.userData.pos       = pos;
    mesh.userData.color     = piece;
    mesh.userData.baseColor = base;
    pieceGroup.add(mesh);
  }
}

// ── Bar rebuild — deduplicated by to_sq ───────────────────────────────────────

const barMeshMap  = new Map(); // notation → { mesh, data }
const barGroupMap = new Map(); // toSq → [meshes]  (all segments for a destination)
const MAX_BAR_HEIGHT = 0.55;
const BAR_W          = 0.14;
const BAR_OFFSET_X   = 0.38;  // trajectory bar: beside piece to the right
const NET_OFFSETS    = [0.57, 0.72, 0.87];  // 3 net bar columns
const NET_W          = 0.10;

// Compute normalized bar height for a net slot.
// vals: array of raw field values for this slot; allVals used for heuristic abs-normalization.
function _computeNetBarH(def, vals, allVals) {
  if (!vals || vals.length === 0) return null;
  if (def.isAbsNorm) {
    const absMax = Math.max(1, ...allVals.map(v => Math.abs(v || 0)));
    const best   = Math.max(...vals.map(v => Math.abs(v || 0)));
    return Math.max(0.04, (best / absMax) * MAX_BAR_HEIGHT);
  }
  return Math.max(0.04, Math.max(...vals) * MAX_BAR_HEIGHT);
}

// Find best capture square according to the first active selected net, falling back to heuristic.
function _getBestCaptureSq(moves) {
  for (const netKey of selectedNets) {
    if (!netKey) continue;
    const def = NET_DEFS[netKey];
    if (!def) continue;
    const capMoves = moves.filter(m => m.capture_sq && m[def.field] != null);
    if (capMoves.length === 0) continue;
    const best = capMoves.reduce((a, b) => {
      const va = def.isAbsNorm ? Math.abs(a[def.field]) : a[def.field];
      const vb = def.isAbsNorm ? Math.abs(b[def.field]) : b[def.field];
      return (def.isHigherBetter ? vb > va : vb < va) ? b : a;
    });
    return best.capture_sq;
  }
  const capMoves = moves.filter(m => m.capture_sq && m.heuristic_score != null);
  if (capMoves.length > 0)
    return capMoves.reduce((a, b) => b.heuristic_score > a.heuristic_score ? b : a).capture_sq;
  return null;
}

function _addBarMesh(barX, z, segH, yBot, colHex, opacity, rep, mvsForSq, needsCapture, toSq) {
  const mat  = new THREE.MeshLambertMaterial({ color: colHex, transparent: true, opacity });
  const geom = new THREE.BoxGeometry(BAR_W, segH, BAR_W);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(barX, yBot + segH / 2, z);
  mesh.castShadow = true;
  mesh.userData.notation     = rep.notation;
  mesh.userData.moveData     = rep;
  mesh.userData.allMoves     = mvsForSq;
  mesh.userData.needsCapture = needsCapture;
  mesh.userData.toSq         = toSq;
  mesh.userData.baseColor    = new THREE.Color(colHex);
  mesh.userData.baseOpacity  = opacity;
  barGroup.add(mesh);
  return mesh;
}

function _rebuildBarsAggregated(movesArray) {
  // Movement/fly phase with no piece selected: show summed traj + best net bars per source piece.
  const bySrc = new Map();
  for (const mv of movesArray) {
    if (!mv.from_sq || !POS_COORDS[mv.from_sq]) continue;
    if (!bySrc.has(mv.from_sq)) bySrc.set(mv.from_sq, []);
    bySrc.get(mv.from_sq).push(mv);
  }
  if (!bySrc.size) return;

  const allHAbsMax  = Math.max(1, ...movesArray.map(m => Math.abs(m.heuristic_score || 0)));
  const allPredVals = movesArray.map(m => m.pred_human_prob ?? 0);
  const maxPred     = Math.max(0, ...allPredVals);
  const srcTotals   = new Map();
  for (const [src, mvs] of bySrc)
    srcTotals.set(src, mvs.reduce((s, m) => s + (m.total || 0), 0));
  const maxSrcTotal = Math.max(1, ...srcTotals.values());

  for (const [src, mvs] of bySrc) {
    const [x,, z]   = POS_COORDS[src];
    const baseY     = 0.07;
    const barX      = x + BAR_OFFSET_X;
    const segMeshes = [];

    const bestHeurAbs = Math.max(...mvs.map(m => Math.abs(m.heuristic_score || 0)));
    const bestPred    = Math.max(...mvs.map(m => m.pred_human_prob ?? -Infinity));
    const dbMvs  = mvs.filter(m => m.has_db_data);
    const wins   = dbMvs.reduce((s, m) => s + (m.wins   || 0), 0);
    const draws  = dbMvs.reduce((s, m) => s + (m.draws  || 0), 0);
    const losses = dbMvs.reduce((s, m) => s + (m.losses || 0), 0);
    const total  = wins + draws + losses;
    const trajSum = mvs.reduce((s, m) => s + (m.traj_freq        || 0), 0);
    const predSum = mvs.reduce((s, m) => s + (m.pred_human_prob  || 0), 0);

    // Compute best value per selected net for tooltip/aggregation
    const _bestByNet = {};
    for (const [k, def] of Object.entries(NET_DEFS)) {
      const vals = mvs.map(m => m[def.field]).filter(v => v != null);
      if (vals.length === 0) { _bestByNet[k] = null; continue; }
      _bestByNet[k] = def.isHigherBetter
        ? Math.max(...vals)
        : Math.min(...vals);
    }

    const synth = {
      notation:        src,
      has_db_data:     total > 0,
      wins, draws, losses, total,
      win_pct:         total > 0 ? wins / total : 0,
      heuristic_score: Math.max(...mvs.map(m => m.heuristic_score || 0)),
      sentinel_score:  _bestByNet.sentinel,
      pred_human_prob: bestPred > -Infinity ? bestPred : null,
      malom_wdl_after: null,
      malom_dtw_after: null,
      avg_moves_to_end: dbMvs.length > 0
        ? dbMvs.reduce((s, m) => s + (m.avg_moves_to_end || 0), 0) / dbMvs.length : 0,
      _isAggregate: true,
      _src:          src,
      _bestHeurAbs:  bestHeurAbs,
      _trajSum:      trajSum,
      _predSum:      predSum,
      _bestByNet,
    };

    // Traj/DB bar — summed W/D/L, height proportional to total games
    if (total > 0) {
      const barH  = Math.max(0.06, (total / maxSrcTotal) * MAX_BAR_HEIGHT);
      const lossH = barH * (losses / (total || 1));
      const drawH = barH * (draws  / (total || 1));
      const winH  = barH * (wins   / (total || 1));
      for (const seg of [
        { h: lossH, col: 0xef4444, yBot: baseY },
        { h: drawH, col: 0xa06040, yBot: baseY + lossH },
        { h: winH,  col: 0x4ade80, yBot: baseY + lossH + drawH },
      ]) {
        if (seg.h < 0.005) continue;
        segMeshes.push(_addBarMesh(barX, z, seg.h, seg.yBot, seg.col, 0.88, synth, mvs, false, src));
      }
    } else {
      // No DB data: pred prob fallback (purple), else grey heuristic
      if (bestPred > -Infinity && bestPred > 0 && maxPred > 0) {
        const barH = Math.max(0.06, (bestPred / maxPred) * MAX_BAR_HEIGHT);
        segMeshes.push(_addBarMesh(barX, z, barH, baseY, 0xa06fe0, 0.75, synth, mvs, false, src));
      } else {
        const barH = 0.06 + (bestHeurAbs / allHAbsMax) * (MAX_BAR_HEIGHT * 0.5);
        segMeshes.push(_addBarMesh(barX, z, barH, baseY, 0x555555, 0.55, synth, mvs, false, src));
      }
    }

    // 3 dynamic net bars
    const allHeurVals = movesArray.map(m => m.heuristic_score).filter(v => v != null);
    for (let slotIdx = 0; slotIdx < 3; slotIdx++) {
      const netKey = selectedNets[slotIdx];
      if (!netKey) continue;
      const def = NET_DEFS[netKey];
      const vals = mvs.map(m => m[def.field]).filter(v => v != null);
      if (vals.length === 0) continue;
      const allVals = movesArray.map(m => m[def.field]).filter(v => v != null);
      const barH = _computeNetBarH(def, vals, allVals.length ? allVals : vals);
      if (!barH) continue;
      const nm = new THREE.Mesh(
        new THREE.BoxGeometry(NET_W, barH, NET_W),
        new THREE.MeshLambertMaterial({ color: def.hexColor, transparent: true, opacity: 0.85 }),
      );
      nm.position.set(x + NET_OFFSETS[slotIdx], baseY + barH / 2, z);
      nm.castShadow = true;
      nm.userData.notation = src; nm.userData.moveData = synth;
      nm.userData.allMoves = mvs; nm.userData.needsCapture = false;
      nm.userData.toSq = src;
      nm.userData.baseColor   = new THREE.Color(def.hexColor);
      nm.userData.baseOpacity = 0.85;
      barGroup.add(nm);
      segMeshes.push(nm);
    }

    barGroupMap.set(src, segMeshes);
  }
}

function rebuildBars(movesArray) {
  barGroup.clear();
  barMeshMap.clear();
  barGroupMap.clear();
  if (!movesArray || movesArray.length === 0) return;

  // In movement/fly phase with no piece selected: aggregate per source piece
  if (currentPhase !== 'place' && selectionState === 'idle') {
    _rebuildBarsAggregated(movesArray);
    return;
  }

  // Group by to_sq so mill-closing positions show one bar per destination
  const byToSq = new Map();
  for (const mv of movesArray) {
    if (!mv.to_sq || !POS_COORDS[mv.to_sq]) continue;
    if (!byToSq.has(mv.to_sq)) byToSq.set(mv.to_sq, []);
    byToSq.get(mv.to_sq).push(mv);
  }

  const dbMoves     = movesArray.filter(m => m.has_db_data);
  const maxTotal    = Math.max(1, ...dbMoves.map(m => m.total || 0));
  const hScores     = movesArray.filter(m => !m.has_db_data).map(m => m.heuristic_score);
  const minH        = hScores.length ? Math.min(...hScores) : 0;
  const maxH        = hScores.length ? Math.max(...hScores) : 1;
  const hRange      = Math.max(1, maxH - minH);
  const predScores  = movesArray.filter(m => !m.has_db_data && m.pred_human_prob != null).map(m => m.pred_human_prob);
  const maxPred     = predScores.length > 0 ? Math.max(...predScores) : 0;

  for (const [toSq, mvsForSq] of byToSq) {
    const rep          = mvsForSq.find(m => !m.capture_sq) || mvsForSq[0];
    const needsCapture = mvsForSq.every(m => m.capture_sq != null);
    const [x,, z]      = POS_COORDS[toSq];
    const baseY        = 0.07;
    const segMeshes    = [];

    // ── Trajectory bar (W/D/L stacked segments, beside piece) ──
    const barX = x + BAR_OFFSET_X;
    if (rep.has_db_data) {
      const totalForSq = mvsForSq.reduce((s, m) => s + (m.total || 0), 0);
      const barH  = Math.max(0.06, (totalForSq / maxTotal) * MAX_BAR_HEIGHT);
      const wins   = rep.wins   || 0;
      const draws  = rep.draws  || 0;
      const losses = rep.losses || 0;
      const total  = Math.max(1, wins + draws + losses);

      const lossH = barH * (losses / total);
      const drawH = barH * (draws  / total);
      const winH  = barH * (wins   / total);

      const segs = [
        { h: lossH, col: 0xef4444, yBot: baseY },
        { h: drawH, col: 0xa06040, yBot: baseY + lossH },
        { h: winH,  col: 0x4ade80, yBot: baseY + lossH + drawH },
      ];
      let primaryMesh = null;
      for (const seg of segs) {
        if (seg.h < 0.005) continue;
        const m = _addBarMesh(barX, z, seg.h, seg.yBot, seg.col, 0.88, rep, mvsForSq, needsCapture, toSq);
        segMeshes.push(m);
        if (!primaryMesh) primaryMesh = m;
      }
      if (primaryMesh) {
        for (const mv of mvsForSq) barMeshMap.set(mv.notation, { mesh: primaryMesh, data: mv });
      }
    } else {
      // No DB data: pred prob fallback (purple), else grey heuristic
      if (rep.pred_human_prob != null && maxPred > 0) {
        const barH = Math.max(0.06, (rep.pred_human_prob / maxPred) * MAX_BAR_HEIGHT);
        const m = _addBarMesh(barX, z, barH, baseY, 0xa06fe0, 0.75, rep, mvsForSq, needsCapture, toSq);
        segMeshes.push(m);
        for (const mv of mvsForSq) barMeshMap.set(mv.notation, { mesh: m, data: mv });
      } else {
        const norm  = (rep.heuristic_score - minH) / hRange;
        const barH  = 0.06 + norm * (MAX_BAR_HEIGHT * 0.5);
        const m = _addBarMesh(barX, z, barH, baseY, 0x555555, 0.55, rep, mvsForSq, needsCapture, toSq);
        segMeshes.push(m);
        for (const mv of mvsForSq) barMeshMap.set(mv.notation, { mesh: m, data: mv });
      }
    }

    // ── 3 dynamic net bars ──
    for (let slotIdx = 0; slotIdx < 3; slotIdx++) {
      const netKey = selectedNets[slotIdx];
      if (!netKey) continue;
      const def = NET_DEFS[netKey];
      const val = rep[def.field];
      if (val == null) continue;
      const allVals = movesArray.map(m => m[def.field]).filter(v => v != null);
      const barH = _computeNetBarH(def, [val], allVals.length ? allVals : [val]);
      if (!barH) continue;
      const nm = new THREE.Mesh(
        new THREE.BoxGeometry(NET_W, barH, NET_W),
        new THREE.MeshLambertMaterial({ color: def.hexColor, transparent: true, opacity: 0.85 }),
      );
      nm.position.set(x + NET_OFFSETS[slotIdx], baseY + barH / 2, z);
      nm.castShadow = true;
      nm.userData.notation     = rep.notation;
      nm.userData.moveData     = rep;
      nm.userData.allMoves     = mvsForSq;
      nm.userData.needsCapture = needsCapture;
      nm.userData.toSq         = toSq;
      nm.userData.baseColor    = new THREE.Color(def.hexColor);
      nm.userData.baseOpacity  = 0.85;
      barGroup.add(nm);
      segMeshes.push(nm);
    }

    barGroupMap.set(toSq, segMeshes);
  }
}

// ── Regret (horizon effect) — async fetch, injected as regret_score into moves ──

async function fetchRegret(fen) {
  try {
    const res = await fetch('/api/explorer/regret?fen=' + encodeURIComponent(fen));
    const data = await res.json();
    if (data.error) { console.warn('Regret error:', data.error); return {}; }
    return data.regret_scores || {};
  } catch { return {}; }
}

function _injectRegretScores(r) {
  _regretFen = currentData?.fen ?? null;
  for (const mv of (currentData?.moves || [])) {
    mv.regret_score = r[mv.notation] ?? null;
  }
}

function _ensureRegret() {
  if (!currentData?.fen) return;
  if (_regretFen === currentData.fen) {
    // Already injected for this position — just re-render
    _refreshAfterStateChange();
    return;
  }
  const fen = currentData.fen;
  fetchRegret(fen).then(r => {
    if (currentData?.fen === fen) {
      _injectRegretScores(r);
      _refreshAfterStateChange();
    }
  });
}

// ── Move arrows (from_sq → to_sq) ────────────────────────────────────────────

const _up = new THREE.Vector3(0, 1, 0);

function rebuildArrows(movesArray) {
  arrowGroup.clear();
  if (!movesArray || movesArray.length === 0) return;
  const dbMoves  = movesArray.filter(m => m.has_db_data);
  const maxTotal = Math.max(1, ...dbMoves.map(m => m.total || 0));

  // Deduplicate arrows by (from_sq, to_sq)
  const seen = new Set();
  for (const mv of movesArray) {
    if (!mv.from_sq || !POS_COORDS[mv.from_sq] || !POS_COORDS[mv.to_sq]) continue;
    const key = `${mv.from_sq}-${mv.to_sq}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const [fx,, fz] = POS_COORDS[mv.from_sq];
    const [tx,, tz] = POS_COORDS[mv.to_sq];
    const from3 = new THREE.Vector3(fx, 0.58, fz);
    const to3   = new THREE.Vector3(tx, 0.58, tz);
    const dir   = new THREE.Vector3().subVectors(to3, from3);
    const len   = dir.length();
    if (len < 0.01) continue;
    const dirN = dir.clone().normalize();
    const q    = new THREE.Quaternion().setFromUnitVectors(_up, dirN);

    let col, opacity, shaftRadius, headRadius;
    if (mv.has_db_data) {
      col         = barColor(mv).getHex();
      opacity     = 0.22 + 0.65 * (mv.total / maxTotal);
      shaftRadius = 0.038;
      headRadius  = 0.115;
    } else {
      col         = 0x555555;
      opacity     = 0.18;
      shaftRadius = 0.020;
      headRadius  = 0.065;
    }

    const mat      = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity });
    const headLen  = Math.min(0.38, len * 0.28);
    const shaftLen = len - headLen - 0.04;

    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLen, 6), mat);
    shaft.position.copy(from3).addScaledVector(dirN, shaftLen / 2);
    shaft.setRotationFromQuaternion(q);

    const head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLen, 8), mat.clone());
    head.position.copy(from3).addScaledVector(dirN, shaftLen + headLen / 2);
    head.setRotationFromQuaternion(q);

    arrowGroup.add(shaft, head);
  }
}

// ── Malom overlay (rings + DTW labels) ───────────────────────────────────────

const malomRingGeo = new THREE.RingGeometry(0.26, 0.41, 24);
malomRingGeo.rotateX(-Math.PI / 2);

function rebuildMalomOverlay(movesArray) {
  malomGroup.clear();
  if (!movesArray) return;
  for (const mv of movesArray) {
    if (!mv.malom_wdl_after || !mv.to_sq || !POS_COORDS[mv.to_sq]) continue;
    const col = mv.malom_wdl_after === 'L' ? 0x22c55e
               : mv.malom_wdl_after === 'W' ? 0xef4444
               : 0xf59e0b;
    const mat  = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(malomRingGeo, mat);
    const [x,, z] = POS_COORDS[mv.to_sq];
    ring.position.set(x, 0.08, z);
    if (mv.malom_dtw_after != null) {
      const labelCol = mv.malom_wdl_after === 'L' ? '#4ade80'
                     : mv.malom_wdl_after === 'W' ? '#fca5a5'
                     : '#fcd34d';
      const sprite = makeDtwSprite(String(Math.abs(mv.malom_dtw_after)), labelCol);
      sprite.position.set(0, 0.6, 0);
      ring.add(sprite);
    }
    malomGroup.add(ring);
  }
}

const malomToggle = document.getElementById('malom-toggle');
if (malomToggle) {
  malomToggle.addEventListener('change', () => {
    malomGroup.visible = malomToggle.checked;
  });
}

const _explorerBandSel = document.getElementById('pred-band-select');
if (_explorerBandSel) {
  _explorerBandSel.addEventListener('change', () => {
    explorerEloBand = _explorerBandSel.value;
    if (currentData?.fen) loadPosition(currentData.fen);
  });
}

for (let _i = 0; _i < 3; _i++) {
  const _sel = document.getElementById(`net-slot-${_i}`);
  if (_sel) {
    const _idx = _i;
    _sel.addEventListener('change', () => {
      selectedNets[_idx] = _sel.value || null;
      updateNetLegend();
      if (_sel.value === 'regret') {
        _ensureRegret();
      } else {
        _refreshAfterStateChange();
      }
    });
  }
}

const _generalistToggle = document.getElementById('generalist-toggle');
if (_generalistToggle) {
  _generalistToggle.addEventListener('change', () => {
    generalistRingEnabled = _generalistToggle.checked;
    updateNetLegend();
    _refreshAfterStateChange();
  });
}

// ── Hint rings (HumanDB gold, Pred Human blue-purple, Sentinel blue, Heuristic purple) ──
// Each indicator uses a distinct ring radius so overlapping hints remain visible as
// concentric rings rather than z-fighting on the same geometry.

function _makeRingGeo(inner, outer) {
  const g = new THREE.RingGeometry(inner, outer, 32);
  g.rotateX(-Math.PI / 2);
  return g;
}

// Outermost → innermost: HumanDB, then 3 net slots, Capture
const _hintGeos = {
  human:   _makeRingGeo(0.44, 0.58),  // gold — largest
  slot0:   _makeRingGeo(0.32, 0.46),  // 2nd ring
  slot1:   _makeRingGeo(0.20, 0.34),  // 3rd ring
  slot2:   _makeRingGeo(0.08, 0.22),  // innermost ring
  capture: _makeRingGeo(0.32, 0.46),  // red/gold capture rings
};

function makeHintRing(sq, hexColor, geoKey, yOffset = 0) {
  if (!POS_COORDS[sq]) return null;
  const geo = _hintGeos[geoKey] || _hintGeos.human;
  const mat = new THREE.MeshBasicMaterial({ color: hexColor, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(geo, mat);
  const [x,, z] = POS_COORDS[sq];
  ring.position.set(x, 0.12 + yOffset, z);
  return ring;
}

function makeGeneralistArrow(toSq, fromSq) {
  if (!POS_COORDS[toSq]) return null;
  const mat = new THREE.MeshLambertMaterial({ color: 0xe07830, transparent: true, opacity: 0.92 });

  if (fromSq && POS_COORDS[fromSq]) {
    // Movement/fly phase: horizontal arrow pointing from fromSq toward toSq
    const [fx,, fz] = POS_COORDS[fromSq];
    const [tx,, tz] = POS_COORDS[toSq];
    const dx = tx - fx, dz = tz - fz;
    const len = Math.sqrt(dx * dx + dz * dz);
    const Y = 1.0;  // height above board
    const dir = new THREE.Vector3(dx / len, 0, dz / len);
    const origin = new THREE.Vector3(fx, Y, fz);
    const arrow = new THREE.ArrowHelper(dir, origin, len, 0xe07830, 0.50, 0.30);
    // ArrowHelper uses LineBasicMaterial for line — swap both sub-meshes to MeshLambertMaterial
    // so opacity works; instead just set the built-in line/cone colors directly
    arrow.line.material.color.set(0xe07830);
    arrow.line.material.linewidth = 3;
    arrow.cone.material.color.set(0xe07830);
    arrow.cone.material.transparent = true;
    arrow.cone.material.opacity = 0.92;
    return arrow;
  }

  // Placement phase: downward-pointing arrow above the destination square
  const [x,, z] = POS_COORDS[toSq];
  const group = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.35, 8), mat);
  shaft.position.set(0, 1.30, 0);
  group.add(shaft);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.42, 8), mat);
  cone.rotation.x = Math.PI;
  cone.position.set(0, 0.90, 0);
  group.add(cone);
  group.position.set(x, 0, z);
  return group;
}

function rebuildHints(allMoves, hasTrajData) {
  hintGroup.clear();
  if (!allMoves) return;

  if (selectionState === 'capture') {
    const capSqs = new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean));
    const bestCapSq = _getBestCaptureSq(pendingCaptureMoves);
    for (const sq of capSqs) {
      const ring = makeHintRing(sq, sq === bestCapSq ? 0xffd700 : 0xff3333, 'capture');
      if (ring) hintGroup.add(ring);
    }
    return;
  }

  const visibleMoves = (selectionState === 'piece_selected' && selectedPieceSq)
    ? allMoves.filter(m => m.from_sq === selectedPieceSq)
    : allMoves;
  if (visibleMoves.length === 0) return;

  // HumanDB best (gold, outermost)
  const dbMoves = visibleMoves.filter(m => m.has_db_data && (m.total || 0) > 0);
  if (hasTrajData !== false && dbMoves.length > 0) {
    const humanBestSq = [...dbMoves].sort((a, b) => wilsonLower(b.wins, b.total) - wilsonLower(a.wins, a.total))[0].to_sq;
    const ring = makeHintRing(humanBestSq, 0xffd700, 'human');
    if (ring) hintGroup.add(ring);
  }

  // 3 dynamic net slots
  const slotGeos = ['slot0', 'slot1', 'slot2'];
  for (let slotIdx = 0; slotIdx < 3; slotIdx++) {
    const netKey = selectedNets[slotIdx];
    if (!netKey) continue;
    const def = NET_DEFS[netKey];
    const scoredMoves = visibleMoves.filter(m => m[def.field] != null);
    if (scoredMoves.length === 0) continue;
    const bestMv = def.isHigherBetter
      ? [...scoredMoves].sort((a, b) => {
          const va = def.isAbsNorm ? Math.abs(a[def.field]) : a[def.field];
          const vb = def.isAbsNorm ? Math.abs(b[def.field]) : b[def.field];
          return vb - va;
        })[0]
      : [...scoredMoves].sort((a, b) => a[def.field] - b[def.field])[0];
    const ring = makeHintRing(bestMv.to_sq, def.hexColor, slotGeos[slotIdx]);
    if (ring) hintGroup.add(ring);
  }

  // Generalist arrow: horizontal (move/fly) or downward (placement)
  if (generalistRingEnabled && currentData?.generalist_top_sq) {
    const arrow = makeGeneralistArrow(
      currentData.generalist_top_sq,
      currentData.generalist_top_from ?? null,
    );
    if (arrow) hintGroup.add(arrow);
  }
}

// ── Piece colour highlights ───────────────────────────────────────────────────

function updatePieceHighlights() {
  const capSqs = (selectionState === 'capture')
    ? new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean))
    : new Set();

  const bestCapSq = selectionState === 'capture' ? _getBestCaptureSq(pendingCaptureMoves) : null;

  for (const mesh of pieceGroup.children) {
    const { pos, baseColor } = mesh.userData;
    if (selectionState === 'capture') {
      if (capSqs.has(pos)) {
        mesh.material.color.setHex(pos === bestCapSq ? 0xffd700 : 0xee2222);
        mesh.material.opacity = 1.0;
      } else {
        mesh.material.color.copy(baseColor);
        mesh.material.opacity = 0.45;
      }
    } else if (selectionState === 'piece_selected' && pos === selectedPieceSq) {
      mesh.material.color.setHex(0xffd700);
      mesh.material.opacity = 1.0;
    } else {
      mesh.material.color.copy(baseColor);
      mesh.material.opacity = 1.0;
    }
  }
}

// ── Selection state helpers ───────────────────────────────────────────────────

function filterMovesForState() {
  if (!currentData) return [];
  const all = currentData.moves || [];
  if (selectionState === 'capture') return [];
  if (selectionState === 'piece_selected' && selectedPieceSq) {
    return all.filter(mv => mv.from_sq === selectedPieceSq);
  }
  // idle + move/fly phase: no bars until a piece is selected
  if (currentPhase !== 'place' && selectionState === 'idle') return [];
  return all;
}

function updateStatusIndicator() {
  const el = document.getElementById('selection-status');
  if (!el) return;
  if (currentPhase === 'place' && selectionState === 'idle') {
    el.style.display = 'none';
    return;
  }
  let msg;
  if (selectionState === 'capture') {
    msg = '▶ Click an opponent piece to capture  ·  Esc or click empty to cancel';
  } else if (selectionState === 'piece_selected') {
    msg = `▶ ${selectedPieceSq} selected — click a destination square`;
  } else {
    msg = '▶ Click one of your pieces to move';
  }
  el.textContent = msg;
  el.style.display = 'block';
}

function updateNetLegend() {
  const legendEl = document.getElementById('hint-ring-legend');
  if (!legendEl) return;
  const slotSizes = [{ w:14, h:14 }, { w:10, h:10 }, { w:7, h:7 }];
  const slotTitles = ['2nd ring', '3rd ring', 'Innermost ring'];
  let html = `<div class="legend-row" style="margin:0;" title="Outermost ring">` +
    `<div style="width:18px;height:18px;border-radius:50%;border:2px solid #ffd700;flex-shrink:0;"></div>&nbsp;HumanDB best</div>`;
  for (let i = 0; i < 3; i++) {
    const netKey = selectedNets[i];
    if (!netKey) continue;
    const def = NET_DEFS[netKey];
    const s = slotSizes[i];
    html += `<div class="legend-row" style="margin:0;" title="${slotTitles[i]}">` +
      `<div style="width:${s.w}px;height:${s.h}px;border-radius:50%;border:2px solid ${def.cssColor};flex-shrink:0;"></div>&nbsp;${def.label} best</div>`;
  }
  if (generalistRingEnabled) {
    html += `<div class="legend-row" style="margin:0;" title="Generalist AI (outer ring, orange)">` +
      `<div style="width:18px;height:18px;border-radius:50%;border:2px solid #e07830;flex-shrink:0;"></div>&nbsp;Generalist best</div>`;
  }
  html += `<div style="font-size:.65rem;color:#666;margin-top:0.3rem;line-height:1.5"` +
    ` title="T = observed move frequency from human trajectory data&#10;P = model-predicted human move probability (Elo-conditioned)&#10;T:— = move not seen in trajectory sample">T=traj &nbsp; P=pred &nbsp; T:—=unseen</div>`;
  legendEl.innerHTML = html;
}

function updateBestBtn() {
  const btn = document.getElementById('btn-best');
  if (!btn) return;
  const hasMalom = currentData?.position_stats?.canonical_winning_move != null;
  btn.textContent = (hasMalom ? 'Malom best' : 'Heuristic best') + ' →';
}

function _refreshAfterStateChange() {
  const filtered = filterMovesForState();
  const allMoves = currentData?.moves || [];
  // Idle movement phase: show aggregated per-piece bars but no arrows (keeps board clean)
  const barsData = (currentPhase !== 'place' && selectionState === 'idle') ? allMoves : filtered;
  rebuildBars(barsData);
  rebuildArrows(filtered);
  rebuildHints(allMoves, currentData?.has_traj_data);
  updatePieceHighlights();
  updateStatusIndicator();
}

// ── Raycasting / hover / click ────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
let   hoveredBar   = null;
let   hoveredPiece = null;
let   hoveredPad   = null;

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  // ── Bar hover ──
  if (hoveredBar) {
    for (const m of (barGroupMap.get(hoveredBar.userData.toSq) || [hoveredBar])) {
      m.material.color.copy(m.userData.baseColor);
      m.material.opacity = m.userData.baseOpacity ?? (m.userData.moveData?.has_db_data ? 0.88 : 0.55);
    }
    hoveredBar = null;
    tooltip.style.display = 'none';
  }
  const barHits = raycaster.intersectObjects(barGroup.children);
  if (barHits.length > 0) {
    const mesh = barHits[0].object;
    hoveredBar = mesh;
    for (const m of (barGroupMap.get(mesh.userData.toSq) || [mesh])) {
      m.material.color.set(C.barHov);
      m.material.opacity = 1.0;
    }
    showTooltip(e.clientX, e.clientY, mesh.userData.moveData);
    document.querySelectorAll('.move-item').forEach(el =>
      el.classList.toggle('highlighted', el.dataset.notation === mesh.userData.notation));
  } else {
    document.querySelectorAll('.move-item.highlighted').forEach(el => el.classList.remove('highlighted'));
  }

  // ── Piece hover ──
  const prevHoveredPiece = hoveredPiece;
  hoveredPiece = null;

  const pieceTargets = [];
  if (selectionState === 'capture') {
    const capSqs = new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean));
    for (const mesh of pieceGroup.children) {
      if (capSqs.has(mesh.userData.pos)) pieceTargets.push(mesh);
    }
  } else if (currentPhase !== 'place') {
    // Own pieces are hoverable in move/fly phase
    for (const mesh of pieceGroup.children) {
      if (mesh.userData.color === currentTurn) pieceTargets.push(mesh);
    }
  }

  if (pieceTargets.length > 0) {
    const hits = raycaster.intersectObjects(pieceTargets);
    if (hits.length > 0) hoveredPiece = hits[0].object;
  }

  if (prevHoveredPiece !== hoveredPiece) {
    updatePieceHighlights();
    if (hoveredPiece) {
      hoveredPiece.material.color.setHex(0xffd700);
      hoveredPiece.material.opacity = 1.0;
    }
  }

  // Show aggregate tooltip when hovering a moveable piece (no bar in the way)
  if (!hoveredBar) {
    if (hoveredPiece && selectionState === 'idle' && currentPhase !== 'place') {
      const src = hoveredPiece.userData.pos;
      const synthMv = barGroupMap.get(src)?.[0]?.userData?.moveData;
      if (synthMv?._isAggregate) {
        showTooltip(e.clientX, e.clientY, synthMv);
      } else if (prevHoveredPiece && !hoveredPiece) {
        tooltip.style.display = 'none';
      }
    } else if (!hoveredPiece && prevHoveredPiece) {
      tooltip.style.display = 'none';
    }
  }

  // ── Pad hover (for placement / destination click without bar) ──
  hoveredPad = null;
  if (!hoveredPiece && !hoveredBar) {
    const padHits = raycaster.intersectObjects(padMeshList);
    if (padHits.length > 0) hoveredPad = padHits[0].object;
  }

  canvas.style.cursor = (hoveredBar || hoveredPiece || hoveredPad) ? 'pointer' : 'default';
}

canvas.addEventListener('mousemove', onMouseMove);

canvas.addEventListener('click', () => {
  // ── Capture mode ──
  if (selectionState === 'capture') {
    if (hoveredPiece) {
      const sq = hoveredPiece.userData.pos;
      const mv = pendingCaptureMoves.find(m => m.capture_sq === sq);
      if (mv) {
        selectionState      = 'idle';
        selectedPieceSq     = null;
        pendingCaptureMoves = [];
        applyMove(mv.notation);
        return;
      }
    }
    // Cancel: click on non-capturable area
    selectionState = captureReturnState;
    if (selectionState !== 'piece_selected') selectedPieceSq = null;
    pendingCaptureMoves = [];
    _refreshAfterStateChange();
    return;
  }

  // ── Piece selected ──
  if (selectionState === 'piece_selected') {
    if (hoveredBar) {
      const { allMoves, needsCapture } = hoveredBar.userData;
      if (needsCapture) {
        pendingCaptureMoves = allMoves;
        captureReturnState  = 'piece_selected';
        selectionState      = 'capture';
        _refreshAfterStateChange();
      } else {
        const notation = allMoves.find(m => !m.capture_sq)?.notation ?? allMoves[0].notation;
        selectionState  = 'idle';
        selectedPieceSq = null;
        applyMove(notation);
      }
      return;
    }
    if (hoveredPad && !hoveredPiece) {
      // Click on destination pad square
      const sq = hoveredPad.userData.pos;
      const mvsForSq = (currentData?.moves || []).filter(m => m.from_sq === selectedPieceSq && m.to_sq === sq);
      if (mvsForSq.length > 0) {
        const needsCapture = mvsForSq.every(m => m.capture_sq != null);
        if (needsCapture) {
          pendingCaptureMoves = mvsForSq;
          captureReturnState  = 'piece_selected';
          selectionState      = 'capture';
          _refreshAfterStateChange();
        } else {
          const notation = mvsForSq.find(m => !m.capture_sq)?.notation ?? mvsForSq[0].notation;
          selectionState  = 'idle';
          selectedPieceSq = null;
          applyMove(notation);
        }
      } else {
        selectionState  = 'idle';
        selectedPieceSq = null;
        _refreshAfterStateChange();
      }
      return;
    }
    if (hoveredPiece) {
      const sq = hoveredPiece.userData.pos;
      if (sq === selectedPieceSq) {
        // Deselect
        selectionState  = 'idle';
        selectedPieceSq = null;
      } else if (hoveredPiece.userData.color === currentTurn) {
        // Switch to different own piece
        selectedPieceSq = sq;
      }
      _refreshAfterStateChange();
      return;
    }
    // Click on empty space — deselect
    selectionState  = 'idle';
    selectedPieceSq = null;
    _refreshAfterStateChange();
    return;
  }

  // ── Idle ──
  if (currentPhase === 'place') {
    if (hoveredBar) {
      const { allMoves, needsCapture } = hoveredBar.userData;
      if (needsCapture) {
        pendingCaptureMoves = allMoves;
        captureReturnState  = 'idle';
        selectionState      = 'capture';
        _refreshAfterStateChange();
      } else {
        const notation = allMoves.find(m => !m.capture_sq)?.notation ?? allMoves[0].notation;
        applyMove(notation);
      }
    } else if (hoveredPad) {
      // Click on a board square pad — find move with matching to_sq
      const sq = hoveredPad.userData.pos;
      const mvsForSq = (currentData?.moves || []).filter(m => m.to_sq === sq);
      if (mvsForSq.length > 0) {
        const needsCapture = mvsForSq.every(m => m.capture_sq != null);
        if (needsCapture) {
          pendingCaptureMoves = mvsForSq;
          captureReturnState  = 'idle';
          selectionState      = 'capture';
          _refreshAfterStateChange();
        } else {
          const notation = mvsForSq.find(m => !m.capture_sq)?.notation ?? mvsForSq[0].notation;
          applyMove(notation);
        }
      }
    }
  } else {
    // move / fly phase — require piece click first (piece or pad with own piece on it)
    const clickSq = hoveredPiece?.userData.pos ?? (hoveredPad?.userData.pos ?? null);
    if (clickSq && hoveredPiece?.userData.color === currentTurn) {
      selectedPieceSq = clickSq;
      selectionState  = 'piece_selected';
      _refreshAfterStateChange();
    }
  }
});

// Escape key cancels any active selection
window.addEventListener('keydown', e => {
  if (e.key === 'Escape' && selectionState !== 'idle') {
    if (selectionState === 'capture') {
      selectionState = captureReturnState;
      if (selectionState !== 'piece_selected') selectedPieceSq = null;
      pendingCaptureMoves = [];
    } else {
      selectionState  = 'idle';
      selectedPieceSq = null;
    }
    _refreshAfterStateChange();
  }
});

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTooltip(cx, cy, mv) {
  if (!mv) return;
  const allHAbsMax = Math.max(1, ...(currentData?.moves || []).map(m => Math.abs(m.heuristic_score || 0)));

  // Aggregated per-piece tooltip (movement phase, no piece selected)
  if (mv._isAggregate) {
    const hasTrajData = currentData?.has_traj_data;
    const winPct  = mv.total > 0 ? `${(mv.win_pct * 100).toFixed(1)}%` : '—';
    const heurPct = mv._bestHeurAbs > 0 ? `${(mv._bestHeurAbs / allHAbsMax * 100).toFixed(0)}%` : '—';
    const trajPct = mv._trajSum > 0 ? Math.round(mv._trajSum * 100) : 0;
    const predPct = mv._predSum > 0 ? Math.round(mv._predSum * 100) : 0;
    const trajRow = hasTrajData !== false && trajPct > 0
      ? `<div class="tt-row"><span class="tt-label">T (piece)</span><span>T:${trajPct}%</span></div>`
      : hasTrajData === false && predPct > 0
        ? `<div class="tt-row"><span class="tt-label">P (piece)</span><span style="color:#5591c7">P:${predPct}%</span></div>`
        : '';
    let netRows = '';
    if (mv._bestByNet) {
      for (const [k, def] of Object.entries(NET_DEFS)) {
        const v = mv._bestByNet[k];
        if (v == null) continue;
        const disp = def.isAbsNorm
          ? `${(Math.abs(v) / allHAbsMax * 100).toFixed(0)}% (${v >= 0 ? '+' : ''}${v})`
          : `${(v * 100).toFixed(1)}%`;
        netRows += `<div class="tt-row"><span class="tt-label" style="color:${def.cssColor}">${def.label}</span><span style="color:${def.cssColor}">${disp}</span></div>`;
      }
    }
    tooltip.innerHTML = `
      <div class="tt-notation">${mv._src} — all moves</div>
      ${mv.total > 0 ? `
      <div class="tt-row"><span class="tt-label">Win%</span><span>${winPct}</span></div>
      <div class="tt-row"><span class="tt-label">W/D/L</span><span>${mv.wins}/${mv.draws}/${mv.losses}</span></div>
      <div class="tt-row"><span class="tt-label">Total games</span><span>${mv.total}</span></div>
      ` : ''}
      ${trajRow}
      ${netRows}
    `;
    const wr = wrap.getBoundingClientRect();
    let tx = cx - wr.left + 14, ty = cy - wr.top - 10;
    if (tx + 180 > wr.width)  tx = cx - wr.left - 180;
    if (ty + 180 > wr.height) ty = cy - wr.top  - 180;
    tooltip.style.left = tx + 'px'; tooltip.style.top = ty + 'px';
    tooltip.style.display = 'block';
    return;
  }

  const sentText = mv.sentinel_score != null ? `${(mv.sentinel_score * 100).toFixed(1)}%` : '—';
  const heurText = mv.heuristic_score != null
    ? `${(Math.abs(mv.heuristic_score) / allHAbsMax * 100).toFixed(0)}% (${mv.heuristic_score >= 0 ? '+' : ''}${mv.heuristic_score})`
    : '—';

  const hasTrajData = currentData?.has_traj_data;
  const posTotal = (currentData?.moves || []).reduce((s, m) => s + (m.total || 0), 0);

  let dbRows = '';
  let trajRow = '';
  if (mv.has_db_data) {
    const wdlText = mv.malom_wdl_after
      ? `${mv.malom_wdl_after}${mv.malom_dtw_after != null ? ` (${mv.malom_dtw_after > 0 ? '+' : ''}${mv.malom_dtw_after} DTW)` : ''}`
      : '—';
    const tPct = posTotal > 0 ? (mv.total / posTotal * 100).toFixed(1) : '—';
    dbRows = `
    <div class="tt-row"><span class="tt-label">Win%</span><span>${(mv.win_pct*100).toFixed(1)}%</span></div>
    <div class="tt-row"><span class="tt-label">W/D/L</span><span>${mv.wins}/${mv.draws}/${mv.losses}</span></div>
    <div class="tt-row"><span class="tt-label">T choice</span><span>T:${tPct}% (n=${mv.total})</span></div>
    <div class="tt-row"><span class="tt-label">Avg plies left</span><span>${mv.avg_moves_to_end.toFixed(0)}</span></div>
    <div class="tt-row"><span class="tt-label">Malom (after)</span><span>${wdlText}</span></div>`;
  } else if (hasTrajData) {
    trajRow = `<div class="tt-row"><span class="tt-label" style="opacity:0.55">T choice</span><span style="opacity:0.55">T:— not observed</span></div>`;
  }

  const predRow = mv.pred_human_prob != null
    ? `<div class="tt-row"><span class="tt-label" style="color:#a06fe0">Pred</span><span style="color:#a06fe0">P:${(mv.pred_human_prob * 100).toFixed(1)}%</span></div>`
    : '';
  const gapnetRow = mv.gapnet_score != null
    ? `<div class="tt-row"><span class="tt-label" style="color:#cc5555">GapNet risk</span><span style="color:#cc5555">${(mv.gapnet_score * 100).toFixed(1)}%</span></div>`
    : '';
  const valueRow = mv.value_score != null
    ? `<div class="tt-row"><span class="tt-label" style="color:#50aaaa">ValueNet</span><span style="color:#50aaaa">${(mv.value_score * 100).toFixed(1)}%</span></div>`
    : '';
  const prefRow = mv.pref_score != null
    ? `<div class="tt-row"><span class="tt-label" style="color:#c4a020">PrefNet</span><span style="color:#c4a020">${(mv.pref_score * 100).toFixed(1)}%</span></div>`
    : '';
  const regretRow = mv.regret_score != null
    ? `<div class="tt-row"><span class="tt-label" style="color:#ff6020">Regret</span><span style="color:#ff6020">${(mv.regret_score * 100).toFixed(1)}%</span></div>`
    : '';

  tooltip.innerHTML = `
    <div class="tt-notation">${mv.notation}</div>
    ${dbRows}
    ${trajRow}
    <div class="tt-row"><span class="tt-label" style="color:#e07030">Sentinel</span><span style="color:#e07030">${sentText}</span></div>
    <div class="tt-row"><span class="tt-label" style="color:#c8a96e">Heuristic</span><span style="color:#c8a96e">${heurText}</span></div>
    ${predRow}${gapnetRow}${valueRow}${prefRow}${regretRow}
  `;
  const wr = wrap.getBoundingClientRect();
  let tx = cx - wr.left + 14;
  let ty = cy - wr.top  - 10;
  if (tx + 180 > wr.width)  tx = cx - wr.left - 180;
  if (ty + 160 > wr.height) ty = cy - wr.top  - 160;
  tooltip.style.left    = tx + 'px';
  tooltip.style.top     = ty + 'px';
  tooltip.style.display = 'block';
}

// ── Side-panel updates ────────────────────────────────────────────────────────

function updatePanel(data) {
  const badge = document.getElementById('turn-badge');
  badge.textContent = data.turn === 'W' ? 'White' : 'Black';
  badge.className   = 'turn-badge ' + (data.turn === 'W' ? 'white' : 'black');

  const posEl = document.getElementById('pos-stats');
  const ps    = data.position_stats;
  if (!ps) {
    posEl.innerHTML = '<div id="no-data-notice">No HumanDB data for this position.</div>';
  } else {
    const tot = Math.max(1, ps.total_games);
    const wp  = (ps.wins   / tot * 100).toFixed(1);
    const dp  = (ps.draws  / tot * 100).toFixed(1);
    const lp  = (ps.losses / tot * 100).toFixed(1);
    const malomHtml = ps.malom_wdl
      ? `<span class="malom-badge malom-${ps.malom_wdl.toLowerCase()}">${ps.malom_wdl}${ps.malom_dtw != null ? ` DTW ${ps.malom_dtw}` : ''}</span>`
      : '';
    posEl.innerHTML = `
      <div class="stat-row"><span class="stat-label">Games</span><span class="stat-val">${ps.total_games.toLocaleString()}</span></div>
      <div class="wdl-bar">
        <div class="w" style="width:${wp}%"></div>
        <div class="d" style="width:${dp}%"></div>
        <div class="l" style="width:${lp}%"></div>
      </div>
      <div class="stat-row"><span class="stat-label">Win%</span><span class="stat-val" style="color:#22c55e">${wp}%</span></div>
      <div class="stat-row"><span class="stat-label">Draw%</span><span class="stat-val" style="color:#f59e0b">${dp}%</span></div>
      <div class="stat-row"><span class="stat-label">Loss%</span><span class="stat-val" style="color:#ef4444">${lp}%</span></div>
      ${malomHtml}
    `;
  }

  const listEl = document.getElementById('move-list');
  listEl.innerHTML = '';
  if (data.moves && data.moves.length > 0) {
    const hAbsMax = Math.max(1, ...data.moves.map(m => Math.abs(m.heuristic_score || 0)));
    const activeNets = selectedNets.filter(Boolean);

    // Helper: format one net value cell for the list
    const _fmtNetCell = (mv, netKey) => {
      const def = NET_DEFS[netKey];
      if (!def) return '<span class="move-net-val" style="color:#555">—</span>';
      const raw = netKey === 'regret' ? mv.regret_score : mv[def.field];
      if (raw == null) return '<span class="move-net-val" style="color:#555">—</span>';
      const pct = def.isAbsNorm
        ? (Math.abs(raw) / hAbsMax * 100).toFixed(0)
        : (raw * 100).toFixed(0);
      return `<span class="move-net-val" style="color:${def.cssColor}">${pct}%</span>`;
    };

    // Column header row (net names, right-aligned to match row values)
    if (activeNets.length > 0) {
      const hdrEl = document.createElement('div');
      hdrEl.className = 'move-list-net-hdr';
      hdrEl.innerHTML =
        '<span class="move-list-net-hdr-spacer"></span>' +
        activeNets.map(k => {
          const def = NET_DEFS[k];
          return def ? `<span class="move-net-hdr" style="color:${def.cssColor}">${def.label}</span>` : '';
        }).join('');
      listEl.appendChild(hdrEl);
    }

    for (const mv of data.moves) {
      const col    = barColor(mv);
      const colHex = '#' + col.getHexString();
      const netCells = activeNets.map(k => _fmtNetCell(mv, k)).join('');

      let rightContent = '';
      if (mv.has_db_data) {
        const wdl = mv.malom_wdl_after
          ? `<span class="move-malom" style="background:${mv.malom_wdl_after==='L'?'#16532a':mv.malom_wdl_after==='W'?'#7f1d1d':'#78350f'};color:${mv.malom_wdl_after==='L'?'#4ade80':mv.malom_wdl_after==='W'?'#fca5a5':'#fcd34d'}">${mv.malom_wdl_after}${mv.malom_dtw_after!=null?' '+mv.malom_dtw_after:''}</span>`
          : '';
        rightContent = `
          <span class="move-sub">${mv.total}</span>
          ${wdl}
          <span class="move-pct" style="color:${colHex}">${(mv.win_pct*100).toFixed(1)}%</span>
          ${netCells}
        `;
      } else {
        rightContent = `<span style="flex:1"></span>${netCells}`;
      }

      const item = document.createElement('div');
      item.className = 'move-item' + (mv.has_db_data ? '' : ' move-item-no-db');
      item.dataset.notation = mv.notation;
      item.innerHTML = `
        <div class="move-bar-swatch" style="background:${colHex}"></div>
        <span class="move-notation">${mv.notation}</span>
        ${rightContent}
      `;
      item.addEventListener('click', () => applyMove(mv.notation));
      item.addEventListener('mouseenter', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) {
          for (const m of (barGroupMap.get(entry.mesh.userData.toSq) || [entry.mesh])) {
            m.material.color.set(C.barHov); m.material.opacity = 1;
          }
        }
      });
      item.addEventListener('mouseleave', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) {
          for (const m of (barGroupMap.get(entry.mesh.userData.toSq) || [entry.mesh])) {
            m.material.color.copy(m.userData.baseColor);
            m.material.opacity = m.userData.baseOpacity ?? (mv.has_db_data ? 0.88 : 0.55);
          }
        }
      });
      listEl.appendChild(item);
    }
  } else {
    listEl.innerHTML = '<div style="padding:0.5rem 0.75rem;color:#8a7a5a;font-size:0.8rem">No move data.</div>';
  }

  const lineEl = document.getElementById('winning-line');
  lineEl.textContent = data.winning_line && data.winning_line.length > 0
    ? data.winning_line.join(' → ') : '—';
}

// ── Navigation ────────────────────────────────────────────────────────────────

const history = [];
let   currentData  = null;
let   _playedMoves = [];   // { notation, color } records for the history panel
let   _startingFen = null; // FEN at the start of the current exploration session

// ── Played-moves panel ────────────────────────────────────────────────────────

const _pmPanel    = document.getElementById('pm-panel');
const _pmBody     = document.getElementById('pm-body');
const _pmCopyBtn  = document.getElementById('pm-copy-btn');
const _pmEmpty    = document.getElementById('pm-empty');

function _renderPlayedMoves() {
  _pmBody.innerHTML = '';
  if (_playedMoves.length === 0) {
    _pmBody.appendChild(Object.assign(document.createElement('div'), { id: 'pm-empty', textContent: 'No moves yet.' }));
    return;
  }
  // Header
  const hdr = document.createElement('div');
  hdr.className = 'pm-row pm-row-hdr';
  hdr.innerHTML = '<span class="pm-num">#</span><span class="pm-w">⬜</span><span class="pm-b">⬛</span>';
  _pmBody.appendChild(hdr);

  // Pair into rows
  const rows = [];
  let i = 0;
  while (i < _playedMoves.length) {
    const mv = _playedMoves[i];
    if (mv.color === 'W') {
      const next = _playedMoves[i + 1];
      rows.push([mv.notation, next?.color === 'B' ? next.notation : '']);
      i += (next?.color === 'B') ? 2 : 1;
    } else {
      rows.push(['—', mv.notation]);
      i += 1;
    }
  }

  rows.forEach((pair, idx) => {
    const row = document.createElement('div');
    row.className = 'pm-row';
    row.innerHTML =
      `<span class="pm-num">${idx + 1}.</span>` +
      `<span class="pm-w">${pair[0] || ''}</span>` +
      `<span class="pm-b">${pair[1] || ''}</span>`;
    _pmBody.appendChild(row);
  });
  _pmBody.scrollTop = _pmBody.scrollHeight;
}

function _copyPlayedMoves() {
  if (_playedMoves.length === 0) return;
  const startFen = _startingFen || '........................|W|0|0';
  const rows = [];
  let i = 0;
  while (i < _playedMoves.length) {
    const mv = _playedMoves[i];
    if (mv.color === 'W') {
      const next = _playedMoves[i + 1];
      rows.push([mv.notation, next?.color === 'B' ? next.notation : '']);
      i += (next?.color === 'B') ? 2 : 1;
    } else {
      rows.push(['—', mv.notation]);
      i += 1;
    }
  }
  const moveText = rows.map((pair, idx) => `${idx + 1}. ${pair[0]}${pair[1] ? ' ' + pair[1] : ''}`).join('\n');
  const text = `FEN: ${startFen}\n${moveText}`;
  navigator.clipboard.writeText(text).then(() => {
    const prev = _pmCopyBtn.textContent;
    _pmCopyBtn.textContent = 'Copied!';
    setTimeout(() => { _pmCopyBtn.textContent = prev; }, 1500);
  }).catch(() => {
    prompt('Copy this game:', text);
  });
}

if (_pmCopyBtn) _pmCopyBtn.addEventListener('click', _copyPlayedMoves);

async function loadPosition(fen) {
  loading.style.display = 'flex';
  // Reset all selection state for new position
  selectionState      = 'idle';
  selectedPieceSq     = null;
  pendingCaptureMoves = [];
  captureReturnState  = 'idle';
  hoveredBar          = null;
  hoveredPiece        = null;
  tooltip.style.display = 'none';

  try {
    const res  = await fetch('/api/explorer/position?fen=' + encodeURIComponent(fen) + '&elo_band=' + explorerEloBand);
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    currentData  = data;
    currentTurn  = data.turn;
    currentPhase = data.phase || 'move';
    document.getElementById('fen-input').value = data.fen;
    rebuildPieces(data.board);
    const initialMoves = filterMovesForState();
    const allMovesInit = data.moves || [];
    const barsInit = (currentPhase !== 'place' && selectionState === 'idle') ? allMovesInit : initialMoves;
    rebuildBars(barsInit);
    rebuildArrows(initialMoves);
    rebuildMalomOverlay(data.moves || []);
    rebuildHints(data.moves || [], data.has_traj_data);
    updatePanel(data);
    updatePieceHighlights();
    updateStatusIndicator();
    updateBestBtn();
    document.getElementById('btn-back').disabled = history.length === 0;
    const backLink = document.querySelector('a.btn-back');
    if (backLink && data.fen) backLink.href = '/?setup_fen=' + encodeURIComponent(data.fen);
    // If any net slot shows regret, fetch async and re-render when ready
    if (selectedNets.includes('regret')) _ensureRegret();
  } catch (err) {
    alert('Failed to load position: ' + err.message);
  } finally {
    loading.style.display = 'none';
  }
}

async function applyMove(notation) {
  if (!currentData) return;
  if (_otToggle && _otToggle.checked) {
    _otPath.push(notation);
    _otRender();
    document.getElementById('btn-back').disabled = false;
    return;
  }
  // Record starting FEN on first move
  if (_playedMoves.length === 0) _startingFen = currentData.fen;
  _playedMoves.push({ notation, color: currentTurn });
  history.push(currentData.fen);
  await loadPosition(await fenAfterMove(currentData.fen, notation));
  _renderPlayedMoves();
}

async function fenAfterMove(fen, move) {
  const res  = await fetch(`/api/explorer/move?fen=${encodeURIComponent(fen)}&move=${encodeURIComponent(move)}`);
  const data = await res.json();
  return data.fen || fen;
}

document.getElementById('btn-back').addEventListener('click', () => {
  if (_otToggle && _otToggle.checked && _otPath.length > 0) {
    _otPath.pop();
    _otRender();
    document.getElementById('btn-back').disabled = _otPath.length === 0;
    return;
  }
  if (history.length === 0) return;
  _playedMoves.pop();
  if (_playedMoves.length === 0) _startingFen = null;
  _renderPlayedMoves();
  loadPosition(history.pop());
});

document.getElementById('btn-reset').addEventListener('click', () => {
  history.length = 0;
  _playedMoves = [];
  _startingFen = null;
  _renderPlayedMoves();
  if (_otToggle && _otToggle.checked) {
    _otPath = [];
    _otRender();
    document.getElementById('btn-back').disabled = true;
  } else {
    loadPosition('........................|W|0|0');
  }
});

document.getElementById('btn-best').addEventListener('click', async () => {
  if (!currentData) return;
  const best = currentData.position_stats?.canonical_winning_move;
  if (best) { await applyMove(best); return; }
  // Fallback: best heuristic move
  const moves = currentData.moves || [];
  if (moves.length === 0) return;
  const bestHeur = moves.reduce((a, b) =>
    (b.heuristic_score ?? -Infinity) > (a.heuristic_score ?? -Infinity) ? b : a
  );
  await applyMove(bestHeur.notation);
});

document.getElementById('btn-go').addEventListener('click', () => {
  const fen = document.getElementById('fen-input').value.trim();
  if (fen) { history.length = 0; loadPosition(fen); }
});
document.getElementById('fen-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-go').click();
});

// ── Resize ────────────────────────────────────────────────────────────────────

function resize() {
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();

// ── Render loop ───────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
animate();

// ── Boot ──────────────────────────────────────────────────────────────────────

updateNetLegend();  // populate legend before first position loads
const _urlFen = new URLSearchParams(window.location.search).get('fen');
loadPosition(_urlFen || '........................|W|0|0');

// ── Opening Tree ──────────────────────────────────────────────────────────────

let _otData      = null;   // full tree JSON from /api/opening_tree
let _otPath      = [];     // current drilldown path (array of move strings)
let _otLoaded    = false;
let _otSyncToken = 0;      // request token to cancel stale board-sync fetches

const _otLeftPanel = document.getElementById('ot-left-panel');
const _otBody      = document.getElementById('ot-body');
const _otBreadcrumb = document.getElementById('ot-breadcrumb');
const _otRows      = document.getElementById('ot-rows');
const _otDepthIn   = document.getElementById('ot-depth');
const _otDepthVal  = document.getElementById('ot-depth-val');
const _otToggle    = document.getElementById('ot-toggle');

function _otNodeAtPath(data, path) {
  let node = data;
  for (const mv of path) {
    const child = (node.children || []).find(c => c.move === mv);
    if (!child) return null;
    node = child;
  }
  return node;
}

function _otRenderBreadcrumb() {
  _otBreadcrumb.innerHTML = '';
  const root = document.createElement('span');
  root.className = 'ot-crumb ot-crumb-root';
  root.textContent = 'Opening tree';
  root.onclick = () => { _otPath = []; _otRender(); };
  _otBreadcrumb.appendChild(root);
  for (let i = 0; i < _otPath.length; i++) {
    const sep = document.createElement('span');
    sep.className = 'ot-crumb-sep';
    sep.textContent = ' › ';
    _otBreadcrumb.appendChild(sep);
    const crumb = document.createElement('span');
    // Even index = White's move (ply 1,3,5…), odd = Black's (ply 2,4,6…)
    crumb.className = 'ot-crumb ' + (i % 2 === 0 ? 'ot-crumb-w' : 'ot-crumb-b');
    crumb.textContent = _otPath[i];
    const idx = i;
    crumb.onclick = () => { _otPath = _otPath.slice(0, idx + 1); _otRender(); };
    _otBreadcrumb.appendChild(crumb);
  }
  if (_otPath.length > 0) {
    const plyLabel = document.createElement('span');
    plyLabel.className = 'ot-ply-label';
    plyLabel.textContent = `(ply ${_otPath.length})`;
    _otBreadcrumb.appendChild(plyLabel);
  }
}

function _otRender(syncBoard = true) {
  if (!_otData) {
    _otRows.innerHTML = '<div id="ot-loading">Loading…</div>';
    return;
  }
  _otRenderBreadcrumb();
  _otUpdateSaveBar();
  const node = _otNodeAtPath(_otData, _otPath);
  const children = node ? (node.children || []) : [];

  _otRows.innerHTML = '';

  // Show divergence message when off-book
  if (_otPath.length > 0 && !node) {
    const divergePly = (() => {
      let n = _otData;
      for (let i = 0; i < _otPath.length; i++) {
        const c = (n.children || []).find(ch => ch.move === _otPath[i]);
        if (!c) return i + 1;
        n = c;
      }
      return _otPath.length;
    })();
    const msg = document.createElement('div');
    msg.className = 'ot-diverge-msg';
    msg.textContent = `Diverged from book at ply ${divergePly}. Enter a name below to save this path.`;
    _otRows.appendChild(msg);
    if (syncBoard) _otSyncBoard();
    return;
  }

  const showNovel = document.getElementById('ot-show-novel')?.checked !== false;
  const visibleChildren = showNovel ? children : children.filter(c => (c.source || 'book') !== 'learned');

  if (!visibleChildren.length) {
    _otRows.innerHTML = '<div id="ot-empty">No data at this depth.</div>';
    if (syncBoard) _otSyncBoard();
    return;
  }

  for (const child of visibleChildren) {
    const total  = child.w_wins + child.draws + child.b_wins;
    const hasWdl = total > 0;
    const wPct   = hasWdl ? child.w_wins / total * 100 : 0;
    const dPct   = hasWdl ? child.draws  / total * 100 : 0;
    const bPct   = hasWdl ? child.b_wins / total * 100 : 0;
    const isLeaf = !child.children || child.children.length === 0;
    // White plays on odd plies (ply 1,3,5…), Black on even (2,4,6…)
    const isWhiteTurn = (child.ply % 2 === 1);
    const src = child.source || 'book';

    const row = document.createElement('div');
    row.className = 'ot-row' + (src === 'learned' ? ' ot-row-novel' : '');

    const turnDot = `<span class="ot-turn-dot ${isWhiteTurn ? 'ot-turn-w' : 'ot-turn-b'}"></span>`;
    // Prefer terminal names; fall back to through_openings when there are ≤2
    const throughNames = child.through_openings || [];
    const displayNames = child.opening_names.length
      ? child.opening_names
      : (throughNames.length <= 2 ? throughNames : [throughNames[0], `+${throughNames.length - 1} more`]);
    const srcBadge = src === 'learned'
      ? '<span class="ot-src ot-src-novel">Novel</span>'
      : src === 'human'
      ? '<span class="ot-src ot-src-human">Human</span>'
      : '';
    const nameHtml = displayNames.length
      ? displayNames.map(n => `<span class="ot-opening-label" title="${throughNames.join(', ')}">${n}</span>`).join('')
      : '';
    const expandHint = !isLeaf
      ? `<span class="ot-expand-hint">${child.children.length}›</span>`
      : '';

    const barWidth = Math.min(100, child.human_pct);
    const pctText  = child.human_pct > 0 ? child.human_pct.toFixed(1) + '%' : '—';

    const wdlHtml = hasWdl
      ? `<div class="ot-wdl-bar">
           <div class="ot-wdl-w" style="width:${wPct.toFixed(1)}%"></div>
           <div class="ot-wdl-d" style="width:${dPct.toFixed(1)}%"></div>
           <div class="ot-wdl-b" style="width:${bPct.toFixed(1)}%"></div>
         </div>
         <span class="ot-wdl-text">${child.w_wins}W ${child.draws}D ${child.b_wins}B</span>`
      : '<span class="ot-wdl-text" style="color:#3a2e1e">—</span>';

    row.innerHTML = `
      <div class="ot-row-top">
        <div class="ot-move">${turnDot}${child.move}</div>
        ${expandHint}
      </div>
      <div class="ot-names-cell">${srcBadge}${nameHtml}</div>
      <div class="ot-pct-cell">
        <div class="ot-pct-track"><div class="ot-pct-fill" style="width:${barWidth}%"></div></div>
        <span class="ot-pct-text">${pctText}</span>
      </div>
      <div class="ot-wdl-cell">${wdlHtml}</div>
    `;

    row.addEventListener('click', () => {
      if (!isLeaf) {
        _otPath.push(child.move);
        _otRender();
        document.getElementById('btn-back').disabled = false;
      } else {
        // Leaf: just sync board to show this position
        _otPath.push(child.move);
        _otSyncBoard();
        _otRenderBreadcrumb();
        _otUpdateSaveBar();
        document.getElementById('btn-back').disabled = false;
      }
    });
    _otRows.appendChild(row);
  }
  if (syncBoard) _otSyncBoard();
}

async function _otLoad(forceRefresh = false) {
  const depth = parseInt(_otDepthIn.value, 10);
  _otRows.innerHTML = '<div id="ot-loading">Loading…</div>';
  try {
    const url = `/api/opening_tree?depth=${depth}${forceRefresh ? '&refresh=true' : ''}`;
    const res = await fetch(url);
    _otData = await res.json();
    _otLoaded = true;
    _otRender(false); // keep current board on initial load
  } catch (e) {
    _otRows.innerHTML = `<div id="ot-empty">Failed to load tree: ${e.message}</div>`;
  }
}

// ── Board ↔ tree sync ─────────────────────────────────────────────────────────

async function _otSyncBoard() {
  if (!_otToggle || !_otToggle.checked) return;
  const token = ++_otSyncToken;
  const url = _otPath.length
    ? `/api/explorer/fen_after_moves?moves=${encodeURIComponent(_otPath.join(','))}`
    : `/api/explorer/fen_after_moves`;
  try {
    const res  = await fetch(url);
    const data = await res.json();
    if (token !== _otSyncToken) return; // superseded by newer sync
    if (data.fen) {
      await loadPosition(data.fen);
      // In tree mode btn-back reflects _otPath depth, not history
      document.getElementById('btn-back').disabled = _otPath.length === 0;
    }
  } catch (_e) { /* ignore network errors */ }
}

function _otApplyCameraPan(open) {
  controls.target.set(open ? 2.2 : 0, 0, 0);
  camera.position.set(open ? 2.2 : 0, 8, 9);
  controls.update();
}

_otToggle.addEventListener('change', () => {
  if (_otToggle.checked) {
    // Opening tree ON — hide played-moves panel, clear history
    if (_pmPanel) _pmPanel.style.display = 'none';
    _playedMoves = [];
    _startingFen = null;
    _otLeftPanel.style.display = 'flex';
    _otApplyCameraPan(true);
    if (!_otLoaded) _otLoad();
  } else {
    _otLeftPanel.style.display = 'none';
    _otApplyCameraPan(false);
    if (_pmPanel) _pmPanel.style.display = 'flex';
  }
});

_otDepthIn.addEventListener('input', () => {
  _otDepthVal.textContent = _otDepthIn.value;
});
_otDepthIn.addEventListener('change', () => {
  _otData   = null;
  _otLoaded = false;
  _otPath   = [];
  if (_otToggle.checked) _otLoad();
});

// Novel toggle: re-render in place (no network fetch needed — data already loaded)
const _otShowNovelChk = document.getElementById('ot-show-novel');
if (_otShowNovelChk) {
  _otShowNovelChk.addEventListener('change', () => { if (_otLoaded) _otRender(false); });
}

// Refresh button: force reload from server (picks up new source tags / openings)
const _otRefreshBtn = document.getElementById('ot-refresh-btn');
if (_otRefreshBtn) {
  _otRefreshBtn.addEventListener('click', () => {
    _otData   = null;
    _otLoaded = false;
    _otPath   = [];
    _otLoad(true);
  });
}

// ── Save path as opening ──────────────────────────────────────────────────────

const _otSaveBar = document.getElementById('ot-save-bar');
const _otSaveName = document.getElementById('ot-save-name');
const _otSaveFamily = document.getElementById('ot-save-family');
const _otSaveFamilyCustom = document.getElementById('ot-save-family-custom');
const _otSaveBtn = document.getElementById('ot-save-btn');
const _otAssessBtn = document.getElementById('ot-assess-btn');
const _otSaveMsg = document.getElementById('ot-save-msg');
let _otLastSavedId = null;

function _otUpdateSaveBar() {
  _otSaveBar.style.display = _otPath.length >= 4 ? 'flex' : 'none';
  if (_otPath.length < 4) _otSaveMsg.textContent = '';
}

_otSaveBtn.addEventListener('click', async () => {
  const name = _otSaveName.value.trim();
  if (!name) { _otSaveMsg.textContent = 'Enter a name first.'; _otSaveMsg.style.color = '#ef4444'; return; }
  _otSaveBtn.disabled = true;
  _otSaveMsg.textContent = 'Saving…';
  _otSaveMsg.style.color = '#8a7a5a';
  try {
    const customFamily = _otSaveFamilyCustom?.value.trim();
    const family = customFamily || _otSaveFamily.value;
    const res = await fetch('/api/opening_tree/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moves: _otPath, name, family }),
    });
    const data = await res.json();
    if (data.ok) {
      _otSaveMsg.textContent = `Saved as "${name}".`;
      _otSaveMsg.style.color = '#4ade80';
      _otSaveName.value = '';
      _otLastSavedId = data.opening_id ?? null;
      if (_otAssessBtn) _otAssessBtn.style.display = _otLastSavedId ? 'inline-block' : 'none';
      _otData   = null;
      _otLoaded = false;
      _otLoad();
    } else {
      _otSaveMsg.textContent = data.error || 'Save failed.';
      _otSaveMsg.style.color = '#ef4444';
    }
  } catch (e) {
    _otSaveMsg.textContent = 'Network error.';
    _otSaveMsg.style.color = '#ef4444';
  } finally {
    _otSaveBtn.disabled = false;
  }
});

if (_otAssessBtn) {
  _otAssessBtn.addEventListener('click', async () => {
    if (!_otLastSavedId) return;
    _otAssessBtn.disabled = true;
    _otSaveMsg.textContent = 'Assessing…';
    _otSaveMsg.style.color = '#8a7a5a';
    try {
      const res = await fetch('/api/opening_tree/audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ opening_id: _otLastSavedId }),
      });
      const data = await res.json();
      if (data.ok) {
        const labels = { W: '♙ favours White', B: '♟ favours Black', equal: '⚖ balanced', unknown: '— unknown' };
        _otSaveMsg.textContent = labels[data.favored_side] || data.favored_side;
        _otSaveMsg.style.color = data.favored_side === 'W' ? '#e8c87a' : data.favored_side === 'B' ? '#aaa' : '#4ade80';
        _otAssessBtn.style.display = 'none';
      } else {
        _otSaveMsg.textContent = 'Assess failed: ' + (data.error || '');
        _otSaveMsg.style.color = '#ef4444';
      }
    } catch (e) {
      _otSaveMsg.textContent = 'Network error.';
      _otSaveMsg.style.color = '#ef4444';
    } finally {
      _otAssessBtn.disabled = false;
    }
  });
}
