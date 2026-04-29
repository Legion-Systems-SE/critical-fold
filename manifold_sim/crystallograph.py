"""
Crystallograph — Rotational Moiré Frequency Viewer
====================================================
Generates a standalone HTML file with controlled rotation for
reading lattice frequencies through visual moiré interference.

Rotation modes:
    continuous  — smooth rotation (Laplacian-like)
    pi          — quantized steps of π/N (crystallographic)

Controls (in browser):
    Space       — pause / resume rotation
    1/2/3       — rotate around X / Y / Z axis
    4           — rotate around body diagonal [1,1,1]
    L / P       — switch to continuous (L) or pi-quantized (P) mode
    Up/Down     — offset elevation (phi)
    Left/Right  — adjust rotation speed
    [ / ]       — offset X position (shift viewing axis)
    R           — reset to default view
    S           — step one frame (when paused)
    F           — toggle fullscreen HUD

Usage:
    python3 crystallograph.py --run runs_emergent/0761 --color shell
    python3 crystallograph.py --run runs_emergent/0761 --color phase --speed 0.3

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def resolve_run_dir(run_arg):
    if not run_arg:
        for base in ['runs_emergent', 'runs_coupled', 'runs']:
            latest = SCRIPT_DIR / base / "latest.txt"
            if latest.exists():
                return SCRIPT_DIR / base / latest.read_text().strip()
        raise FileNotFoundError("No runs found")
    p = Path(run_arg)
    if p.exists():
        return p
    for base in ['runs_emergent', 'runs_coupled']:
        candidate = SCRIPT_DIR / base / f"{int(run_arg):04d}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Run not found: {run_arg}")


def to_sim_space(coords_idx, grid_size):
    return (coords_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)


def load_run(run_dir, step=0):
    with open(run_dir / 'meta.json') as f:
        meta = json.load(f)
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.int32)
    gs = meta.get('grid_size', 89)
    positions = to_sim_space(registry.astype(np.float64), gs)

    phase_path = run_dir / 'phase.npy'
    phase = np.load(str(phase_path)) if phase_path.exists() else None

    values = None
    clouds_path = run_dir / 'clouds.npz'
    if clouds_path.exists():
        npz = np.load(str(clouds_path))
        key = f"s{step:04d}_values"
        if key in npz:
            values = npz[key]

    return positions, phase, values, meta


def compute_colors(positions, phase, values, mode):
    n = len(positions)
    colors = np.ones((n, 3)) * 0.5
    R = np.sqrt((positions**2).sum(axis=1))

    if mode == 'shell':
        R_sorted = np.sort(R)
        gaps = np.diff(R_sorted)
        big_gap = int(np.argmax(gaps))
        gap_inner = R_sorted[big_gap]
        inner = R <= gap_inner
        colors[inner] = [0.1, 0.7, 0.8]
        colors[~inner] = [0.9, 0.75, 0.2]
    elif mode == 'phase' and phase is not None:
        t = (phase + np.pi/2) / np.pi
        colors[:, 0] = np.clip(t, 0, 1)
        colors[:, 1] = 0.15
        colors[:, 2] = np.clip(1.0 - t, 0, 1)
    elif mode == 'omega' and values is not None:
        v = np.clip(values[:n], 0, 1)
        colors[:, 0] = v
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - v
    elif mode == 'active' and values is not None:
        active = values[:n] > 0.5
        colors[active] = [0.2, 0.9, 0.3]
        colors[~active] = [0.25, 0.25, 0.3]

    return colors


def generate_crystallograph_html(positions, colors, meta, initial_speed=0.5):
    pos_list = positions.tolist()
    col_list = colors.tolist()
    n_nodes = len(positions)
    gs = meta.get('grid_size', '?')
    run_id = meta.get('run_id', '?')

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Crystallograph — Run {run_id}</title>
<style>
  body {{ margin: 0; background: #0a0a0f; overflow: hidden; font-family: 'Courier New', monospace; }}
  #hud {{
    position: absolute; top: 12px; left: 12px; color: #7faacc;
    font-size: 11px; line-height: 1.7; pointer-events: none;
    background: rgba(10,10,15,0.9); padding: 12px 16px; border-left: 2px solid #334466;
    min-width: 280px;
  }}
  #hud .title {{ color: #aaccee; font-size: 13px; margin-bottom: 6px; }}
  #hud .val {{ color: #88ccff; }}
  #hud .mode {{ color: #ffaa44; }}
  #hud .key {{ color: #556677; font-size: 9px; }}
  #angle-bar {{
    position: absolute; bottom: 40px; left: 50%; transform: translateX(-50%);
    width: 80%; height: 3px; background: #1a1a2e; pointer-events: none;
  }}
  #angle-marker {{
    position: absolute; top: -4px; width: 10px; height: 10px;
    background: #88ccff; border-radius: 50%; transform: translateX(-50%);
  }}
  #angle-label {{
    position: absolute; bottom: 50px; left: 50%; transform: translateX(-50%);
    color: #556677; font-size: 10px; font-family: 'Courier New', monospace;
    pointer-events: none;
  }}
</style>
</head><body>

<div id="hud">
  <div class="title">Crystallograph — Run {run_id}</div>
  <div>Grid: {gs}³ | Nodes: {n_nodes}</div>
  <div>Axis: <span class="val" id="h-axis">Y</span>
       Mode: <span class="mode" id="h-mode">continuous</span></div>
  <div>θ: <span class="val" id="h-theta">0.000</span>
       φ offset: <span class="val" id="h-phi">0.000</span></div>
  <div>Speed: <span class="val" id="h-speed">{initial_speed:.2f}</span>
       X offset: <span class="val" id="h-xoff">0.000</span></div>
  <div id="h-state" class="mode">ROTATING</div>
  <div class="key" style="margin-top:8px">
    [Space] pause &nbsp; [S] step &nbsp; [R] reset<br>
    [1-4] axis: X Y Z diag &nbsp; [L/P] mode<br>
    [↑↓] elevation &nbsp; [←→] speed<br>
    [Q/E] x-offset &nbsp; [&lt; &gt;] pi divisor &nbsp; [Z] zoom<br>
    [M] mark angle &nbsp; [X] export marks to console<br>
    [Shift+←→] scrub back/forward
  </div>
</div>

<div id="angle-bar"><div id="angle-marker"></div></div>
<div id="angle-label">0.000 rad</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const positions = {json.dumps(pos_list)};
const colors = {json.dumps(col_list)};

// State
let rotAxis = 'Y';
let rotMode = 'continuous';
let speed = {initial_speed};
let paused = false;
let theta = 0;
let phiOffset = 0;
let xOffset = 0;
let radius = 20;
let piN = 12;
let markers = [];

// Scene
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.015);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 200);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Build point cloud
const geo = new THREE.BufferGeometry();
const posArr = new Float32Array(positions.length * 3);
const colArr = new Float32Array(colors.length * 3);
for (let i = 0; i < positions.length; i++) {{
  posArr[i*3] = positions[i][0];
  posArr[i*3+1] = positions[i][1];
  posArr[i*3+2] = positions[i][2];
  colArr[i*3] = colors[i][0];
  colArr[i*3+1] = colors[i][1];
  colArr[i*3+2] = colors[i][2];
}}
geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

const mat = new THREE.PointsMaterial({{
  size: 0.12, vertexColors: true, transparent: true, opacity: 0.85,
  sizeAttenuation: true, blending: THREE.AdditiveBlending
}});
scene.add(new THREE.Points(geo, mat));

// Axes
const axLen = 12;
const axMat = new THREE.LineBasicMaterial({{ color: 0x1a2a3a }});
[[[axLen,0,0],[-axLen,0,0]], [[0,axLen,0],[0,-axLen,0]], [[0,0,axLen],[0,0,-axLen]]].forEach(pair => {{
  const g = new THREE.BufferGeometry().setFromPoints(pair.map(p => new THREE.Vector3(...p)));
  scene.add(new THREE.Line(g, axMat));
}});

// Core
const coreGeo = new THREE.SphereGeometry(0.3, 12, 8);
const coreMat = new THREE.MeshBasicMaterial({{ color: 0x223344, wireframe: true, transparent: true, opacity: 0.1 }});
scene.add(new THREE.Mesh(coreGeo, coreMat));

// HUD references
const hAxis = document.getElementById('h-axis');
const hMode = document.getElementById('h-mode');
const hTheta = document.getElementById('h-theta');
const hPhi = document.getElementById('h-phi');
const hSpeed = document.getElementById('h-speed');
const hXoff = document.getElementById('h-xoff');
const hState = document.getElementById('h-state');
const angleMarker = document.getElementById('angle-marker');
const angleLabel = document.getElementById('angle-label');

function updateHUD() {{
  hAxis.textContent = rotAxis === 'D' ? '[1,1,1]' : rotAxis;
  hMode.textContent = rotMode === 'continuous' ? 'continuous' : 'π/' + piN;
  hTheta.textContent = theta.toFixed(3);
  hPhi.textContent = phiOffset.toFixed(3);
  hSpeed.textContent = speed.toFixed(2);
  hXoff.textContent = xOffset.toFixed(3);
  hState.textContent = paused ? 'PAUSED' : 'ROTATING';
  hState.style.color = paused ? '#ff6644' : '#44ff88';

  const pct = ((theta % (2 * Math.PI)) / (2 * Math.PI)) * 100;
  angleMarker.style.left = pct + '%';

  const piRatio = theta / Math.PI;
  angleLabel.textContent = theta.toFixed(3) + ' rad (' + piRatio.toFixed(3) + 'π)';
}}

function getCameraPos() {{
  const inv3 = 1.0 / Math.sqrt(3.0);
  let cx, cy, cz;

  if (rotAxis === 'X') {{
    cx = xOffset;
    cy = radius * Math.sin(theta) * Math.cos(phiOffset) + radius * Math.sin(phiOffset) * Math.cos(theta) * 0;
    cz = radius * Math.cos(theta);
    cy = radius * Math.sin(theta + phiOffset);
    cz = radius * Math.cos(theta + phiOffset);
    cx = xOffset;
  }} else if (rotAxis === 'Y') {{
    cx = radius * Math.sin(theta);
    cy = radius * Math.sin(phiOffset);
    cz = radius * Math.cos(theta);
  }} else if (rotAxis === 'Z') {{
    cx = radius * Math.sin(theta);
    cy = radius * Math.cos(theta);
    cz = radius * Math.sin(phiOffset);
  }} else {{
    // Body diagonal: rotate around [1,1,1]
    const c = Math.cos(theta), s = Math.sin(theta);
    const ux = inv3, uy = inv3, uz = inv3;
    // Rodrigues rotation of initial camera position [0, 0, radius]
    const vx = 0, vy = 0, vz = radius;
    const dot = uz * vz;
    // cross = u × v
    const crossX = uy * vz;
    const crossY = -ux * vz;
    const crossZ = 0;
    cx = vx * c + crossX * s + ux * dot * (1 - c) + xOffset;
    cy = vy * c + crossY * s + uy * dot * (1 - c) + phiOffset * 5;
    cz = vz * c + crossZ * s + uz * dot * (1 - c);
  }}

  // Apply x-offset for non-diagonal modes
  if (rotAxis !== 'D') {{
    if (rotAxis === 'Y') cx += xOffset;
    else if (rotAxis === 'X') cz += xOffset;
    else cx += xOffset;
  }}

  return [cx, cy, cz];
}}

// Keyboard
document.addEventListener('keydown', e => {{
  const k = e.key.toLowerCase();
  if (e.shiftKey && (k === 'arrowleft' || k === 'arrowright')) {{
    // Shift+Arrow: scrub theta backward/forward
    const step = rotMode === 'pi' ? Math.PI / piN : 0.005;
    theta += (k === 'arrowright') ? step : -step;
    e.preventDefault();
    updateHUD();
    return;
  }}
  if (k === ' ') {{ paused = !paused; e.preventDefault(); }}
  else if (k === '1') rotAxis = 'X';
  else if (k === '2') rotAxis = 'Y';
  else if (k === '3') rotAxis = 'Z';
  else if (k === '4') rotAxis = 'D';
  else if (k === 'l') rotMode = 'continuous';
  else if (k === 'p') {{
    rotMode = 'pi';
    // Snap theta to nearest pi/N
    const step = Math.PI / piN;
    theta = Math.round(theta / step) * step;
  }}
  else if (k === 'arrowup') phiOffset += 0.05;
  else if (k === 'arrowdown') phiOffset -= 0.05;
  else if (k === 'arrowleft') speed = Math.max(0.01, speed - 0.05);
  else if (k === 'arrowright') speed += 0.05;
  else if (k === '[' || k === 'q') xOffset -= 0.2;
  else if (k === ']' || k === 'e') xOffset += 0.2;
  else if (k === 'z') radius = radius > 15 ? 12 : 20;
  else if (k === 'r') {{
    theta = 0; phiOffset = 0; xOffset = 0; speed = {initial_speed}; rotAxis = 'Y'; rotMode = 'continuous';
  }}
  else if (k === 's' && paused) {{
    if (rotMode === 'pi') theta += Math.PI / piN;
    else theta += 0.01;
  }}
  else if (k === ',' || k === '<') piN = Math.max(2, piN - 1);
  else if (k === '.' || k === '>') piN = Math.min(48, piN + 1);
  else if (k === 'm') {{
    // Mark current angle — log to console and flash HUD
    const piRatio = theta / Math.PI;
    const entry = {{
      theta: theta.toFixed(6),
      piRatio: piRatio.toFixed(6),
      axis: rotAxis,
      phi: phiOffset.toFixed(4),
      xOff: xOffset.toFixed(4)
    }};
    markers.push(entry);
    console.log('MARK', JSON.stringify(entry));
    hState.textContent = 'MARKED θ=' + theta.toFixed(3);
    hState.style.color = '#ffff44';
    setTimeout(() => updateHUD(), 800);
  }}
  else if (k === 'x') {{
    // Export all marks to console
    console.log('=== MARKED ANGLES ===');
    markers.forEach((m, i) => console.log(i, JSON.stringify(m)));
    console.log('Total marks:', markers.length);
    hState.textContent = 'EXPORTED ' + markers.length + ' marks';
    hState.style.color = '#44ffff';
    setTimeout(() => updateHUD(), 1200);
  }}
  updateHUD();
}});

// Scroll for zoom
renderer.domElement.addEventListener('wheel', e => {{
  radius = Math.max(8, Math.min(50, radius + e.deltaY * 0.02));
}});

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

let lastTime = 0;
function animate(time) {{
  requestAnimationFrame(animate);
  const dt = Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;

  if (!paused) {{
    if (rotMode === 'continuous') {{
      theta += speed * dt;
    }} else {{
      // Pi-quantized: accumulate and step
      theta += speed * dt;
      const step = Math.PI / piN;
      const snapped = Math.round(theta / step) * step;
      // Snap camera to quantized position
      const pos = getCameraPos();
      const snapTheta = theta;
      theta = snapped;
      const snapPos = getCameraPos();
      theta = snapTheta;
      // Use snapped position for rendering
      camera.position.set(snapPos[0], snapPos[1], snapPos[2]);
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      updateHUD();
      return;
    }}
  }}

  const pos = getCameraPos();
  camera.position.set(pos[0], pos[1], pos[2]);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  updateHUD();
}}
requestAnimationFrame(animate);
</script>
</body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description='Crystallograph — Rotational Moiré Viewer')
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--color', choices=['shell', 'phase', 'omega', 'active'], default='shell')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    print(f"Run: {run_dir}")

    positions, phase, values, meta = load_run(run_dir, args.step)
    colors = compute_colors(positions, phase, values, args.color)
    print(f"  Nodes: {len(positions)} | Color: {args.color} | Step: {args.step}")

    html = generate_crystallograph_html(positions, colors, meta, args.speed)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = run_dir / 'crystallograph.html'

    out_path.write_text(html)
    print(f"  Written: {out_path}")


if __name__ == '__main__':
    main()
