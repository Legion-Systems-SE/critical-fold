"""
Resonant Field Visualizer — 3D Interactive View
=================================================
Generates a standalone HTML file with Three.js for interactive
3D visualization of the node cloud, phase structure, and voids.

Usage:
    python visualize_3d.py                    # Latest run
    python visualize_3d.py --run 0001         # Specific run
    python visualize_3d.py --run 0001 --step 50  # Specific timestep
    python visualize_3d.py --run 0001 --color phase  # Color by phase

Color modes:
    omega    — color by omega value (default, blue=low, red=high)
    phase    — color by injection phase (-π/2=blue, +π/2=red)
    shell    — color by radial shell membership (inner=cyan, outer=gold)
    active   — color by active/inactive (green=active, gray=inactive)

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

def resolve_run_dir(run_arg, base='runs_coupled'):
    runs_dir = SCRIPT_DIR / base
    if run_arg:
        if run_arg.isdigit():
            return runs_dir / f"{int(run_arg):04d}"
        return Path(run_arg)
    latest = runs_dir / "latest.txt"
    if latest.exists():
        return runs_dir / latest.read_text().strip()
    # Try regular runs
    runs_dir = SCRIPT_DIR / 'runs'
    latest = runs_dir / "latest.txt"
    if latest.exists():
        return runs_dir / latest.read_text().strip()
    print("No runs found.")
    sys.exit(1)


def load_cloud_at_step(clouds_file, step):
    npz = np.load(clouds_file)
    prefix = f"s{step:04d}"
    key = f"{prefix}_values"
    if key not in npz:
        return None, 0, 0
    values = npz[key]
    active = int(npz[f"{prefix}_active"][0])
    mean_t = float(npz[f"{prefix}_mean_t"][0])
    return values, active, mean_t


def get_cloud_steps(clouds_file):
    npz = np.load(clouds_file)
    steps = set()
    for key in npz.files:
        if key.startswith('s') and '_values' in key:
            try:
                steps.add(int(key.split('_')[0][1:]))
            except ValueError:
                pass
    return sorted(steps)


def to_sim_space(coords_idx, grid_size):
    return (coords_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)


def generate_html(positions, colors, sizes, meta, title="Resonant Field",
                  void_centers=None, phase_data=None):
    """Generate standalone Three.js HTML visualization."""

    # Encode data as JSON
    pos_list = positions.tolist()
    col_list = colors.tolist()
    size_list = sizes.tolist()
    void_list = void_centers.tolist() if void_centers is not None else []

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #0a0a0f; overflow: hidden; font-family: 'Courier New', monospace; }}
  #info {{
    position: absolute; top: 12px; left: 12px; color: #7faacc;
    font-size: 11px; line-height: 1.6; pointer-events: none;
    background: rgba(10,10,15,0.85); padding: 10px 14px; border-left: 2px solid #334466;
  }}
  #controls {{
    position: absolute; bottom: 12px; left: 12px; color: #556677;
    font-size: 10px; pointer-events: none;
  }}
</style>
</head><body>
<div id="info">
  <div style="color:#aaccee;font-size:13px;margin-bottom:4px">⬡ {title}</div>
  <div>Grid: {meta.get('grid_size','?')}³ | Nodes: {meta.get('n_nodes','?')}</div>
  <div>Engine: {meta.get('engine_version','?')}</div>
  <div id="step-info"></div>
</div>
<div id="controls">drag: rotate | scroll: zoom | shift+drag: pan</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// Data
const positions = {json.dumps(pos_list)};
const colors = {json.dumps(col_list)};
const sizes = {json.dumps(size_list)};
const voids = {json.dumps(void_list)};

// Scene
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.02);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 200);
camera.position.set(15, 10, 15);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Node cloud
const geometry = new THREE.BufferGeometry();
const posArr = new Float32Array(positions.length * 3);
const colArr = new Float32Array(colors.length * 3);
const sizeArr = new Float32Array(sizes.length);

for (let i = 0; i < positions.length; i++) {{
  posArr[i*3] = positions[i][0];
  posArr[i*3+1] = positions[i][1];
  posArr[i*3+2] = positions[i][2];
  colArr[i*3] = colors[i][0];
  colArr[i*3+1] = colors[i][1];
  colArr[i*3+2] = colors[i][2];
  sizeArr[i] = sizes[i];
}}

geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
geometry.setAttribute('size', new THREE.BufferAttribute(sizeArr, 1));

const material = new THREE.PointsMaterial({{
  size: 0.15, vertexColors: true, transparent: true, opacity: 0.85,
  sizeAttenuation: true, blending: THREE.AdditiveBlending
}});
const points = new THREE.Points(geometry, material);
scene.add(points);

// Void markers (larger, different color)
if (voids.length > 0) {{
  const voidGeo = new THREE.BufferGeometry();
  const voidPos = new Float32Array(voids.length * 3);
  for (let i = 0; i < voids.length; i++) {{
    voidPos[i*3] = voids[i][0];
    voidPos[i*3+1] = voids[i][1];
    voidPos[i*3+2] = voids[i][2];
  }}
  voidGeo.setAttribute('position', new THREE.BufferAttribute(voidPos, 3));
  const voidMat = new THREE.PointsMaterial({{
    size: 0.4, color: 0xffaa00, transparent: true, opacity: 0.6,
    sizeAttenuation: true, blending: THREE.AdditiveBlending
  }});
  scene.add(new THREE.Points(voidGeo, voidMat));
}}

// Core sphere (wireframe, subtle)
const coreGeo = new THREE.SphereGeometry(0.5, 16, 12);
const coreMat = new THREE.MeshBasicMaterial({{
  color: 0x223344, wireframe: true, transparent: true, opacity: 0.15
}});
scene.add(new THREE.Mesh(coreGeo, coreMat));

// Axes (subtle)
const axLen = 12;
const axMat = new THREE.LineBasicMaterial({{ color: 0x1a2a3a }});
[[[axLen,0,0],[-axLen,0,0]], [[0,axLen,0],[0,-axLen,0]], [[0,0,axLen],[0,0,-axLen]]].forEach(pair => {{
  const g = new THREE.BufferGeometry().setFromPoints(pair.map(p => new THREE.Vector3(...p)));
  scene.add(new THREE.Line(g, axMat));
}});

// Mouse controls (orbit)
let isDragging = false, isShift = false;
let prevX = 0, prevY = 0;
let theta = Math.PI/4, phi = Math.PI/6, radius = 22;
let panX = 0, panY = 0;

renderer.domElement.addEventListener('mousedown', e => {{ isDragging = true; prevX = e.clientX; prevY = e.clientY; isShift = e.shiftKey; }});
renderer.domElement.addEventListener('mouseup', () => isDragging = false);
renderer.domElement.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  const dx = e.clientX - prevX, dy = e.clientY - prevY;
  if (isShift) {{ panX -= dx * 0.02; panY += dy * 0.02; }}
  else {{ theta -= dx * 0.005; phi = Math.max(-Math.PI/2+0.1, Math.min(Math.PI/2-0.1, phi + dy * 0.005)); }}
  prevX = e.clientX; prevY = e.clientY;
}});
renderer.domElement.addEventListener('wheel', e => {{ radius = Math.max(5, Math.min(60, radius + e.deltaY * 0.02)); }});
window.addEventListener('resize', () => {{ camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }});

function animate() {{
  requestAnimationFrame(animate);
  camera.position.x = panX + radius * Math.cos(phi) * Math.sin(theta);
  camera.position.y = panY + radius * Math.sin(phi);
  camera.position.z = panX + radius * Math.cos(phi) * Math.cos(theta);
  camera.lookAt(panX, panY, 0);
  renderer.render(scene, camera);
}}
animate();
</script>
</body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description='3D Resonant Field Visualizer')
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--color', choices=['omega', 'phase', 'shell', 'active', 'body'],
                        default='phase')
    parser.add_argument('--output', type=str, default='field_3d.html')
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    print(f"Run: {run_dir}")

    # Load data
    with open(run_dir / 'meta.json') as f:
        meta = json.load(f)
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.int32)
    grid_size = meta.get('grid_size', 89)
    positions = to_sim_space(registry.astype(np.float64), grid_size)

    # Phase data
    phase_path = run_dir / 'phase.npy'
    phase = np.load(str(phase_path)) if phase_path.exists() else None

    # Cloud data for specific step
    clouds_path = run_dir / 'clouds.npz'
    values = None
    if clouds_path.exists():
        steps = get_cloud_steps(str(clouds_path))
        if args.step in steps:
            values, active, mean_t = load_cloud_at_step(str(clouds_path), args.step)
        elif steps:
            # Use closest step
            closest = min(steps, key=lambda s: abs(s - args.step))
            values, active, mean_t = load_cloud_at_step(str(clouds_path), closest)
            print(f"  Using closest step {closest}")

    # Compute colors
    n = len(positions)
    colors = np.ones((n, 3)) * 0.5  # default gray

    R = np.sqrt((positions**2).sum(axis=1))

    if args.color == 'phase' and phase is not None:
        # Blue = -π/2, Red = +π/2
        t = (phase + np.pi/2) / np.pi  # 0 to 1
        colors[:, 0] = t          # red channel
        colors[:, 1] = 0.15       # dim green
        colors[:, 2] = 1.0 - t    # blue channel
        print(f"  Color: phase ({int((phase < 0).sum())} blue, {int((phase > 0).sum())} red)")

    elif args.color == 'omega' and values is not None:
        v = np.clip(values[:n], 0, 1)
        colors[:, 0] = v
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - v

    elif args.color == 'shell':
        R_sorted = np.sort(R)
        gaps = np.diff(R_sorted)
        big_gap = int(np.argmax(gaps))
        gap_inner = R_sorted[big_gap]
        inner = R <= gap_inner
        outer = ~inner
        colors[inner, :] = [0.1, 0.7, 0.8]   # cyan
        colors[outer, :] = [0.9, 0.75, 0.2]   # gold

    elif args.color == 'active' and values is not None:
        active_mask = values[:n] > 0.5
        colors[active_mask, :] = [0.2, 0.9, 0.3]    # green
        colors[~active_mask, :] = [0.25, 0.25, 0.3]  # gray

    elif args.color == 'body':
        fid_path = run_dir / 'field_ids.npy'
        if fid_path.exists():
            fids = np.load(str(fid_path))
            body_a = fids == 0
            body_b = fids == 1
            colors[body_a, :] = [0.1, 0.6, 0.9]     # blue — body A
            colors[body_b, :] = [0.95, 0.4, 0.15]    # orange — body B
            if values is not None:
                dead = values[:n] < 0.5
                colors[dead] *= 0.3
            print(f"  Color: body (A={int(body_a.sum())} blue, "
                  f"B={int(body_b.sum())} orange)")
        else:
            print("  No field_ids.npy — falling back to gray")

    # Sizes
    sizes = np.ones(n) * 0.15
    if values is not None:
        sizes = 0.08 + 0.12 * np.clip(values[:n], 0, 1)

    # Void centers from Delaunay (if scipy available)
    void_centers = None
    try:
        from scipy.spatial import Delaunay
        R_sorted = np.sort(R)
        gaps = np.diff(R_sorted)
        big_gap = int(np.argmax(gaps))
        gap_inner = R_sorted[big_gap]
        gap_outer = R_sorted[big_gap + 1]
        gap_ratio = (gap_outer - gap_inner) / (np.sort(gaps)[::-1][1] + 1e-6)

        if gap_ratio > 5:
            tri = Delaunay(positions)
            simplices = tri.simplices
            tet_pts = positions[simplices]
            p0 = tet_pts[:, 0, :]
            rest = tet_pts[:, 1:, :]
            A = 2.0 * (p0[:, None, :] - rest)
            b = (p0**2).sum(axis=1, keepdims=True) - (rest**2).sum(axis=2)
            dets = np.linalg.det(A)
            valid = np.abs(dets) > 1e-10
            centers = np.zeros_like(p0)
            radii = np.full(len(tet_pts), np.inf)
            if valid.any():
                c = np.linalg.solve(A[valid], b[valid][..., None]).squeeze(-1)
                centers[valid] = c
                radii[valid] = np.linalg.norm(p0[valid] - c, axis=1)

            in_box = np.abs(centers).max(axis=1) < 10.0
            R_tet = np.sqrt((positions[simplices]**2).sum(axis=2))
            spans = valid & in_box & (R_tet.min(axis=1) <= gap_inner) & (R_tet.max(axis=1) >= gap_outer)

            if spans.sum() > 0:
                top_idx = np.argsort(-radii[spans])[:20]
                void_centers = centers[spans][top_idx]
                print(f"  Void centers: {len(void_centers)} gap-spanning voids")
    except ImportError:
        print("  (scipy not available — skipping void detection)")

    # Generate HTML
    title = f"Run {meta.get('run_id', '?'):04d} — {args.color} — step {args.step}"
    html = generate_html(positions, colors, sizes, meta, title, void_centers, phase)

    out_path = run_dir / args.output
    out_path.write_text(html)
    print(f"  Written: {out_path}")
    return str(out_path)


if __name__ == '__main__':
    main()
