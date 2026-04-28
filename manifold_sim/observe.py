"""
Resonant Field Observer — Void-Based Sampling
===============================================
Observes the manifold from inside the voids.

Each void is defined by its 4 surrounding nodes (Delaunay tetrahedron
vertices). The observer records those 4 nodes' values over time,
plus the tension vectors pointing through the void.

Quantization modes for WHICH steps to observe:
  all       — every available step
  prime     — steps where step number is prime
  pi        — steps nearest to n·π·10 (beat-locked to π)
  beat      — steps at beat phase extrema (0, π, 2π)
  fibonacci — Fibonacci-numbered steps

Usage:
    python observe.py --run runs_coupled/0001
    python observe.py --run runs_coupled/0001 --mode prime
    python observe.py --run runs_coupled/0001 --mode pi
    python observe.py --run runs_coupled/0001 --top 6

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path
from sympy import isprime

SCRIPT_DIR = Path(__file__).parent.resolve()


def resolve_run_dir(run_arg):
    if run_arg:
        p = Path(run_arg)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if p.exists():
            return p
    # Try latest
    for base in ['runs_coupled', 'runs']:
        latest = SCRIPT_DIR / base / 'latest.txt'
        if latest.exists():
            return SCRIPT_DIR / base / latest.read_text().strip()
    print("No run found.")
    sys.exit(1)


def to_sim_space(coords_idx, grid_size):
    return (coords_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)


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


def load_cloud_at_step(clouds_file, step):
    npz = np.load(clouds_file)
    prefix = f"s{step:04d}"
    values = npz[f"{prefix}_values"]
    active = int(npz[f"{prefix}_active"][0])
    t_mag = npz.get(f"{prefix}_tvec_mag", None)
    t_unit = npz.get(f"{prefix}_tvec_unit", None)
    return values, active, t_mag, t_unit


def find_voids(positions, top_n=10):
    """Find void centers from Delaunay triangulation of gap-spanning tets."""
    from scipy.spatial import Delaunay

    R = np.sqrt((positions**2).sum(axis=1))
    R_sorted = np.sort(R)
    gaps = np.diff(R_sorted)
    big_gap = int(np.argmax(gaps))
    gap_inner = R_sorted[big_gap]
    gap_outer = R_sorted[big_gap + 1]
    gap_ratio = (gap_outer - gap_inner) / (np.sort(gaps)[::-1][1] + 1e-6)

    if gap_ratio < 5:
        print("  No clear shell structure — using largest circumradii instead")
        gap_inner = R.max() * 0.3
        gap_outer = R.max() * 0.7

    tri = Delaunay(positions)
    simplices = tri.simplices
    tet_pts = positions[simplices]

    # Circumcenters
    p0 = tet_pts[:, 0, :]
    rest = tet_pts[:, 1:, :]
    A = 2.0 * (p0[:, None, :] - rest)
    b = (p0**2).sum(axis=1, keepdims=True) - (rest**2).sum(axis=2)
    dets = np.linalg.det(A)
    valid = np.abs(dets) > 1e-10
    centers = np.zeros_like(p0)
    radii = np.full(len(tet_pts), 0.0)

    if valid.any():
        c = np.linalg.solve(A[valid], b[valid][..., None]).squeeze(-1)
        centers[valid] = c
        radii[valid] = np.linalg.norm(p0[valid] - c, axis=1)

    in_box = np.abs(centers).max(axis=1) < 10.0
    R_per_tet = R[simplices]
    spans = valid & in_box & (R_per_tet.min(axis=1) <= gap_inner) & (R_per_tet.max(axis=1) >= gap_outer)

    if spans.sum() < 4:
        spans = valid & in_box
        print(f"  Warning: only {spans.sum()} spanning tets, using all valid")

    top_idx = np.argsort(-radii[spans])[:top_n]
    void_indices = np.where(spans)[0][top_idx]

    return void_indices, centers, radii, simplices, gap_inner, gap_outer


def quantize_steps(all_steps, mode, beat_detune=0.1):
    """Select observation steps based on quantization mode."""
    if mode == 'all':
        return all_steps

    elif mode == 'prime':
        return [s for s in all_steps if s > 1 and isprime(s)]

    elif mode == 'pi':
        # Steps nearest to n·π·10 (spatial beat period)
        targets = [int(round(n * np.pi * 10)) for n in range(1, 20)]
        selected = []
        for t in targets:
            closest = min(all_steps, key=lambda s: abs(s - t))
            if closest not in selected:
                selected.append(closest)
        return sorted(selected)

    elif mode == 'beat':
        # Steps at beat phase extrema: cos(2π·detune·step) = ±1
        # cos = 1 when step = n/detune, cos = -1 when step = (n+0.5)/detune
        period = 1.0 / beat_detune
        targets = []
        for n in range(50):
            targets.append(int(round(n * period)))          # cos = 1
            targets.append(int(round((n + 0.5) * period)))  # cos = -1
        selected = []
        for t in targets:
            if t in all_steps and t not in selected:
                selected.append(t)
        return sorted(selected)

    elif mode == 'fibonacci':
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        return [s for s in all_steps if s in fibs]

    else:
        print(f"Unknown mode: {mode}")
        return all_steps


def run_observer(run_dir, mode='all', top_n=6):
    run_dir = Path(run_dir)

    with open(run_dir / 'meta.json') as f:
        meta = json.load(f)

    registry = np.load(str(run_dir / 'registry.npy')).astype(np.int32)
    grid_size = meta.get('grid_size', 89)
    positions = to_sim_space(registry.astype(np.float64), grid_size)
    beat_detune = meta.get('beat_detune', 0.1)

    # Load phase if available
    phase_path = run_dir / 'phase.npy'
    phase = np.load(str(phase_path)) if phase_path.exists() else None

    clouds_path = run_dir / 'clouds.npz'
    if not clouds_path.exists():
        print("No clouds.npz — run the engine first.")
        return
    all_steps = get_cloud_steps(str(clouds_path))

    print(f"\n{'='*72}")
    print(f"VOID OBSERVER — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid_size}³ | Nodes: {len(registry)} | Steps: {len(all_steps)}")
    print(f"Quantization: {mode}")
    print(f"{'='*72}")

    # Find voids
    print(f"\n[VOID DETECTION]")
    void_idx, centers, radii, simplices, gap_inner, gap_outer = find_voids(
        positions, top_n=top_n)
    print(f"  Gap: R = {gap_inner:.3f} → {gap_outer:.3f}")
    print(f"  Top {len(void_idx)} voids selected")

    # Select observation steps
    obs_steps = quantize_steps(all_steps, mode, beat_detune)
    print(f"\n[OBSERVATION STEPS — {mode}]")
    print(f"  {len(obs_steps)} steps selected from {len(all_steps)} available")
    if len(obs_steps) <= 30:
        print(f"  Steps: {obs_steps}")

    # Observe from each void
    print(f"\n[VOID OBSERVATIONS]")
    print(f"  Each void is observed through its 4 surrounding nodes.")
    print(f"  'view' = mean omega of the 4 nodes")
    print(f"  'contrast' = std of the 4 node values (local anisotropy)")
    print(f"  'phase_mix' = fraction of surrounding nodes at +π/2 phase")

    for vi, void_i in enumerate(void_idx):
        tet_nodes = simplices[void_i]  # 4 node indices
        tet_pos = positions[tet_nodes]
        center = centers[void_i]
        radius = radii[void_i]

        # Phase composition of surrounding nodes
        if phase is not None:
            tet_phase = phase[tet_nodes]
            n_pos_phase = int((tet_phase > 0).sum())
            phase_mix = n_pos_phase / 4.0
            phase_label = f"{n_pos_phase}/4 at +π/2"
        else:
            phase_mix = -1
            phase_label = "no phase data"

        print(f"\n  --- Void {vi+1} ---")
        print(f"  Center: ({center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f})")
        print(f"  Circumradius: {radius:.3f}")
        print(f"  Phase: {phase_label}")
        print(f"  Surrounding node indices: {tet_nodes}")

        # Time series
        print(f"\n  {'Step':>6} | {'View':>8} | {'Contrast':>8} | "
              f"{'BeatΦ':>7} | {'Active':>6} | Notes")
        print(f"  {'-'*65}")

        prev_view = None
        for step in obs_steps:
            values, active, t_mag, t_unit = load_cloud_at_step(
                str(clouds_path), step)

            tet_values = values[tet_nodes]
            view = float(tet_values.mean())
            contrast = float(tet_values.std())
            beat_phase = np.cos(2 * np.pi * beat_detune * step)

            # Notes: detect transitions
            notes = []
            n_active_nodes = int((tet_values > 0.5).sum())
            if n_active_nodes == 0:
                notes.append("COLD")
            elif n_active_nodes == 4:
                notes.append("HOT")
            elif n_active_nodes == 2:
                notes.append("SPLIT")

            if prev_view is not None:
                delta = view - prev_view
                if abs(delta) > 0.1:
                    notes.append(f"Δ={delta:+.3f}")

            # Check if view value is near a simple fraction
            for num in range(1, 10):
                for den in range(num + 1, 16):
                    if abs(view - num / den) < 0.005 and view > 0.01:
                        notes.append(f"≈{num}/{den}")
                        break

            prev_view = view
            notes_str = ' '.join(notes) if notes else ''
            print(f"  {step:>6} | {view:>8.4f} | {contrast:>8.4f} | "
                  f"{beat_phase:>+7.3f} | {n_active_nodes:>4}/4 | {notes_str}")

    # Cross-void comparison at selected steps
    if len(void_idx) >= 3 and len(obs_steps) >= 3:
        print(f"\n[CROSS-VOID CORRELATION]")
        print(f"  Do different voids see the same transitions?")
        print(f"\n  {'Step':>6} |", end='')
        for vi in range(min(len(void_idx), 6)):
            print(f" V{vi+1:>2}   ", end='')
        print(f" | {'Range':>6} | Notes")
        print(f"  {'-'*72}")

        for step in obs_steps[:30]:
            values, active, _, _ = load_cloud_at_step(str(clouds_path), step)
            views = []
            for vi, void_i in enumerate(void_idx[:6]):
                tet_nodes = simplices[void_i]
                v = float(values[tet_nodes].mean())
                views.append(v)

            row = f"  {step:>6} |"
            for v in views:
                row += f" {v:.3f}"
            vrange = max(views) - min(views)
            notes = ''
            if vrange < 0.01:
                notes = 'SYNC'
            elif vrange > 0.3:
                notes = 'DIVERGE'
            row += f" | {vrange:.4f} | {notes}"
            print(row)

    # Observer summary with ratio analysis
    print(f"\n[RATIO ANALYSIS — final step observations]")
    last_step = obs_steps[-1] if obs_steps else all_steps[-1]
    values, active, t_mag, t_unit = load_cloud_at_step(str(clouds_path), last_step)

    void_views = []
    for vi, void_i in enumerate(void_idx):
        tet_nodes = simplices[void_i]
        v = float(values[tet_nodes].mean())
        void_views.append(v)

    if len(void_views) >= 2:
        print(f"  Void views at step {last_step}: {[f'{v:.4f}' for v in void_views]}")
        print(f"\n  Pairwise ratios:")
        for i in range(len(void_views)):
            for j in range(i + 1, len(void_views)):
                if void_views[j] > 0.001:
                    r = void_views[i] / void_views[j]
                    from fractions import Fraction
                    frac = Fraction(r).limit_denominator(50)
                    print(f"    V{i+1}/V{j+1} = {r:.6f} ≈ {frac}")

    print(f"\n{'='*72}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Void Observer')
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--mode', choices=['all', 'prime', 'pi', 'beat', 'fibonacci'],
                        default='prime', help='Step quantization mode')
    parser.add_argument('--top', type=int, default=6,
                        help='Number of top voids to observe')
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    run_observer(run_dir, mode=args.mode, top_n=args.top)
