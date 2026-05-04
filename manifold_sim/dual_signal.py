"""
Dual Signal Decomposition — Zeta vs Lattice Axis Analysis
==========================================================
Decomposes the node distribution's anisotropy into projections
along two known orthogonal directions:

  1. ZETA axis:    θ=34.3°, φ=-135.0°  [-0.3985, -0.3984, 0.8261]
  2. LATTICE axis: θ=54.7°, φ=+45.0°   [0.5774,  0.5774, 0.5774] = (1,1,1)/√3

Method: compute the weighted covariance matrix of node positions,
project onto each axis. σ²_axis = axis^T · Cov · axis gives variance
along that direction. Compare to isotropic baseline (trace/3).

Modes:
  run      — analyze a single engine run (emergent or coupled)
  sweep    — sweep n_zeros in the injection geometry (coupled only)
  compare  — run + sweep overlay

Usage:
    python dual_signal.py                          # latest run
    python dual_signal.py --run runs_emergent/0761
    python dual_signal.py --sweep                  # injection sweep n=5→50
    python dual_signal.py --sweep --grid 89        # specific grid
    python dual_signal.py --sweep --grids 65,89,97

Ported from axis_prediction_v3.py (April 2026) with:
  - Full-precision zeta zeros (12 decimal places)
  - Current run data format (clouds.npz, registry.npy)
  - Emergent engine compatibility
  - Structural constant tagging

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import math
import sys
import json
import argparse
from pathlib import Path

# === THE TWO SIGNALS ===
ZETA_AXIS = np.array([-0.3985, -0.3984, 0.8261])
ZETA_AXIS /= np.linalg.norm(ZETA_AXIS)

LATTICE_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

CROSS_AXIS = np.cross(ZETA_AXIS, LATTICE_AXIS)
CROSS_AXIS /= np.linalg.norm(CROSS_AXIS)

SOFT_ALPHA = 1.0
DRAIN_RADIUS = 0.15

ZETA_ZEROS = [
    14.134725141734, 21.022039638771, 25.010857580145, 30.424876125859,
    32.935061587739, 37.586178158825, 40.918719012147, 43.327073280914,
    48.005150881167, 49.773832477672, 52.970321477714, 56.446247697063,
    59.347044002602, 60.831778524609, 65.112544048081, 67.079810529494,
    69.546401711696, 72.067157674228, 75.704690699083, 77.144840068874,
    79.337375020249, 82.910380854086, 84.735492980517, 87.425274613125,
    88.809111207634, 92.491899270806, 94.651344040519, 95.870634228245,
    98.831194218193, 101.317851005731, 103.725538040478, 105.446623052705,
    107.168611184276, 111.029535543002, 111.874659176999, 114.320220915452,
    116.226680320857, 118.790782865996, 121.370125002400, 122.946829293884,
    124.256818554347, 127.516683879991, 129.578704199956, 131.087688530933,
    133.497737203552, 134.756509753374, 138.116042054514, 139.736208952121,
    141.123707404282, 143.111845807620,
]

STRUCTURAL = {
    7:   "k (fold number)",
    13:  "π(41) (shadow number)",
    14:  "⌊γ₁⌋",
    24:  "|O| (survivor count)",
    36:  "h²=0 Hodge count",
    41:  "smallest odd core",
    96:  "beat quantum",
}


def to_sim_space(registry, grid_size):
    dx = 20.0 / (grid_size - 1)
    return (registry.astype(np.float64) - (grid_size - 1) / 2.0) * dx


def decompose_dual(positions, weights=None):
    if len(positions) < 10:
        return None

    if weights is not None:
        w = weights / (weights.sum() + 1e-12)
        mean = (positions * w[:, np.newaxis]).sum(axis=0)
        centered = positions - mean
        cov = (centered * w[:, np.newaxis]).T @ centered
    else:
        mean = positions.mean(axis=0)
        centered = positions - mean
        cov = (centered.T @ centered) / len(centered)

    sigma_mean = np.trace(cov) / 3.0

    sigma_zeta = float(ZETA_AXIS @ cov @ ZETA_AXIS)
    sigma_lattice = float(LATTICE_AXIS @ cov @ LATTICE_AXIS)
    sigma_cross = float(CROSS_AXIS @ cov @ CROSS_AXIS)

    zeta_excess = (sigma_zeta / sigma_mean - 1.0) * 100 if sigma_mean > 0 else 0
    lattice_excess = (sigma_lattice / sigma_mean - 1.0) * 100 if sigma_mean > 0 else 0
    cross_excess = (sigma_cross / sigma_mean - 1.0) * 100 if sigma_mean > 0 else 0

    coupling = float(ZETA_AXIS @ cov @ LATTICE_AXIS)
    coupling_pct = (coupling / sigma_mean * 100) if sigma_mean > 0 else 0

    return {
        'sigma_zeta': sigma_zeta,
        'sigma_lattice': sigma_lattice,
        'sigma_cross': sigma_cross,
        'sigma_mean': sigma_mean,
        'zeta_excess': zeta_excess,
        'lattice_excess': lattice_excess,
        'cross_excess': cross_excess,
        'dominance': zeta_excess - lattice_excess,
        'coupling': coupling_pct,
        'n_nodes': len(positions),
        'com': mean,
    }


def print_decomposition(d, label=""):
    if d is None:
        print(f"  {label}: insufficient nodes")
        return
    z = d['zeta_excess']
    l = d['lattice_excess']
    dom = d['dominance']
    cpl = d['coupling']

    if dom > 0.5:
        bar = "Z" + "█" * min(int(abs(dom) * 2), 30)
    elif dom < -0.5:
        bar = "L" + "█" * min(int(abs(dom) * 2), 30)
    else:
        bar = "=" + "░" * min(int(max(abs(z), abs(l)) * 2), 30)

    print(f"  {label:>6} | Z:{z:>+7.2f}% | L:{l:>+7.2f}% | "
          f"dom:{dom:>+7.2f} | cpl:{cpl:>+6.2f}% | "
          f"N={d['n_nodes']:>6,} | {bar}")


# =====================================================================
# RUN ANALYSIS (reads from engine output)
# =====================================================================

def analyze_run(run_dir):
    run_dir = Path(run_dir)

    meta_path = run_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    reg_path = run_dir / 'registry.npy'
    if not reg_path.exists():
        print(f"No registry.npy in {run_dir}")
        return
    registry = np.load(str(reg_path)).astype(np.int32)

    grid_size = meta.get('grid_size', 89)
    positions = to_sim_space(registry, grid_size)

    clouds_path = run_dir / 'clouds.npz'
    has_clouds = clouds_path.exists()

    print("=" * 70)
    print(f"DUAL SIGNAL DECOMPOSITION — {run_dir.name}")
    print(f"Grid: {grid_size}³ | Nodes: {len(registry):,}")
    dot = abs(np.dot(ZETA_AXIS, LATTICE_AXIS))
    print(f"Zeta axis:    θ=34.3°, φ=-135.0°")
    print(f"Lattice axis: θ=54.7°, φ=+45.0° (cubic body diagonal)")
    print(f"Orthogonality: {dot:.4f}")
    print("=" * 70)

    # Static decomposition (all registered nodes, uniform weight)
    print(f"\n--- Static (injection geometry) ---")
    d = decompose_dual(positions)
    print_decomposition(d, "all")

    if d:
        print(f"\n  σ²_zeta:    {d['sigma_zeta']:.6f}")
        print(f"  σ²_lattice: {d['sigma_lattice']:.6f}")
        print(f"  σ²_cross:   {d['sigma_cross']:.6f}")
        print(f"  σ²_mean:    {d['sigma_mean']:.6f}")
        print(f"  ratio Z/L:  {d['sigma_zeta']/(d['sigma_lattice']+1e-20):.6f}")

    # Time evolution (if clouds available)
    if has_clouds:
        print(f"\n--- Time evolution (weighted by field values) ---")
        clouds = np.load(str(clouds_path))
        step_keys = sorted([k for k in clouds.files if k.endswith('_values')],
                           key=lambda k: int(k.split('_')[0][1:]))

        results = []
        for vk in step_keys:
            step = int(vk.split('_')[0][1:])
            values = clouds[vk]
            ak = vk.replace('_values', '_active')
            active = int(clouds[ak][0]) if ak in clouds else len(values)

            vals = values[:len(positions)]
            if len(vals) < 10:
                continue

            active_mask = vals > 0
            if active_mask.sum() < 10:
                continue

            d = decompose_dual(positions[active_mask], vals[active_mask])
            if d:
                d['step'] = step
                results.append(d)
                print_decomposition(d, f"s{step:04d}")

        if results:
            print_evolution_summary(results)

    return d


def print_evolution_summary(results):
    print(f"\n{'='*70}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*70}")

    best_zeta = max(results, key=lambda r: r['zeta_excess'])
    best_lattice = max(results, key=lambda r: r['lattice_excess'])
    best_dom = max(results, key=lambda r: r['dominance'])

    print(f"  Peak zeta:    step {best_zeta.get('step','?'):>5} "
          f"({best_zeta['zeta_excess']:+.2f}%)")
    print(f"  Peak lattice: step {best_lattice.get('step','?'):>5} "
          f"({best_lattice['lattice_excess']:+.2f}%)")
    print(f"  Best Z dom:   step {best_dom.get('step','?'):>5} "
          f"(dom={best_dom['dominance']:+.2f})")

    # Crossovers
    crossovers = []
    for i in range(1, len(results)):
        d_prev = results[i-1]['dominance']
        d_curr = results[i]['dominance']
        if d_prev * d_curr < 0:
            crossovers.append((results[i-1].get('step', i-1),
                               results[i].get('step', i)))
    if crossovers:
        print(f"  Crossovers ({len(crossovers)}):")
        for a, b in crossovers:
            print(f"    between step {a} and {b}")

    # Isotropic points
    iso = [r for r in results
           if abs(r['zeta_excess']) < 0.1 and abs(r['lattice_excess']) < 0.1]
    if iso:
        print(f"  Isotropic points ({len(iso)}):")
        for r in iso:
            print(f"    step {r.get('step','?'):>5} "
                  f"(Z={r['zeta_excess']:+.3f}%, L={r['lattice_excess']:+.3f}%)")

    # Periodicity
    dom_vals = np.array([r['dominance'] for r in results])
    if len(dom_vals) > 10:
        centered = dom_vals - dom_vals.mean()
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0] + 1e-10
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if (autocorr[i] > autocorr[i-1] and
                autocorr[i] > autocorr[i+1] and
                autocorr[i] > 0.2):
                peaks.append((i, autocorr[i]))
        if peaks:
            print(f"  Dominance periodicity:")
            for lag, strength in peaks[:3]:
                print(f"    period ≈ {lag} steps (autocorr = {strength:.3f})")

    print(f"{'='*70}")


# =====================================================================
# INJECTION SWEEP (pure geometry, no engine run needed)
# =====================================================================

def compute_injection(grid_size, n_zeros=50, use_hann=False,
                      tune_scalar=1.0, beat_detune=0.1):
    dx = 20.0 / (grid_size - 1)
    coords = np.linspace(-10, 10, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R_soft = np.sqrt(X**2 + Y**2 + Z**2 + (SOFT_ALPHA * dx)**2)
    del X, Y, Z

    amplitude = 1.0 / np.sqrt(R_soft)
    log_R = np.log(R_soft)

    zeros = ZETA_ZEROS[:n_zeros]
    n = len(zeros)
    complex_field = np.zeros_like(R_soft, dtype=np.complex128)

    for i, gamma in enumerate(zeros):
        w = (0.5 * (1.0 + math.cos(math.pi * i / n))) if use_hann else 1.0
        phase = gamma * log_R
        complex_field += (amplitude * np.exp(1j * phase) -
                          amplitude * np.exp(-1j * phase)) * w

    base_freq = (2.0 * np.pi / 20.0) * tune_scalar
    beat_envelope = (np.exp(1j * base_freq * R_soft) +
                     np.exp(1j * (base_freq + beat_detune) * R_soft))
    complex_field = complex_field * beat_envelope.real

    tension = np.abs(complex_field)
    t_min, t_max = tension.min(), tension.max()
    tension = (tension - t_min) / (t_max - t_min + 1e-8)

    threshold = np.quantile(tension.flatten(), 0.999)
    above = tension >= threshold

    node_idx = np.array(np.nonzero(above)).T
    coords_sim = (node_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)
    R_nodes = np.sqrt((coords_sim**2).sum(axis=1))
    mask = R_nodes >= DRAIN_RADIUS
    coords_sim = coords_sim[mask]

    return coords_sim


def sweep(grid_size, n_range, use_hann=False):
    print(f"\n--- Grid {grid_size}³ ---")
    results = []

    for nz in n_range:
        coords = compute_injection(grid_size, n_zeros=nz, use_hann=use_hann)
        d = decompose_dual(coords)
        if d is None:
            continue
        d['n_zeros'] = nz
        results.append(d)
        print_decomposition(d, f"n={nz:>3}")

    return results


def print_sweep_summary(results, grid_size):
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY — Grid {grid_size}³")
    print(f"{'='*70}")

    best_zeta = max(results, key=lambda r: r['zeta_excess'])
    best_lattice = max(results, key=lambda r: r['lattice_excess'])
    best_dom = max(results, key=lambda r: r['dominance'])

    print(f"  Peak zeta signal:    n={best_zeta['n_zeros']:>3} "
          f"({best_zeta['zeta_excess']:+.2f}%)")
    print(f"  Peak lattice signal: n={best_lattice['n_zeros']:>3} "
          f"({best_lattice['lattice_excess']:+.2f}%)")
    print(f"  Best zeta dominance: n={best_dom['n_zeros']:>3} "
          f"(dom={best_dom['dominance']:+.2f})")

    # Crossovers
    crossovers = []
    for i in range(1, len(results)):
        d_prev = results[i-1]['dominance']
        d_curr = results[i]['dominance']
        if d_prev * d_curr < 0:
            n_a = results[i-1]['n_zeros']
            n_b = results[i]['n_zeros']
            crossovers.append((n_a, n_b))
    if crossovers:
        print(f"  Crossovers ({len(crossovers)}):")
        for a, b in crossovers:
            print(f"    between n={a} and n={b}")

        # Tag crossover values
        for a, b in crossovers:
            mid = (a + b) / 2.0
            for v, tag in STRUCTURAL.items():
                if abs(mid - v) < 1.5:
                    print(f"      → n≈{mid:.1f} near {v}: {tag}")

    # Isotropic points
    iso = [r for r in results
           if abs(r['zeta_excess']) < 0.1 and abs(r['lattice_excess']) < 0.1]
    if iso:
        n_iso = len(iso)
        iso_ns = [r['n_zeros'] for r in iso]
        print(f"  Isotropic points ({n_iso}): n = {iso_ns}")
        for v, tag in STRUCTURAL.items():
            if v == n_iso:
                print(f"    → count {n_iso} = {tag}")
            if v == int(ZETA_ZEROS[0]):
                if n_iso == v:
                    print(f"    → count = ⌊γ₁⌋")

    # Periodicity
    dom_vals = np.array([r['dominance'] for r in results])
    if len(dom_vals) > 10:
        centered = dom_vals - dom_vals.mean()
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0] + 1e-10
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if (autocorr[i] > autocorr[i-1] and
                autocorr[i] > autocorr[i+1] and
                autocorr[i] > 0.2):
                peaks.append((i, autocorr[i]))
        if peaks:
            print(f"  Periodicity (autocorrelation):")
            for lag, strength in peaks[:3]:
                print(f"    period ≈ {lag} zeros (autocorr = {strength:.3f})")
                for v, tag in STRUCTURAL.items():
                    if v == lag:
                        print(f"      → {tag}")

    print(f"{'='*70}")


# =====================================================================
# CLI
# =====================================================================

def resolve_run_dir(run_arg):
    if run_arg:
        p = Path(run_arg)
        if p.exists():
            return p
        for base in [Path('runs_emergent'), Path('runs_coupled')]:
            candidate = base / run_arg
            if candidate.exists():
                return candidate
        print(f"Run not found: {run_arg}")
        sys.exit(1)

    for runs_dir in [Path('runs_emergent'), Path('runs_coupled')]:
        latest = runs_dir / 'latest.txt'
        if latest.exists():
            run_id = latest.read_text().strip()
            candidate = runs_dir / run_id
            if candidate.exists():
                return candidate

    print("No run found. Use --run or --sweep.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Dual Signal Decomposition — Zeta vs Lattice Analysis')
    parser.add_argument('--run', type=str, default=None,
                        help='Run directory to analyze')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep n_zeros in injection geometry')
    parser.add_argument('--grid', type=int, default=89,
                        help='Grid size for sweep (default: 89)')
    parser.add_argument('--grids', type=str, default=None,
                        help='Comma-separated grid sizes for multi-grid sweep')
    parser.add_argument('--nmin', type=int, default=5)
    parser.add_argument('--nmax', type=int, default=50)
    parser.add_argument('--hann', action='store_true',
                        help='Use Hann window on zeros')
    args = parser.parse_args()

    dot = abs(np.dot(ZETA_AXIS, LATTICE_AXIS))
    print("=" * 70)
    print("DUAL SIGNAL DECOMPOSITION")
    print(f"Zeta axis:    θ=34.3°, φ=-135.0°")
    print(f"Lattice axis: θ=54.7°, φ=+45.0° (cubic body diagonal)")
    print(f"Dot product:  {dot:.4f} (orthogonal)")
    print("=" * 70)

    if args.sweep:
        n_range = list(range(args.nmin, args.nmax + 1))

        if args.grids:
            grid_sizes = [int(x.strip()) for x in args.grids.split(',')]
        else:
            grid_sizes = [args.grid]

        for grid in grid_sizes:
            results = sweep(grid, n_range, use_hann=args.hann)
            print_sweep_summary(results, grid)
    else:
        run_dir = resolve_run_dir(args.run)
        analyze_run(run_dir)


if __name__ == '__main__':
    main()
