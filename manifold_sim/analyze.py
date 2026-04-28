"""
Resonant Field Analyzer — Unified Analysis Dispatcher
=======================================================
Single entry point for all post-run analysis tools.

Usage:
    python analyze.py prime              # Prime lock scanner on latest run
    python analyze.py symmetry           # PCA axis + antipodal correlation
    python analyze.py voids              # Void center tracking
    python analyze.py waterfall          # FFT waterfall
    python analyze.py waterfall --arcs   # Arc detection + zeta comparison
    python analyze.py axis               # Dual signal decomposition
    python analyze.py summary            # Print run summary

    python analyze.py prime --run 0014   # Analyze specific run
    python analyze.py prime --save p.png # Save plot

All tools default to latest run (reads runs/latest.txt).

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from sympy import isprime

# =============================================================================
# SHARED DATA LOADER
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_DIR   = SCRIPT_DIR / "runs"

def resolve_run_dir(run_arg=None):
    """Resolve run directory from --run argument or latest.txt."""
    if run_arg is not None:
        # Accept bare number or full path
        if run_arg.isdigit():
            run_dir = RUNS_DIR / f"{int(run_arg):04d}"
        else:
            run_dir = Path(run_arg)
        if not run_dir.exists():
            print(f"Run not found: {run_dir}")
            sys.exit(1)
        return run_dir

    latest_path = RUNS_DIR / "latest.txt"
    if not latest_path.exists():
        print("No latest.txt — run the engine first.")
        sys.exit(1)
    run_id = latest_path.read_text().strip()
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        print(f"Latest run not found: {run_dir}")
        sys.exit(1)
    return run_dir


def load_run(run_dir):
    """
    Load all data from a run directory.
    Returns dict with meta, registry, clouds, probes, slices.
    """
    run_dir = Path(run_dir)
    data = {'run_dir': run_dir}

    # Metadata
    meta_path = run_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            data['meta'] = json.load(f)
    else:
        data['meta'] = {}

    # Registry
    reg_path = run_dir / 'registry.npy'
    if reg_path.exists():
        data['registry'] = np.load(str(reg_path)).astype(np.int32)
    else:
        data['registry'] = None

    # Clouds (lazy — return the NPZ handle, not all data)
    clouds_path = run_dir / 'clouds.npz'
    if clouds_path.exists():
        data['clouds_file'] = str(clouds_path)
    else:
        data['clouds_file'] = None

    # Probes
    probes_path = run_dir / 'probes.npz'
    if probes_path.exists():
        data['probes_file'] = str(probes_path)
    else:
        data['probes_file'] = None

    # Slices
    slices_path = run_dir / 'slices.npz'
    if slices_path.exists():
        data['slices_file'] = str(slices_path)
    else:
        data['slices_file'] = None

    return data


def get_cloud_steps(clouds_file):
    """Extract sorted step numbers from clouds.npz keys."""
    if clouds_file is None:
        return []
    npz = np.load(clouds_file)
    steps = set()
    for key in npz.files:
        # keys like "s0042_values"
        if key.startswith('s') and '_' in key:
            try:
                step = int(key.split('_')[0][1:])
                steps.add(step)
            except ValueError:
                pass
    return sorted(steps)


def load_cloud_at_step(clouds_file, step):
    """Load cloud data for a specific step."""
    npz = np.load(clouds_file)
    prefix = f"s{step:04d}"
    values = npz[f"{prefix}_values"]
    active = int(npz[f"{prefix}_active"][0])
    mean_t = float(npz[f"{prefix}_mean_t"][0])
    t_unit = npz.get(f"{prefix}_tvec_unit", None)
    t_mag  = npz.get(f"{prefix}_tvec_mag", None)
    return values, active, mean_t, t_unit, t_mag


def to_sim_space(coords_idx, grid_size):
    """Grid index → simulation space [-10, 10]."""
    return (coords_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)


# =============================================================================
# PRIME LOCK SCANNER
# =============================================================================

def check_prime_adjacency(delta, max_k=7):
    """
    Test Δ for prime adjacency at harmonic tiers k=1,3,5,7,...

    k=1: Δ ± 1² = Δ ± 1   → Fundamental Lock
    k=3: Δ ± 3² = Δ ± 9   → Harmonic Lock
    k=5: Δ ± 5² = Δ ± 25  → Stress-State Lock
    k=7: Δ ± 7² = Δ ± 49  → (next tier)

    Returns list of (k, offset, candidate, is_prime) tuples.
    """
    results = []
    for k in range(1, max_k + 1, 2):  # odd k: 1, 3, 5, 7
        offset = k * k
        for sign in [+1, -1]:
            candidate = delta + sign * offset
            if candidate > 1:
                p = isprime(candidate)
                results.append({
                    'k': k,
                    'tier': {1: 'Fundamental', 3: 'Harmonic',
                             5: 'Stress-State', 7: 'Extended'}.get(k, f'k={k}'),
                    'sign': '+' if sign > 0 else '-',
                    'offset': offset,
                    'candidate': candidate,
                    'is_prime': p,
                })
    return results


def scan_prime_locks(values_int, labels=None, max_k=7):
    """
    Scan all pairs for prime lock relationships.

    values_int: array of integers (reduced node values, Δ counts, etc.)
    labels:     optional names for each value
    max_k:      highest odd k to test

    Returns summary dict.
    """
    n = len(values_int)
    if labels is None:
        labels = [f"v{i}" for i in range(n)]

    all_locks = []

    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(int(values_int[i]) - int(values_int[j]))
            if delta < 2:
                continue
            checks = check_prime_adjacency(delta, max_k)
            locks_found = [c for c in checks if c['is_prime']]
            if locks_found:
                all_locks.append({
                    'pair': (labels[i], labels[j]),
                    'values': (int(values_int[i]), int(values_int[j])),
                    'delta': delta,
                    'locks': locks_found,
                })

    return all_locks


def scan_square_relations(values_int, labels=None, tolerance=0):
    """
    Check for exact or near-exact square relationships between values.
    a² + b² = c² (Pythagorean), a² = b (perfect square), a*b = c² etc.

    Returns list of found relations.
    """
    n = len(values_int)
    if labels is None:
        labels = [f"v{i}" for i in range(n)]

    vals = [int(v) for v in values_int]
    val_set = set(vals)
    relations = []

    # Check for perfect squares
    for i, v in enumerate(vals):
        if v > 0:
            root = int(round(v ** 0.5))
            if abs(root * root - v) <= tolerance:
                relations.append({
                    'type': 'perfect_square',
                    'desc': f"{labels[i]}={v} = {root}²",
                    'values': [v],
                })

    # Pythagorean: a² + b² = c²
    for i in range(n):
        for j in range(i + 1, n):
            sq_sum = vals[i]**2 + vals[j]**2
            root = int(round(sq_sum ** 0.5))
            if abs(root * root - sq_sum) <= tolerance:
                relations.append({
                    'type': 'pythagorean',
                    'desc': f"{labels[i]}²+{labels[j]}²={vals[i]}²+{vals[j]}²"
                            f"={root}²={root*root}",
                    'values': [vals[i], vals[j], root],
                })

    # Product = square: a * b = c²
    for i in range(n):
        for j in range(i + 1, n):
            prod = vals[i] * vals[j]
            if prod > 0:
                root = int(round(prod ** 0.5))
                if abs(root * root - prod) <= tolerance:
                    relations.append({
                        'type': 'product_square',
                        'desc': f"{labels[i]}×{labels[j]}={vals[i]}×{vals[j]}"
                                f"={root}²={prod}",
                        'values': [vals[i], vals[j], root],
                    })

    return relations


def run_prime_analysis(run_data, verbose=True):
    """
    Full prime lock + square relation analysis on a run.

    Analyzes:
    1. Total vs active node count Δ (the partition lock)
    2. Pairwise Δ between cloud snapshots (temporal lock)
    3. Square relations in the node counts
    """
    meta     = run_data['meta']
    n_nodes  = meta.get('n_nodes', 0)
    grid     = meta.get('grid_size', 0)

    print(f"\n{'='*70}")
    print(f"PRIME LOCK SCANNER — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid}³ | Nodes: {n_nodes:,}")
    print(f"{'='*70}")

    # 1. Partition lock: total - active = Δ
    clouds_file = run_data['clouds_file']
    if clouds_file is None:
        print("No cloud data found.")
        return

    steps = get_cloud_steps(clouds_file)
    if not steps:
        print("No cloud steps found.")
        return

    print(f"\n[PARTITION LOCK — Δ = total − active]")
    print(f"  {'Step':>6} | {'Total':>7} | {'Active':>7} | {'Δ':>7} | "
          f"{'Δ-1':>7} {'P?':>3} | {'Δ+1':>7} {'P?':>3} | "
          f"{'Δ-9':>7} {'P?':>3} | {'Δ+9':>7} {'P?':>3} | "
          f"{'Δ-25':>7} {'P?':>3} | {'Δ+25':>7} {'P?':>3}")
    print(f"  {'-'*100}")

    partition_deltas = []

    for step in steps:
        values, active, mean_t, _, _ = load_cloud_at_step(clouds_file, step)
        delta = n_nodes - active
        partition_deltas.append((step, n_nodes, active, delta))

        # Check k=1,3,5
        checks = check_prime_adjacency(delta, max_k=5)
        row = f"  {step:>6} | {n_nodes:>7,} | {active:>7,} | {delta:>7,}"
        for c in checks:
            tag = '✓' if c['is_prime'] else ' '
            row += f" | {c['candidate']:>7} {tag:>3}"
        print(row)

    # 2. Temporal evolution: Δ between consecutive active counts
    if len(partition_deltas) > 1:
        print(f"\n[TEMPORAL Δ — change in active count between steps]")
        print(f"  {'Step A→B':>12} | {'ΔActive':>8} | k=1 locks | k=3 locks")
        print(f"  {'-'*60}")

        for idx in range(1, len(partition_deltas)):
            s_a, _, act_a, _ = partition_deltas[idx-1]
            s_b, _, act_b, _ = partition_deltas[idx]
            d_active = abs(act_b - act_a)
            if d_active < 2:
                continue
            checks = check_prime_adjacency(d_active, max_k=5)
            k1 = [c for c in checks if c['k'] == 1 and c['is_prime']]
            k3 = [c for c in checks if c['k'] == 3 and c['is_prime']]
            k1_str = ','.join(str(c['candidate']) for c in k1) if k1 else '—'
            k3_str = ','.join(str(c['candidate']) for c in k3) if k3 else '—'
            print(f"  {s_a:>5}→{s_b:<5} | {d_active:>8} | {k1_str:>9} | {k3_str:>9}")

    # 3. Square relations on key values
    key_values = [n_nodes]
    key_labels = ['N_total']
    if partition_deltas:
        _, _, first_active, first_delta = partition_deltas[0]
        _, _, last_active, last_delta = partition_deltas[-1]
        key_values += [first_active, first_delta, last_active, last_delta]
        key_labels += ['active_0', 'Δ_0', 'active_final', 'Δ_final']

    print(f"\n[SQUARE RELATIONS]")
    sq_rels = scan_square_relations(key_values, key_labels, tolerance=1)
    if sq_rels:
        for r in sq_rels:
            print(f"  {r['type']:18s}: {r['desc']}")
    else:
        print(f"  No exact square relations found (tolerance=1)")

    # 4. Orthogonality check: dot products of node position vectors
    registry = run_data['registry']
    if registry is not None and len(partition_deltas) > 0:
        grid_size = meta.get('grid_size', 255)
        positions = to_sim_space(registry.astype(np.float64), grid_size)

        # Get active mask at first cloud step
        first_step = steps[0]
        values, active, _, _, _ = load_cloud_at_step(clouds_file, first_step)
        active_mask = values > 0.5

        n_active = int(active_mask.sum())
        n_inactive = len(values) - n_active
        delta = n_inactive

        print(f"\n[ORTHOGONALITY — Step {first_step}]")
        print(f"  Active: {n_active} | Inactive: {n_inactive} | Δ: {delta}")

        # Check Δ through all k tiers
        print(f"\n  Prime adjacency of Δ={delta}:")
        for c in check_prime_adjacency(delta, max_k=7):
            tag = "✓ PRIME" if c['is_prime'] else "  —"
            print(f"    k={c['k']} ({c['tier']:14s}): "
                  f"Δ{c['sign']}{c['offset']:>3} = {c['candidate']:>7} {tag}")

    print(f"\n{'='*70}")


# =============================================================================
# SYMMETRY ANALYSIS (ported from symmetry_analyzer.py)
# =============================================================================

def run_symmetry_analysis(run_data, save_path=None, verbose=True):
    """PCA axis tracking, antipodal correlation, axis stability."""
    meta     = run_data['meta']
    registry = run_data['registry']
    clouds_f = run_data['clouds_file']

    if registry is None or clouds_f is None:
        print("Missing registry or clouds data.")
        return

    grid_size = meta.get('grid_size', 255)
    positions = to_sim_space(registry.astype(np.float64), grid_size)
    steps     = get_cloud_steps(clouds_f)

    print(f"\n{'='*70}")
    print(f"SYMMETRY ANALYSIS — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid_size}³ | Nodes: {len(registry):,} | Steps: {len(steps)}")
    print(f"{'='*70}")

    print(f"\n{'Step':>6} | {'Active':>8} | {'Axis x':>7} {'y':>7} {'z':>7} | "
          f"{'VarR':>5} | {'θ°':>6} | {'φ°':>6}")
    print('-' * 70)

    axes_history = []

    for step in steps:
        values, active, mean_t, _, _ = load_cloud_at_step(clouds_f, step)
        vals = values[:len(positions)]
        if len(vals) < 10:
            continue

        # Weighted PCA
        w = vals / (vals.sum() + 1e-12)
        mean_pos = (positions[:len(w)] * w[:, np.newaxis]).sum(axis=0)
        centered = positions[:len(w)] - mean_pos
        wcov = (centered * w[:, np.newaxis]).T @ centered

        try:
            _, s, Vt = np.linalg.svd(wcov)
            axis = Vt[0]
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            var_ratio = float(s[0] / s.sum()) if s.sum() > 0 else 0.0
        except np.linalg.LinAlgError:
            axis = np.array([0., 0., 1.])
            var_ratio = 0.0

        theta = np.degrees(np.arccos(np.clip(axis[2], -1, 1)))
        phi   = np.degrees(np.arctan2(axis[1], axis[0]))
        axes_history.append(axis)

        print(f"{step:>6} | {active:>8,} | {axis[0]:>7.4f} {axis[1]:>7.4f} "
              f"{axis[2]:>7.4f} | {var_ratio:>5.3f} | {theta:>6.1f} | {phi:>6.1f}")

    # Axis stability
    if len(axes_history) > 2:
        axes_arr = np.array(axes_history)
        mean_ax  = axes_arr.mean(axis=0)
        mean_ax  = mean_ax / (np.linalg.norm(mean_ax) + 1e-12)
        deviations = []
        for ax in axes_arr:
            dot = np.clip(np.abs(np.dot(ax, mean_ax)), 0, 1)
            deviations.append(np.degrees(np.arccos(dot)))
        mean_dev = float(np.mean(deviations))
        final_dev = deviations[-1]

        print(f"\nAxis stability: final={final_dev:.2f}°  mean={mean_dev:.2f}°")
        if final_dev < 5.0:
            print("→ STABLE AXIS: genuine preferred direction")
        elif final_dev < 20.0:
            print("→ WEAKLY STABLE: possible preferred direction")
        else:
            print("→ UNSTABLE: no reliable preferred axis")

    print(f"{'='*70}")


# =============================================================================
# VOID TRACKER (ported from void_tracker.py)
# =============================================================================

def run_void_analysis(run_data, save_path=None, verbose=True):
    """Find and track void centers on outer sphere."""
    from scipy.ndimage import gaussian_filter

    meta     = run_data['meta']
    registry = run_data['registry']
    clouds_f = run_data['clouds_file']

    if registry is None or clouds_f is None:
        print("Missing registry or clouds data.")
        return

    grid_size = meta.get('grid_size', 255)
    positions = to_sim_space(registry.astype(np.float64), grid_size)
    steps     = get_cloud_steps(clouds_f)

    R = np.sqrt((positions**2).sum(axis=1))
    R_max = R.max()
    outer_mask = R > 0.6 * R_max

    print(f"\n{'='*70}")
    print(f"VOID TRACKER — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid_size}³ | Outer nodes: {outer_mask.sum():,}")
    print(f"{'='*70}")

    for step in steps:
        values, active, mean_t, _, _ = load_cloud_at_step(clouds_f, step)
        vals = values[:len(positions)]
        if outer_mask.sum() < 10:
            continue

        outer_pos = positions[outer_mask]
        outer_val = vals[outer_mask] if len(vals) > outer_mask.sum() else vals[:outer_mask.sum()]

        # Spherical coordinates
        r_o = np.sqrt((outer_pos**2).sum(axis=1))
        theta = np.degrees(np.arccos(np.clip(outer_pos[:,2] / (r_o + 1e-10), -1, 1)))
        phi   = np.degrees(np.arctan2(outer_pos[:,1], outer_pos[:,0]))

        # Build density map
        bin_deg = 5.0
        n_theta = int(180 / bin_deg)
        n_phi   = int(360 / bin_deg)
        theta_bins = np.linspace(0, 180, n_theta + 1)
        phi_bins   = np.linspace(-180, 180, n_phi + 1)

        # Area-corrected weighting — guard against sin(0) and sin(180)
        sin_theta = np.sin(np.radians(theta))
        area_weight = np.where(sin_theta > 0.01, 1.0 / sin_theta, 1.0 / 0.01)

        density, _, _ = np.histogram2d(theta, phi,
                                        bins=[theta_bins, phi_bins],
                                        weights=outer_val * area_weight)
        counts, _, _ = np.histogram2d(theta, phi,
                                       bins=[theta_bins, phi_bins])
        # Safe division
        density = np.where(counts > 0, density / counts, 0.0)

        smoothed = gaussian_filter(density, sigma=1.5)
        mean_d = smoothed.mean()
        if mean_d < 1e-10:
            continue

        inverted = np.clip(mean_d - smoothed, 0, None)
        inv_max = inverted.max()
        if inv_max <= 0:
            continue
        inverted_norm = inverted / inv_max

        # Find void centers (local maxima in inverted map)
        voids = []
        rows, cols = inverted_norm.shape
        for ii in range(1, rows-1):
            for jj in range(cols):
                jm = (jj-1) % cols
                jp = (jj+1) % cols
                val = inverted_norm[ii, jj]
                if val < 0.4:
                    continue
                neighbors = [
                    inverted_norm[ii-1, jm], inverted_norm[ii-1, jj],
                    inverted_norm[ii-1, jp], inverted_norm[ii, jm],
                    inverted_norm[ii, jp],   inverted_norm[ii+1, jm],
                    inverted_norm[ii+1, jj], inverted_norm[ii+1, jp],
                ]
                if val >= max(neighbors):
                    tc = (theta_bins[ii] + theta_bins[ii+1]) / 2
                    pc = (phi_bins[jj] + phi_bins[jj+1]) / 2
                    voids.append((tc, pc, float(val)))

        voids.sort(key=lambda v: v[2], reverse=True)

        print(f"\nStep {step:>6}: {len(voids)} voids")
        if voids:
            print(f"  {'θ°':>8} | {'φ°':>8} | {'Depth':>6}")
            print(f"  {'-'*28}")
            for t, p, d in voids[:8]:
                print(f"  {t:>8.2f} | {p:>8.2f} | {d:>6.3f}")

    print(f"\n{'='*70}")


# =============================================================================
# SUMMARY
# =============================================================================

def run_summary(run_data):
    """Print the summary.txt or reconstruct from meta."""
    run_dir = run_data['run_dir']
    summary_path = run_dir / 'summary.txt'
    if summary_path.exists():
        print(summary_path.read_text(encoding='utf-8'))
    else:
        meta = run_data['meta']
        print(json.dumps(meta, indent=2))


# =============================================================================
# PHASE QUANTIZATION — ln(p) TEST (Euler product prediction)
# =============================================================================

def get_zeta_zeros(count=50):
    zeros = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425274, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846
    ]
    return zeros[:count]


def run_phase_analysis(run_data=None, save_path=None, verbose=True):
    """
    Test Gemini's Euler product prediction:
    Do fractional deltas between consecutive zeta zeros quantize
    into intervals proportional to ln(p) and 2·ln(p)?

    This operates on the zeta zeros themselves — no simulation run needed.
    If run_data has extra_freq in meta, those are appended for comparison.
    """
    zeros = get_zeta_zeros(50)

    # If we have run metadata with extra freqs, append them
    if run_data and run_data.get('meta'):
        extra = run_data['meta'].get('extra_freq')
        if extra:
            zeros = zeros + extra
            print(f"Including {len(extra)} extra frequencies from run config")

    zeros = np.array(zeros, dtype=np.float64)
    n = len(zeros)

    # Reference primes and their logs
    ref_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    ln_p = {p: np.log(p) for p in ref_primes}

    print(f"\n{'='*70}")
    print(f"PHASE QUANTIZATION — ln(p) TEST")
    print(f"{'='*70}")
    print(f"Zeros: {n} | Testing against ln(p) for p = {ref_primes[:8]}...")
    print(f"\nReference values:")
    for p in ref_primes[:8]:
        print(f"  ln({p:>2}) = {ln_p[p]:.6f}    2·ln({p:>2}) = {2*ln_p[p]:.6f}")

    # --- Method 1: Consecutive gap fractional parts ---
    gaps = np.diff(zeros)
    frac_gaps = gaps - np.floor(gaps)

    print(f"\n[CONSECUTIVE GAPS — fractional parts]")
    print(f"  {'n→n+1':>8} | {'Gap':>8} | {'Frac':>8} | Best ln(p) match")
    print(f"  {'-'*60}")

    gap_matches = {p: 0 for p in ref_primes}
    tolerance = 0.05

    for i in range(len(gaps)):
        fg = frac_gaps[i]
        # Test against ln(p) mod 1 and 2·ln(p) mod 1
        best_p = None
        best_k = None
        best_residual = 999.0

        for p in ref_primes:
            for k_mult, k_label in [(1, '1'), (2, '2')]:
                target = (k_mult * ln_p[p]) % 1.0
                # Check distance on the circle [0,1)
                residual = min(abs(fg - target), abs(fg - target + 1), abs(fg - target - 1))
                if residual < best_residual:
                    best_residual = residual
                    best_p = p
                    best_k = k_label

        hit = '✓' if best_residual < tolerance else ' '
        if best_residual < tolerance:
            gap_matches[best_p] += 1

        print(f"  γ{i+1:>2}→γ{i+2:<2} | {gaps[i]:>8.4f} | {fg:>8.4f} | "
              f"{best_k}·ln({best_p:>2})={float((int(best_k)*ln_p[best_p])%1):.4f} "
              f"Δ={best_residual:.4f} {hit}")

    print(f"\n  Match counts (tolerance={tolerance}):")
    for p in ref_primes[:8]:
        bar = '█' * gap_matches[p]
        print(f"    ln({p:>2}): {gap_matches[p]:>3} {bar}")

    total_hits = sum(1 for i in range(len(gaps))
                     if min(min(abs(frac_gaps[i] - (k * ln_p[p]) % 1),
                                abs(frac_gaps[i] - (k * ln_p[p]) % 1 + 1),
                                abs(frac_gaps[i] - (k * ln_p[p]) % 1 - 1))
                            for p in ref_primes for k in [1, 2]) < tolerance)

    # --- Method 2: All pairwise Δ_frac ---
    print(f"\n[ALL PAIRWISE Δ_frac — clustering test]")

    all_deltas = []
    for i in range(n):
        for j in range(i+1, n):
            d = abs(zeros[i] - zeros[j])
            all_deltas.append(d - np.floor(d))
    all_deltas = np.array(all_deltas)

    # Histogram the fractional deltas and check if peaks align with ln(p) mod 1
    n_bins = 100
    hist, bin_edges = np.histogram(all_deltas, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=np.mean(hist) * 1.3,
                                    distance=3)

    print(f"  Total pairs: {len(all_deltas)}")
    print(f"  Histogram peaks (>{1.3:.1f}× mean):")

    for pk in peaks:
        fc = bin_centers[pk]
        count = hist[pk]
        # Match against ln(p) mod 1
        best_match = None
        best_res = 999.0
        for p in ref_primes:
            for k_mult, k_label in [(1, ''), (2, '2·')]:
                target = (k_mult * ln_p[p]) % 1.0
                res = min(abs(fc - target), abs(fc - target + 1), abs(fc - target - 1))
                if res < best_res:
                    best_res = res
                    best_match = f"{k_label}ln({p})"
        hit = '✓' if best_res < tolerance else ' '
        print(f"    frac={fc:.4f} count={count:>4} → "
              f"{best_match}={(float(best_res)+fc):.4f} Δ={best_res:.4f} {hit}")

    # --- Method 3: Raw gaps (not fractional) against ln(p) ratios ---
    print(f"\n[RAW GAPS — ratio test: Δγ / ln(p)]")
    print(f"  If gaps are integer multiples of ln(p), ratio should be near-integer")
    print(f"\n  {'n→n+1':>8} | {'Gap':>8} | {'÷ln2':>8} | {'÷ln3':>8} | "
          f"{'÷ln5':>8} | {'÷ln7':>8}")
    print(f"  {'-'*60}")

    clean_ratio_count = {p: 0 for p in [2, 3, 5, 7]}
    for i in range(min(len(gaps), 49)):
        g = gaps[i]
        ratios = {}
        row = f"  γ{i+1:>2}→γ{i+2:<2} | {g:>8.4f}"
        for p in [2, 3, 5, 7]:
            r = g / ln_p[p]
            frac_r = abs(r - round(r))
            ratios[p] = (r, frac_r)
            tag = '*' if frac_r < 0.1 else ' '
            if frac_r < 0.1:
                clean_ratio_count[p] += 1
            row += f" | {r:>7.3f}{tag}"
        print(row)

    print(f"\n  Near-integer ratio counts (|frac| < 0.1):")
    for p in [2, 3, 5, 7]:
        pct = 100 * clean_ratio_count[p] / len(gaps)
        bar = '█' * clean_ratio_count[p]
        print(f"    ÷ln({p}): {clean_ratio_count[p]:>3}/{len(gaps)} "
              f"({pct:.0f}%) {bar}")

    # Random baseline: how many would we expect by chance?
    # For |frac| < 0.1, a uniform random variable has P = 0.2 (two tails of 0.1)
    expected = int(0.2 * len(gaps))
    print(f"    Random baseline: ~{expected}/{len(gaps)} (20%)")

    print(f"\n{'='*70}")


# =============================================================================
# VORONOI / DELAUNAY VOID ANALYSIS (3D structure of empty regions)
# =============================================================================

def _batch_circumsphere(tet_pts):
    """
    tet_pts: (N, 4, 3). Returns (centers (N,3), radii (N,), valid_mask (N,)).

    For each tetrahedron, finds the circumcenter c by solving:
        2 (P_0 - P_i) · c = |P_0|^2 - |P_i|^2   for i = 1, 2, 3
    and circumradius = |P_0 - c|.
    The Delaunay property guarantees the circumsphere is node-empty,
    so circumradius at each tet = radius of largest empty ball touching 4 nodes.
    """
    p0   = tet_pts[:, 0, :]
    rest = tet_pts[:, 1:, :]
    A    = 2.0 * (p0[:, None, :] - rest)
    b    = (p0 ** 2).sum(axis=1, keepdims=True) - (rest ** 2).sum(axis=2)

    dets  = np.linalg.det(A)
    valid = np.abs(dets) > 1e-10
    centers = np.zeros_like(p0)
    radii   = np.full(len(tet_pts), np.inf)

    if valid.any():
        c = np.linalg.solve(A[valid], b[valid][..., None]).squeeze(-1)
        centers[valid] = c
        radii[valid]   = np.linalg.norm(p0[valid] - c, axis=1)

    return centers, radii, valid


def run_voronoi_analysis(run_data, save_path=None, verbose=True):
    """
    3D Voronoi / Delaunay void analysis.

    Produces:
      - Detection of the primary radial gap in the node registry (if any)
      - Delaunay tetrahedra classified as gap-spanning or intra-shell
      - Circumradii as empty-ball radii at each Voronoi vertex
      - Top-N largest voids with positions
      - Sphericity test on gap-void center distribution
      - Per-shell temporal activity trajectory
      - Per-void coolness evolution (mean ω over 4 surrounding nodes)

    Notes on interpretation:
      - Empty-ball radius = circumradius of a Delaunay tet (provably empty)
      - Centers lying outside the box are filtered out (hull artifacts)
      - Centers clustering at grid body-diagonals (±k, ±k, ±k) are cubic-grid
        symmetry artifacts, not physical voids
      - Intra-shell tet circumradii do NOT equal "within-shell void size";
        they reflect tet geometry, not field structure. Moiré voids require
        a different metric (k-NN asymmetry or cell-volume analysis).
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        print("scipy required for voronoi analysis.")
        return

    meta     = run_data['meta']
    registry = run_data['registry']
    clouds_f = run_data['clouds_file']

    if registry is None:
        print("Missing registry.")
        return

    grid_size = meta.get('grid_size', 255)
    positions = to_sim_space(registry.astype(np.float64), grid_size)
    n_nodes   = len(positions)
    R         = np.sqrt((positions ** 2).sum(axis=1))

    print(f"\n{'='*72}")
    print(f"VORONOI / DELAUNAY VOID ANALYSIS — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid_size}³ | Nodes: {n_nodes}")
    print(f"{'='*72}")

    # -- Radial shell detection -------------------------------------------
    R_sorted    = np.sort(R)
    gaps        = np.diff(R_sorted)
    big_gap_idx = int(np.argmax(gaps))
    gap_inner   = float(R_sorted[big_gap_idx])
    gap_outer   = float(R_sorted[big_gap_idx + 1])
    gap_width   = gap_outer - gap_inner

    # Shell structure detection: compare the largest radial gap to the
    # second-largest. If the biggest is much bigger than the next, we've
    # found a real discontinuity. (Median and percentile baselines fail
    # here because many consecutive R values are identical on a grid.)
    gaps_sorted    = np.sort(gaps)[::-1]
    second_biggest = float(gaps_sorted[1]) if len(gaps_sorted) > 1 else 0.0
    gap_ratio      = gap_width / max(second_biggest, 1e-6)

    print(f"\n[RADIAL SHELL STRUCTURE]")
    print(f"  R range:            [{R.min():.3f}, {R.max():.3f}]")
    print(f"  Largest radial gap: R = {gap_inner:.3f} → {gap_outer:.3f}  (width {gap_width:.3f})")
    print(f"  2nd-largest gap:    {second_biggest:.4f}")
    print(f"  Ratio 1st/2nd:      {gap_ratio:.1f}x  "
          f"{'(SHELL STRUCTURE DETECTED)' if gap_ratio > 5 else '(no clear shell structure)'}")

    has_shells = gap_ratio > 5
    if has_shells:
        n_inner = int((R <= gap_inner).sum())
        n_outer = int((R >= gap_outer).sum())
        print(f"  Inner shell:        {n_inner} nodes (R ≤ {gap_inner:.3f})")
        print(f"  Outer shell:        {n_outer} nodes (R ≥ {gap_outer:.3f})")
    else:
        n_inner = n_nodes
        n_outer = 0

    # -- Delaunay triangulation ------------------------------------------
    print(f"\n[DELAUNAY TRIANGULATION]")
    tri       = Delaunay(positions)
    simplices = tri.simplices
    n_tets    = len(simplices)
    print(f"  Tetrahedra: {n_tets}")

    tet_pts            = positions[simplices]
    centers, radii, vld = _batch_circumsphere(tet_pts)

    in_box = np.abs(centers).max(axis=1) < 10.0
    valid  = vld & in_box
    tet_R  = np.sqrt((centers ** 2).sum(axis=1))
    print(f"  Valid circumspheres (in box): {int(valid.sum())} / {n_tets}")

    # -- Classification by shell membership ------------------------------
    if has_shells:
        R_per_tet = R[simplices]
        min_R     = R_per_tet.min(axis=1)
        max_R     = R_per_tet.max(axis=1)
        spans_gap = valid & (min_R <= gap_inner) & (max_R >= gap_outer)
        in_inner  = valid & (max_R <= gap_inner)
        in_outer  = valid & (min_R >= gap_outer)
        mixed     = valid & ~(spans_gap | in_inner | in_outer)

        print(f"\n[TET CLASSIFICATION]")
        print(f"  Gap-spanning:       {int(spans_gap.sum())}")
        print(f"  Within inner shell: {int(in_inner.sum())}")
        outer_note = ' — outer shell is too thin radially for interior tets' if int(in_outer.sum()) == 0 and n_outer > 0 else ''
        print(f"  Within outer shell: {int(in_outer.sum())}{outer_note}")
        print(f"  Mixed/partial:      {int(mixed.sum())}")

        print(f"\n[CIRCUMRADIUS STATISTICS BY CLASS]")
        print(f"  {'class':<18} | {'count':>5} | {'min':>6} | {'median':>6} | "
              f"{'mean':>6} | {'max':>6}")
        print(f"  {'-'*72}")
        for label, mask in [('all valid', valid),
                            ('gap-spanning', spans_gap),
                            ('within inner', in_inner),
                            ('within outer', in_outer),
                            ('mixed', mixed)]:
            if mask.any():
                r = radii[mask]
                print(f"  {label:<18} | {int(mask.sum()):>5} | "
                      f"{r.min():>6.3f} | {np.median(r):>6.3f} | "
                      f"{r.mean():>6.3f} | {r.max():>6.3f}")
    else:
        spans_gap = np.zeros_like(valid)
        in_inner  = valid.copy()
        in_outer  = np.zeros_like(valid)
        mixed     = np.zeros_like(valid)

    # -- Top-N largest voids ---------------------------------------------
    print(f"\n[TOP 15 LARGEST VOIDS BY CIRCUMRADIUS]")
    all_sorted = np.argsort(-radii)
    all_sorted = [i for i in all_sorted if valid[i]][:15]
    print(f"  {'idx':>5} | {'R_ctr':>6} | {'radius':>6} | "
          f"{'x':>6} {'y':>6} {'z':>6} | class")
    print(f"  {'-'*68}")
    for i in all_sorted:
        c = centers[i]
        cls = ('gap' if spans_gap[i] else
               'inner' if in_inner[i] else
               'outer' if in_outer[i] else 'mixed')
        print(f"  {i:>5} | {tet_R[i]:>6.3f} | {radii[i]:>6.3f} | "
              f"{c[0]:>6.2f} {c[1]:>6.2f} {c[2]:>6.2f} | {cls}")

    # -- Gap-void sphericity test ---------------------------------------
    if has_shells and spans_gap.sum() > 10:
        gR          = tet_R[spans_gap]
        expected_R  = (gap_inner + gap_outer) / 2.0
        deviation   = gR.mean() - expected_R

        print(f"\n[GAP-VOID SPHERICITY]")
        print(f"  Gap-void count:         {int(spans_gap.sum())}")
        print(f"  Center R range:         [{gR.min():.3f}, {gR.max():.3f}]")
        print(f"  Center R mean±std:      {gR.mean():.3f} ± {gR.std():.3f}")
        print(f"  Expected if spherical:  {expected_R:.3f}")
        print(f"  Deviation:              {deviation:+.3f}  "
              f"(std/mean = {gR.std()/max(gR.mean(),1e-9):.3f})")

    # -- Temporal trajectory --------------------------------------------
    if clouds_f is not None and has_shells:
        steps = get_cloud_steps(clouds_f)
        if len(steps) > 0:
            print(f"\n[TEMPORAL SHELL ACTIVITY — 9 sample steps]")
            print(f"  {'step':>6} | {'inner active':>14} | {'outer active':>14} | "
                  f"{'inner frac':>10}")
            print(f"  {'-'*58}")
            inner_reg = (R <= gap_inner)
            outer_reg = (R >= gap_outer)
            sample    = np.linspace(0, len(steps) - 1, 9).astype(int)
            for si in sample:
                step = steps[si]
                vals, _, _, _, _ = load_cloud_at_step(clouds_f, step)
                in_act  = int((vals[inner_reg] > 0.5).sum())
                out_act = int((vals[outer_reg] > 0.5).sum())
                frac    = in_act / max(1, n_inner)
                print(f"  {step:>6} | {in_act:>6}/{n_inner:<6} | "
                      f"{out_act:>6}/{n_outer:<6} | {frac:>10.3f}")

            # Per-gap-void coolness evolution
            if spans_gap.sum() >= 6:
                top_gap = [t for t in np.argsort(-radii) if spans_gap[t]][:6]
                print(f"\n[GAP-VOID COOLNESS EVOLUTION — top 6 by radius]")
                print(f"  Coolness = 1 - mean(ω over 4 surrounding nodes)")
                print(f"  {'step':>6} | " + ' '.join(f'g{i+1:>5}' for i in range(6)))
                print(f"  {'-'*50}")
                for si in sample:
                    step = steps[si]
                    vals, _, _, _, _ = load_cloud_at_step(clouds_f, step)
                    row = f"  {step:>6} |"
                    for t in top_gap:
                        four     = vals[simplices[t]]
                        coolness = 1.0 - float(np.clip(four.mean(), 0, 1))
                        row += f" {coolness:>5.2f}"
                    print(row)

    print(f"\n{'='*72}")
    print(f"NOTE: Intra-shell tet circumradii do NOT represent moiré voids;")
    print(f"      they're Delaunay geometry artifacts. Interior void detection")
    print(f"      requires k-NN asymmetry or per-node cell-volume analysis,")
    print(f"      which this tool does not yet provide.")
    print(f"{'='*72}")


# =============================================================================
# ENTRY POINT
# =============================================================================

TOOLS = {
    'prime':     ('Prime Lock Scanner',       run_prime_analysis),
    'symmetry':  ('Symmetry Analyzer',        run_symmetry_analysis),
    'voids':     ('Void Tracker (angular)',   run_void_analysis),
    'voronoi':   ('Voronoi/Delaunay Void Analysis', run_voronoi_analysis),
    'phases':    ('Phase Quantization (ln(p))', run_phase_analysis),
    'summary':   ('Run Summary',              run_summary),
}

def main():
    parser = argparse.ArgumentParser(
        description='Resonant Field Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tools:
  prime       Prime lock scanner (Δ±k² primality, square relations)
  symmetry    PCA axis tracking, antipodal correlation
  voids       Void tracker (angular, outer shell only)
  voronoi     Voronoi/Delaunay 3D void analysis (shells, gap, sphericity)
  phases      Phase quantization: test ln(p) clustering in zero gaps
  summary     Print run summary

Examples:
  python analyze.py prime
  python analyze.py voronoi
  python analyze.py phases
  python analyze.py symmetry --run 0014
  python analyze.py voids --save voids.png
        """
    )
    parser.add_argument('tool', choices=list(TOOLS.keys()),
                        help='Analysis tool to run')
    parser.add_argument('--run',  type=str, default=None,
                        help='Run ID (default: latest)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save output to file')
    parser.add_argument('--arcs', action='store_true',
                        help='(waterfall) Enable arc detection')

    args = parser.parse_args()

    tool_name, tool_func = TOOLS[args.tool]
    print(f"Tool: {tool_name}")

    # Phases can run without a simulation run (operates on zeta zeros)
    if args.tool == 'phases':
        run_data = None
        if args.run:
            run_dir = resolve_run_dir(args.run)
            run_data = load_run(run_dir)
            print(f"Run directory: {run_dir}")
        tool_func(run_data, save_path=args.save)
    else:
        run_dir  = resolve_run_dir(args.run)
        run_data = load_run(run_dir)
        print(f"Run directory: {run_dir}")

        if args.tool == 'summary':
            tool_func(run_data)
        elif args.save:
            tool_func(run_data, save_path=args.save)
        else:
            tool_func(run_data)


if __name__ == '__main__':
    main()
