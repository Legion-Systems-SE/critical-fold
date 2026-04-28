"""
Möbius Twist Curvature Analyzer
=================================
Maps the primitive root orbit (ρ=3163 mod Ω=3677) onto the engine's
node manifold and measures local twist curvature. Checks whether
void positions correlate with twist extrema.

The primitive root orbit is a complete permutation of Z/ΩZ* that
encodes angular structure via its first differences. This tool
projects that structure onto the spatial node positions and asks:
does the engine's void geometry know about the algebraic twist?

Three mapping methods:
  phi       — legacy φ-only azimuthal mapping (known degeneracy)
  quadrant  — (θ,φ) joint mapping via the orbit's 4-column grid
              structure. Respects the {0,1}/{2,3} column pairing
              and the 11×Ω hemisphere imbalance.
  radial    — log(r) mapping along the spectral injection axis.

Usage:
    python twist.py                                    # quadrant mapping (default)
    python twist.py --run runs_coupled/0001
    python twist.py --mapping phi                      # legacy comparison
    python twist.py --mapping radial                   # test radial hypothesis
    python twist.py --mapping quadrant --mapping phi   # (run twice to compare)
    python twist.py --step 100                         # analyze specific step
    python twist.py --orbit-only                       # just orbit diagnostics
    python twist.py --top 8                            # number of voids

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from sympy import isprime, factorint

SCRIPT_DIR = Path(__file__).parent.resolve()

# =============================================================================
# THE ALGEBRAIC CONSTANTS
# =============================================================================

RHO   = 3163           # Primitive root = 4 × 7 × 113 − 1
OMEGA = 3677           # Anchor prime, (Ω−1)/4 = 919 (also prime)
N_ORB = OMEGA - 1      # Orbit length = 3676

# Precomputed structural constants
RHO_INV = pow(RHO, -1, OMEGA)  # = 3584 = 2⁹ × 7
FORBIDDEN_DIFF = (1 - RHO_INV) % OMEGA  # = 94 = 2 × 47
MIDPOINT_VAL = OMEGA - 1       # seq[N/2] = −1 mod Ω always


# =============================================================================
# ORBIT GENERATION
# =============================================================================

def generate_orbit():
    """
    Generate the full primitive root orbit: seq[n] = ρ^n mod Ω.
    Returns the sequence and its first differences (twist rates).
    """
    seq = np.zeros(N_ORB, dtype=np.int64)
    val = 1
    for n in range(N_ORB):
        seq[n] = val
        val = (val * RHO) % OMEGA

    # First differences mod Ω (the twist rate at each step)
    diffs = np.array([(seq[n+1] - seq[n]) % OMEGA for n in range(N_ORB - 1)],
                     dtype=np.int64)

    return seq, diffs


def orbit_diagnostics(seq, diffs):
    """Print structural diagnostics of the primitive root orbit."""

    print(f"\n{'='*72}")
    print(f"PRIMITIVE ROOT ORBIT — ρ={RHO}, Ω={OMEGA}")
    print(f"{'='*72}")

    print(f"\n  Orbit length: {N_ORB} = {factorint(N_ORB)}")
    print(f"  ρ = {RHO} = 4 × 7 × 113 − 1 (π convergent construction)")
    print(f"  ρ⁻¹ = {RHO_INV} = {factorint(RHO_INV)} = 2⁹ × 7")

    # Verify midpoint
    mid = N_ORB // 2
    print(f"\n  Midpoint seq[{mid}] = {seq[mid]}")
    print(f"  ≡ −1 (mod Ω): {seq[mid] == OMEGA - 1}")

    # Verify mirror symmetry
    n_checked = 0
    all_unity = True
    for n in [1, 2, 5, 14, 50, 100, 500, 1000, 1838]:
        if n < N_ORB:
            product = (seq[n] * seq[N_ORB - n]) % OMEGA
            if product != 1:
                all_unity = False
            n_checked += 1
    print(f"  Mirror symmetry (seq[n]·seq[N-n] ≡ 1): {all_unity}")
    print(f"    (checked {n_checked} pairs)")

    # First differences
    unique_diffs = len(set(diffs.tolist()))
    print(f"\n  First differences:")
    print(f"    Distinct values: {unique_diffs} / {len(diffs)}")

    all_possible = set(range(OMEGA))
    present = set(diffs.tolist())
    missing = sorted(all_possible - present)
    print(f"    Missing values: {missing}")
    for m in missing:
        if m > 0:
            print(f"      {m} = {factorint(m)}")
    print(f"    Forbidden difference: {FORBIDDEN_DIFF} = 2 × 47")

    # 4-column grid structure
    W = 4
    H = N_ORB // W
    if N_ORB % W == 0:
        print(f"\n  4-column grid ({H} × {W}):")
        col_sums = [0] * W
        for row in range(H):
            for col in range(W):
                col_sums[col] += int(seq[row * W + col])

        for col in range(W):
            q = col_sums[col] // OMEGA
            print(f"    Col {col}: {col_sums[col]} = {q} × Ω")

        q0 = col_sums[0] // OMEGA
        q2 = col_sums[2] // OMEGA
        print(f"    Pair imbalance: {q0} − {q2} = {q0 - q2}")
        print(f"    Total imbalance: 2 × {q0 - q2} × Ω = {2*(q0-q2)} × Ω")


# =============================================================================
# SPATIAL MAPPING
# =============================================================================

def to_sim_space(coords_idx, grid_size):
    """Grid index → simulation space [-10, 10]."""
    return (coords_idx.astype(np.float64) - grid_size / 2.0) * (20.0 / grid_size)


def nodes_to_polar(positions):
    """
    Convert 3D node positions to (r, θ, φ) spherical coordinates.
    Returns r, theta (polar), phi (azimuthal).
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-12), -1, 1))   # polar angle
    phi = np.arctan2(y, x)                                 # azimuthal angle
    return r, theta, phi


def map_angles_to_orbit_phi(phi, seq):
    """
    LEGACY: φ-only mapping. Retained for comparison.
    Creates heavy degeneracy (orbit index 766 dominates).
    """
    phi_norm = (phi + np.pi) / (2 * np.pi)
    orbit_norm = seq.astype(np.float64) / OMEGA

    orbit_idx = np.zeros(len(phi), dtype=np.int64)
    for i in range(len(phi)):
        d = np.abs(orbit_norm - phi_norm[i])
        d = np.minimum(d, 1.0 - d)
        orbit_idx[i] = np.argmin(d)

    return orbit_idx


def map_nodes_to_orbit_quadrant(r, theta, phi, seq):
    """
    Joint (θ, φ) mapping respecting the orbit's 4-column grid structure.

    The orbit naturally folds into a 4 × 919 grid (N=3676 = 4 × 919).
    The 4 columns pair as {0,1} and {2,3} with an imbalance of 11×Ω
    between pairs. This maps onto angular quadrants:

      Col 0: θ ∈ [0, π/2),  φ ∈ [0, π)    — upper-front
      Col 1: θ ∈ [0, π/2),  φ ∈ [-π, 0)   — upper-back
      Col 2: θ ∈ [π/2, π],  φ ∈ [0, π)    — lower-front
      Col 3: θ ∈ [π/2, π],  φ ∈ [-π, 0)   — lower-back

    The hemisphere split (cols {0,1} vs {2,3}) carries the 11-imbalance.
    Within each column, a combined angular coordinate selects the row
    (one of 919 positions), mapped to orbit index via nearest-value match.

    Returns: orbit_index for each node
    """
    W = 4
    H = N_ORB // W  # 919

    n_nodes = len(r)
    orbit_idx = np.zeros(n_nodes, dtype=np.int64)

    # Assign each node to a quadrant (column)
    upper = theta < (np.pi / 2)
    front = phi >= 0

    col = np.zeros(n_nodes, dtype=np.int64)
    col[upper & front]  = 0
    col[upper & ~front] = 1
    col[~upper & front] = 2
    col[~upper & ~front] = 3

    # Within-column angular coordinate: combine θ and φ into a single
    # monotonic parameter. Use the "unwound" angle:
    #   α = θ_local × cos(φ_local) + φ_local × sin(θ_local)
    # This mixes both angles so nodes at different (θ,φ) but similar
    # φ alone get different positions — breaking the degeneracy.
    theta_norm = theta / np.pi          # [0, 1]
    phi_norm = (phi + np.pi) / (2*np.pi)  # [0, 1]

    # Combined angular parameter: weighted sum with golden ratio to
    # avoid rational resonances creating new degeneracies
    GOLDEN = (1 + np.sqrt(5)) / 2
    alpha = np.mod(theta_norm + GOLDEN * phi_norm, 1.0)

    # Extract the column subsequences from the orbit
    col_seqs = []
    for c in range(W):
        col_vals = np.array([int(seq[row * W + c]) for row in range(H)],
                            dtype=np.int64)
        col_seqs.append(col_vals)

    # Normalize each column's values to [0, 1) for matching
    col_norms = []
    for c in range(W):
        col_norms.append(col_seqs[c].astype(np.float64) / OMEGA)

    # Map each node to its column's nearest orbit position
    for i in range(n_nodes):
        c = col[i]
        a = alpha[i]
        # Find the row whose normalized orbit value is nearest to α
        d = np.abs(col_norms[c] - a)
        d = np.minimum(d, 1.0 - d)  # circular distance
        row = np.argmin(d)
        # Convert back to global orbit index
        orbit_idx[i] = row * W + c

    return orbit_idx, col


def map_nodes_to_orbit_radial(r, theta, phi, seq):
    """
    Radial-harmonic mapping: uses r as the primary coordinate.

    The zeta zeros seed structure at specific log-radial distances.
    This mapping projects each node's radial position through the
    orbit, using log(r) as the natural coordinate (matching the
    engine's log-radial phase injection).

    Breaks angular degeneracy entirely by using a different axis.
    Useful for testing whether the twist structure is radial rather
    than angular.

    Returns: orbit_index for each node
    """
    n_nodes = len(r)
    orbit_idx = np.zeros(n_nodes, dtype=np.int64)

    # Use log-radial coordinate, normalized to [0, 1)
    log_r = np.log(r + 1e-8)
    log_r_norm = (log_r - log_r.min()) / (log_r.max() - log_r.min() + 1e-12)

    orbit_norm = seq.astype(np.float64) / OMEGA

    for i in range(n_nodes):
        d = np.abs(orbit_norm - log_r_norm[i])
        d = np.minimum(d, 1.0 - d)
        orbit_idx[i] = np.argmin(d)

    return orbit_idx


def map_angles_to_orbit(r, theta, phi, seq, method='quadrant'):
    """
    Dispatcher for orbit mapping methods.

    Methods:
      'phi'       — legacy φ-only (known degeneracy at index 766)
      'quadrant'  — (θ,φ) joint mapping via 4-column grid structure
      'radial'    — log(r) mapping along the spectral injection axis

    Returns: orbit_index array, and optionally column assignments
    """
    if method == 'phi':
        return map_angles_to_orbit_phi(phi, seq), None
    elif method == 'quadrant':
        return map_nodes_to_orbit_quadrant(r, theta, phi, seq)
    elif method == 'radial':
        return map_nodes_to_orbit_radial(r, theta, phi, seq), None
    else:
        raise ValueError(f"Unknown mapping method: {method}")


def compute_twist_curvature(orbit_idx, diffs):
    """
    Compute local twist curvature at each node based on its
    position in the primitive root orbit.

    The twist rate is the first difference of the orbit at the
    node's orbit index. The twist curvature is the second difference
    (rate of change of twist rate).

    Returns: twist_rate, twist_curvature arrays (one per node)
    """
    n_nodes = len(orbit_idx)

    # Twist rate: first difference at orbit position
    twist_rate = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        idx = orbit_idx[i]
        if idx < len(diffs):
            # Normalize to [-0.5, 0.5] range
            d = diffs[idx] / float(OMEGA)
            if d > 0.5:
                d -= 1.0
            twist_rate[i] = d

    # Twist curvature: second difference (change in twist rate)
    # Use neighboring orbit positions
    twist_curv = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        idx = orbit_idx[i]
        if 0 < idx < len(diffs) - 1:
            d_prev = diffs[idx - 1] / float(OMEGA)
            d_curr = diffs[idx] / float(OMEGA)
            if d_prev > 0.5: d_prev -= 1.0
            if d_curr > 0.5: d_curr -= 1.0
            twist_curv[i] = d_curr - d_prev

    return twist_rate, twist_curv


# =============================================================================
# VOID DETECTION (from observe.py, adapted)
# =============================================================================

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
    spans = (valid & in_box &
             (R_per_tet.min(axis=1) <= gap_inner) &
             (R_per_tet.max(axis=1) >= gap_outer))

    if spans.sum() < 4:
        spans = valid & in_box
        print(f"  Warning: only {spans.sum()} spanning tets, using all valid")

    top_idx = np.argsort(-radii[spans])[:top_n]
    void_indices = np.where(spans)[0][top_idx]

    return void_indices, centers, radii, simplices, gap_inner, gap_outer


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def resolve_run_dir(run_arg):
    """Resolve run directory from argument or latest.txt."""
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


def load_cloud_at_step(clouds_file, step):
    """Load cloud data for a specific step."""
    npz = np.load(clouds_file)
    prefix = f"s{step:04d}"
    values = npz[f"{prefix}_values"]
    active = int(npz[f"{prefix}_active"][0])
    t_mag = npz.get(f"{prefix}_tvec_mag", None)
    t_unit = npz.get(f"{prefix}_tvec_unit", None)
    return values, active, t_mag, t_unit


def get_cloud_steps(clouds_file):
    """Extract sorted step numbers from clouds.npz."""
    npz = np.load(clouds_file)
    steps = set()
    for key in npz.files:
        if key.startswith('s') and '_values' in key:
            try:
                steps.add(int(key.split('_')[0][1:]))
            except ValueError:
                pass
    return sorted(steps)


def run_twist_analysis(run_dir, step=None, top_n=6, mapping='quadrant'):
    """
    Full twist curvature analysis:
    1. Generate orbit and compute twist rates
    2. Map engine nodes to orbit positions (via selected mapping method)
    3. Compute twist curvature at each node
    4. Detect voids
    5. Correlate void positions with twist extrema

    mapping: 'phi' (legacy), 'quadrant' (4-column grid), 'radial' (log-r)
    """
    run_dir = Path(run_dir)

    with open(run_dir / 'meta.json') as f:
        meta = json.load(f)

    registry = np.load(str(run_dir / 'registry.npy')).astype(np.int32)
    grid_size = meta.get('grid_size', 89)
    positions = to_sim_space(registry.astype(np.float64), grid_size)

    # Phase data
    phase_path = run_dir / 'phase.npy'
    phase = np.load(str(phase_path)) if phase_path.exists() else None

    # Clouds
    clouds_path = run_dir / 'clouds.npz'
    if not clouds_path.exists():
        print("No clouds.npz — run the engine first.")
        return

    all_steps = get_cloud_steps(str(clouds_path))

    print(f"\n{'='*72}")
    print(f"MÖBIUS TWIST CURVATURE — Run {meta.get('run_id', '?'):04d}")
    print(f"Grid: {grid_size}³ | Nodes: {len(registry)} | Steps: {len(all_steps)}")
    print(f"Mapping: {mapping}")
    print(f"{'='*72}")

    # --- Step 1: Generate orbit ---
    seq, diffs = generate_orbit()
    orbit_diagnostics(seq, diffs)

    # --- Step 2: Map nodes to spherical coordinates ---
    r, theta, phi = nodes_to_polar(positions)

    print(f"\n{'='*72}")
    print(f"NODE → ORBIT MAPPING (method: {mapping})")
    print(f"{'='*72}")

    print(f"\n  Spherical coordinate ranges:")
    print(f"    r:     [{r.min():.3f}, {r.max():.3f}]")
    print(f"    θ:     [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]")
    print(f"    φ:     [{np.degrees(phi.min()):.1f}°, {np.degrees(phi.max()):.1f}°]")

    # --- Step 3: Map to orbit positions ---
    orbit_idx, col_assignments = map_angles_to_orbit(r, theta, phi, seq,
                                                      method=mapping)

    print(f"\n  Orbit index range: [{orbit_idx.min()}, {orbit_idx.max()}]")
    print(f"  Unique orbit positions mapped: {len(np.unique(orbit_idx))}")
    print(f"  Orbit coverage: {len(np.unique(orbit_idx)) / N_ORB * 100:.1f}%")

    # Quadrant distribution (if available)
    if col_assignments is not None:
        print(f"\n  Quadrant distribution (4-column grid mapping):")
        col_names = ['Col 0 (upper-front)', 'Col 1 (upper-back)',
                     'Col 2 (lower-front)', 'Col 3 (lower-back)']
        for c in range(4):
            n_c = int((col_assignments == c).sum())
            print(f"    {col_names[c]}: {n_c} nodes ({n_c/len(r)*100:.1f}%)")

        # Check hemisphere imbalance — compare to orbit's 11×Ω prediction
        upper = int((col_assignments <= 1).sum())
        lower = int((col_assignments >= 2).sum())
        imbalance = upper - lower
        print(f"\n  Hemisphere imbalance: {upper} upper − {lower} lower = {imbalance}")
        print(f"  Imbalance / total: {imbalance/len(r)*100:+.2f}%")
        if imbalance != 0:
            print(f"  |imbalance| mod 11 = {abs(imbalance) % 11}")

    # --- Step 4: Compute twist curvature ---
    twist_rate, twist_curv = compute_twist_curvature(orbit_idx, diffs)

    print(f"\n  Twist rate statistics:")
    print(f"    Mean:  {twist_rate.mean():+.6f}")
    print(f"    Std:   {twist_rate.std():.6f}")
    print(f"    Range: [{twist_rate.min():.6f}, {twist_rate.max():.6f}]")

    print(f"\n  Twist curvature statistics:")
    print(f"    Mean:  {twist_curv.mean():+.6f}")
    print(f"    Std:   {twist_curv.std():.6f}")
    print(f"    Range: [{twist_curv.min():.6f}, {twist_curv.max():.6f}]")

    # Nodes at extremal twist
    top_twist = np.argsort(np.abs(twist_curv))[-10:][::-1]
    print(f"\n  Top 10 nodes by |twist curvature|:")
    print(f"  {'Node':>6} | {'r':>6} | {'θ°':>6} | {'φ°':>7} | "
          f"{'Rate':>8} | {'Curv':>8} | {'OrbIdx':>6}")
    print(f"  {'-'*65}")
    for ni in top_twist:
        print(f"  {ni:>6} | {r[ni]:>6.2f} | {np.degrees(theta[ni]):>6.1f} | "
              f"{np.degrees(phi[ni]):>+7.1f} | {twist_rate[ni]:>+8.5f} | "
              f"{twist_curv[ni]:>+8.5f} | {orbit_idx[ni]:>6}")

    # --- Step 5: Detect voids ---
    print(f"\n{'='*72}")
    print(f"VOID DETECTION")
    print(f"{'='*72}")

    void_idx, centers, radii, simplices, gap_inner, gap_outer = find_voids(
        positions, top_n=top_n)

    print(f"  Gap: R = {gap_inner:.3f} → {gap_outer:.3f}")
    print(f"  Top {len(void_idx)} voids detected")

    # Void properties + twist at void surrounding nodes
    print(f"\n  {'Void':>4} | {'Center':>24} | {'CRad':>6} | "
          f"{'Mean Twist':>10} | {'Mean |Curv|':>11} | {'Phase':>5}")
    print(f"  {'-'*80}")

    void_twist_vals = []
    void_curv_vals = []

    for vi, void_i in enumerate(void_idx):
        tet_nodes = simplices[void_i]
        center = centers[void_i]
        crad = radii[void_i]

        # Twist at surrounding nodes
        tet_twist = twist_rate[tet_nodes]
        tet_curv = twist_curv[tet_nodes]
        mean_twist = float(tet_twist.mean())
        mean_abs_curv = float(np.abs(tet_curv).mean())

        void_twist_vals.append(mean_twist)
        void_curv_vals.append(mean_abs_curv)

        # Phase info
        if phase is not None:
            tet_phase = phase[tet_nodes]
            n_pos = int((tet_phase > 0).sum())
            phase_str = f"{n_pos}/4"
        else:
            phase_str = "  —"

        center_str = f"({center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f})"
        print(f"  {vi+1:>4} | {center_str:>24} | {crad:>6.3f} | "
              f"{mean_twist:>+10.5f} | {mean_abs_curv:>11.5f} | {phase_str:>5}")

    # --- Step 6: Correlation ---
    print(f"\n{'='*72}")
    print(f"TWIST-VOID CORRELATION")
    print(f"{'='*72}")

    # Compare void twist curvature to background
    all_abs_curv = np.abs(twist_curv)
    bg_mean = float(all_abs_curv.mean())
    bg_std = float(all_abs_curv.std())
    void_mean_curv = float(np.mean(void_curv_vals))

    print(f"\n  Background |twist curvature|:")
    print(f"    Mean: {bg_mean:.6f}")
    print(f"    Std:  {bg_std:.6f}")
    print(f"\n  Void region |twist curvature|:")
    print(f"    Mean: {void_mean_curv:.6f}")

    if bg_std > 0:
        z_score = (void_mean_curv - bg_mean) / bg_std
        print(f"    Z-score vs background: {z_score:+.3f}")

        if abs(z_score) > 2:
            direction = "HIGHER" if z_score > 0 else "LOWER"
            print(f"    → Voids show significantly {direction} twist curvature")
            print(f"    → The algebraic structure CORRELATES with spatial voids")
        elif abs(z_score) > 1:
            direction = "higher" if z_score > 0 else "lower"
            print(f"    → Voids show marginally {direction} twist curvature")
            print(f"    → Suggestive but not definitive")
        else:
            print(f"    → No significant deviation from background")
            print(f"    → Twist structure may operate at a different scale")

    # --- Step 7: Time evolution at voids (if step specified) ---
    if step is not None:
        analyze_step = step
    elif len(all_steps) > 0:
        # Use a few representative steps
        analyze_step = None

    if analyze_step is not None and analyze_step in all_steps:
        print(f"\n{'='*72}")
        print(f"TWIST × FIELD at step {analyze_step}")
        print(f"{'='*72}")

        values, active, t_mag, t_unit = load_cloud_at_step(
            str(clouds_path), analyze_step)

        print(f"\n  {'Void':>4} | {'View':>8} | {'Twist':>8} | "
              f"{'|Curv|':>8} | {'T×V':>8} | Notes")
        print(f"  {'-'*58}")

        for vi, void_i in enumerate(void_idx):
            tet_nodes = simplices[void_i]
            tet_values = values[tet_nodes]
            view = float(tet_values.mean())
            t = void_twist_vals[vi]
            c = void_curv_vals[vi]
            product = view * t

            notes = []
            if abs(product) < 0.001:
                notes.append("ORTHOGONAL")
            elif product > 0:
                notes.append("ALIGNED")
            else:
                notes.append("OPPOSED")

            notes_str = ' '.join(notes)
            print(f"  {vi+1:>4} | {view:>8.4f} | {t:>+8.5f} | "
                  f"{c:>8.5f} | {product:>+8.5f} | {notes_str}")
    else:
        # Multi-step overview
        if len(all_steps) >= 3:
            sample_steps = [all_steps[0],
                            all_steps[len(all_steps)//2],
                            all_steps[-1]]

            print(f"\n{'='*72}")
            print(f"TWIST × FIELD — sample steps")
            print(f"{'='*72}")

            for s in sample_steps:
                values, active, t_mag, t_unit = load_cloud_at_step(
                    str(clouds_path), s)

                # Compute mean product across all voids
                products = []
                for vi, void_i in enumerate(void_idx):
                    tet_nodes = simplices[void_i]
                    view = float(values[tet_nodes].mean())
                    products.append(view * void_twist_vals[vi])

                mean_product = np.mean(products)
                spread = np.std(products)

                alignment = "ORTHOGONAL" if abs(mean_product) < 0.001 else \
                           "ALIGNED" if mean_product > 0 else "OPPOSED"

                print(f"  Step {s:>4} | active: {active:>4} | "
                      f"⟨view×twist⟩ = {mean_product:>+.5f} ± {spread:.5f} | "
                      f"{alignment}")

    # --- Ratio analysis ---
    print(f"\n{'='*72}")
    print(f"TWIST RATIO ANALYSIS")
    print(f"{'='*72}")

    if len(void_twist_vals) >= 2:
        print(f"\n  Pairwise twist rate ratios:")
        from fractions import Fraction
        for i in range(len(void_twist_vals)):
            for j in range(i + 1, len(void_twist_vals)):
                if abs(void_twist_vals[j]) > 1e-8:
                    ratio = void_twist_vals[i] / void_twist_vals[j]
                    frac = Fraction(ratio).limit_denominator(50)
                    print(f"    V{i+1}/V{j+1} = {ratio:>+10.6f} ≈ {frac}")

    print(f"\n{'='*72}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Möbius Twist Curvature Analyzer')
    parser.add_argument('--run', type=str, default=None,
                        help='Run directory (default: latest)')
    parser.add_argument('--step', type=int, default=None,
                        help='Analyze specific step')
    parser.add_argument('--top', type=int, default=6,
                        help='Number of top voids to track')
    parser.add_argument('--mapping', choices=['phi', 'quadrant', 'radial'],
                        default='quadrant',
                        help='Orbit mapping method (default: quadrant)')
    parser.add_argument('--orbit-only', action='store_true',
                        help='Print orbit diagnostics only (no run needed)')

    args = parser.parse_args()

    if args.orbit_only:
        seq, diffs = generate_orbit()
        orbit_diagnostics(seq, diffs)
    else:
        run_dir = resolve_run_dir(args.run)
        run_twist_analysis(run_dir, step=args.step, top_n=args.top,
                           mapping=args.mapping)
