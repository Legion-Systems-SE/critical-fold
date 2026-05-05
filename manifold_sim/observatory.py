"""
Radio Observatory — Windowed Spectral Analysis & Beamforming
=============================================================
Compound tool unifying the spectral observation chain:

  1. WINDOWED FFT — Hann, Hamming, Blackman, Kaiser (β adjustable)
     Applied to rotational moiré spectra and time-series data.

  2. BEAMFORMING — Conventional (delay-and-sum) and MVDR/Capon
     minimum-variance beamformer over a Uniform Circular Array
     (UCA) of observation axes in the plane perpendicular to
     the look direction.

  3. TELESCOPES — Fixed structural axes (X, Y, Z, body diagonal,
     zeta axis, lattice axis) plus user-defined movable telescopes
     at arbitrary (θ, φ) pointing angles.

  4. SPECTRAL MAP — Full angle-vs-frequency heatmap showing where
     structural power concentrates as the observation axis rotates.

Modes:
  spectrum   — windowed FFT of a single axis, compare windows
  array      — UCA beamforming with steerable null/look direction
  sweep      — angle-frequency spectral map (the full sky survey)
  compare    — side-by-side window comparison on same data

Usage:
    python3 observatory.py --run NNNN                       # spectrum, latest axis
    python3 observatory.py --run NNNN --mode sweep          # full sky survey
    python3 observatory.py --run NNNN --mode array --look 34.3,-135  # beam at zeta axis
    python3 observatory.py --run NNNN --window kaiser --beta 8.6     # Kaiser god mode
    python3 observatory.py --run NNNN --mode compare        # all windows side by side
    python3 observatory.py --run NNNN --telescope 45,90     # add movable telescope
    python3 observatory.py --run NNNN --mode sweep --save sky.png

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import numpy as np
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# ── Structural axes (fixed dishes) ───────────────────────────────────

FIXED_AXES = {
    'X':       np.array([1.0, 0.0, 0.0]),
    'Y':       np.array([0.0, 1.0, 0.0]),
    'Z':       np.array([0.0, 0.0, 1.0]),
    'diag':    np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    'zeta':    np.array([-0.3985, -0.3984, 0.8261]),
    'lattice': np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
}
FIXED_AXES['zeta'] /= np.linalg.norm(FIXED_AXES['zeta'])

STRUCTURAL_FREQS = [1, 3, 5, 7, 9, 15, 27, 45]
N_ANGLES = 720

STICK = [2, 3, 4, 5, 6, 7, 8, 9]
STICK_PRIMES = {2, 3, 5, 7}
BRIDGE = 5.0 / np.pi


# ── Window functions ─────────────────────────────────────────────────

def get_window(name, N, beta=8.6):
    """Return a window function of length N."""
    if name == 'rectangular' or name == 'rect':
        return np.ones(N)
    elif name == 'hann':
        return np.hanning(N)
    elif name == 'hamming':
        return np.hamming(N)
    elif name == 'blackman':
        return np.blackman(N)
    elif name == 'kaiser':
        return np.kaiser(N, beta)
    elif name == 'flattop':
        # Flat-top window for accurate amplitude measurement
        n = np.arange(N)
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        w = np.zeros(N)
        for k, ak in enumerate(a):
            w += ak * np.cos(2 * np.pi * k * n / N) * ((-1) ** k)
        return w
    else:
        raise ValueError(f"Unknown window: {name}")


def window_properties(name, N, beta=8.6):
    """Compute key window properties: main lobe width, sidelobe level, ENBW."""
    w = get_window(name, N, beta)
    W = np.fft.rfft(w, n=4*N)
    mag = np.abs(W)
    mag_db = 20 * np.log10(mag / mag.max() + 1e-30)

    # Main lobe width (3dB)
    above_3db = np.where(mag_db >= -3.0)[0]
    lobe_3db = 2 * len(above_3db) / (4 * N) * N if len(above_3db) > 0 else 0

    # Peak sidelobe level
    # Find first null after main lobe
    below_null = np.where(mag_db[1:] < -60)[0]
    if len(below_null) > 0:
        first_null = below_null[0] + 1
        if first_null < len(mag_db):
            sidelobe_db = mag_db[first_null:].max()
        else:
            sidelobe_db = -100.0
    else:
        sidelobe_db = mag_db[len(above_3db):].max() if len(above_3db) < len(mag_db) else -100.0

    # Equivalent noise bandwidth
    enbw = N * np.sum(w**2) / (np.sum(w))**2

    return {
        'main_lobe_bins': lobe_3db,
        'sidelobe_db': sidelobe_db,
        'enbw': enbw,
        'coherent_gain': np.sum(w) / N,
    }


# ── Geometry ─────────────────────────────────────────────────────────

def spherical_to_cart(theta_deg, phi_deg):
    """Convert (θ, φ) in degrees to unit vector."""
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def orthonormal_frame(axis):
    a = axis / np.linalg.norm(axis)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def uca_steering_vectors(n_elements, axis, freqs, look_dir):
    """
    Compute steering vectors for a Uniform Circular Array.
    The UCA sits in the plane perpendicular to `axis`,
    with `n_elements` equally spaced sensors.
    """
    e1, e2 = orthonormal_frame(axis)
    element_angles = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)

    # Element positions on the unit circle in the e1-e2 plane
    positions = np.array([
        np.cos(a) * e1 + np.sin(a) * e2 for a in element_angles
    ])

    look_unit = look_dir / np.linalg.norm(look_dir)
    delays = positions @ look_unit

    # Steering vectors: one per frequency
    steering = {}
    for f in freqs:
        if f == 0:
            continue
        phase = np.exp(-1j * 2 * np.pi * f * delays)
        steering[f] = phase / np.sqrt(n_elements)

    return steering, positions


# ── Signal extraction ────────────────────────────────────────────────

def load_positions(run_dir):
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.float64)
    meta = json.load(open(run_dir / 'meta.json'))
    gs = meta.get('grid_size', 89)
    pos = (registry - gs / 2.0) * (20.0 / gs)
    return pos, meta


def rotational_signal(positions, axis, n_angles=N_ANGLES, n_bins=200):
    """Compute variance-of-histogram signal as the view rotates around axis."""
    e1, e2 = orthonormal_frame(axis)
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    signal = np.zeros(n_angles)
    for i, theta in enumerate(thetas):
        view = np.cos(theta) * e1 + np.sin(theta) * e2
        proj = positions @ view
        counts, _ = np.histogram(proj, bins=n_bins)
        signal[i] = np.var(counts)
    return thetas, signal


def windowed_fft(signal, window_name='hann', beta=8.6):
    """Apply window and compute FFT. Returns frequencies, complex spectrum, power."""
    N = len(signal)
    w = get_window(window_name, N, beta)
    windowed = signal * w

    # Normalize for coherent gain
    cg = w.sum() / N
    fft_result = np.fft.rfft(windowed) / (cg * N / 2)
    freqs = np.fft.rfftfreq(N, d=1.0/N)
    power = np.abs(fft_result) ** 2

    return freqs, fft_result, power


# ── Wannier decomposition ───────────────────────────────────────────

def wannier_decompose(signal, structural_freqs, bandwidth=1):
    """
    Decompose periodic signal into Wannier-localized structural components.
    Each structural frequency is isolated via bandpass, inverse-FFT'd back
    to real space. The residual is everything outside the structural bands.
    """
    N = len(signal)
    S = np.fft.rfft(signal)

    components = {}
    mask_all = np.zeros(len(S), dtype=bool)

    for f in structural_freqs:
        idx = int(round(f))
        if idx >= len(S):
            continue
        mask = np.zeros(len(S), dtype=bool)
        for di in range(-bandwidth, bandwidth + 1):
            ii = idx + di
            if 0 <= ii < len(S):
                mask[ii] = True
                mask_all[ii] = True
        S_band = np.zeros_like(S)
        S_band[mask] = S[mask]
        components[f] = {
            'signal': np.fft.irfft(S_band, n=N),
            'amplitude': float(np.abs(S[idx])),
            'phase': float(np.angle(S[idx])),
            'power': float(np.abs(S[idx]) ** 2),
        }

    S_res = S.copy()
    S_res[mask_all] = 0
    residual = np.fft.irfft(S_res, n=N)

    total = float(np.sum(np.abs(S[1:]) ** 2))
    res_e = float(np.sum(np.abs(S_res[1:]) ** 2))
    struct_e = total - res_e

    return components, residual, {
        'total': total,
        'structural': struct_e,
        'residual': res_e,
        'ratio': struct_e / total if total > 0 else 0,
    }


def wannier_3d(run_dir, meta, structural_freqs):
    """
    3D Wannier: scatter field onto full grid, 3D FFT, isolate |k| shells
    at each structural frequency, inverse FFT back → per-node localization.
    For two-body runs, compares frequency content between bodies.
    """
    gs = meta.get('grid_size', 89)
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.int32)
    n = len(registry)

    clouds = run_dir / 'clouds.npz'
    if not clouds.exists():
        return None
    npz = np.load(str(clouds))
    steps = sorted(int(k.split('_')[0][1:]) for k in npz.files if '_values' in k)
    if not steps:
        return None
    step = steps[-1]
    values = npz[f's{step:04d}_values']

    grid = np.zeros((gs, gs, gs))
    ix, iy, iz = registry[:, 0], registry[:, 1], registry[:, 2]
    v = values[:n] if len(values) >= n else np.pad(values, (0, n - len(values)))
    grid[ix, iy, iz] = v

    G = np.fft.fftn(grid)
    power_3d = np.abs(G) ** 2

    kx = np.fft.fftfreq(gs) * gs
    ky = np.fft.fftfreq(gs) * gs
    kz = np.fft.fftfreq(gs) * gs
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_max = gs // 2
    radial_power = np.zeros(k_max + 1)
    for ki in range(k_max + 1):
        shell = (K_mag >= ki - 0.5) & (K_mag < ki + 0.5)
        radial_power[ki] = power_3d[shell].sum()

    node_amps = np.zeros((n, len(structural_freqs)))
    for fi, f in enumerate(structural_freqs):
        if f > k_max:
            continue
        shell = (K_mag >= f - 0.5) & (K_mag < f + 0.5)
        G_band = np.zeros_like(G)
        G_band[shell] = G[shell]
        band = np.fft.ifftn(G_band).real
        node_amps[:, fi] = np.abs(band[ix, iy, iz])

    structural_total = node_amps.sum(axis=1)

    # Residual: subtract all structural shells
    mask_struct = np.zeros_like(K_mag, dtype=bool)
    for f in structural_freqs:
        if f <= k_max:
            mask_struct |= (K_mag >= f - 0.5) & (K_mag < f + 0.5)
    G_res = G.copy()
    G_res[mask_struct] = 0
    residual_field = np.fft.ifftn(G_res).real
    node_residual = np.abs(residual_field[ix, iy, iz])

    body_stats = None
    fid_path = run_dir / 'field_ids.npy'
    if fid_path.exists():
        fids = np.load(str(fid_path))[:n]
        body_a = fids == 0
        body_b = fids == 1
        body_stats = {}
        for fi, f in enumerate(structural_freqs):
            a_mean = float(node_amps[body_a, fi].mean()) if body_a.any() else 0
            b_mean = float(node_amps[body_b, fi].mean()) if body_b.any() else 0
            body_stats[f] = {'A': a_mean, 'B': b_mean,
                             'ratio': a_mean / (b_mean + 1e-30)}

    return {
        'node_amps': node_amps,
        'node_residual': node_residual,
        'structural_total': structural_total,
        'radial_power': radial_power,
        'step': step,
        'body_stats': body_stats,
        'n_body_A': int(body_a.sum()) if body_stats else 0,
        'n_body_B': int(body_b.sum()) if body_stats else 0,
    }


def hodge_dual_test(axis_results, structural_freqs):
    """
    Test: is the residual the Hodge dual of the structural content?

    If the 8 structural frequencies are grade-1 elements of Cl(8,0),
    then their products (higher grades) should generate the residual:
      - Grade 1 × pseudoscalar: freq × bus_width(8) → Hodge dual
      - Grade 2: pairwise sums/differences → bivector content
      - Grade 2 × pseudoscalar: (f_i + f_j) × 8 → dressed bivectors
    """
    bus_width = 8

    print(f"\n  [4] HODGE DUAL TEST — is the residual the other side of the sheet?")
    print(f"      Hypothesis: residual = higher grades of Cl(8,0)")
    print(f"      Bus width (pseudoscalar period): {bus_width}")

    freqs_list = sorted(structural_freqs)

    # Build predicted frequency table by grade
    predictions = {}

    # Grade 1 × pseudoscalar (Hodge dual of each generator)
    for f in freqs_list:
        fp = f * bus_width
        key = fp
        if key not in predictions:
            predictions[key] = []
        predictions[key].append(f"g1×I: {f}×{bus_width}")

    # Grade 2: pairwise sums and differences
    for i, f1 in enumerate(freqs_list):
        for f2 in freqs_list[i+1:]:
            fs = f1 + f2
            fd = abs(f1 - f2)
            if fs not in predictions:
                predictions[fs] = []
            predictions[fs].append(f"g2+: {f1}+{f2}")
            if fd > 0:
                if fd not in predictions:
                    predictions[fd] = []
                predictions[fd].append(f"g2-: |{f1}-{f2}|")

    # Grade 2 × pseudoscalar
    for i, f1 in enumerate(freqs_list):
        for f2 in freqs_list[i+1:]:
            fp = (f1 + f2) * bus_width
            if fp not in predictions:
                predictions[fp] = []
            predictions[fp].append(f"g2×I: ({f1}+{f2})×{bus_width}")

    # Test against each axis
    for name in ['X', 'Y', 'Z', 'zeta']:
        if name not in axis_results:
            continue
        comp, residual, energy = axis_results[name]

        N = len(residual)
        S_res = np.fft.rfft(residual)
        res_power = np.abs(S_res) ** 2
        res_freqs = np.fft.rfftfreq(N, d=1.0 / N)

        struct_set = set(structural_freqs)
        residual_peaks = [(int(round(res_freqs[i])), res_power[i])
                          for i in range(1, len(res_power))
                          if int(round(res_freqs[i])) not in struct_set
                          and res_power[i] > 0]
        residual_peaks.sort(key=lambda x: -x[1])
        top_peaks = residual_peaks[:12]

        explained_count = 0
        explained_power = 0.0
        total_res_power = sum(p for _, p in top_peaks)

        print(f"\n  {name} axis — residual peaks vs Cl(8,0) predictions:")
        print(f"    {'freq':>4s}  {'power':>10s}  {'grade':>6s}  {'source':>35s}")
        print(f"    {'─'*60}")

        for f_peak, power in top_peaks:
            matches = []
            for fp, sources in predictions.items():
                if abs(f_peak - fp) <= 1:
                    matches.extend(sources)

            if matches:
                explained_count += 1
                explained_power += power
                grade_str = matches[0].split(':')[0]
                source_str = ', '.join(m.split(': ')[1] for m in matches[:3])
                if len(matches) > 3:
                    source_str += f' (+{len(matches)-3})'
            else:
                grade_str = '?'
                source_str = 'unexplained'

            print(f"    {f_peak:4d}  {power:>10.4e}  {grade_str:>6s}  {source_str}")

        pct = explained_power / total_res_power * 100 if total_res_power > 0 else 0
        print(f"    ───")
        print(f"    Peaks explained: {explained_count}/{len(top_peaks)}")
        print(f"    Power explained: {pct:.1f}%")

        # Signal product verification: multiply Wannier pairs, compare with residual
        product_sum = np.zeros(N)
        n_pairs = 0
        for i, f1 in enumerate(freqs_list):
            for f2 in freqs_list[i+1:]:
                if f1 in comp and f2 in comp:
                    product_sum += comp[f1]['signal'] * comp[f2]['signal']
                    n_pairs += 1

        if n_pairs > 0:
            S_prod = np.fft.rfft(product_sum)
            prod_power = np.abs(S_prod) ** 2

            non_struct = [i for i in range(1, min(len(res_power), len(prod_power)))
                          if int(round(res_freqs[i])) not in struct_set]
            if non_struct:
                r_vals = np.array([res_power[i] for i in non_struct])
                p_vals = np.array([prod_power[i] for i in non_struct])
                if r_vals.std() > 0 and p_vals.std() > 0:
                    corr = float(np.corrcoef(r_vals, p_vals)[0, 1])
                else:
                    corr = 0.0
                ratio = float(p_vals.sum() / (r_vals.sum() + 1e-30))
                print(f"    Grade-2 product correlation with residual: {corr:+.4f}")
                print(f"    Power ratio (products / residual): {ratio:.6f}")


# ── Cl(8,0) grade structure ─────────────────────────────────────────

def cl8_grade_table(structural_freqs):
    """
    Build full Cl(8,0) product frequency table from 8 generators.
    Each grade-k basis element is a k-fold product of generators.
    In signal domain: products → sum/difference frequencies.
    """
    from itertools import combinations
    grades = {0: {0: ['scalar']}}
    for k in range(1, len(structural_freqs) + 1):
        freq_map = {}
        for combo in combinations(range(len(structural_freqs)), k):
            fs = [structural_freqs[i] for i in combo]
            label = '·'.join(str(f) for f in fs)
            for signs in range(1 << (k - 1)):
                val = fs[0]
                for j in range(1, k):
                    val += fs[j] if not (signs & (1 << (j - 1))) else -fs[j]
                f_abs = abs(val)
                if f_abs not in freq_map:
                    freq_map[f_abs] = []
                if label not in freq_map[f_abs]:
                    freq_map[f_abs].append(label)
        grades[k] = freq_map
    return grades


# ── MVDR / Capon beamformer ──────────────────────────────────────────

def mvdr_beamform(data_matrix, steering_vector):
    """
    MVDR (Capon) beamformer.
    data_matrix: (n_elements, n_snapshots) complex
    steering_vector: (n_elements,) complex
    Returns: beamformer output power (scalar)
    """
    n_elements = data_matrix.shape[0]
    R = (data_matrix @ data_matrix.conj().T) / data_matrix.shape[1]

    # Diagonal loading for numerical stability
    R += np.eye(n_elements) * np.trace(R).real * 1e-6

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    a = steering_vector.reshape(-1, 1)
    numerator = a.conj().T @ R_inv @ a
    w_mvdr = R_inv @ a / numerator
    out = w_mvdr.conj().T @ R @ w_mvdr
    output_power = float(np.abs(out).flatten()[0])
    return output_power, w_mvdr.flatten()


def conventional_beamform(data_matrix, steering_vector):
    """Delay-and-sum beamformer. Returns output power."""
    a = steering_vector.reshape(-1, 1)
    w = a / len(steering_vector)
    output = w.conj().T @ data_matrix
    power = float(np.mean(np.abs(output)**2))
    return power


# ── Observatory modes ────────────────────────────────────────────────

def mode_spectrum(positions, meta, args):
    """Single-axis windowed spectrum analysis."""
    axis_raw = args.axis if args.axis else 'Y'
    axis_name = axis_raw
    axis_lookup = axis_raw.upper() if axis_raw.upper() in {k.upper() for k in FIXED_AXES} else axis_raw
    matched = {k: v for k, v in FIXED_AXES.items() if k.upper() == axis_lookup.upper()}
    if matched:
        axis_name = list(matched.keys())[0]
        axis = list(matched.values())[0]
    else:
        parts = axis_raw.split(',')
        if len(parts) == 2:
            axis = spherical_to_cart(float(parts[0]), float(parts[1]))
            axis_name = f"θ={parts[0]}°,φ={parts[1]}°"
        else:
            axis = FIXED_AXES['Y']
            axis_name = 'Y'

    print(f"\n  Axis: {axis_name} → [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print(f"  Window: {args.window}" + (f" (β={args.beta})" if args.window == 'kaiser' else ''))

    # Window properties
    wp = window_properties(args.window, N_ANGLES, args.beta)
    print(f"  Main lobe: {wp['main_lobe_bins']:.1f} bins | "
          f"Sidelobes: {wp['sidelobe_db']:.1f} dB | "
          f"ENBW: {wp['enbw']:.3f}")

    # Compute signal
    print(f"\n  Scanning {N_ANGLES} angles...", end='', flush=True)
    thetas, signal = rotational_signal(positions, axis)
    print(" done")

    # Windowed FFT
    freqs, spectrum, power = windowed_fft(signal, args.window, args.beta)

    # Extract structural frequencies
    print(f"\n  {'freq':>4s}  {'power':>12s}  {'amplitude':>10s}  "
          f"{'phase/π':>8s}  {'SNR_dB':>8s}  {'level':>30s}")
    print(f"  {'─'*80}")

    noise_floor = np.median(power[power > 0]) if np.any(power > 0) else 1e-30

    for f in STRUCTURAL_FREQS:
        idx = int(round(f))
        if idx < len(power):
            p = power[idx]
            a = np.abs(spectrum[idx])
            ph = np.angle(spectrum[idx]) / np.pi
            snr = 10 * np.log10(p / noise_floor + 1e-30)
            max_p = power[1:].max() if len(power) > 1 else 1
            bar_len = int(a / (np.sqrt(max_p) + 1e-30) * 30)
            bar = "█" * min(bar_len, 30)
            print(f"  {f:4d}  {p:12.4e}  {a:10.4f}  "
                  f"{ph:+8.4f}  {snr:+8.1f}  {bar}")

    # Non-structural peaks
    print(f"\n  Top 5 non-structural peaks:")
    struct_set = set(STRUCTURAL_FREQS)
    non_struct = [(i, power[i]) for i in range(1, len(power))
                  if int(round(freqs[i])) not in struct_set and power[i] > noise_floor]
    non_struct.sort(key=lambda x: -x[1])
    for i, (idx, p) in enumerate(non_struct[:5]):
        print(f"    f={freqs[idx]:>6.1f}  power={p:.4e}")

    return freqs, spectrum, power


def mode_compare(positions, meta, args):
    """Compare all window functions on the same axis."""
    axis_raw = args.axis if args.axis else 'Y'
    matched = {k: v for k, v in FIXED_AXES.items() if k.upper() == axis_raw.upper()}
    if matched:
        axis_name = list(matched.keys())[0]
        axis = list(matched.values())[0]
    else:
        axis_name = axis_raw
        axis = FIXED_AXES['Y']

    print(f"\n  Axis: {axis_name}")
    print(f"  Scanning...", end='', flush=True)
    thetas, signal = rotational_signal(positions, axis)
    print(" done")

    windows = ['rectangular', 'hann', 'hamming', 'blackman', 'kaiser']
    betas = [4.0, 8.6, 14.0]

    print(f"\n  {'Window':>14s}  {'ENBW':>6s}  {'Sidelobe':>9s}  ", end='')
    for f in STRUCTURAL_FREQS:
        print(f"  f={f:<3d}", end='')
    print(f"  {'struct/total':>12s}")
    print(f"  {'─'*120}")

    for win in windows:
        if win == 'kaiser':
            for beta in betas:
                freqs, spectrum, power = windowed_fft(signal, win, beta)
                wp = window_properties(win, N_ANGLES, beta)
                struct_power = sum(power[int(round(f))] for f in STRUCTURAL_FREQS
                                   if int(round(f)) < len(power))
                total_power = power[1:].sum()
                ratio = struct_power / total_power if total_power > 0 else 0

                label = f"kaiser(β={beta})"
                print(f"  {label:>14s}  {wp['enbw']:>6.3f}  {wp['sidelobe_db']:>+8.1f}  ", end='')
                for f in STRUCTURAL_FREQS:
                    idx = int(round(f))
                    p = power[idx] if idx < len(power) else 0
                    print(f"  {p:>6.1e}", end='')
                print(f"  {ratio:>12.4f}")
        else:
            freqs, spectrum, power = windowed_fft(signal, win)
            wp = window_properties(win, N_ANGLES)
            struct_power = sum(power[int(round(f))] for f in STRUCTURAL_FREQS
                               if int(round(f)) < len(power))
            total_power = power[1:].sum()
            ratio = struct_power / total_power if total_power > 0 else 0

            print(f"  {win:>14s}  {wp['enbw']:>6.3f}  {wp['sidelobe_db']:>+8.1f}  ", end='')
            for f in STRUCTURAL_FREQS:
                idx = int(round(f))
                p = power[idx] if idx < len(power) else 0
                print(f"  {p:>6.1e}", end='')
            print(f"  {ratio:>12.4f}")


def mode_array(positions, meta, args):
    """UCA beamforming with steerable look direction."""
    # Parse look direction
    if args.look:
        parts = args.look.split(',')
        look_theta, look_phi = float(parts[0]), float(parts[1])
    else:
        look_theta, look_phi = 34.3, -135.0  # default: zeta axis
    look_dir = spherical_to_cart(look_theta, look_phi)

    n_elements = args.array_size
    array_axis = FIXED_AXES.get('Z', np.array([0, 0, 1.0]))

    print(f"\n  Look direction: θ={look_theta}°, φ={look_phi}°")
    print(f"  Array: {n_elements}-element UCA in XY plane")
    print(f"  Window: {args.window}")

    # Get rotational signals from each element direction
    e1, e2 = orthonormal_frame(array_axis)
    element_angles = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)

    print(f"\n  Scanning {n_elements} elements...", flush=True)
    signals = []
    for i, ea in enumerate(element_angles):
        elem_axis = np.cos(ea) * e1 + np.sin(ea) * e2
        _, sig = rotational_signal(positions, elem_axis)
        signals.append(sig)
        if (i + 1) % 4 == 0:
            print(f"    {i+1}/{n_elements} complete", flush=True)

    # Windowed FFT of each element
    all_spectra = []
    for sig in signals:
        _, spectrum, _ = windowed_fft(sig, args.window, args.beta)
        all_spectra.append(spectrum)

    data_matrix = np.array(all_spectra)  # (n_elements, n_freqs)

    # Steering vectors
    steering, elem_pos = uca_steering_vectors(
        n_elements, array_axis, STRUCTURAL_FREQS, look_dir)

    # Beamform at each structural frequency
    print(f"\n  {'freq':>4s}  {'Conv_dB':>8s}  {'MVDR_dB':>8s}  {'Gain':>6s}  {'level':>30s}")
    print(f"  {'─'*70}")

    for f in STRUCTURAL_FREQS:
        idx = int(round(f))
        if idx >= data_matrix.shape[1]:
            continue

        sv = steering.get(f)
        if sv is None:
            continue

        freq_data = data_matrix[:, idx:idx+1]

        conv_power = conventional_beamform(freq_data, sv)
        mvdr_power, w_mvdr = mvdr_beamform(freq_data, sv)

        conv_db = 10 * np.log10(conv_power + 1e-30)
        mvdr_db = 10 * np.log10(mvdr_power + 1e-30)
        gain = mvdr_db - conv_db

        bar = "█" * max(0, int((mvdr_db + 50) / 2))
        print(f"  {f:4d}  {conv_db:>+8.1f}  {mvdr_db:>+8.1f}  {gain:>+6.1f}  {bar}")

    # Scan the beam across the sky at the dominant frequency
    print(f"\n  [BEAM PATTERN — f=1 (fundamental)]")
    scan_thetas = np.linspace(0, 180, 37)
    scan_phis = np.linspace(-180, 180, 73)

    f_scan = 1
    idx_scan = int(round(f_scan))
    if idx_scan < data_matrix.shape[1]:
        freq_data = data_matrix[:, idx_scan:idx_scan+1]
        peak_power = -np.inf
        peak_dir = (0, 0)

        print(f"  Scanning {len(scan_thetas)}×{len(scan_phis)} directions...", end='', flush=True)
        for theta_s in scan_thetas:
            for phi_s in scan_phis:
                scan_dir = spherical_to_cart(theta_s, phi_s)
                sv_scan = steering.get(f_scan)
                if sv_scan is None:
                    continue
                # Recompute steering for scan direction
                delays = elem_pos @ scan_dir
                sv_scan = np.exp(-1j * 2 * np.pi * f_scan * delays) / np.sqrt(n_elements)
                p = conventional_beamform(freq_data, sv_scan)
                p_db = 10 * np.log10(p + 1e-30)
                if p_db > peak_power:
                    peak_power = p_db
                    peak_dir = (theta_s, phi_s)

        print(f" done")
        print(f"  Peak beam: θ={peak_dir[0]:.1f}°, φ={peak_dir[1]:.1f}° ({peak_power:+.1f} dB)")


def mode_sweep(positions, meta, args):
    """Full angle-frequency spectral map."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    axis_name = args.axis.upper() if args.axis else 'Z'
    axis = FIXED_AXES.get(axis_name, FIXED_AXES['Z'])

    print(f"\n  Sweep axis: {axis_name}")
    print(f"  Window: {args.window}")

    # Sweep observation angle (elevation from the sweep axis)
    n_elevations = 90
    elevations = np.linspace(0, np.pi, n_elevations, endpoint=False)

    e1, e2 = orthonormal_frame(axis)
    all_power = []

    print(f"  Scanning {n_elevations} elevations...", flush=True)
    for i, elev in enumerate(elevations):
        obs_axis = np.cos(elev) * axis + np.sin(elev) * e1
        obs_axis /= np.linalg.norm(obs_axis)
        _, sig = rotational_signal(positions, obs_axis)
        _, _, power = windowed_fft(sig, args.window, args.beta)
        all_power.append(power)
        if (i + 1) % 15 == 0:
            print(f"    {i+1}/{n_elevations}", flush=True)

    power_map = np.array(all_power)  # (n_elevations, n_freqs)
    max_freq = min(60, power_map.shape[1])
    power_map = power_map[:, :max_freq]

    # Normalize per frequency for visibility
    for j in range(power_map.shape[1]):
        col = power_map[:, j]
        if col.max() > 0:
            power_map[:, j] = col / col.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0a0a0f',
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Spectral map
    ax1.set_facecolor('#0d0d1a')
    elev_deg = np.degrees(elevations)
    im = ax1.pcolormesh(np.arange(max_freq), elev_deg,
                        10 * np.log10(power_map + 1e-30),
                        cmap='inferno', shading='auto')
    for f in STRUCTURAL_FREQS:
        if f < max_freq:
            ax1.axvline(x=f, color='#00ff88', alpha=0.3, lw=0.5)
    ax1.set_xlabel('Frequency (cycles/revolution)', color='#888', fontsize=9)
    ax1.set_ylabel('Elevation from axis (°)', color='#888', fontsize=9)
    ax1.set_title(f'Spectral Map — {axis_name} axis, {args.window} window',
                  color='#ccc', fontsize=10)
    ax1.tick_params(colors='#666', labelsize=7)
    cb = fig.colorbar(im, ax=ax1, label='dB (normalized)')
    cb.ax.tick_params(labelsize=7)
    cb.set_label('dB (normalized)', color='#888', fontsize=8)

    # Structural frequency power vs elevation
    ax2.set_facecolor('#0d0d1a')
    colors = ['#00d4ff', '#ff6b35', '#00ff88', '#ff4444',
              '#ffaa00', '#7b2dff', '#ff69b4', '#00ffff']
    for fi, f in enumerate(STRUCTURAL_FREQS):
        if f < max_freq:
            ax2.plot(elev_deg, power_map[:, f],
                     color=colors[fi % len(colors)], lw=1.2,
                     label=f'f={f}', alpha=0.8)
    ax2.set_xlabel('Elevation (°)', color='#888', fontsize=9)
    ax2.set_ylabel('Normalized power', color='#888', fontsize=9)
    ax2.legend(fontsize=6, facecolor='#1a1a2e', labelcolor='#ccc', ncol=4)
    ax2.tick_params(colors='#666', labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a2a3a')

    plt.tight_layout()

    save_path = args.save or str(SCRIPT_DIR / 'audio_output' / 'sky_survey.png')
    fig.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()

    # Print structural summary
    print(f"\n  ELEVATION OF PEAK POWER PER STRUCTURAL FREQUENCY:")
    print(f"  {'freq':>4s}  {'peak_elev':>10s}  {'peak_power':>10s}")
    print(f"  {'─'*30}")
    for f in STRUCTURAL_FREQS:
        if f < max_freq:
            peak_idx = np.argmax(power_map[:, f])
            print(f"  {f:4d}  {elev_deg[peak_idx]:>10.1f}°  {power_map[peak_idx, f]:>10.4f}")


def mode_wannier(positions, meta, args):
    """Wannier decomposition: structural localization + residual extraction."""
    run_dir = resolve_run_dir(args.run)

    print(f"\n  ── WANNIER DECOMPOSITION ──")
    print(f"  Lattice-matched spectral isolation of structural frequencies")
    print(f"  What comes in with Green's function, comes out with Green's function")

    # ── 1. Rotational Wannier (per-axis) ──
    print(f"\n  [1] ROTATIONAL WANNIER — per-axis structural isolation")
    print(f"      Bandwidth: ±1 bin around each structural frequency")
    print(f"      Scanning {N_ANGLES} angles per axis...", flush=True)

    axis_results = {}
    for name, axis in FIXED_AXES.items():
        _, signal = rotational_signal(positions, axis)
        components, residual, energy = wannier_decompose(signal, STRUCTURAL_FREQS)
        axis_results[name] = (components, residual, energy)

    print(f"\n  {'Axis':>8s}  {'struct%':>8s}  {'resid%':>8s}  ", end='')
    for f in STRUCTURAL_FREQS:
        print(f" f={f:<3d}", end='')
    print()
    print(f"  {'─'*90}")

    for name in FIXED_AXES:
        comp, res, energy = axis_results[name]
        s_pct = energy['ratio'] * 100
        r_pct = (1 - energy['ratio']) * 100
        print(f"  {name:>8s}  {s_pct:>7.1f}%  {r_pct:>7.1f}%  ", end='')
        for f in STRUCTURAL_FREQS:
            if f in comp:
                print(f" {comp[f]['amplitude']:>5.3f}", end='')
            else:
                print(f" {'---':>5s}", end='')
        print()

    # Phase coherence
    print(f"\n  Phase structure (φ/π per frequency, all axes):")
    print(f"  {'freq':>4s}", end='')
    for name in FIXED_AXES:
        print(f"  {name:>8s}", end='')
    print(f"  {'locked?':>8s}")
    print(f"  {'─'*72}")

    for f in STRUCTURAL_FREQS:
        phases = []
        print(f"  {f:4d}", end='')
        for name in FIXED_AXES:
            comp = axis_results[name][0]
            if f in comp:
                ph = comp[f]['phase'] / np.pi
                phases.append(ph)
                print(f"  {ph:>+7.3f}π", end='')
            else:
                print(f"  {'---':>8s}", end='')
        if phases:
            locked = all(abs(abs(p) - 1.0) < 0.15 for p in phases)
            quad = all(abs(abs(p) - 0.5) < 0.15 for p in phases)
            tag = '±π' if locked else ('±π/2' if quad else 'free')
            print(f"  {tag:>8s}")
        else:
            print()

    # ── 2. Residual analysis ──
    print(f"\n  [2] RESIDUAL ANALYSIS — after structural subtraction")

    for name in ['X', 'Y', 'Z', 'zeta']:
        if name not in axis_results:
            continue
        comp, residual, energy = axis_results[name]
        freqs_r = np.fft.rfftfreq(len(residual), d=1.0 / len(residual))
        S_res = np.fft.rfft(residual)
        res_power = np.abs(S_res) ** 2

        struct_set = set(STRUCTURAL_FREQS)
        peaks = [(i, res_power[i]) for i in range(1, len(res_power))
                 if int(round(freqs_r[i])) not in struct_set and res_power[i] > 0]
        peaks.sort(key=lambda x: -x[1])

        r_pct = (1 - energy['ratio']) * 100
        rms = float(np.sqrt(np.mean(residual ** 2)))
        print(f"\n  {name}: residual = {r_pct:.2f}% of total energy | RMS = {rms:.6f}")
        if peaks[:5]:
            for idx, p in peaks[:5]:
                snr = 10 * np.log10(p / (np.median(res_power[1:]) + 1e-30) + 1e-30)
                print(f"    f={freqs_r[idx]:>6.1f}  power={p:.4e}  SNR={snr:+.1f} dB")
        else:
            print(f"    (no residual peaks)")

    # ── 3. 3D Wannier (per-node) ──
    print(f"\n  [3] 3D WANNIER — per-node frequency localization")

    result_3d = wannier_3d(run_dir, meta, STRUCTURAL_FREQS)

    if result_3d is not None:
        node_amps = result_3d['node_amps']
        node_res = result_3d['node_residual']
        print(f"  Field step: {result_3d['step']}")

        # Radial k-space power spectrum
        rp = result_3d['radial_power']
        print(f"\n  Radial k-space power (top 10 |k| shells):")
        top_k = np.argsort(-rp)[:10]
        for ki in sorted(top_k):
            marker = ' ◆' if ki in STRUCTURAL_FREQS else ''
            print(f"    |k|={ki:3d}  power={rp[ki]:.4e}{marker}")

        print(f"\n  Per-frequency node localization:")
        print(f"  {'freq':>4s}  {'mean_amp':>10s}  {'std':>10s}  "
              f"{'max':>10s}  {'IPR':>8s}  {'coverage':>10s}")
        print(f"  {'─'*60}")

        n_nodes = len(node_amps)
        for fi, f in enumerate(STRUCTURAL_FREQS):
            a = node_amps[:, fi]
            if a.max() == 0:
                continue
            mean_a = float(a.mean())
            std_a = float(a.std())
            max_a = float(a.max())
            a2 = a ** 2
            total_a2 = np.sum(a2)
            ipr = float(np.sum(a2 ** 2) / (total_a2 ** 2 + 1e-30)) * n_nodes
            above = float((a > mean_a).sum()) / n_nodes * 100
            print(f"  {f:4d}  {mean_a:>10.6f}  {std_a:>10.6f}  "
                  f"{max_a:>10.6f}  {ipr:>8.2f}  {above:>9.1f}%")

        # Structural vs residual per node
        struct_mean = float(result_3d['structural_total'].mean())
        res_mean = float(node_res.mean())
        total_mean = struct_mean + res_mean
        s_frac = struct_mean / (total_mean + 1e-30) * 100
        print(f"\n  Node-averaged: structural {s_frac:.1f}% | residual {100-s_frac:.1f}%")

        # Two-body comparison
        if result_3d['body_stats']:
            n_a = result_3d['n_body_A']
            n_b = result_3d['n_body_B']
            print(f"\n  Two-body frequency partition (A={n_a:,} nodes, B={n_b:,} nodes):")
            print(f"  {'freq':>4s}  {'body_A':>10s}  {'body_B':>10s}  "
                  f"{'A/B':>8s}  {'dominance':>12s}")
            print(f"  {'─'*50}")
            for f in STRUCTURAL_FREQS:
                bs = result_3d['body_stats'].get(f)
                if bs:
                    r = bs['ratio']
                    dom = 'A' if r > 1.1 else ('B' if r < 0.9 else 'equal')
                    print(f"  {f:4d}  {bs['A']:>10.6f}  {bs['B']:>10.6f}  "
                          f"{r:>8.3f}  {dom:>12s}")
    else:
        print(f"  (no cloud data — 3D decomposition requires clouds.npz)")

    # ── 4. Hodge dual test ──
    hodge_dual_test(axis_results, STRUCTURAL_FREQS)

    # ── Summary ──
    avg_struct = np.mean([axis_results[n][2]['ratio'] for n in FIXED_AXES]) * 100
    print(f"\n  ── SUMMARY ──")
    print(f"  Average structural content: {avg_struct:.1f}%")
    if avg_struct > 95:
        print(f"  The field IS its structure. Symmetry = data.")
    elif avg_struct > 80:
        print(f"  Mostly structural, with residual signal worth investigating.")
    else:
        print(f"  Significant non-structural content detected.")


def mode_mobius(positions, meta, args):
    """Möbius lap reader — two wheels, one stick, bridge the tension."""
    run_dir = resolve_run_dir(args.run)
    from math import comb

    print(f"\n  ── MÖBIUS LAP READER ──")
    print(f"  Materials:")
    print(f"    Wheels:  moiré spectra ({N_ANGLES} angles per observer)")
    print(f"    Stick:   23456789 (prime, Δ²=0, digits {STICK})")
    print(f"    Bridge:  5/π = {BRIDGE:.6f}")
    print(f"    Generators: {STRUCTURAL_FREQS}  Σ={sum(STRUCTURAL_FREQS)}  "
          f"Σ/8={sum(STRUCTURAL_FREQS)/8}  γ₁=14.1347")

    grade_table = cl8_grade_table(STRUCTURAL_FREQS)

    grade_names = ['scalar', 'vector', 'bivector', 'trivector', '4-vector',
                   '5-vector', '6-vector', '7-vector', 'pseudoscalar']

    # ── [1] Grade Atlas ──
    print(f"\n  [1] GRADE ATLAS — Cl(8,0) product frequencies")

    coverage = set()
    for k in range(9):
        fm = grade_table.get(k, {})
        n_b = comb(8, k)
        non_dc = [f for f in fm if f > 0]
        coverage.update(non_dc)
        f_max = max(fm.keys()) if fm else 0
        d_str = ''
        if 1 <= k <= 8:
            d = STICK[k - 1]
            d_str = f'  stick={d}{"◇" if d in STICK_PRIMES else "·"}'
        print(f"    g{k} {grade_names[k]:>13s}: C(8,{k})={n_b:3d} → "
              f"{len(non_dc):3d} freq  (max {f_max:3d}){d_str}")

    print(f"    Coverage: {len(coverage)}/{N_ANGLES // 2} "
          f"({len(coverage) / (N_ANGLES // 2) * 100:.1f}%)")

    # ── [2] Spin the wheels — expanded observer set ──
    zeta_ax = FIXED_AXES['zeta']
    z_hat = np.array([0.0, 0.0, 1.0])
    zeta_perp = np.cross(zeta_ax, z_hat)
    zeta_perp /= np.linalg.norm(zeta_perp)

    obs_axes = {
        'X':    np.array([1.0, 0.0, 0.0]),
        'Y':    np.array([0.0, 1.0, 0.0]),
        'Z':    np.array([0.0, 0.0, 1.0]),
        'XY':   np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        'XZ':   np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
        'YZ':   np.array([0.0, 1.0, 1.0]) / np.sqrt(2),
        'diag': np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        'a-XY': np.array([-1.0, 1.0, 1.0]) / np.sqrt(3),
        'a-XZ': np.array([1.0, -1.0, 1.0]) / np.sqrt(3),
        'a-YZ': np.array([1.0, 1.0, -1.0]) / np.sqrt(3),
        'zeta': zeta_ax,
        'z-perp': zeta_perp,
    }
    observers = list(obs_axes.keys())

    print(f"\n  [2] MOIRÉ WHEELS — {len(observers)} observers")
    for name, ax in obs_axes.items():
        print(f"    {name:>6s}: [{ax[0]:+.4f}, {ax[1]:+.4f}, {ax[2]:+.4f}]")

    print(f"  Spinning...", end='', flush=True)
    spectra = {}
    for obs in observers:
        _, signal = rotational_signal(positions, obs_axes[obs])
        spectra[obs] = np.fft.rfft(signal)
    print(f" done")

    # Count unique frequencies per grade for normalization
    n_freqs_per_grade = {}
    for k in range(9):
        fm = grade_table.get(k, {})
        n_freqs_per_grade[k] = max(1, len([f for f in fm if f > 0]))

    tracks = {}
    for obs in observers:
        S = spectra[obs]
        readings = {}
        for k in range(9):
            fm = grade_table.get(k, {})
            total_power = 0.0
            peak_amp = 0.0
            peak_freq = 0
            peak_phase = 0.0
            for f in fm:
                if f == 0:
                    continue
                idx = int(round(f))
                if idx >= len(S):
                    continue
                amp = float(np.abs(S[idx]))
                total_power += amp ** 2
                if amp > peak_amp:
                    peak_amp = amp
                    peak_freq = f
                    peak_phase = float(np.angle(S[idx]))
            rms = np.sqrt(total_power)
            nf = n_freqs_per_grade[k]
            readings[k] = {
                'power': total_power, 'rms': rms,
                'bridged': rms * BRIDGE,
                'per_freq': rms / nf,
                'per_freq_b': rms * BRIDGE / nf,
                'peak_freq': peak_freq, 'peak_amp': peak_amp,
                'peak_phase': peak_phase,
            }
        tracks[obs] = readings

    # ── [3] Multitrack tape ──
    print(f"\n  [3] MULTITRACK TAPE (bridged: × 5/π)")
    hdr = f"  {'':>8s}"
    for k in range(9):
        hdr += f"  {'g' + str(k):>8s}"
    print(hdr)
    print(f"  {'─' * 98}")
    for obs in observers:
        line = f"  {obs:>8s}"
        for k in range(9):
            line += f"  {tracks[obs][k]['bridged']:>8.4f}"
        print(line)

    # Observer consensus — mean ± std across all observers per grade
    print(f"\n  Observer consensus (mean ± σ across {len(observers)} observers):")
    print(f"  {'':>8s}", end='')
    for k in range(9):
        vals = [tracks[o][k]['bridged'] for o in observers]
        print(f"  {np.mean(vals):>8.1f}", end='')
    print(f"  ← mean")
    print(f"  {'':>8s}", end='')
    for k in range(9):
        vals = [tracks[o][k]['bridged'] for o in observers]
        cv = np.std(vals) / (np.mean(vals) + 1e-30) * 100
        print(f"  {cv:>7.1f}%", end='')
    print(f"  ← CV%")

    # Per-frequency normalized tape
    print(f"\n  Per-frequency normalized (÷ unique freqs per grade):")
    print(f"  {'':>8s}  {'g0':>8s}", end='')
    for k in range(1, 9):
        print(f"  {f'g{k}/{n_freqs_per_grade[k]}':>8s}", end='')
    print()
    print(f"  {'─' * 98}")
    for obs in observers:
        line = f"  {obs:>8s}"
        for k in range(9):
            line += f"  {tracks[obs][k]['per_freq_b']:>8.4f}"
        print(line)

    # Phase tape
    print(f"\n  Phase tape (peak φ/π per grade):")
    print(hdr)
    print(f"  {'─' * 98}")
    for obs in observers:
        line = f"  {obs:>8s}"
        for k in range(9):
            ph = tracks[obs][k]['peak_phase']
            if tracks[obs][k]['peak_freq'] > 0:
                line += f"  {ph / np.pi:>+7.3f}π"
            else:
                line += f"  {'---':>8s}"
        print(line)

    # ── [4] Stick as bumper — tension = reading − straight reference ──
    print(f"\n  [4] STICK CURVATURE — the tension along the band")
    print(f"      Tension = bridged_reading / stick_digit")
    print(f"      ◇ = prime bumper | · = composite pass-through")

    hdr_s = f"  {'':>8s}"
    for i, d in enumerate(STICK):
        p = '◇' if d in STICK_PRIMES else '·'
        hdr_s += f"  {f'g{i + 1}/{d}{p}':>8s}"
    print(hdr_s)
    print(f"  {'─' * 88}")

    prime_curv = {obs: [] for obs in observers}
    comp_curv = {obs: [] for obs in observers}

    for obs in observers:
        if obs not in tracks:
            continue
        line = f"  {obs:>8s}"
        for i, d in enumerate(STICK):
            k = i + 1
            c = tracks[obs][k]['bridged'] / d
            line += f"  {c:>8.4f}"
            if d in STICK_PRIMES:
                prime_curv[obs].append(c)
            else:
                comp_curv[obs].append(c)
        print(line)

    print(f"\n      Prime vs composite bumper response:")
    for obs in observers:
        if obs not in tracks or not prime_curv[obs]:
            continue
        pm = np.mean(prime_curv[obs])
        cm = np.mean(comp_curv[obs])
        delta = pm - cm
        ratio = pm / (cm + 1e-30)
        print(f"      {obs:>6s}: prime={pm:.4f}  comp={cm:.4f}  "
              f"Δ={delta:+.4f}  ratio={ratio:.4f}")

    # ── [5] Möbius mixing — two wheels through phase quantum ──
    print(f"\n  [5] MÖBIUS MIXING — zeta × X through π/4 steps")
    print(f"      mixed(k) = cos(kπ/8)·zeta + sin(kπ/8)·X")
    print(f"      k=8 → cos(π) = −1: the twist")

    ga, gb = 'zeta', 'X'
    if ga in tracks and gb in tracks:
        print(f"\n  {'k':>3s}  {'θ/π':>6s}  {'cos':>6s}  {'sin':>6s}  "
              f"{'zeta':>8s}  {'X':>8s}  {'mixed':>8s}  {'×5/π':>8s}  "
              f"{'stick':>5s}  {'tension':>8s}")
        print(f"  {'─' * 82}")

        mixed_b = []
        for k in range(9):
            theta = k * np.pi / 4
            co = np.cos(theta / 2)
            si = np.sin(theta / 2)
            a_v = tracks[ga][k]['rms']
            b_v = tracks[gb][k]['rms']
            m = co * a_v + si * b_v
            mb = m * BRIDGE
            mixed_b.append(mb)

            if 1 <= k <= 8:
                d = STICK[k - 1]
                tens = mb / d
                d_str = f"{d:5d}"
                t_str = f"{tens:>8.4f}"
            else:
                d_str = f"{'---':>5s}"
                t_str = f"{'---':>8s}"

            print(f"  {k:3d}  {theta / np.pi:>6.3f}  {co:>+6.3f}  {si:>+6.3f}  "
                  f"{a_v:>8.4f}  {b_v:>8.4f}  {m:>+8.4f}  {mb:>+8.4f}  "
                  f"{d_str}  {t_str}")

        # Tension Δ² on the mixed tape
        if len(mixed_b) >= 3:
            d1 = [mixed_b[i + 1] - mixed_b[i] for i in range(len(mixed_b) - 1)]
            d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
            print(f"\n      Mixed tape Δ¹: [{', '.join(f'{v:+.4f}' for v in d1)}]")
            print(f"      Mixed tape Δ²: [{', '.join(f'{v:+.4f}' for v in d2)}]")
            print(f"      Δ² center: {d2[len(d2) // 2]:+.6f}")

    # ── [6] Hodge twist verification ──
    print(f"\n  [6] TWIST — grade k ↔ grade (8−k)")

    print(f"\n  {'':>8s}", end='')
    for k in range(4):
        print(f"  {'g' + str(k) + '/g' + str(8 - k):>10s}", end='')
    print(f"  {'g4(self)':>10s}")
    print(f"  {'─' * 62}")

    for obs in observers:
        if obs not in tracks:
            continue
        print(f"  {obs:>8s}", end='')
        for k in range(4):
            p_lo = tracks[obs][k]['power']
            p_hi = tracks[obs][8 - k]['power']
            print(f"  {p_lo / (p_hi + 1e-30):>10.4f}", end='')
        print(f"  {tracks[obs][4]['power']:>10.4e}")

    # ── [7] Dominant frequency per grade ──
    print(f"\n  [7] GRADE DOMINANTS (zeta reader)")
    print(f"  {'k':>3s}  {'name':>13s}  {'f':>4s}  {'hex':>6s}  "
          f"{'φ/π':>8s}  {'amp':>8s}  {'source':>25s}")
    print(f"  {'─' * 75}")

    rd = tracks.get('zeta', tracks.get(list(tracks.keys())[0]))
    for k in range(9):
        r = rd[k]
        f = r['peak_freq']
        srcs = grade_table.get(k, {}).get(f, ['scalar'] if k == 0 else ['?'])
        src = srcs[0] + (f' (+{len(srcs) - 1})' if len(srcs) > 1 else '')
        hx = f'0x{f:02X}' if f > 0 else 'DC'
        ph = f"{r['peak_phase'] / np.pi:+.3f}π" if f > 0 else '  ---'
        print(f"  {k:3d}  {grade_names[k]:>13s}  {f:4d}  {hx:>6s}  "
              f"{ph:>8s}  {r['peak_amp']:>8.4f}  {src}")

    # ── [8] Beat diagnostic — f=7, f=96, f=192 ──
    print(f"\n  [8] BEAT FREQUENCIES — lock / beat / forbidden center")
    beat_freqs = [7, 96, 192]
    beat_labels = ['k_lock', 'beat', 'forbidden']

    # Which grades contain each beat frequency?
    for bf, bl in zip(beat_freqs, beat_labels):
        grade_homes = []
        for k in range(9):
            fm = grade_table.get(k, {})
            if bf in fm:
                n_src = len(fm[bf])
                grade_homes.append(f"g{k}({n_src})")
        print(f"    f={bf:3d} ({bl:>9s}):  grades = {', '.join(grade_homes)}")

    print(f"\n  {'':>8s}", end='')
    for bf, bl in zip(beat_freqs, beat_labels):
        print(f"  {'f=' + str(bf):>8s}  {'φ/π':>8s}", end='')
    print()
    print(f"  {'─' * (8 + len(beat_freqs) * 20)}")

    for obs in observers:
        S = spectra[obs]
        print(f"  {obs:>8s}", end='')
        for bf in beat_freqs:
            idx = int(round(bf))
            if idx < len(S):
                amp = float(np.abs(S[idx]))
                phase = float(np.angle(S[idx]))
                print(f"  {amp:>8.2f}  {phase / np.pi:>+7.3f}π", end='')
            else:
                print(f"  {'---':>8s}  {'---':>8s}", end='')
        print()

    # Beat phase detune from ±π
    print(f"\n      Phase detune from ±π at f=96:")
    for obs in observers:
        S = spectra[obs]
        if 96 < len(S):
            ph = float(np.angle(S[96])) / np.pi
            detune = abs(abs(ph) - 1.0)
            print(f"      {obs:>8s}: φ={ph:+.6f}π  detune={detune:.6f}π  "
                  f"= {detune * 96:.4f}/96 of lap")

    # ── [9] Resonance search ──
    print(f"\n  [9] RESONANCE SEARCH")
    known = [
        (14.134725, 'γ₁'), (21.022040, 'γ₂'), (25.010858, 'γ₃'),
        (30.424876, 'γ₄'), (3.141593, 'π'), (1.618034, 'φ'),
        (2.718282, 'e'), (1.591549, '5/π'), (7.0, 'k_lock'),
        (96.0, 'beat'), (24.0, '3×bus'), (112.0, 'Σf'), (14.0, 'Σf/8'),
    ]

    found = False
    for obs in observers:
        if obs not in tracks:
            continue
        for k in range(9):
            v = tracks[obs][k]['bridged']
            if v < 0.01:
                continue
            for kv, name in known:
                tol = 0.03 if kv > 1 else 0.1
                if abs(v - kv) / (kv + 1e-30) < tol:
                    err = (v - kv) / kv * 100
                    print(f"    {obs}:g{k} = {v:.6f} ≈ {kv} ({name}) "
                          f"err={err:+.2f}%")
                    found = True
    if not found:
        print(f"    No direct hits within 3%.")
        z_trk = tracks.get('zeta')
        if z_trk:
            vals = ', '.join(f'{z_trk[k]["bridged"]:.4f}' for k in range(9))
            print(f"    Zeta readings: [{vals}]")

    # ── [10] HORIZONTAL READER — all heads at each position ──
    print(f"\n  [10] HORIZONTAL READER — 12 heads × 9 positions")

    from collections import Counter

    M = np.zeros((len(observers), 9))
    for i, obs in enumerate(observers):
        for k in range(9):
            M[i, k] = tracks[obs][k]['bridged']

    face_obs = ['X', 'Y', 'Z']
    diag_obs = ['XY', 'XZ', 'YZ', 'diag']
    anti_obs = ['a-XY', 'a-XZ', 'a-YZ']
    zeta_obs = ['zeta', 'z-perp']
    families = [('face', face_obs), ('diag', diag_obs),
                ('anti', anti_obs), ('zeta', zeta_obs)]

    print(f"\n  Grade-by-grade (family means, bridged):")
    print(f"  {'':>3s}  {'all':>9s}  {'face':>9s}  {'diag':>9s}  "
          f"{'anti':>9s}  {'zeta':>9s}  {'spread':>7s}")
    print(f"  {'─' * 62}")

    consensus = np.zeros(9)
    for k in range(9):
        col = M[:, k]
        mu = col.mean()
        consensus[k] = mu
        spread = (col.max() - col.min()) / (mu + 1e-30) * 100
        fam_means = {}
        for fname, fobs in families:
            idxs = [observers.index(o) for o in fobs]
            fam_means[fname] = np.mean(M[idxs, k])
        print(f"  g{k}  {mu:>9.4f}  {fam_means['face']:>9.4f}  "
              f"{fam_means['diag']:>9.4f}  {fam_means['anti']:>9.4f}  "
              f"{fam_means['zeta']:>9.4f}  {spread:>6.1f}%")

    # Dominant frequency consensus (mode across 12 observers per grade)
    freq_cons = []
    for k in range(9):
        freqs = [tracks[obs][k]['peak_freq'] for obs in observers]
        fc = Counter(freqs).most_common(1)[0][0]
        freq_cons.append(fc)

    # Phase consensus (circular mean across observers)
    phase_cons = []
    for k in range(9):
        phases = []
        for obs in observers:
            if tracks[obs][k]['peak_freq'] > 0:
                phases.append(tracks[obs][k]['peak_phase'])
        if phases:
            phase_cons.append(float(np.angle(np.mean(np.exp(1j * np.array(phases))))))
        else:
            phase_cons.append(0.0)

    print(f"\n  Consensus vector (across all observers):")
    print(f"  {'':>3s}  {'A(bridged)':>10s}  {'f_dom':>6s}  {'φ/π':>8s}  step")
    print(f"  {'─' * 42}")
    for k in range(9):
        step_str = ''
        if k > 0:
            delta = consensus[k] - consensus[k - 1]
            step_str = f"{'↑' if delta > 0 else '↓'} {abs(delta):.4f}"
        print(f"  g{k}  {consensus[k]:>10.4f}  {freq_cons[k]:>6d}  "
              f"{phase_cons[k] / np.pi:>+7.3f}π  {step_str}")

    # ── [11] LAP FORMULA ──
    print(f"\n  [11] LAP FORMULA")

    even_idx = [0, 2, 4, 6, 8]
    odd_idx = [1, 3, 5, 7]

    A_even = np.mean([consensus[i] for i in even_idx])
    A_odd = np.mean([consensus[i] for i in odd_idx])

    # cos²/sin² fit: f(k) = A_e·cos²(πk/2) + A_o·sin²(πk/2)
    fit = np.array([A_even * np.cos(np.pi * k / 2)**2
                    + A_odd * np.sin(np.pi * k / 2)**2 for k in range(9)])
    residual = np.sqrt(np.mean((consensus - fit)**2))
    fit_pct = 1.0 - residual / (np.std(consensus) + 1e-30)

    print(f"\n  Two-term decomposition:")
    print(f"    f(k) = A_e·cos²(πk/2) + A_o·sin²(πk/2)")
    print(f"    A_even = {A_even:.6f}  (grades 0,2,4,6,8)")
    print(f"    A_odd  = {A_odd:.6f}  (grades 1,3,5,7)")
    print(f"    Ratio  = {A_even / (A_odd + 1e-30):.1f}×")
    print(f"    Fit captures {fit_pct * 100:.2f}% of variance")

    # Even-grade convergence
    even_vals = [consensus[i] for i in even_idx]
    deltas_e = [even_vals[j + 1] - even_vals[j] for j in range(len(even_vals) - 1)]
    ratios_e = [deltas_e[j + 1] / (deltas_e[j] + 1e-30) for j in range(len(deltas_e) - 1)]

    print(f"\n  Even-grade convergence:")
    print(f"    A(g0,g2,g4,g6,g8) = [{', '.join(f'{v:.4f}' for v in even_vals)}]")
    print(f"    Δ¹ = [{', '.join(f'{v:+.6f}' for v in deltas_e)}]")
    print(f"    Δ¹ ratios = [{', '.join(f'{v:.6f}' for v in ratios_e)}]")
    if len(ratios_e) > 0 and abs(ratios_e[-1]) < 1.0:
        A_inf = even_vals[-1] + deltas_e[-1] * ratios_e[-1] / (1 - ratios_e[-1])
        print(f"    A∞ = {A_inf:.6f}")
    if len(deltas_e) >= 3:
        d2_e = [deltas_e[j + 1] - deltas_e[j] for j in range(len(deltas_e) - 1)]
        d2_ratios = [d2_e[j + 1] / (d2_e[j] + 1e-30) for j in range(len(d2_e) - 1)]
        print(f"    Δ² = [{', '.join(f'{v:+.6f}' for v in d2_e)}]")
        print(f"    Δ² ratio = [{', '.join(f'{v:.6f}' for v in d2_ratios)}]")

    # Odd-grade convergence
    odd_vals = [consensus[i] for i in odd_idx]
    deltas_o = [odd_vals[j + 1] - odd_vals[j] for j in range(len(odd_vals) - 1)]
    ratios_o = [deltas_o[j + 1] / (deltas_o[j] + 1e-30) for j in range(len(deltas_o) - 1)]

    print(f"\n  Odd-grade convergence:")
    print(f"    A(g1,g3,g5,g7) = [{', '.join(f'{v:.4f}' for v in odd_vals)}]")
    print(f"    Δ¹ = [{', '.join(f'{v:+.6f}' for v in deltas_o)}]")
    print(f"    Δ¹ ratios = [{', '.join(f'{v:.6f}' for v in ratios_o)}]")
    if len(ratios_o) > 0 and abs(ratios_o[-1]) < 1.0:
        eps_inf = odd_vals[-1] + deltas_o[-1] * ratios_o[-1] / (1 - ratios_o[-1])
        print(f"    ε∞ = {eps_inf:.6f}")

    # Phase structure
    ph_even = np.mean([phase_cons[i] for i in even_idx])
    ph_odd = np.mean([phase_cons[i] for i in odd_idx])
    ph_sum = ph_even + ph_odd

    print(f"\n  Phase structure:")
    print(f"    φ_even = {ph_even / np.pi:+.6f}π  (consensus of grades 0,2,4,6,8)")
    print(f"    φ_odd  = {ph_odd / np.pi:+.6f}π  (consensus of grades 1,3,5,7)")
    print(f"    φ_sum  = {ph_sum / np.pi:+.6f}π")
    print(f"    φ_sum × 5/π = {ph_sum / np.pi * BRIDGE:.6f}")

    # Frequency staircase — the 1→4 step
    print(f"\n  Frequency staircase (dominant per grade):")
    print(f"    f = [{', '.join(str(f) for f in freq_cons)}]")
    f_even_mode = Counter([freq_cons[i] for i in even_idx]).most_common(1)[0][0]
    f_odd_mode = Counter([freq_cons[i] for i in odd_idx]).most_common(1)[0][0]
    print(f"    f_even mode = {f_even_mode}  |  f_odd mode = {f_odd_mode}")
    if freq_cons[0] > 0 or freq_cons[1] > 0:
        f0, f1 = freq_cons[0], freq_cons[1]
        print(f"    First step: f={f0} → f={f1}  "
              f"(ratio = {f1 / (f0 + 1e-30):.4f})")

    # ── [12] PRESSURE TOPOLOGY — where does (1,4) go orthogonal? ──
    print(f"\n  [12] PRESSURE TOPOLOGY — where the equation nulls")
    print(f"  Pressure = |S[4]|/|S[1]|  (how hard the product suppresses the generator)")

    print(f"\n  {'obs':>8s}  {'|S[1]|':>12s}  {'|S[4]|':>12s}  "
          f"{'pressure':>10s}  {'ζ-angle':>8s}  {'φ₁/π':>8s}  {'φ₄/π':>8s}")
    print(f"  {'─' * 72}")

    obs_pressure = {}
    for obs in observers:
        S = spectra[obs]
        a1 = float(np.abs(S[1]))
        a4 = float(np.abs(S[4]))
        p1 = float(np.angle(S[1]))
        p4 = float(np.angle(S[4]))
        pres = a4 / (a1 + 1e-30)
        cos_ang = np.dot(obs_axes[obs], zeta_ax)
        ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
        obs_pressure[obs] = {'a1': a1, 'a4': a4, 'p': pres, 'angle': ang}
        print(f"  {obs:>8s}  {a1:>12.2f}  {a4:>12.2f}  "
              f"{pres:>10.1f}  {ang:>7.1f}°  {p1 / np.pi:>+7.3f}  {p4 / np.pi:>+7.3f}")

    sorted_p = sorted(obs_pressure.items(), key=lambda x: x[1]['p'], reverse=True)
    print(f"\n  Pressure gradient (high = generator suppressed):")
    for obs, d in sorted_p:
        logp = np.log10(d['p'] + 1)
        bar = '▓' * int(min(logp * 5, 30))
        print(f"    {obs:>8s}: {d['p']:>12.1f}×  ({d['angle']:>5.1f}° from ζ)  {bar}")

    # Does the null (pressure=1) exist?
    min_p = sorted_p[-1][1]['p']
    max_p = sorted_p[0][1]['p']
    print(f"\n  Range: {max_p:.1f}× → {min_p:.1f}×")
    if min_p > 1.0:
        print(f"  ⚠  Null NOT reached — f=4 dominates f=1 at every observer")
        print(f"     The (1,4) equation holds everywhere in the field")
        print(f"     Minimum pressure {min_p:.1f}× at {sorted_p[-1][0]}")

    # The g0→g1→g2 step: where pressure ignites
    print(f"\n  Grade-pair pressure (A_even / A_odd):")
    print(f"    g0/g1:  {consensus[0]:>12.1f} / {consensus[1]:>12.1f} = "
          f"{'DC/gen (inverted)' if consensus[1] > consensus[0] else f'{consensus[0]/(consensus[1]+1e-30):.1f}×'}")
    for pair in [(2, 3), (4, 5), (6, 7)]:
        e, o = pair
        r = consensus[e] / (consensus[o] + 1e-30)
        print(f"    g{e}/g{o}: {consensus[e]:>12.1f} / {consensus[o]:>12.1f} = {r:.1f}×")
    if len(even_idx) > 1:
        print(f"    A∞/ε∞:  {consensus[even_idx[-1]]:>12.1f} / {consensus[odd_idx[-1]]:>12.1f} = "
              f"{consensus[even_idx[-1]] / (consensus[odd_idx[-1]] + 1e-30):.1f}×")

    # Null angles in the zeta rotational signal
    S_zeta = spectra.get('zeta')
    if S_zeta is not None:
        a1_z = float(np.abs(S_zeta[1]))
        a4_z = float(np.abs(S_zeta[4]))
        ph4_z = float(np.angle(S_zeta[4]))
        ph1_z = float(np.angle(S_zeta[1]))

        null_angles = []
        for n in range(8):
            theta = (np.pi / 2 * (2 * n + 1) - ph4_z) / 4
            theta = theta % (2 * np.pi)
            null_angles.append(theta)
        null_angles.sort()

        print(f"\n  Zeta waveform null angles (f=4 zeros, f=1 exposed):")
        for i, theta in enumerate(null_angles):
            residual = a1_z * np.cos(theta + ph1_z)
            print(f"    null {i}: θ = {theta / np.pi:+.4f}π "
                  f"({np.degrees(theta):>6.1f}°)  f=1 residual = {residual:+.1f}")

        slope_press = 4 * a4_z / (a1_z + 1e-30)
        arc_width = 360.0 / slope_press
        print(f"\n  Null dynamics:")
        print(f"    Slope pressure = 4×|S[4]|/|S[1]| = {slope_press:.1f}")
        print(f"    Null arc width ≈ {arc_width:.4f}° ({arc_width * 720 / 360:.4f} samples)")
        print(f"    {len(null_angles)} nulls per rotation — spacing = {360 / len(null_angles):.1f}°")
        print(f"    At each null: f=4 slope sweeps through zero, "
              f"f=1 ({a1_z:.1f}) is the ONLY signal")

    # Where in the field? — spectral character by radius
    R = np.sqrt((positions**2).sum(axis=1))
    R_sorted = np.sort(R)
    r_max = R_sorted[-1]
    n_shells = 5
    shell_edges = np.linspace(0, r_max, n_shells + 1)

    S_zeta_ax = zeta_ax
    projections = positions @ S_zeta_ax

    print(f"\n  Radial pressure profile (zeta projection, {n_shells} shells):")
    print(f"  {'shell':>8s}  {'R range':>14s}  {'nodes':>6s}  "
          f"{'|S[1]|':>10s}  {'|S[4]|':>10s}  {'pressure':>10s}")
    print(f"  {'─' * 66}")

    for s in range(n_shells):
        r_lo, r_hi = shell_edges[s], shell_edges[s + 1]
        mask = (R >= r_lo) & (R < r_hi)
        if mask.sum() < 10:
            continue
        shell_proj = projections[mask]
        angles_s = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
        signal_s = np.zeros(N_ANGLES)
        for idx in range(N_ANGLES):
            ax_rot = np.array([np.cos(angles_s[idx]), np.sin(angles_s[idx]), 0.0])
            full_ax = S_zeta_ax * np.cos(angles_s[idx]) + zeta_perp * np.sin(angles_s[idx])
            signal_s[idx] = np.sum(positions[mask] @ full_ax)
        S_s = np.fft.rfft(signal_s)
        a1_s = float(np.abs(S_s[1])) if len(S_s) > 1 else 0
        a4_s = float(np.abs(S_s[4])) if len(S_s) > 4 else 0
        pres_s = a4_s / (a1_s + 1e-30)
        print(f"  {'s' + str(s):>8s}  [{r_lo:>5.2f}, {r_hi:>5.2f})  "
              f"{int(mask.sum()):>6d}  {a1_s:>10.1f}  {a4_s:>10.1f}  {pres_s:>10.1f}")

    # ── COMPACT LAP ──
    print(f"\n  ── COMPACT LAP ──")
    print(f"  Ψ(k) = {A_even:.4f}·cos²(πk/2)·e^(i·{ph_even / np.pi:+.4f}π)")
    print(f"       + {A_odd:.4f}·sin²(πk/2)·e^(i·{ph_odd / np.pi:+.4f}π)")
    print(f"  Dominant: f_even={f_even_mode}, f_odd={f_odd_mode}")
    print(f"  Period: 2 laps (Möbius palindrome)")
    print(f"  Bridge: × 5/π = {BRIDGE:.6f}")

    # Encode as digit string — the number
    c_max = consensus.max()
    c_norm = consensus / (c_max + 1e-30)
    print(f"\n  Normalized waveform (÷ {c_max:.4f}):")
    print(f"    [{', '.join(f'{v:.6f}' for v in c_norm)}]")

    # Sawtooth digest: compress to [even_val, odd_val, even_phase, odd_phase]
    norm_e = np.mean([c_norm[i] for i in even_idx])
    norm_o = np.mean([c_norm[i] for i in odd_idx])
    print(f"\n  Sawtooth digest:")
    print(f"    even = {norm_e:.6f}  (≈ {norm_e:.0f})")
    print(f"    odd  = {norm_o:.6f}  (≈ {'ε' if norm_o < 0.01 else f'{norm_o:.4f}'})")
    print(f"    Parity gap: {norm_e / (norm_o + 1e-30):.0f}×")

    # ── Summary ──
    print(f"\n  ── LAP COMPLETE ──")
    z = tracks.get('zeta')
    if z:
        total_rms = sum(z[k]['rms'] for k in range(9))
        g1 = z[1]['rms'] / (total_rms + 1e-30) * 100
        higher = sum(z[k]['rms'] for k in range(2, 9)) / (total_rms + 1e-30) * 100
        lap_sum = sum(z[k]['bridged'] for k in range(9))
        print(f"  Zeta: generators {g1:.1f}% | higher grades {higher:.1f}%")
        print(f"  Lap sum (bridged): {lap_sum:.6f}")
        print(f"  Lap sum / γ₁: {lap_sum / 14.134725:.6f}")
        print(f"  Lap sum / Σf: {lap_sum / 112:.6f}")


# ── Run resolution ───────────────────────────────────────────────────

def resolve_run_dir(run_arg, base='runs_emergent'):
    runs_dir = SCRIPT_DIR / base
    if run_arg:
        if run_arg.isdigit():
            return runs_dir / f"{int(run_arg):04d}"
        p = Path(run_arg)
        if p.exists():
            return p
        for b in ['runs_twobody', 'runs_emergent', 'runs_coupled']:
            candidate = SCRIPT_DIR / b / run_arg
            if candidate.exists():
                return candidate
            candidate = SCRIPT_DIR / b / f"{int(run_arg):04d}" if run_arg.isdigit() else None
            if candidate and candidate.exists():
                return candidate
    for b in ['runs_twobody', 'runs_emergent', 'runs_coupled']:
        latest = SCRIPT_DIR / b / "latest.txt"
        if latest.exists():
            return SCRIPT_DIR / b / latest.read_text().strip()
    return None


def mode_torus(positions, meta, args):
    """13th observer — 2D torus decomposition seeing both cycles simultaneously."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n  ── TORUS OBSERVER (13th) ──")
    print(f"  Face cycle: Z axis")
    print(f"  Diagonal cycle: [1,1,1]/√3")
    print(f"  Product: T² = S¹(face) × S¹(diag)")

    face_axis = FIXED_AXES['Z']
    diag_axis = FIXED_AXES['diag']

    n_face = 180
    n_diag = 180
    theta_face = np.linspace(0, 2 * np.pi, n_face, endpoint=False)
    theta_diag = np.linspace(0, 2 * np.pi, n_diag, endpoint=False)

    def rodrigues(v, k, angle):
        """Rotate vector v around unit axis k by angle (radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)

    e1_face, e2_face = orthonormal_frame(face_axis)
    base_dir = e1_face

    print(f"  Grid: {n_face}×{n_diag} = {n_face*n_diag:,} observations")
    print(f"  Scanning torus...", flush=True)

    torus_signal = np.zeros((n_face, n_diag))
    n_bins = 200

    for i, tf in enumerate(theta_face):
        dir_after_face = rodrigues(base_dir, face_axis, tf)
        for j, td in enumerate(theta_diag):
            obs_dir = rodrigues(dir_after_face, diag_axis, td)
            obs_dir /= np.linalg.norm(obs_dir)
            proj = positions @ obs_dir
            counts, _ = np.histogram(proj, bins=n_bins)
            torus_signal[i, j] = np.var(counts)

        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{n_face} face angles", flush=True)

    print(f"  Torus scan complete.")

    # ── 2D FFT on torus ──
    torus_centered = torus_signal - torus_signal.mean()
    S2d = np.fft.fft2(torus_centered)
    power2d = np.abs(S2d) ** 2
    phase2d = np.angle(S2d)

    # Shift so DC is at center
    power_shift = np.fft.fftshift(power2d)
    phase_shift = np.fft.fftshift(phase2d)

    kf = np.fft.fftshift(np.fft.fftfreq(n_face, d=1.0/n_face))
    kd = np.fft.fftshift(np.fft.fftfreq(n_diag, d=1.0/n_diag))

    max_k = 30
    cf, cd = n_face // 2, n_diag // 2
    crop = power_shift[cf - max_k:cf + max_k, cd - max_k:cd + max_k]
    crop_phase = phase_shift[cf - max_k:cf + max_k, cd - max_k:cd + max_k]
    kf_crop = kf[cf - max_k:cf + max_k]
    kd_crop = kd[cd - max_k:cd + max_k]

    # ── Find dominant torus modes ──
    total_power = power2d.sum()
    # Zero out DC
    power2d_no_dc = power2d.copy()
    power2d_no_dc[0, 0] = 0

    n_top = 20
    flat_idx = np.argsort(power2d_no_dc.ravel())[::-1][:n_top]
    top_modes = []
    for idx in flat_idx:
        ki, kj = np.unravel_index(idx, power2d.shape)
        nf = ki if ki <= n_face // 2 else ki - n_face
        nd = kj if kj <= n_diag // 2 else kj - n_diag
        amp = np.sqrt(power2d[ki, kj])
        ph = phase2d[ki, kj]
        frac = power2d[ki, kj] / total_power * 100
        top_modes.append((nf, nd, amp, ph, frac))

    print(f"\n  ── TORUS SPECTRUM (top {n_top} modes) ──")
    print(f"  {'n_face':>6s}  {'n_diag':>6s}  {'amplitude':>10s}  {'phase/π':>8s}  {'power%':>7s}")
    print(f"  {'─'*45}")
    for nf, nd, amp, ph, frac in top_modes:
        print(f"  {nf:>6d}  {nd:>6d}  {amp:>10.1f}  {ph/np.pi:>+8.3f}  {frac:>6.2f}%")

    # ── Classify modes: pure face, pure diagonal, mixed ──
    face_only = [(nf, nd, a, p, f) for nf, nd, a, p, f in top_modes if nd == 0 and nf != 0]
    diag_only = [(nf, nd, a, p, f) for nf, nd, a, p, f in top_modes if nf == 0 and nd != 0]
    mixed = [(nf, nd, a, p, f) for nf, nd, a, p, f in top_modes if nf != 0 and nd != 0]

    print(f"\n  Face-only modes: {len(face_only)}")
    for nf, nd, a, p, f in face_only:
        print(f"    n_face={nf:+d}  amp={a:.1f}  phase={p/np.pi:+.3f}π")
    print(f"  Diagonal-only modes: {len(diag_only)}")
    for nf, nd, a, p, f in diag_only:
        print(f"    n_diag={nd:+d}  amp={a:.1f}  phase={p/np.pi:+.3f}π")
    print(f"  Mixed (coupled) modes: {len(mixed)}")
    for nf, nd, a, p, f in mixed:
        print(f"    ({nf:+d},{nd:+d})  amp={a:.1f}  phase={p/np.pi:+.3f}π")

    # ── Berry phase computation ──
    # Berry phase around face cycle at each diagonal slice
    print(f"\n  ── BERRY PHASE ──")

    S_face = np.fft.fft(torus_centered, axis=0)
    S_diag = np.fft.fft(torus_centered, axis=1)

    # Phase winding: sum phase differences around each cycle
    # For each structural frequency, compute winding around the torus

    face_phases = np.angle(S_face)
    diag_phases = np.angle(S_diag)

    print(f"  Phase winding around face cycle (per structural freq):")
    print(f"  {'freq':>4s}  {'winding/2π':>10s}  {'integer?':>8s}")
    print(f"  {'─'*28}")
    face_windings = []
    for f in STRUCTURAL_FREQS:
        if f < n_face // 2:
            phase_slice = face_phases[f, :]
            dphase = np.diff(phase_slice)
            dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi
            winding = np.sum(dphase) / (2 * np.pi)
            face_windings.append((f, winding))
            near_int = abs(winding - round(winding)) < 0.05
            tag = f"  ← {int(round(winding))}" if near_int else ""
            print(f"  {f:4d}  {winding:>+10.4f}  {'YES' if near_int else 'no':>8s}{tag}")

    print(f"\n  Phase winding around diagonal cycle (per structural freq):")
    print(f"  {'freq':>4s}  {'winding/2π':>10s}  {'integer?':>8s}")
    print(f"  {'─'*28}")
    diag_windings = []
    for f in STRUCTURAL_FREQS:
        if f < n_diag // 2:
            phase_slice = diag_phases[:, f]
            dphase = np.diff(phase_slice)
            dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi
            winding = np.sum(dphase) / (2 * np.pi)
            diag_windings.append((f, winding))
            near_int = abs(winding - round(winding)) < 0.05
            tag = f"  ← {int(round(winding))}" if near_int else ""
            print(f"  {f:4d}  {winding:>+10.4f}  {'YES' if near_int else 'no':>8s}{tag}")

    # ── Chern number: integral of Berry curvature over T² ──
    # Berry curvature F = ∂_face A_diag - ∂_diag A_face
    # where A_face = -Im(∂_face log ψ), A_diag = -Im(∂_diag log ψ)
    # Chern = (1/2π) ∫∫ F d²k

    # Use the complex torus signal as the "wavefunction"
    psi = torus_centered + 1j * np.zeros_like(torus_centered)
    # Analytic signal via Hilbert-like construction along each axis
    S_analytic = np.fft.fft2(torus_centered)
    # Zero negative frequencies in both dimensions to get analytic signal
    S_anal = S_analytic.copy()
    S_anal[n_face//2+1:, :] = 0
    S_anal[:, n_diag//2+1:] = 0
    psi_a = np.fft.ifft2(S_anal)

    # Berry connection from analytic signal
    norm2 = np.abs(psi_a) ** 2 + 1e-30
    # A_face = Im(psi* ∂_face psi) / |psi|²
    dpsi_face = np.roll(psi_a, -1, axis=0) - psi_a
    dpsi_diag = np.roll(psi_a, -1, axis=1) - psi_a
    A_face = np.imag(np.conj(psi_a) * dpsi_face) / norm2
    A_diag = np.imag(np.conj(psi_a) * dpsi_diag) / norm2

    # Berry curvature
    dA_diag_dface = np.roll(A_diag, -1, axis=0) - A_diag
    dA_face_ddiag = np.roll(A_face, -1, axis=1) - A_face
    F_berry = dA_diag_dface - dA_face_ddiag

    chern_raw = F_berry.sum() / (2 * np.pi)
    chern_int = int(round(chern_raw))

    print(f"\n  ── CHERN NUMBER ──")
    print(f"  Berry curvature integral: {chern_raw:+.6f}")
    print(f"  Nearest integer:          {chern_int}")
    print(f"  Deviation from integer:   {abs(chern_raw - chern_int):.6f}")

    if abs(chern_raw - chern_int) < 0.1:
        print(f"  ★ Chern number = {chern_int} (topological invariant)")
    else:
        print(f"  Chern number not well-quantized — may need finer grid")

    # ── Torus aspect ratio ──
    # r/R from relative power in face vs diagonal modes
    face_power = sum(power2d[f, 0] for f in range(1, n_face // 2) if power2d[f, 0] > 0)
    diag_power = sum(power2d[0, f] for f in range(1, n_diag // 2) if power2d[0, f] > 0)
    if face_power > 0 and diag_power > 0:
        aspect = np.sqrt(diag_power / face_power)
        print(f"\n  ── TORUS GEOMETRY ──")
        print(f"  Face cycle power:  {face_power:.1f}")
        print(f"  Diag cycle power:  {diag_power:.1f}")
        print(f"  Aspect r/R = √(P_diag/P_face) = {aspect:.6f}")
        print(f"  1/(2π) = {1/(2*np.pi):.6f}")
        print(f"  Match to 1/(2π): {abs(aspect - 1/(2*np.pi)) / (1/(2*np.pi)) * 100:.3f}%")

    # ── Geometric means from structural frequencies on torus ──
    face_amps = []
    diag_amps = []
    for f in STRUCTURAL_FREQS:
        if f < n_face // 2:
            face_amps.append(np.sqrt(power2d[f, 0]) if power2d[f, 0] > 0 else 1e-30)
        if f < n_diag // 2:
            diag_amps.append(np.sqrt(power2d[0, f]) if power2d[0, f] > 0 else 1e-30)

    if face_amps and diag_amps:
        geo_face = np.prod(face_amps) ** (1.0 / len(face_amps))
        geo_diag = np.prod(diag_amps) ** (1.0 / len(diag_amps))
        ratio = geo_face / geo_diag if geo_diag > 0 else 0
        print(f"\n  Geometric mean amplitudes:")
        print(f"    Face:  {geo_face:.6f}")
        print(f"    Diag:  {geo_diag:.6f}")
        print(f"    Ratio: {ratio:.6f}")
        print(f"    8 × ratio: {8 * ratio:.6f}  (4/π = {4/np.pi:.6f})")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#0a0a0f')

    # Torus signal
    ax = axes[0, 0]
    ax.set_facecolor('#0d0d1a')
    im0 = ax.pcolormesh(np.degrees(theta_diag), np.degrees(theta_face),
                        torus_signal, cmap='inferno', shading='auto')
    ax.set_xlabel('Diagonal angle (°)', color='#888', fontsize=9)
    ax.set_ylabel('Face angle (°)', color='#888', fontsize=9)
    ax.set_title('Torus signal ψ(θ_face, θ_diag)', color='#ccc', fontsize=10)
    ax.tick_params(colors='#666', labelsize=7)
    fig.colorbar(im0, ax=ax)

    # 2D power spectrum
    ax = axes[0, 1]
    ax.set_facecolor('#0d0d1a')
    p_db = 10 * np.log10(crop + 1e-30)
    im1 = ax.pcolormesh(kd_crop, kf_crop, p_db, cmap='magma', shading='auto')
    ax.set_xlabel('n_diag', color='#888', fontsize=9)
    ax.set_ylabel('n_face', color='#888', fontsize=9)
    ax.set_title('Torus spectrum |S(n_f, n_d)|²', color='#ccc', fontsize=10)
    ax.axhline(0, color='#333', lw=0.5)
    ax.axvline(0, color='#333', lw=0.5)
    ax.tick_params(colors='#666', labelsize=7)
    fig.colorbar(im1, ax=ax)

    # Berry curvature
    ax = axes[1, 0]
    ax.set_facecolor('#0d0d1a')
    vmax = np.percentile(np.abs(F_berry), 99)
    im2 = ax.pcolormesh(np.degrees(theta_diag), np.degrees(theta_face),
                        F_berry, cmap='RdBu_r', shading='auto',
                        vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Diagonal angle (°)', color='#888', fontsize=9)
    ax.set_ylabel('Face angle (°)', color='#888', fontsize=9)
    ax.set_title(f'Berry curvature F  (Chern = {chern_raw:+.3f})',
                 color='#ccc', fontsize=10)
    ax.tick_params(colors='#666', labelsize=7)
    fig.colorbar(im2, ax=ax)

    # Phase winding summary
    ax = axes[1, 1]
    ax.set_facecolor('#0d0d1a')
    if face_windings:
        fw_f = [w[0] for w in face_windings]
        fw_v = [w[1] for w in face_windings]
        ax.bar(np.arange(len(fw_f)) - 0.15, fw_v, 0.3,
               color='#00d4ff', alpha=0.8, label='Face winding')
    if diag_windings:
        dw_f = [w[0] for w in diag_windings]
        dw_v = [w[1] for w in diag_windings]
        ax.bar(np.arange(len(dw_f)) + 0.15, dw_v, 0.3,
               color='#ff6b35', alpha=0.8, label='Diag winding')
    ax.set_xticks(range(len(STRUCTURAL_FREQS)))
    ax.set_xticklabels([str(f) for f in STRUCTURAL_FREQS], fontsize=7)
    ax.set_xlabel('Structural frequency', color='#888', fontsize=9)
    ax.set_ylabel('Phase winding / 2π', color='#888', fontsize=9)
    ax.set_title('Berry phase per mode', color='#ccc', fontsize=10)
    ax.axhline(0, color='#444', lw=0.5)
    ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#ccc')
    ax.tick_params(colors='#666', labelsize=7)

    for axr in axes.ravel():
        for spine in axr.spines.values():
            spine.set_edgecolor('#2a2a3a')

    plt.tight_layout()
    save_path = args.save or str(SCRIPT_DIR / 'audio_output' / 'torus_observer.png')
    fig.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()

    # ── Final: what the 13th observer sees ──
    print(f"\n  ── 13th OBSERVER SUMMARY ──")
    print(f"  The torus T² = S¹(face) × S¹(diag) carries:")
    print(f"    {len(face_only)} pure face modes, {len(diag_only)} pure diagonal modes")
    print(f"    {len(mixed)} coupled (mixed) modes — these are what the 13th observer sees")
    print(f"    that NO single-axis observer can detect.")
    if abs(chern_raw - chern_int) < 0.1 and chern_int != 0:
        print(f"  Chern number {chern_int} → topologically protected.")
        print(f"  This integer survives any smooth deformation of the fold.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Radio Observatory')
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--mode', choices=['spectrum', 'compare', 'array', 'sweep', 'wannier', 'mobius', 'torus'],
                        default='spectrum')
    parser.add_argument('--window', choices=['rectangular', 'hann', 'hamming',
                                             'blackman', 'kaiser', 'flattop'],
                        default='hann')
    parser.add_argument('--beta', type=float, default=8.6,
                        help='Kaiser window β parameter (default: 8.6)')
    parser.add_argument('--axis', type=str, default=None,
                        help='Observation axis: X,Y,Z,diag,zeta,lattice or θ,φ')
    parser.add_argument('--look', type=str, default=None,
                        help='Beamformer look direction as θ,φ (degrees)')
    parser.add_argument('--array-size', type=int, default=8,
                        help='Number of UCA elements (default: 8)')
    parser.add_argument('--telescope', type=str, default=None,
                        help='Add movable telescope at θ,φ (degrees)')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    if run_dir is None or not run_dir.exists():
        print("No run found. Use --run NNNN")
        return

    positions, meta = load_positions(run_dir)
    grid = meta.get('grid_size', '?')
    n_nodes = len(positions)

    print("=" * 72)
    print("  RADIO OBSERVATORY")
    print(f"  Run {run_dir.name} | Grid {grid} | {n_nodes:,} nodes")
    print(f"  Mode: {args.mode} | Window: {args.window}")
    print("=" * 72)

    # Add telescope axes
    if args.telescope:
        parts = args.telescope.split(',')
        t_theta, t_phi = float(parts[0]), float(parts[1])
        t_axis = spherical_to_cart(t_theta, t_phi)
        name = f"T({t_theta:.0f},{t_phi:.0f})"
        FIXED_AXES[name] = t_axis
        print(f"  Telescope added: {name} → [{t_axis[0]:.4f}, {t_axis[1]:.4f}, {t_axis[2]:.4f}]")

    if args.mode == 'spectrum':
        mode_spectrum(positions, meta, args)
    elif args.mode == 'compare':
        mode_compare(positions, meta, args)
    elif args.mode == 'array':
        mode_array(positions, meta, args)
    elif args.mode == 'sweep':
        mode_sweep(positions, meta, args)
    elif args.mode == 'wannier':
        mode_wannier(positions, meta, args)
    elif args.mode == 'mobius':
        mode_mobius(positions, meta, args)
    elif args.mode == 'torus':
        mode_torus(positions, meta, args)

    print(f"\n{'='*72}")


if __name__ == '__main__':
    main()
