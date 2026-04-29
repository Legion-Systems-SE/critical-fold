"""
Waveform Decomposition Test — Square vs Sawtooth
=================================================
Tests whether the engine's crystallographic spectrum shows structurally
different harmonic content along face axes vs diagonal axes.

Square wave prediction: face axes → power concentrated at {1,3,5,7}
                        with 1/n amplitude decay
Sawtooth prediction:    diagonal → power spreads to {9,15,27,45}

Null hypothesis: all axes show the same harmonic ratios.

Usage:
    python3 manifold_sim/waveform_test.py
"""

import numpy as np
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

RUNS = {
    45:  SCRIPT_DIR / 'runs_emergent' / '0763',
    65:  SCRIPT_DIR / 'runs_emergent' / '0761',
    89:  SCRIPT_DIR / 'runs_emergent' / '0762',
}

TARGET_FREQS = [1, 3, 5, 7, 9, 15, 27, 45]
FACE_SET = {1, 3, 5, 7}
DIAG_SET = {9, 15, 27, 45}

N_ANGLES = 720


def load_positions(run_dir):
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.float64)
    meta = json.load(open(run_dir / 'meta.json'))
    gs = meta.get('grid_size', 89)
    pos = (registry - gs / 2.0) * (20.0 / gs)
    return pos, gs, len(registry)


def get_axes():
    axes = {
        'X':    np.array([1.0, 0.0, 0.0]),
        'Y':    np.array([0.0, 1.0, 0.0]),
        'Z':    np.array([0.0, 0.0, 1.0]),
        'diag': np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }
    return axes


def orthonormal_frame(axis):
    a = axis / np.linalg.norm(axis)
    if abs(a[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def rotational_spectrum(positions, axis, n_angles=N_ANGLES, n_bins=200):
    e1, e2 = orthonormal_frame(axis)
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    variance_signal = np.zeros(n_angles)

    for i, theta in enumerate(thetas):
        view = np.cos(theta) * e1 + np.sin(theta) * e2
        proj = positions @ view
        counts, _ = np.histogram(proj, bins=n_bins)
        variance_signal[i] = np.var(counts)

    fft = np.fft.rfft(variance_signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n_angles, d=1.0/n_angles)
    return thetas, variance_signal, freqs, power


def ideal_square_amplitudes(freqs):
    """Ideal square wave: amplitude = 1/n for odd n, 0 for even n."""
    amp = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        f_int = int(round(f))
        if f_int > 0 and f_int % 2 == 1:
            amp[i] = 1.0 / f_int
    return amp


def main():
    print("=" * 78)
    print("  WAVEFORM DECOMPOSITION TEST — Square vs Sawtooth")
    print("  Testing: does the field distinguish two harmonic types by axis?")
    print("=" * 78)
    print()
    print("  Target frequencies:  {1, 3, 5, 7, 9, 15, 27, 45}")
    print("  Face set (square):   {1, 3, 5, 7}")
    print("  Diagonal set:        {9, 15, 27, 45}")
    print("  Null hypothesis:     same power ratios on all axes")
    print()

    all_results = {}

    for grid_size in sorted(RUNS.keys()):
        run_dir = RUNS[grid_size]
        if not run_dir.exists():
            print(f"  [SKIP] Grid {grid_size}: run dir not found")
            continue

        positions, gs, n_nodes = load_positions(run_dir)
        print(f"{'=' * 78}")
        print(f"  GRID {grid_size}  |  {n_nodes} nodes  |  {run_dir.name}")
        print(f"{'=' * 78}")

        axes = get_axes()
        grid_results = {}

        for ax_name, axis in axes.items():
            thetas, vsig, freqs, power = rotational_spectrum(positions, axis)

            # Extract power at target frequencies
            target_power = {}
            for tf in TARGET_FREQS:
                idx = int(round(tf))
                if idx < len(power):
                    target_power[tf] = power[idx]
                else:
                    target_power[tf] = 0.0

            total_target = sum(target_power.values())
            if total_target == 0:
                total_target = 1e-30

            face_power = sum(target_power[f] for f in FACE_SET)
            diag_power = sum(target_power[f] for f in DIAG_SET)
            squareness = face_power / (face_power + diag_power) if (face_power + diag_power) > 0 else 0

            # Amplitude at each frequency (sqrt of power)
            amps = {f: np.sqrt(p) for f, p in target_power.items()}
            max_amp = max(amps.values()) if max(amps.values()) > 0 else 1.0
            norm_amps = {f: a / max_amp for f, a in amps.items()}

            # Compare to ideal square wave (1/n decay)
            ideal = {f: 1.0 / f if f > 0 else 1.0 for f in TARGET_FREQS}
            max_ideal = max(ideal.values())
            ideal_norm = {f: v / max_ideal for f, v in ideal.items()}

            # Correlation with ideal square wave
            obs_vec = np.array([norm_amps[f] for f in TARGET_FREQS])
            ideal_vec = np.array([ideal_norm[f] for f in TARGET_FREQS])
            if np.linalg.norm(obs_vec) > 0 and np.linalg.norm(ideal_vec) > 0:
                corr = np.dot(obs_vec, ideal_vec) / (np.linalg.norm(obs_vec) * np.linalg.norm(ideal_vec))
            else:
                corr = 0.0

            grid_results[ax_name] = {
                'target_power': target_power,
                'amps': amps,
                'norm_amps': norm_amps,
                'squareness': squareness,
                'sq_correlation': corr,
                'face_power': face_power,
                'diag_power': diag_power,
            }

            # Print detailed results
            print(f"\n  Axis: {ax_name:4s}  |  Squareness: {squareness:.4f}  |  "
                  f"Square-wave corr: {corr:.4f}")
            print(f"  {'─' * 70}")
            print(f"  {'freq':>4s}  {'power':>12s}  {'amplitude':>10s}  "
                  f"{'norm_amp':>8s}  {'ideal_sq':>8s}  {'set':>6s}")
            print(f"  {'─' * 70}")
            for f in TARGET_FREQS:
                p = target_power[f]
                a = amps[f]
                na = norm_amps[f]
                iq = ideal_norm[f]
                s = "FACE" if f in FACE_SET else "DIAG"
                bar = "█" * int(na * 40)
                print(f"  {f:4d}  {p:12.2e}  {a:10.2f}  {na:8.4f}  "
                      f"{iq:8.4f}  {s:>6s}  {bar}")

        all_results[grid_size] = grid_results
        print()

    # Cross-grid summary
    print()
    print("=" * 78)
    print("  CROSS-GRID SUMMARY")
    print("=" * 78)
    print()
    print(f"  {'grid':>4s}  {'axis':>4s}  {'squareness':>10s}  {'sq_corr':>8s}  "
          f"{'face_pwr':>10s}  {'diag_pwr':>10s}  {'ratio':>8s}")
    print(f"  {'─' * 70}")

    for grid_size in sorted(all_results.keys()):
        for ax_name in ['X', 'Y', 'Z', 'diag']:
            r = all_results[grid_size][ax_name]
            ratio = r['face_power'] / r['diag_power'] if r['diag_power'] > 0 else float('inf')
            print(f"  {grid_size:4d}  {ax_name:>4s}  {r['squareness']:10.4f}  "
                  f"{r['sq_correlation']:8.4f}  {r['face_power']:10.2e}  "
                  f"{r['diag_power']:10.2e}  {ratio:8.2f}")
        print()

    # Axis-averaged squareness
    print()
    print(f"  AXIS-AVERAGED SQUARENESS (across all grids):")
    print(f"  {'─' * 50}")
    for ax_name in ['X', 'Y', 'Z', 'diag']:
        sq_vals = [all_results[g][ax_name]['squareness'] for g in sorted(all_results.keys())]
        co_vals = [all_results[g][ax_name]['sq_correlation'] for g in sorted(all_results.keys())]
        mean_sq = np.mean(sq_vals)
        std_sq = np.std(sq_vals)
        mean_co = np.mean(co_vals)
        label = "FACE" if ax_name in ('X', 'Y', 'Z') else "DIAGONAL"
        print(f"  {ax_name:>4s} ({label:>8s}):  squareness = {mean_sq:.4f} ± {std_sq:.4f}  "
              f"  sq_corr = {mean_co:.4f}")

    # The verdict
    print()
    print("=" * 78)
    face_sq = []
    diag_sq = []
    for g in all_results:
        for ax in ['X', 'Y', 'Z']:
            face_sq.append(all_results[g][ax]['squareness'])
        diag_sq.append(all_results[g]['diag']['squareness'])

    face_mean = np.mean(face_sq)
    diag_mean = np.mean(diag_sq)
    separation = face_mean - diag_mean

    print(f"  Face axes mean squareness:     {face_mean:.4f}")
    print(f"  Diagonal mean squareness:      {diag_mean:.4f}")
    print(f"  Separation (face - diagonal):  {separation:.4f}")
    print()

    if separation > 0.05:
        print("  RESULT: STRUCTURAL DISTINCTION DETECTED")
        print("  Face axes carry more square-wave content than diagonal.")
        print("  The field geometry provides a physical basis for binary classification.")
    elif separation > 0.01:
        print("  RESULT: WEAK DISTINCTION")
        print("  Small difference detected. May need more rotation samples.")
    else:
        print("  RESULT: NULL HYPOTHESIS SURVIVES")
        print("  No significant difference between face and diagonal spectra.")
        print("  The binary bucket classification lacks structural support.")

    print("=" * 78)


if __name__ == '__main__':
    main()
