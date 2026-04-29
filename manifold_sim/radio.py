"""
Crystallographic Radio — Sonification of Field Resonances
==========================================================
Tunes into the 8 grid-invariant frequencies {1,3,5,7,9,15,27,45}
using amplitudes measured from the field's rotational moiré spectrum.

Each rotation axis (X, Y, Z, diagonal) produces a different filter.
Sawtooth test: broadband input → field filter → what survives.

Output (to audio_output/):
    radio_{axis}_full.wav       — all 8 frequencies, field-weighted
    radio_{axis}_square.wav     — face set {1,3,5,7} only
    radio_{axis}_extended.wav   — diagonal set {9,15,27,45} only
    radio_{axis}_sawtooth.wav   — sawtooth through field filter
    radio_face_vs_diag.wav      — stereo comparison (X left, diag right)

Usage:
    python3 manifold_sim/radio.py
    python3 manifold_sim/radio.py --run 0761 --base-hz 110
    python3 manifold_sim/radio.py --run 0762 --base-hz 55 --duration 8
"""

import numpy as np
import json
import wave
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

STRUCTURAL_FREQS = [1, 3, 5, 7, 9, 15, 27, 45]
FACE_SET = {1, 3, 5, 7}
DIAG_SET = {9, 15, 27, 45}

FS = 44100
N_ANGLES = 720


# ── Audio utilities ──────────────────────────────────────────────────

def write_wav(filename, signal, fs=FS):
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85
    sig16 = np.int16(signal * 32767)
    with wave.open(str(filename), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(sig16.tobytes())


def write_stereo_wav(filename, left, right, fs=FS):
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left = left / peak * 0.85
        right = right / peak * 0.85
    stereo = np.empty(len(left) + len(right), dtype=np.float64)
    stereo[0::2] = left
    stereo[1::2] = right
    sig16 = np.int16(stereo * 32767)
    with wave.open(str(filename), 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(sig16.tobytes())


# ── Field analysis (from waveform_test.py) ───────────────────────────

def load_positions(run_dir):
    registry = np.load(str(run_dir / 'registry.npy')).astype(np.float64)
    meta = json.load(open(run_dir / 'meta.json'))
    gs = meta.get('grid_size', 89)
    pos = (registry - gs / 2.0) * (20.0 / gs)
    return pos, meta


def orthonormal_frame(axis):
    a = axis / np.linalg.norm(axis)
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def rotational_spectrum(positions, axis):
    e1, e2 = orthonormal_frame(axis)
    thetas = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    variance_signal = np.zeros(N_ANGLES)
    for i, theta in enumerate(thetas):
        view = np.cos(theta) * e1 + np.sin(theta) * e2
        proj = positions @ view
        counts, _ = np.histogram(proj, bins=200)
        variance_signal[i] = np.var(counts)
    fft = np.fft.rfft(variance_signal)
    return fft


def extract_field_response(fft_result):
    """Extract amplitude and phase at each structural frequency."""
    response = {}
    for f in STRUCTURAL_FREQS:
        if f < len(fft_result):
            c = fft_result[f]
            response[f] = {'amplitude': np.abs(c), 'phase': np.angle(c)}
        else:
            response[f] = {'amplitude': 0.0, 'phase': 0.0}
    return response


# ── Synthesis ────────────────────────────────────────────────────────

def synthesize_tones(response, base_hz, duration, freq_set=None):
    """Additive synthesis using field amplitudes at structural frequencies."""
    t = np.linspace(0, duration, int(FS * duration), endpoint=False)
    signal = np.zeros_like(t)

    if freq_set is None:
        freq_set = set(STRUCTURAL_FREQS)

    max_amp = max(r['amplitude'] for r in response.values())
    if max_amp == 0:
        return signal

    for f in STRUCTURAL_FREQS:
        if f not in freq_set:
            continue
        hz = base_hz * f
        if hz >= FS / 2:
            continue
        amp = response[f]['amplitude'] / max_amp
        phi = response[f]['phase']
        signal += amp * np.sin(2 * np.pi * hz * t + phi)

    return signal


def synthesize_sawtooth_filtered(response, base_hz, duration):
    """Generate sawtooth, filter through field response."""
    n_samples = int(FS * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Generate sawtooth via additive synthesis (all harmonics up to Nyquist)
    sawtooth = np.zeros_like(t)
    max_harmonic = int(FS / (2 * base_hz))
    for n in range(1, max_harmonic + 1):
        sawtooth += ((-1) ** (n + 1)) * np.sin(2 * np.pi * n * base_hz * t) / n

    # FFT the sawtooth
    saw_fft = np.fft.rfft(sawtooth)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/FS)

    # Build transfer function from field response
    max_amp = max(r['amplitude'] for r in response.values())
    if max_amp == 0:
        return sawtooth, sawtooth

    transfer = np.zeros(len(saw_fft), dtype=complex)
    for f in STRUCTURAL_FREQS:
        target_hz = base_hz * f
        if target_hz >= FS / 2:
            continue
        idx = np.argmin(np.abs(freqs - target_hz))
        amp = response[f]['amplitude'] / max_amp
        phi = response[f]['phase']
        transfer[idx] = amp * np.exp(1j * phi)

    # Apply filter
    filtered_fft = saw_fft * transfer
    filtered = np.fft.irfft(filtered_fft, n=n_samples)

    return sawtooth, filtered


# ── Main ─────────────────────────────────────────────────────────────

def resolve_run_dir(run_arg, base='runs_emergent'):
    runs_dir = SCRIPT_DIR / base
    if run_arg:
        if run_arg.isdigit():
            return runs_dir / f"{int(run_arg):04d}"
        return Path(run_arg)
    latest = runs_dir / "latest.txt"
    if latest.exists():
        return runs_dir / latest.read_text().strip()
    return None


def main():
    parser = argparse.ArgumentParser(description='Crystallographic Radio')
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--base-hz', type=float, default=110.0,
                        help='Base frequency in Hz (default: 110 = A2)')
    parser.add_argument('--duration', type=float, default=6.0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    if run_dir is None or not run_dir.exists():
        print("No run found. Specify --run NNNN")
        return

    positions, meta = load_positions(run_dir)
    grid = meta.get('grid_size', '?')
    n_nodes = len(positions)

    print("=" * 72)
    print("  CRYSTALLOGRAPHIC RADIO")
    print(f"  Run {run_dir.name} | Grid {grid} | {n_nodes} nodes")
    print(f"  Base frequency: {args.base_hz} Hz | Duration: {args.duration}s")
    print("=" * 72)
    print()

    # Map frequencies to musical notes
    print("  DIAL POSITIONS:")
    print(f"  {'freq':>4s}  {'Hz':>8s}  {'set':>6s}  {'harmonic':>10s}")
    print(f"  {'─' * 50}")
    for f in STRUCTURAL_FREQS:
        hz = args.base_hz * f
        s = "FACE" if f in FACE_SET else "DIAG"
        print(f"  {f:4d}  {hz:8.1f}  {s:>6s}  {f:>3d}th harmonic of {args.base_hz:.0f}")
    print()

    # Axes
    axes = {
        'X':    np.array([1.0, 0.0, 0.0]),
        'Y':    np.array([0.0, 1.0, 0.0]),
        'Z':    np.array([0.0, 0.0, 1.0]),
        'diag': np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }

    out_dir = SCRIPT_DIR / 'audio_output'
    out_dir.mkdir(exist_ok=True)

    all_responses = {}
    all_full_signals = {}

    for ax_name, axis in axes.items():
        print(f"  Scanning axis {ax_name}...", end='', flush=True)
        fft = rotational_spectrum(positions, axis)
        response = extract_field_response(fft)
        all_responses[ax_name] = response
        print(" done")

        # Print field response
        print(f"\n  AXIS {ax_name} — FIELD RESPONSE:")
        print(f"  {'freq':>4s}  {'amplitude':>10s}  {'phase/π':>8s}  {'norm':>6s}  {'set':>5s}  {'level':>40s}")
        print(f"  {'─' * 72}")

        max_a = max(r['amplitude'] for r in response.values())
        for f in STRUCTURAL_FREQS:
            a = response[f]['amplitude']
            p = response[f]['phase'] / np.pi
            norm = a / max_a if max_a > 0 else 0
            s = "FACE" if f in FACE_SET else "DIAG"
            bar = "█" * int(norm * 40)
            print(f"  {f:4d}  {a:10.1f}  {p:+8.4f}  {norm:6.3f}  {s:>5s}  {bar}")
        print()

        # Synthesize
        sig_full = synthesize_tones(response, args.base_hz, args.duration)
        sig_face = synthesize_tones(response, args.base_hz, args.duration, FACE_SET)
        sig_diag = synthesize_tones(response, args.base_hz, args.duration, DIAG_SET)
        _, sig_saw = synthesize_sawtooth_filtered(response, args.base_hz, args.duration)

        all_full_signals[ax_name] = sig_full

        # Write WAVs
        write_wav(out_dir / f'radio_{ax_name}_full.wav', sig_full)
        write_wav(out_dir / f'radio_{ax_name}_square.wav', sig_face)
        write_wav(out_dir / f'radio_{ax_name}_extended.wav', sig_diag)
        write_wav(out_dir / f'radio_{ax_name}_sawtooth.wav', sig_saw)

        print(f"  Written: radio_{ax_name}_*.wav (full, square, extended, sawtooth)")
        print()

    # Stereo comparison: face axis (X) left, diagonal right
    if 'X' in all_full_signals and 'diag' in all_full_signals:
        write_stereo_wav(
            out_dir / 'radio_face_vs_diag.wav',
            all_full_signals['X'],
            all_full_signals['diag']
        )
        print("  Written: radio_face_vs_diag.wav (X=left, diag=right)")

    # Also: raw sawtooth for reference
    n_samples = int(FS * args.duration)
    t = np.linspace(0, args.duration, n_samples, endpoint=False)
    raw_saw = np.zeros_like(t)
    max_h = int(FS / (2 * args.base_hz))
    for n in range(1, min(max_h + 1, 200)):
        raw_saw += ((-1) ** (n + 1)) * np.sin(2 * np.pi * n * args.base_hz * t) / n
    write_wav(out_dir / 'radio_sawtooth_raw.wav', raw_saw)
    print("  Written: radio_sawtooth_raw.wav (unfiltered reference)")

    # Phase analysis
    print()
    print("=" * 72)
    print("  PHASE STRUCTURE (radians / π)")
    print("=" * 72)
    print(f"  {'freq':>4s}", end='')
    for ax in axes:
        print(f"  {ax:>10s}", end='')
    print()
    print(f"  {'─' * 52}")
    for f in STRUCTURAL_FREQS:
        print(f"  {f:4d}", end='')
        for ax in axes:
            p = all_responses[ax][f]['phase'] / np.pi
            print(f"  {p:+10.4f}", end='')
        print()

    # Summary: squareness from audio perspective
    print()
    print("=" * 72)
    print("  SPECTRAL ENERGY DISTRIBUTION")
    print("=" * 72)
    for ax_name in axes:
        r = all_responses[ax_name]
        face_e = sum(r[f]['amplitude'] ** 2 for f in FACE_SET)
        diag_e = sum(r[f]['amplitude'] ** 2 for f in DIAG_SET)
        total = face_e + diag_e
        sq = face_e / total if total > 0 else 0
        print(f"  {ax_name:>4s}:  face={face_e/total:.3f}  diag={diag_e/total:.3f}  "
              f"squareness={sq:.4f}")

    print()
    print(f"  Audio files in: {out_dir}")
    print("=" * 72)


if __name__ == '__main__':
    main()
