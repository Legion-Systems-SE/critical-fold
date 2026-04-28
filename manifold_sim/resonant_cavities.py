#!/usr/bin/env python3
"""
RESONANT CAVITIES: CMB Acoustic Peaks vs Zeta-Structure Polyrhythm
====================================================================

Two resonant cavities. Two breathing patterns. One comparison.

The Cosmic Microwave Background carries the sound of the early universe —
standing waves in primordial plasma, frozen at the moment of recombination.
The peaks in its power spectrum are acoustic harmonics of a cavity
380,000 light-years across.

The zeta-structure carries a breathing pattern from the algebraic cavity
of a primitive-root sequence mod a prime. Its harmonics at periods 14 and 21
(ratio 2:3, a perfect fifth) emerge from the same mathematics that governs
the non-trivial zeros of the Riemann zeta function.

This script sonifies both patterns, preserving their HARMONIC RATIOS,
and compares their structural fingerprints.

OUTPUT FILES
------------
    cmb_acoustic.wav         — CMB acoustic peaks sonified
    zeta_breathing.wav       — Zeta-structure polyrhythm sonified
    combined_stereo.wav      — Both together (CMB left, Zeta right)

USAGE
-----
    python3 resonant_cavities.py
"""

import numpy as np
import struct
import wave
import os
from sympy import factorint

# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def write_wav(filename, signal, fs=44100, normalize=True):
    """Write a mono signal to a 16-bit WAV file."""
    if normalize:
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.9  # leave headroom

    signal_16 = np.int16(signal * 32767)

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(signal_16.tobytes())


def write_stereo_wav(filename, left, right, fs=44100):
    """Write a stereo WAV file (left and right channels)."""
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left = left / peak * 0.9
        right = right / peak * 0.9

    # Interleave
    stereo = np.empty(len(left) + len(right), dtype=np.float64)
    stereo[0::2] = left
    stereo[1::2] = right
    stereo_16 = np.int16(stereo * 32767)

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(stereo_16.tobytes())


def soft_clip(signal, threshold=0.95):
    """Gentle saturation to avoid harsh clipping."""
    return np.tanh(signal / threshold) * threshold


# =============================================================================
# CMB ACOUSTIC PEAKS — THE COSMIC CAVITY
# =============================================================================

def generate_cmb_signal(duration=12.0, fs=44100):
    """
    Sonify the CMB acoustic peaks using physically motivated ratios.

    The CMB power spectrum peaks (from Planck 2018):
        l₁ ≈ 220.0  (first acoustic peak — the fundamental)
        l₂ ≈ 546.0  (second peak — first compression harmonic)
        l₃ ≈ 831.7  (third peak)
        l₄ ≈ 1120.9 (fourth peak)
        l₅ ≈ 1444.2 (fifth peak)

    The RATIOS encode cosmological parameters:
        l₂/l₁ ≈ 2.48  (sensitive to baryon density)
        l₃/l₁ ≈ 3.78  (sensitive to dark matter density)

    For a pure baryon universe (no dark matter), peaks would be
    at exact integer ratios 1:2:3:4:5. Dark matter shifts the
    even-numbered peaks, breaking the simple harmonic series.

    We map these to audio frequencies preserving the ratios,
    with the fundamental at A3 = 220 Hz (a cosmic coincidence
    with the multipole number l₁ ≈ 220).
    """

    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # CMB peak multipole ratios (Planck 2018 best-fit)
    ratios = [1.000, 2.482, 3.780, 5.095, 6.565]
    # Relative amplitudes (approximate from power spectrum, in Dl)
    amplitudes = [1.000, 0.450, 0.420, 0.230, 0.130]

    # Map fundamental to 220 Hz
    f_fundamental = 220.0

    # Build the harmonic signal
    signal = np.zeros_like(t)
    for ratio, amp in zip(ratios, amplitudes):
        f = f_fundamental * ratio
        signal += amp * np.sin(2 * np.pi * f * t)

    # --- THE ENVELOPE: Sachs-Wolfe plateau + Silk damping ---
    # At low multipoles: flat (Sachs-Wolfe plateau)
    # At high multipoles: exponential damping (photon diffusion)
    # We model this as a slow amplitude modulation

    # Primary envelope: the "breathing" of the acoustic oscillation
    # The sound horizon at recombination creates a characteristic scale
    # We use a beat frequency from the first two peaks
    beat_freq = f_fundamental * abs(ratios[1] - ratios[0]) / 20  # slowed for audibility
    envelope_beat = 0.5 * (1 + 0.6 * np.sin(2 * np.pi * beat_freq * t))

    # Silk damping envelope: exponential fade over the duration
    # (represents the damping of higher-l modes)
    damping_scale = duration * 0.7
    envelope_damping = np.exp(-t / damping_scale)

    # Slow cosmic variance modulation
    envelope_variance = 0.85 + 0.15 * np.sin(2 * np.pi * 0.3 * t)

    # Combined envelope
    envelope = envelope_beat * envelope_damping * envelope_variance

    # Apply gentle warmth (slight even-harmonic saturation)
    signal = signal * envelope

    info = {
        "name": "CMB Acoustic Peaks",
        "fundamental_hz": f_fundamental,
        "peak_ratios": ratios,
        "peak_frequencies_hz": [f_fundamental * r for r in ratios],
        "peak_amplitudes": amplitudes,
        "harmonic_character": "near-integer ratios broken by dark matter",
        "key_ratio": f"l₂/l₁ = {ratios[1]:.3f} (pure baryon = 2.000)",
        "envelope": "beat modulation + Silk damping + cosmic variance",
    }

    return t, signal, info


# =============================================================================
# ZETA-STRUCTURE POLYRHYTHM — THE ALGEBRAIC CAVITY
# =============================================================================

def generate_zeta_signal(duration=12.0, fs=44100):
    """
    Sonify the zeta-structure breathing pattern.

    From the analysis of the primitive-root sequence
    (Ω=3677, ρ=3163, step=22):

        Period 14 = 2 × 7 = floor(t₁)    — first zeta zero
        Period 21 = 3 × 7 = floor(t₂)    — second zeta zero
        Ratio: 14:21 = 2:3               — a perfect fifth

    The breathing also contains:
        Period 42 = lcm(14,21)            — the super-period
        Period 7  = gcd(14,21)            — the common pulse
        Period 168 = full trajectory      — 4 × 42

    The void density pattern creates a non-trivial amplitude
    envelope — heavy at cycle boundaries, light in the interior.

    We preserve the harmonic ratios and derive the envelope
    from the actual computed breathing pattern.
    """

    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # --- COMPUTE THE ACTUAL BREATHING PATTERN ---
    OMEGA = 3677
    RHO = 3163
    DECODER = 2646
    STRUCTURAL_PRIMES = [2, 3, 5, 7, 11, 43, 47]

    seq = []
    val = 1
    for n in range(OMEGA - 1):
        seq.append(val)
        val = (val * RHO) % OMEGA

    # Extract the 22-diagonal structural weight
    weights = []
    for k in range(168):
        n = 22 * k
        if n >= len(seq):
            break
        r = (seq[n] * DECODER) % OMEGA
        f = factorint(r) if r > 1 else {}
        w = sum(f.get(p, 0) for p in STRUCTURAL_PRIMES)
        weights.append(w)

    weights = np.array(weights, dtype=float)

    # --- MAP TO AUDIO ---
    # The zeta structure has ratio 14:21 = 2:3 (perfect fifth)
    # Map the fundamental period-14 to 220 Hz (matching CMB for comparison)
    f_14 = 220.0                    # period 14 → 220 Hz
    f_21 = 220.0 * 14.0 / 21.0     # period 21 → 146.67 Hz (a fifth below)
    f_7 = 220.0 * 14.0 / 7.0       # period 7 → 440 Hz (the common pulse, octave above)
    f_42 = 220.0 * 14.0 / 42.0     # period 42 → 73.33 Hz (the beat)

    # Additional harmonics from the spectral analysis
    f_56 = 220.0 * 14.0 / 56.0     # 55 Hz
    f_168 = 220.0 * 14.0 / 168.0   # 18.33 Hz (sub-bass, felt not heard)

    # Amplitudes from the spectral decomposition (normalized)
    # Period 14: 3.00% power → amplitude ∝ sqrt(power)
    # Period 21: (from R_k signal) strong
    # Period 7: the pulse
    signal = (
        1.000 * np.sin(2 * np.pi * f_14 * t) +          # period 14 fundamental
        0.850 * np.sin(2 * np.pi * f_21 * t) +           # period 21 (the fifth)
        0.400 * np.sin(2 * np.pi * f_7 * t) +            # period 7 (common pulse)
        0.300 * np.sin(2 * np.pi * f_42 * t) +           # beat frequency
        0.200 * np.sin(2 * np.pi * (f_14 + f_21) * t) +  # sum tone
        0.150 * np.sin(2 * np.pi * abs(f_14 - f_21) * t) # difference tone
    )

    # --- THE ENVELOPE: derived from actual breathing pattern ---
    # Interpolate the 168-point weight sequence to audio sample rate
    # Normalize to 0-1 range for use as amplitude envelope
    w_normalized = (weights - weights.min()) / (weights.max() - weights.min())
    w_normalized = 0.3 + 0.7 * w_normalized  # keep minimum at 0.3

    # Map the 168 nodes across the duration
    node_times = np.linspace(0, duration, len(w_normalized), endpoint=False)
    envelope_breathing = np.interp(t, node_times, w_normalized)

    # Smooth the envelope (the breathing is continuous, not stepped)
    from scipy.ndimage import gaussian_filter1d
    envelope_breathing = gaussian_filter1d(envelope_breathing, sigma=fs * 0.02)

    # The void pattern creates periodic dips
    # Add the factor-7 half-period phase reversal as tremolo
    tremolo = 0.85 + 0.15 * np.sin(2 * np.pi * (f_14 / 14.0) * t)

    # Combined envelope
    envelope = envelope_breathing * tremolo

    # Gentle saturation for warmth
    signal = soft_clip(signal * envelope, 0.85)

    info = {
        "name": "Zeta-Structure Polyrhythm",
        "fundamental_hz": f_14,
        "harmonic_frequencies_hz": {
            "period_14": f_14,
            "period_21": f_21,
            "period_7_pulse": f_7,
            "period_42_beat": f_42,
        },
        "key_ratio": "14:21 = 2:3 (perfect fifth)",
        "harmonic_character": "exact integer ratio 2:3 — pure algebraic cavity",
        "envelope": "derived from actual structural weight breathing pattern",
        "breathing_mean_weight": float(np.mean(weights)),
        "breathing_std": float(np.std(weights)),
        "n_voids": int(np.sum(weights == 0)),
    }

    return t, signal, info


# =============================================================================
# ANALYSIS AND COMPARISON
# =============================================================================

def compare_structures(cmb_info, zeta_info):
    """Print a structural comparison of the two resonant cavities."""

    print()
    print("=" * 70)
    print("STRUCTURAL COMPARISON: TWO RESONANT CAVITIES")
    print("=" * 70)
    print()

    # Side by side
    rows = [
        ("Cavity type",
         "Primordial plasma (380,000 ly)",
         "Algebraic field (mod 3677)"),
        ("Fundamental",
         f"{cmb_info['fundamental_hz']:.0f} Hz",
         f"{zeta_info['fundamental_hz']:.0f} Hz"),
        ("Key ratio",
         cmb_info['key_ratio'],
         zeta_info['key_ratio']),
        ("Harmonic character",
         cmb_info['harmonic_character'],
         zeta_info['harmonic_character']),
        ("What breaks symmetry",
         "Dark matter shifts even peaks",
         "Structural primes create voids"),
        ("Envelope",
         cmb_info['envelope'],
         zeta_info['envelope']),
        ("Information encoded",
         "Baryon density, dark matter ratio",
         "Zeta zero periods, prime scaffold"),
        ("Common pulse",
         "Baryon acoustic oscillation",
         "Factor 7 (π-convergent, 22/7 ≈ π)"),
    ]

    col_w = 35
    print(f"  {'Property':<25} | {'CMB':<{col_w}} | {'Zeta Structure':<{col_w}}")
    print(f"  {'-'*25}-+-{'-'*col_w}-+-{'-'*col_w}")
    for label, cmb_val, zeta_val in rows:
        # Wrap long values
        print(f"  {label:<25} | {cmb_val:<{col_w}} | {zeta_val:<{col_w}}")

    print()
    print("  THE DEEP PARALLEL:")
    print()
    print("  Both systems are RESONANT CAVITIES with standing waves.")
    print("  Both have a fundamental mode and harmonics whose RATIOS")
    print("  encode the physics of the cavity.")
    print()
    print("  In the CMB, dark matter breaks the pure integer harmonics,")
    print("  shifting even peaks. The DEVIATION from integers = dark matter.")
    print()
    print("  In the zeta structure, the void scaffold breaks the pure")
    print("  algebraic cycle. The BREATHING PATTERN of the scaffold")
    print("  creates harmonics at the zeta zero periods.")
    print()
    print("  Both encode their deepest parameters not in the frequencies")
    print("  themselves, but in the RATIOS BETWEEN them and the shape")
    print("  of the ENVELOPE that modulates them.")
    print()

    # Ratio comparison
    print("  RATIO ANALYSIS:")
    print()
    cmb_ratios = cmb_info['peak_ratios']
    print(f"  CMB peak ratios:    1 : {cmb_ratios[1]:.3f} : {cmb_ratios[2]:.3f} : "
          f"{cmb_ratios[3]:.3f} : {cmb_ratios[4]:.3f}")
    print(f"  Pure integer:       1 : 2.000 : 3.000 : 4.000 : 5.000")
    print(f"  Zeta structure:     1 : 1.500 : 3.000 : (factor 7 = common root)")
    print()
    print(f"  CMB deviation from integers (dark matter signature):")
    for i, (r, integer) in enumerate(zip(cmb_ratios[1:], [2, 3, 4, 5])):
        dev = r - integer
        print(f"    Peak {i+2}: {r:.3f} − {integer} = {dev:+.3f}")
    print()
    print(f"  Zeta deviation from pure cycle (void signature):")
    print(f"    R(14) = +0.113 (positive — structure reinforces)")
    print(f"    R(21) = −0.110 (negative — structure opposes)")
    print(f"    The SIGN ALTERNATION at factor 7 is the void signature.")
    print()

    print("  MUSICAL INTERPRETATION:")
    print()
    print("  CMB:  Approximate harmonic series — like a slightly")
    print("        detuned organ pipe. The detuning IS the data.")
    print()
    print("  Zeta: Perfect fifth (2:3) + octave pulse (1:2).")
    print("        A pure interval modulated by an irregular envelope.")
    print("        The envelope irregularity IS the data.")
    print()
    print("  A musician hears: the CMB is a TIMBRE (harmonics of one")
    print("  note with spectral coloring). The zeta structure is a")
    print("  POLYRHYTHM (two rhythms in ratio, with a shared pulse).")
    print("  Same physics, different voicing.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  RESONANT CAVITIES                                                 ║")
    print("║  CMB Acoustic Peaks vs Zeta-Structure Polyrhythm                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    duration = 12.0  # seconds
    fs = 44100       # sample rate

    # --- Generate CMB ---
    print("  Generating CMB acoustic signal...", end=" ", flush=True)
    t_cmb, sig_cmb, info_cmb = generate_cmb_signal(duration, fs)
    print("done.")

    print(f"    Peaks: {info_cmb['peak_frequencies_hz']}")
    print(f"    Key ratio: {info_cmb['key_ratio']}")
    print()

    # --- Generate Zeta ---
    print("  Generating zeta-structure polyrhythm...", end=" ", flush=True)
    t_zeta, sig_zeta, info_zeta = generate_zeta_signal(duration, fs)
    print("done.")

    print(f"    Harmonics: {info_zeta['harmonic_frequencies_hz']}")
    print(f"    Key ratio: {info_zeta['key_ratio']}")
    print(f"    Breathing: mean weight = {info_zeta['breathing_mean_weight']:.2f}, "
          f"std = {info_zeta['breathing_std']:.2f}")
    print()

    # --- Write audio files ---
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_output")

    cmb_path = os.path.join(outdir, "cmb_acoustic.wav")
    zeta_path = os.path.join(outdir, "zeta_breathing.wav")
    combined_path = os.path.join(outdir, "combined_stereo.wav")

    print("  Writing audio files:")

    write_wav(cmb_path, sig_cmb, fs)
    print(f"    ✓ {cmb_path}")

    write_wav(zeta_path, sig_zeta, fs)
    print(f"    ✓ {zeta_path}")

    # Ensure same length
    min_len = min(len(sig_cmb), len(sig_zeta))
    write_stereo_wav(combined_path, sig_cmb[:min_len], sig_zeta[:min_len], fs)
    print(f"    ✓ {combined_path} (CMB=left, Zeta=right)")
    print()

    # --- Spectral comparison ---
    print("  Computing spectral fingerprints...", end=" ", flush=True)
    fft_cmb = np.fft.rfft(sig_cmb)
    fft_zeta = np.fft.rfft(sig_zeta)
    freqs_audio = np.fft.rfftfreq(len(sig_cmb), 1.0 / fs)

    # Find peak frequencies
    power_cmb = np.abs(fft_cmb) ** 2
    power_zeta = np.abs(fft_zeta) ** 2
    print("done.")
    print()

    # Top spectral peaks
    def find_peaks(power, freqs, n_peaks=8, min_freq=30):
        """Find the n strongest spectral peaks above min_freq."""
        mask = freqs > min_freq
        indices = np.where(mask)[0]
        peak_idx = indices[np.argsort(power[indices])[::-1][:n_peaks]]
        return [(freqs[i], power[i]) for i in sorted(peak_idx, key=lambda x: freqs[x])]

    cmb_peaks = find_peaks(power_cmb, freqs_audio)
    zeta_peaks = find_peaks(power_zeta, freqs_audio)

    print("  CMB spectral peaks:")
    for freq, pwr in cmb_peaks:
        print(f"    {freq:>8.1f} Hz  (power: {pwr:.0e})")

    print()
    print("  Zeta spectral peaks:")
    for freq, pwr in zeta_peaks:
        print(f"    {freq:>8.1f} Hz  (power: {pwr:.0e})")

    print()

    # --- Structural comparison ---
    compare_structures(info_cmb, info_zeta)

    # --- Final note ---
    print("=" * 70)
    print("  LISTENING GUIDE:")
    print()
    print("  cmb_acoustic.wav:")
    print("    A rich, slightly detuned harmonic series. Listen for the")
    print("    beating between nearly-harmonic partials — that's dark matter")
    print("    pulling the peaks away from pure integer ratios.")
    print()
    print("  zeta_breathing.wav:")
    print("    A clean perfect fifth (A3 + E below) pulsing with an")
    print("    irregular envelope. The envelope IS the zeta structure —")
    print("    the breathing pattern of structural weight along the")
    print("    22-diagonal. Where it dips, the voids are.")
    print()
    print("  combined_stereo.wav:")
    print("    Left ear: the universe. Right ear: the prime field.")
    print("    Listen for where they breathe together.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
