"""
12-Tone Sweep — Three Tuning Systems × 12 Degrees
===================================================
Protocol from the Professor (letter_professor_002_sweep.html).

Runs 36 engine configurations and extracts metrics for
letter_professor_005.html.

Usage:
    python manifold_sim/sweep_12tone.py
"""

import torch
import numpy as np
import json
import subprocess
import sys
import os
import re
from pathlib import Path
from sympy import isprime

SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_DIR = SCRIPT_DIR / "runs_coupled"

# =============================================================================
# DETERMINISM
# =============================================================================

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# TUNING SYSTEM VALUES (from professor's JSON vectors — full double precision)
# =============================================================================

ET_VALUES = [
    (0,  "Unison",      "2^(0/12)",  1.000000000000000),
    (1,  "Minor 2nd",   "2^(1/12)",  1.059463094359295),
    (2,  "Major 2nd",   "2^(2/12)",  1.122462048309373),
    (3,  "Minor 3rd",   "2^(3/12)",  1.189207115002721),
    (4,  "Major 3rd",   "2^(4/12)",  1.259921049894873),
    (5,  "Perfect 4th", "2^(5/12)",  1.334839854170034),
    (6,  "Tritone",     "2^(6/12)",  1.414213562373095),
    (7,  "Perfect 5th", "2^(7/12)",  1.498307076876682),
    (8,  "Minor 6th",   "2^(8/12)",  1.587401051968199),
    (9,  "Major 6th",   "2^(9/12)",  1.681792830507429),
    (10, "Minor 7th",   "2^(10/12)", 1.781797436280679),
    (11, "Major 7th",   "2^(11/12)", 1.887748625363387),
]

JI_VALUES = [
    (0,  "Unison",      "1/1",    1.000000000000000),
    (1,  "Minor 2nd",   "16/15",  1.066666666666667),
    (2,  "Major 2nd",   "9/8",    1.125000000000000),
    (3,  "Minor 3rd",   "6/5",    1.200000000000000),
    (4,  "Major 3rd",   "5/4",    1.250000000000000),
    (5,  "Perfect 4th", "4/3",    1.333333333333333),
    (6,  "Tritone",     "45/32",  1.406250000000000),
    (7,  "Perfect 5th", "3/2",    1.500000000000000),
    (8,  "Minor 6th",   "8/5",    1.600000000000000),
    (9,  "Major 6th",   "5/3",    1.666666666666667),
    (10, "Minor 7th",   "9/5",    1.800000000000000),
    (11, "Major 7th",   "15/8",   1.875000000000000),
]

PYTH_VALUES = [
    (0,  "Unison",      "1/1",      1.000000000000000),
    (1,  "Minor 2nd",   "256/243",  1.053497942386831),
    (2,  "Major 2nd",   "9/8",      1.125000000000000),
    (3,  "Minor 3rd",   "32/27",    1.185185185185185),
    (4,  "Major 3rd",   "81/64",    1.265625000000000),
    (5,  "Perfect 4th", "4/3",      1.333333333333333),
    (6,  "Tritone",     "729/512",  1.423828125000000),
    (7,  "Perfect 5th", "3/2",      1.500000000000000),
    (8,  "Minor 6th",   "128/81",   1.580246913580247),
    (9,  "Major 6th",   "27/16",    1.687500000000000),
    (10, "Minor 7th",   "16/9",     1.777777777777778),
    (11, "Major 7th",   "243/128",  1.898437500000000),
]

# =============================================================================
# PRIME ADJACENCY (imported logic from analyze.py)
# =============================================================================

def check_prime_adjacency(delta, max_k=7):
    results = []
    for k in range(1, max_k + 1, 2):
        offset = k * k
        for sign in [+1, -1]:
            candidate = delta + sign * offset
            if candidate > 1:
                results.append({
                    'k': k,
                    'sign': '+' if sign > 0 else '-',
                    'offset': offset,
                    'candidate': candidate,
                    'is_prime': isprime(candidate),
                })
    return results

# =============================================================================
# ENGINE RUNNER
# =============================================================================

def run_engine(tune_scalar):
    """Run a single engine invocation with deterministic seeds."""
    set_deterministic()

    sys.path.insert(0, str(SCRIPT_DIR))
    from engine_coupled import run_coupled_simulation

    run_dir = run_coupled_simulation(
        steps=300,
        grid_size=89,
        use_zeta=True,
        tune_scalar=tune_scalar,
        beat_detune=0.1,
        use_hann=False,
        use_laplacian=True,
        severity=5.0,
        core_radius=0.0,
        absorption=0.1,
        n_zeros=36,
        no_carrier=True,
        no_beat=False,
        no_core=True,
        single_shot=True,
        probe_interval=1,
        slice_interval=5,
    )
    return run_dir

# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_prime_metrics(run_dir):
    """Extract prime lock metrics from a completed run."""
    meta_path = Path(run_dir) / "meta.json"
    clouds_path = Path(run_dir) / "clouds.npz"

    with open(meta_path) as f:
        meta = json.load(f)

    n_nodes = meta['n_nodes']
    npz = np.load(str(clouds_path))

    # Get step 0 active count
    values_0 = npz['s0000_values']
    active_0 = int(npz['s0000_active'][0])
    delta_0 = n_nodes - active_0

    # Get step 299 active count
    values_299 = npz['s0299_values']
    active_299 = int(npz['s0299_active'][0])

    # Prime adjacency at delta_0
    checks = check_prime_adjacency(delta_0, max_k=7)
    hits = [c for c in checks if c['is_prime']]
    total_checks = len(checks)

    prime_values = sorted(set(c['candidate'] for c in hits))
    k_levels_hit = sorted(set(c['k'] for c in hits))

    return {
        'n_nodes': n_nodes,
        'delta_0': delta_0,
        'active_0': active_0,
        'active_299': active_299,
        'prime_hits': f"{len(hits)}/{total_checks}",
        'prime_hit_count': len(hits),
        'prime_check_count': total_checks,
        'prime_values': prime_values,
        'k_levels_hit': k_levels_hit,
    }


def extract_twist_metrics(run_dir):
    """Run twist.py and parse key metrics."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "twist.py"),
         "--run", str(Path(run_dir).relative_to(SCRIPT_DIR)),
         "--mapping", "quadrant"],
        capture_output=True, text=True, cwd=str(SCRIPT_DIR),
        timeout=120,
    )
    output = result.stdout

    metrics = {
        'twist_x_field_sign': None,
        'twist_x_field_magnitude': None,
        'void_phase': None,
        'hemisphere_imbalance': None,
        'hemisphere_mod_11': None,
    }

    # Parse twist × field at step 0
    for line in output.split('\n'):
        m = re.search(r'Step\s+0\s+\|.*?⟨view×twist⟩\s*=\s*([+-]?\d+\.\d+).*?\|\s*(ALIGNED|OPPOSED)', line)
        if m:
            metrics['twist_x_field_magnitude'] = float(m.group(1))
            metrics['twist_x_field_sign'] = m.group(2)

        m = re.search(r'Hemisphere imbalance:\s*(\d+)\s*upper\s*[−-]\s*(\d+)\s*lower\s*=\s*([+-]?\d+)', line)
        if m:
            metrics['hemisphere_imbalance'] = int(m.group(3))

        m = re.search(r'\|imbalance\|\s*mod\s*11\s*=\s*(\d+)', line)
        if m:
            metrics['hemisphere_mod_11'] = int(m.group(1))

    # Parse void phases
    void_phases = []
    in_void_section = False
    for line in output.split('\n'):
        if 'VOID DETECTION' in line:
            in_void_section = True
            continue
        if in_void_section and re.match(r'\s+\d+\s+\|', line):
            parts = line.strip().split('|')
            if len(parts) >= 6:
                phase_str = parts[-1].strip()
                void_phases.append(phase_str)
        if in_void_section and ('CORRELATION' in line or '========' in line):
            if void_phases:
                break

    if void_phases:
        metrics['void_phase'] = '; '.join(void_phases)

    return metrics


def is_rational(tune_val):
    """Check if tune_scalar is a ratio of integers <= 256."""
    for denom in range(1, 257):
        numer = round(tune_val * denom)
        if numer <= 256 and abs(tune_val - numer / denom) < 1e-12:
            return True
    return False

# =============================================================================
# MAIN SWEEP
# =============================================================================

def main():
    all_results = []
    systems = [
        ("ET", ET_VALUES),
        ("JI", JI_VALUES),
        ("PYTH", PYTH_VALUES),
    ]

    total = sum(len(v) for _, v in systems)
    count = 0

    for system_name, tunings in systems:
        for degree, interval_name, ratio_exact, tune_val in tunings:
            count += 1
            print(f"\n{'#'*72}")
            print(f"# SWEEP {count}/{total}: {system_name} deg={degree} "
                  f"({interval_name}) tune={tune_val:.15f}")
            print(f"{'#'*72}")

            # Run engine
            run_dir = run_engine(tune_val)

            # Verify tune_scalar in meta.json
            with open(Path(run_dir) / "meta.json") as f:
                meta = json.load(f)
            recorded_tune = meta.get('tune_scalar', None)
            if recorded_tune is not None and abs(recorded_tune - tune_val) > 1e-15:
                print(f"WARNING: tune_scalar mismatch! intended={tune_val:.15f} "
                      f"recorded={recorded_tune:.15f}")

            # Extract metrics
            prime_m = extract_prime_metrics(run_dir)
            twist_m = extract_twist_metrics(run_dir)

            result = {
                'run_id': meta['run_id'],
                'tune_scalar': tune_val,
                'tune_scalar_recorded': recorded_tune,
                'tuning_system': system_name,
                'degree': degree,
                'interval_name': interval_name,
                'ratio_exact': ratio_exact,
                'n_nodes': prime_m['n_nodes'],
                'delta_0': prime_m['delta_0'],
                'active_0': prime_m['active_0'],
                'prime_hits': prime_m['prime_hits'],
                'prime_hit_count': prime_m['prime_hit_count'],
                'prime_check_count': prime_m['prime_check_count'],
                'prime_values': prime_m['prime_values'],
                'k_levels_hit': prime_m['k_levels_hit'],
                'active_299': prime_m['active_299'],
                'twist_x_field_sign': twist_m['twist_x_field_sign'],
                'twist_x_field_magnitude': twist_m['twist_x_field_magnitude'],
                'void_phase': twist_m['void_phase'],
                'hemisphere_imbalance': twist_m['hemisphere_imbalance'],
                'hemisphere_mod_11': twist_m['hemisphere_mod_11'],
                'prime_hit_rate': prime_m['prime_hit_count'] / max(prime_m['prime_check_count'], 1),
                'field_survival_ratio': prime_m['active_299'] / max(prime_m['n_nodes'], 1),
                'is_rational': is_rational(tune_val),
                'wall_seconds': meta.get('wall_seconds', None),
            }

            all_results.append(result)

            print(f"\n  → Run {result['run_id']:04d}: n_nodes={result['n_nodes']}, "
                  f"Δ₀={result['delta_0']}, prime_hits={result['prime_hits']}, "
                  f"active_299={result['active_299']}, "
                  f"twist={result['twist_x_field_sign']}")

    # Save raw results
    output_path = SCRIPT_DIR / "sweep_12tone_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")
    print(f"Total runs: {len(all_results)}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'System':>6} {'Deg':>3} {'Interval':>14} {'Tune':>10} "
          f"{'Nodes':>6} {'Δ₀':>5} {'Hits':>5} {'Act299':>6} "
          f"{'Twist':>8} {'Hemi':>5} {'%11':>3}")
    print(f"{'-'*100}")
    for r in all_results:
        print(f"{r['tuning_system']:>6} {r['degree']:>3} {r['interval_name']:>14} "
              f"{r['tune_scalar']:>10.6f} {r['n_nodes']:>6} {r['delta_0']:>5} "
              f"{r['prime_hits']:>5} {r['active_299']:>6} "
              f"{r['twist_x_field_sign'] or '?':>8} "
              f"{r['hemisphere_imbalance'] or 0:>5} "
              f"{r['hemisphere_mod_11'] or 0:>3}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
