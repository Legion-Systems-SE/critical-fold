"""
Reproducibility Script — Resonant Field Engine
================================================
Reproduces the 7 results from Letters 001–005 from a clean state.

Executes 8 engine runs across two grid sizes and 5 zero counts,
then analyzes each with the prime lock scanner and twist curvature
analyzer. Outputs structured results and a PASS/FAIL summary.

Orchestrator only — does not modify engine_coupled.py, analyze.py,
or twist.py. Calls them as subprocesses.

Usage:
    python reproduce.py              # full run
    python reproduce.py --skip-runs  # reuse existing runs, just analyze

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import subprocess
import json
import re
import sys
import time
import math
import numpy as np
from pathlib import Path
from sympy import isprime

SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_DIR = SCRIPT_DIR / 'runs_coupled'
PYTHON = sys.executable

TORCH_DETERMINISM = [
    "import torch",
    "torch.manual_seed(42)",
    "torch.cuda.manual_seed_all(42)",
    "torch.backends.cudnn.deterministic = True",
    "torch.backends.cudnn.benchmark = False",
]

PROTOCOL = [
    {"id": "A", "grid": 181, "n_zeros": 1,  "label": "baseline_dead"},
    {"id": "B", "grid": 181, "n_zeros": 28, "label": "below_threshold"},
    {"id": "C", "grid": 181, "n_zeros": 36, "label": "lock_onset_fixed_point"},
    {"id": "D", "grid": 181, "n_zeros": 42, "label": "lock_present_sign_flip"},
    {"id": "E", "grid": 181, "n_zeros": 50, "label": "full_lock"},
    {"id": "F", "grid": 89,  "n_zeros": 28, "label": "coarse_below_threshold"},
    {"id": "G", "grid": 89,  "n_zeros": 36, "label": "coarse_lock_onset"},
    {"id": "H", "grid": 89,  "n_zeros": 50, "label": "coarse_lock_present"},
]

COMMON_FLAGS = "--single-shot --no-core --no-carrier --steps 300"


def get_system_info():
    code = ";".join([
        "import torch, sys, numpy, scipy, sympy",
        "print(f'python={sys.version.split()[0]}')",
        "print(f'pytorch={torch.__version__}')",
        "print(f'cuda={torch.version.cuda}')",
        "print(f'gpu={torch.cuda.get_device_name(0)}')",
        "print(f'gpu_mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')",
        "print(f'numpy={numpy.__version__}')",
        "print(f'scipy={scipy.__version__}')",
        "print(f'sympy={sympy.__version__}')",
    ])
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    info = {}
    for line in r.stdout.strip().split('\n'):
        k, v = line.split('=', 1)
        info[k] = v
    return info


def inject_determinism():
    det_code = "; ".join(TORCH_DETERMINISM)
    return det_code


def run_engine(grid, n_zeros):
    cmd = (f"{PYTHON} {SCRIPT_DIR / 'engine_coupled.py'} "
           f"--grid {grid} --n-zeros {n_zeros} {COMMON_FLAGS}")
    det = inject_determinism()
    env_cmd = f"{PYTHON} -c \"{det}; import os; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\""
    subprocess.run(env_cmd, shell=True, capture_output=True)

    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(SCRIPT_DIR), timeout=600)
    wall = time.time() - t0

    run_id = None
    for line in r.stdout.split('\n'):
        m = re.search(r'Run:\s+(\d{4})', line)
        if m:
            run_id = m.group(1)
            break

    if run_id is None:
        print(f"  ERROR: no run ID found in output for grid={grid} n_zeros={n_zeros}")
        print(f"  stdout: {r.stdout[:200]}")
        print(f"  stderr: {r.stderr[:200]}")
        return None, wall

    return run_id, wall


def find_latest_run_for(grid, n_zeros, exclude=None):
    if exclude is None:
        exclude = set()
    candidates = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / 'meta.json'
        if not meta_path.exists():
            continue
        if d.name in exclude:
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get('grid_size') == grid and
                meta.get('n_zeros') == n_zeros and
                meta.get('single_shot') is True and
                meta.get('no_core') is True and
                meta.get('no_carrier') is True and
                meta.get('total_steps') == 300):
                candidates.append(d.name)
        except (json.JSONDecodeError, KeyError):
            pass
    return candidates[-1] if candidates else None


def analyze_prime(run_dir):
    cmd = f"{PYTHON} {SCRIPT_DIR / 'analyze.py'} prime --run {run_dir}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(SCRIPT_DIR), timeout=300)
    return r.stdout


def analyze_twist(run_dir):
    run_rel = str(Path(run_dir).relative_to(SCRIPT_DIR))
    cmd = f"{PYTHON} {SCRIPT_DIR / 'twist.py'} --run {run_rel} --mapping quadrant"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(SCRIPT_DIR), timeout=300)
    return r.stdout


def extract_prime_metrics(prime_output):
    metrics = {
        'delta_0': None,
        'prime_adjacency_hits': 0,
        'prime_adjacency_total': 8,
        'prime_adjacency_values': [],
        'k_levels_hit': [],
        'k3_hit': False,
    }

    ortho_block = False
    for line in prime_output.split('\n'):
        if 'ORTHOGONALITY' in line:
            ortho_block = True
            continue
        if ortho_block:
            m = re.search(r'Δ:\s*(\d+)', line)
            if m:
                metrics['delta_0'] = int(m.group(1))
            if '✓ PRIME' in line:
                metrics['prime_adjacency_hits'] += 1
                val_m = re.search(r'=\s+(\d+)\s+✓', line)
                if val_m:
                    metrics['prime_adjacency_values'].append(int(val_m.group(1)))
                k_m = re.search(r'k=(\d+)', line)
                if k_m:
                    k = int(k_m.group(1))
                    if k not in metrics['k_levels_hit']:
                        metrics['k_levels_hit'].append(k)
                    if k == 3:
                        metrics['k3_hit'] = True

    # Temporal deltas
    deltas = []
    for line in prime_output.split('\n'):
        m = re.match(r'\s+\d+→\d+\s+\|\s+(\d+)', line)
        if m:
            deltas.append(int(m.group(1)))
    metrics['temporal_deltas_first5'] = deltas[:5]

    return metrics


def extract_twist_metrics(twist_output):
    metrics = {
        'orbit_coverage': None,
        'twist_x_field_sign': None,
        'twist_x_field_magnitude': None,
        'twist_x_field_steps': [],
        'hemisphere_imbalance': None,
        'hemisphere_mod11': None,
        'void_phase': [],
        'void_circumradius': None,
    }

    for line in twist_output.split('\n'):
        m = re.search(r'Orbit coverage:\s+([\d.]+)%', line)
        if m:
            metrics['orbit_coverage'] = float(m.group(1)) / 100

        m = re.search(r'Hemisphere imbalance:.*=\s*([-\d]+)', line)
        if m:
            metrics['hemisphere_imbalance'] = int(m.group(1))
            metrics['hemisphere_mod11'] = abs(int(m.group(1))) % 11

        m = re.search(r'Step\s+(\d+)\s+\|.*⟨view×twist⟩\s*=\s*([-+\d.]+).*\|\s+(\w+)', line)
        if m:
            step = int(m.group(1))
            mag = float(m.group(2))
            sign = m.group(3)
            metrics['twist_x_field_steps'].append({
                'step': step, 'magnitude': mag, 'alignment': sign
            })
            if step == 0:
                metrics['twist_x_field_sign'] = sign
                metrics['twist_x_field_magnitude'] = mag

        m = re.search(r'Phase\s*$', line)

        m = re.search(r'^\s+\d+\s+\|.*\|\s+(\d+)/4\s*$', line)
        if m:
            metrics['void_phase'].append(int(m.group(1)))

        m = re.search(r'\|\s+([\d.]+)\s+\|.*\d+/4', line)
        if m and metrics['void_circumradius'] is None:
            crad_m = re.search(r'\|\s+([\d.]+)\s+\|', line)

    # Get circumradius from void table
    for line in twist_output.split('\n'):
        m = re.match(r'\s+\d+\s+\|.*\|\s+([\d.]+)\s+\|.*\|.*\|.*\d+/4', line)
        if m and metrics['void_circumradius'] is None:
            metrics['void_circumradius'] = float(m.group(1))

    return metrics


def extract_active_counts(run_dir):
    meta_path = Path(run_dir) / 'meta.json'
    clouds_path = Path(run_dir) / 'clouds.npz'

    with open(meta_path) as f:
        meta = json.load(f)

    n_nodes = meta.get('n_nodes', 0)

    if not clouds_path.exists():
        return {'active_0': 0, 'active_150': 0, 'active_299': 0, 'n_nodes': n_nodes}

    npz = np.load(str(clouds_path))
    counts = {'n_nodes': n_nodes}

    for step, key in [(0, 'active_0'), (150, 'active_150'), (299, 'active_299')]:
        prefix = f"s{step:04d}_active"
        if prefix in npz:
            counts[key] = int(npz[prefix][0])
        else:
            counts[key] = 0

    return counts


def run_protocol(skip_runs=False):
    print("=" * 72)
    print("REPRODUCIBILITY PROTOCOL — Resonant Field Engine")
    print("=" * 72)

    sysinfo = get_system_info()
    print(f"\nSystem: {sysinfo.get('gpu', '?')} | "
          f"Python {sysinfo.get('python', '?')} | "
          f"PyTorch {sysinfo.get('pytorch', '?')} | "
          f"CUDA {sysinfo.get('cuda', '?')}")

    results = {
        'meta': {
            'generated_by': 'reproduce.py',
            'system': sysinfo,
            'protocol_version': 1,
        },
        'runs': {},
    }

    # Phase 1: Execute runs
    print(f"\n{'='*72}")
    print("PHASE 1 — ENGINE RUNS")
    print(f"{'='*72}")

    used_run_ids = set()
    for spec in PROTOCOL:
        rid = spec['id']
        grid = spec['grid']
        nz = spec['n_zeros']
        label = spec['label']

        if skip_runs:
            run_id = find_latest_run_for(grid, nz, exclude=used_run_ids)
            if run_id:
                used_run_ids.add(run_id)
                run_dir = str(RUNS_DIR / run_id)
                print(f"  [{rid}] Reusing run {run_id} (grid={grid}, n_zeros={nz})")
                results['runs'][rid] = {
                    'spec': spec,
                    'run_id': run_id,
                    'run_dir': run_dir,
                    'wall_time': None,
                }
                continue
            else:
                print(f"  [{rid}] No existing run found, executing...")

        print(f"  [{rid}] Running grid={grid}, n_zeros={nz} ({label})...", end=' ', flush=True)
        run_id, wall = run_engine(grid, nz)
        if run_id:
            used_run_ids.add(run_id)
            run_dir = str(RUNS_DIR / run_id)
            print(f"→ {run_id} ({wall:.1f}s)")
            results['runs'][rid] = {
                'spec': spec,
                'run_id': run_id,
                'run_dir': run_dir,
                'wall_time': round(wall, 2),
            }
        else:
            print("FAILED")
            results['runs'][rid] = {'spec': spec, 'run_id': None, 'error': True}

    # Phase 2: Analyze
    print(f"\n{'='*72}")
    print("PHASE 2 — ANALYSIS")
    print(f"{'='*72}")

    for rid, rdata in results['runs'].items():
        if rdata.get('error'):
            continue
        run_dir = rdata['run_dir']
        nz = rdata['spec']['n_zeros']
        grid = rdata['spec']['grid']
        print(f"  [{rid}] Analyzing run {rdata['run_id']}...", end=' ', flush=True)

        prime_out = analyze_prime(run_dir)
        prime_metrics = extract_prime_metrics(prime_out)
        rdata['prime'] = prime_metrics

        twist_out = analyze_twist(run_dir)
        twist_metrics = extract_twist_metrics(twist_out)
        rdata['twist'] = twist_metrics

        active = extract_active_counts(run_dir)
        rdata['active'] = active

        hits = prime_metrics['prime_adjacency_hits']
        sign = twist_metrics.get('twist_x_field_sign', '?')
        nodes = active.get('n_nodes', '?')
        a299 = active.get('active_299', '?')
        print(f"nodes={nodes} Δ₀={prime_metrics.get('delta_0','?')} "
              f"hits={hits}/8 T×F={sign} active₂₉₉={a299}")

    # Phase 3: Report
    print(f"\n{'='*72}")
    print("PHASE 3 — RESULTS")
    print(f"{'='*72}")

    # Staircase table
    print(f"\n--- STAIRCASE (grid 181) ---")
    print(f"  {'nz':>4} | {'Run':>4} | {'Nodes':>5} | {'Δ₀':>5} | {'Hits':>5} | "
          f"{'A₂₉₉':>5} | {'T×F':>8} | {'VoidPh':>6}")
    print(f"  {'-'*60}")
    for rid in ['A', 'B', 'C', 'D', 'E']:
        r = results['runs'].get(rid, {})
        if r.get('error'):
            continue
        nz = r['spec']['n_zeros']
        run_id = r.get('run_id', '?')
        nodes = r['active']['n_nodes']
        d0 = r['prime']['delta_0']
        hits = r['prime']['prime_adjacency_hits']
        a299 = r['active']['active_299']
        sign = r['twist'].get('twist_x_field_sign', '?')
        vp = r['twist'].get('void_phase', [])
        vp_str = f"{vp[0]}/4" if vp else '?'
        print(f"  {nz:>4} | {run_id:>4} | {nodes:>5} | {d0:>5} | "
              f"{hits:>3}/8 | {a299:>5} | {sign:>8} | {vp_str:>6}")

    print(f"\n--- STAIRCASE (grid 89) ---")
    print(f"  {'nz':>4} | {'Run':>4} | {'Nodes':>5} | {'Δ₀':>5} | {'Hits':>5} | "
          f"{'A₂₉₉':>5} | {'T×F':>8} | {'VoidPh':>6}")
    print(f"  {'-'*60}")
    for rid in ['F', 'G', 'H']:
        r = results['runs'].get(rid, {})
        if r.get('error'):
            continue
        nz = r['spec']['n_zeros']
        run_id = r.get('run_id', '?')
        nodes = r['active']['n_nodes']
        d0 = r['prime']['delta_0']
        hits = r['prime']['prime_adjacency_hits']
        a299 = r['active']['active_299']
        sign = r['twist'].get('twist_x_field_sign', '?')
        vp = r['twist'].get('void_phase', [])
        vp_str = f"{vp[0]}/4" if vp else '?'
        print(f"  {nz:>4} | {run_id:>4} | {nodes:>5} | {d0:>5} | "
              f"{hits:>3}/8 | {a299:>5} | {sign:>8} | {vp_str:>6}")

    # Delta staircase
    print(f"\n--- Δ₀ PROGRESSION (grid 181) ---")
    deltas_181 = {}
    for rid in ['C', 'D', 'E']:
        r = results['runs'].get(rid, {})
        if not r.get('error') and r.get('prime'):
            deltas_181[r['spec']['n_zeros']] = r['prime']['delta_0']
    if 36 in deltas_181 and 42 in deltas_181 and 50 in deltas_181:
        d36, d42, d50 = deltas_181[36], deltas_181[42], deltas_181[50]
        print(f"  36z: Δ₀={d36}  42z: Δ₀={d42}  50z: Δ₀={d50}")
        print(f"  36→42: {d36}-{d42} = {d36-d42}")
        print(f"  42→50: {d42}-{d50} = {d42-d50}")

    # k-level rotation
    print(f"\n--- k-LEVEL ROTATION ---")
    for rid in ['C', 'D', 'E']:
        r = results['runs'].get(rid, {})
        if r.get('error'):
            continue
        nz = r['spec']['n_zeros']
        k_hit = r['prime'].get('k_levels_hit', [])
        vals = r['prime'].get('prime_adjacency_values', [])
        print(f"  {nz}z: k-levels={k_hit} primes={vals}")

    # 193 convergence
    print(f"\n--- 193 CONVERGENCE (grid 89) ---")
    for rid in ['G', 'H']:
        r = results['runs'].get(rid, {})
        if r.get('error'):
            continue
        nz = r['spec']['n_zeros']
        vals = r['prime'].get('prime_adjacency_values', [])
        has_193 = 193 in vals
        print(f"  {nz}z: primes={vals} → 193 present: {has_193}")

    # Phase 4: PASS/FAIL
    print(f"\n{'='*72}")
    print("PHASE 4 — VERIFICATION")
    print(f"{'='*72}\n")

    verdicts = {}

    def get(rid):
        return results['runs'].get(rid, {})

    # R1: Moiré mechanism
    r1_f = get('F').get('prime', {}).get('prime_adjacency_hits', -1)
    r1_g = get('G').get('prime', {}).get('prime_adjacency_hits', -1)
    r1_pass = (r1_f == 0 and r1_g > 0)
    verdicts['R1_moire'] = {
        'claim': 'Prime lock onset at 28→36 zeros persists at grid 89 (all zeros > Nyquist)',
        'check': f'hits[F]={r1_f}==0 AND hits[G]={r1_g}>0',
        'pass': r1_pass,
    }

    # R2: 36-zero fixed point
    nodes_181 = [get(r).get('active', {}).get('n_nodes', 0) for r in ['A','B','C','D','E']]
    nodes_89  = [get(r).get('active', {}).get('n_nodes', 0) for r in ['F','G','H']]
    c_nodes = get('C').get('active', {}).get('n_nodes', 0)
    g_nodes = get('G').get('active', {}).get('n_nodes', 0)
    c_sign = get('C').get('twist', {}).get('twist_x_field_sign', '')
    g_sign = get('G').get('twist', {}).get('twist_x_field_sign', '')
    r2_max181 = c_nodes == max(nodes_181) if nodes_181 else False
    r2_max89  = g_nodes == max(nodes_89) if nodes_89 else False
    r2_ali_c = c_sign == 'ALIGNED'
    r2_ali_g = g_sign == 'ALIGNED'
    r2_pass = r2_max181 and r2_max89 and r2_ali_c and r2_ali_g
    verdicts['R2_fixed_point'] = {
        'claim': '36 zeros produces max nodes and ALIGNED at both grids',
        'check': (f'max181={r2_max181}(nodes={nodes_181}) '
                  f'max89={r2_max89}(nodes={nodes_89}) '
                  f'ali_C={r2_ali_c} ali_G={r2_ali_g}'),
        'pass': r2_pass,
    }

    # R3: Three phase transitions
    a_alive = get('A').get('active', {}).get('active_299', -1)
    b_alive = get('B').get('active', {}).get('active_299', -1)
    b_hits = get('B').get('prime', {}).get('prime_adjacency_hits', -1)
    c_hits = get('C').get('prime', {}).get('prime_adjacency_hits', -1)
    d_hits = get('D').get('prime', {}).get('prime_adjacency_hits', -1)
    e_hits = get('E').get('prime', {}).get('prime_adjacency_hits', -1)
    r3_dead = (a_alive == 0)
    r3_alive = (b_alive > 0)
    r3_lock_off = (b_hits == 0)
    r3_lock_on = (c_hits > 0)
    r3_strengthen = (e_hits > d_hits) if (e_hits >= 0 and d_hits >= 0) else False
    r3_pass = r3_dead and r3_alive and r3_lock_off and r3_lock_on and r3_strengthen
    verdicts['R3_three_transitions'] = {
        'claim': 'Three phase transitions: survival(1→28), lock(28→36), strengthen(42→50)',
        'check': (f'dead_A={r3_dead}(a299={a_alive}) alive_B={r3_alive}(b299={b_alive}) '
                  f'lock_off_B={r3_lock_off}({b_hits}/8) lock_on_C={r3_lock_on}({c_hits}/8) '
                  f'strengthen={r3_strengthen}({d_hits}/8→{e_hits}/8)'),
        'pass': r3_pass,
    }

    # R4: Twist-field opposition
    b_sign = get('B').get('twist', {}).get('twist_x_field_sign', '')
    d_sign = get('D').get('twist', {}).get('twist_x_field_sign', '')
    e_sign = get('E').get('twist', {}).get('twist_x_field_sign', '')
    r4_pass = (b_sign == 'OPPOSED' and d_sign == 'OPPOSED' and e_sign == 'OPPOSED')
    verdicts['R4_opposition'] = {
        'claim': 'Twist×field OPPOSED at void positions (except 36z)',
        'check': f'B={b_sign} D={d_sign} E={e_sign}',
        'pass': r4_pass,
    }

    # R5: Delta staircase
    if 36 in deltas_181 and 42 in deltas_181 and 50 in deltas_181:
        step1 = deltas_181[36] - deltas_181[42]
        step2 = deltas_181[42] - deltas_181[50]
        r5_pass = (step1 == 6 and step2 == 6)
        verdicts['R5_delta_staircase'] = {
            'claim': 'Δ₀ decreases in steps of 6 at grid 181 for n=36,42,50',
            'check': f'36→42: {step1}  42→50: {step2}',
            'pass': r5_pass,
        }
    else:
        verdicts['R5_delta_staircase'] = {
            'claim': 'Δ₀ decreases in steps of 6',
            'check': 'insufficient data',
            'pass': False,
        }

    # R6: k=3 never hits
    any_k3 = False
    for rid in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        r = get(rid)
        if r.get('prime', {}).get('k3_hit', False):
            any_k3 = True
    r6_pass = not any_k3
    verdicts['R6_k_rotation'] = {
        'claim': 'k=3 never hits at any zero count',
        'check': f'any_k3={any_k3}',
        'pass': r6_pass,
    }

    # R7: 193 convergence
    g_vals = get('G').get('prime', {}).get('prime_adjacency_values', [])
    h_vals = get('H').get('prime', {}).get('prime_adjacency_values', [])
    r7_pass = (193 in g_vals and 193 in h_vals)
    verdicts['R7_193_convergence'] = {
        'claim': 'Prime 193 reachable from both 36z and 50z at grid 89',
        'check': f'G_primes={g_vals} H_primes={h_vals}',
        'pass': r7_pass,
    }

    # Print verdicts
    n_pass = sum(1 for v in verdicts.values() if v['pass'])
    n_total = len(verdicts)

    for k, v in verdicts.items():
        status = "PASS" if v['pass'] else "FAIL"
        marker = "✓" if v['pass'] else "✗"
        print(f"  {marker} {status}  {k}")
        print(f"          {v['claim']}")
        print(f"          {v['check']}")
        print()

    print(f"{'='*72}")
    print(f"  TOTAL: {n_pass}/{n_total} PASSED")
    print(f"{'='*72}")

    results['verdicts'] = verdicts
    results['summary'] = {
        'passed': n_pass,
        'total': n_total,
        'all_pass': n_pass == n_total,
    }

    # Save results
    output_path = SCRIPT_DIR / 'reproduce_results.json'

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = make_serializable(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    skip = '--skip-runs' in sys.argv
    run_protocol(skip_runs=skip)
