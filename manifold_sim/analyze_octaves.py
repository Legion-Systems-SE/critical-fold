"""
Octave analysis of the fold byte at variable resolution.
Reads fields.npz from a completed ternary run and measures:
1. Standing wave node positions (where spherical mean of delta crosses zero)
2. Octave intervals between nodes nearest to known constants
3. Grid resolution in bits (dynamic range in octaves)
4. Angular structure (octant divergence) at node positions
5. Phase angle (ω₁/ω₂) profile

Usage:
  python manifold_sim/analyze_octaves.py --run 0740
  python manifold_sim/analyze_octaves.py --run 0740 --run2 0739  # compare two runs
"""

import numpy as np
import json
import argparse
import os

def load_run(run_id):
    run_dir = f'manifold_sim/runs_emergent/{run_id}'
    d = np.load(os.path.join(run_dir, 'fields.npz'))
    meta = json.load(open(os.path.join(run_dir, 'meta.json')))
    return d, meta

def analyze(run_id):
    d, meta = load_run(run_id)
    omega1, omega2 = d['omega1'], d['omega2']
    membrane = d['membrane']
    
    grid = meta['grid_size']
    dx = 20.0 / (grid - 1)
    scale = meta['scale']
    center = grid // 2
    
    ci = np.mgrid[0:grid, 0:grid, 0:grid].astype(np.float32) - center
    x, y, z = ci[0]*dx, ci[1]*dx, ci[2]*dx
    R = np.sqrt(x**2 + y**2 + z**2)
    delta = omega1 - omega2
    
    # Octant index
    oct_idx = ((x >= 0).astype(int) * 4 + 
               (y >= 0).astype(int) * 2 + 
               (z >= 0).astype(int))
    
    # Resolution
    fold_R_est = 8.08  # will refine
    dyn_range = fold_R_est / (2*dx)
    octaves = np.log2(dyn_range)
    
    print(f"\n{'='*80}")
    print(f"OCTAVE ANALYSIS — Run {run_id}")
    print(f"{'='*80}")
    print(f"Grid: {grid}, dx: {dx:.5f}, scale: {scale:.4f}")
    print(f"Nodes: {meta.get('n_nodes', '?')}, Steps: {meta.get('total_steps', '?')}")
    print(f"Resolution: {dyn_range:.0f}:1 = {octaves:.2f} octaves = {octaves:.2f} bits")
    
    # ---- Spherical mean at fine resolution ----
    bin_size = max(dx, 0.02)  # at least dx, never smaller than 0.02
    r_edges = np.arange(0, 10.01, bin_size)
    r_mid = (r_edges[:-1] + r_edges[1:]) / 2
    
    sph_mean = np.zeros(len(r_mid))
    sph_var = np.zeros(len(r_mid))
    oct_std = np.zeros(len(r_mid))
    phase_angle = np.zeros(len(r_mid))
    mem_mean = np.zeros(len(r_mid))
    
    for i in range(len(r_mid)):
        mask = (R >= r_edges[i]) & (R < r_edges[i+1])
        n = mask.sum()
        if n < 4: continue
        
        d_vals = delta[mask]
        sph_mean[i] = float(d_vals.mean())
        sph_var[i] = float(d_vals.var())
        
        o1 = float(omega1[mask].mean())
        o2 = float(omega2[mask].mean())
        phase_angle[i] = np.degrees(np.arctan2(o2, o1))
        mem_mean[i] = float(membrane[mask].mean())
        
        # Octant standard deviation
        oct_vals = []
        for o in range(8):
            omask = mask & (oct_idx == o)
            if omask.sum() > 0:
                oct_vals.append(float(delta[omask].mean()))
        if len(oct_vals) >= 4:
            oct_std[i] = float(np.std(oct_vals))
    
    # ---- Find standing wave nodes ----
    nodes = []
    for i in range(1, len(sph_mean)):
        if sph_mean[i] * sph_mean[i-1] < 0:
            r0, r1 = r_mid[i-1], r_mid[i]
            v0, v1 = sph_mean[i-1], sph_mean[i]
            r_cross = r0 - v0 * (r1 - r0) / (v1 - v0)
            nodes.append(r_cross)
    
    nodes = np.array(nodes)
    
    # ---- Match constants to nearest nodes ----
    consts = [
        ('hbar', 1.054572),
        ('phi',  1.618034),
        ('e',    2.718282),
        ('c',    2.997925),
        ('pi',   3.141593),
    ]
    
    zeros_t = [14.13472514, 21.02203964, 25.01086, 30.42488, 32.93506]
    zeros = [(f'z{i+1}', t/scale) for i, t in enumerate(zeros_t) if t/scale < 10]
    
    print(f"\nStanding wave nodes found: {len(nodes)} (in R=0 to 10)")
    
    # Fold R: use z₂ position (the channel-killing zero) as the structural marker
    # z₂ defines the end of the usable channel; fold_R ≈ z₂ + 1 bit
    if len(zeros) >= 2:
        z2_r = zeros[1][1]
        fold_R = z2_r + z2_r / 7  # z₂ at bit 7 → fold_R = z₂ × 8/7
    else:
        fold_R = 8.08
    
    bit_width = fold_R / 8
    
    print(f"Estimated fold_R: {fold_R:.3f}")
    print(f"Bit width: {bit_width:.4f}")
    
    # ---- Constants at nodes ----
    print(f"\n{'const':>8s}  {'R_true':>8s}  {'R_node':>8s}  {'offset':>8s}  {'phase°':>8s}  {'oct_std':>8s}  {'membrane':>8s}")
    print("-"*70)
    
    const_nodes = {}
    for name, r_true in consts:
        if len(nodes) == 0:
            print(f"{name:>8s}  {r_true:8.4f}  no nodes found")
            continue
        dists = np.abs(nodes - r_true)
        j_near = np.argmin(dists)
        r_near = nodes[j_near]
        offset = r_true - r_near
        
        # Get phase and octant_std at this radius
        idx = np.argmin(np.abs(r_mid - r_true))
        ph = phase_angle[idx]
        oc = oct_std[idx]
        mm = mem_mean[idx]
        
        const_nodes[name] = r_near
        print(f"{name:>8s}  {r_true:8.4f}  {r_near:8.4f}  {offset:+8.4f}  {ph:8.2f}  {oc:8.5f}  {mm:8.4f}")
    
    # ---- Zeros between nodes ----
    print(f"\n{'zero':>8s}  {'R_true':>8s}  {'gap':>8s}  {'membrane':>8s}")
    print("-"*45)
    for name, r_true in zeros:
        if len(nodes) == 0: continue
        dists = np.abs(nodes - r_true)
        sorted_idx = np.argsort(dists)
        j1, j2 = sorted(sorted_idx[:2])
        if j2 < len(nodes):
            gap = nodes[j2] - nodes[j1]
        else:
            gap = 0
        idx = np.argmin(np.abs(r_mid - r_true))
        mm = mem_mean[idx]
        print(f"{name:>8s}  {r_true:8.3f}  {gap:8.3f}  {mm:8.4f}")
    
    # ---- Octave intervals ----
    print(f"\n{'='*80}")
    print(f"OCTAVE INTERVALS")
    print(f"{'='*80}")
    
    cnames = list(const_nodes.keys())
    cvals = [const_nodes[n] for n in cnames]
    
    targets = {
        '1/phi':   1/1.618034,
        'log2(3)': np.log2(3),
        '1/7':     1/7,
        '1':       1.0,
        '1/2':     0.5,
    }
    
    print(f"\n  {'from→to':>16s}  {'ratio':>8s}  {'octaves':>8s}  {'match':>30s}")
    print("  " + "-"*70)
    
    for i in range(len(cnames)):
        for j in range(i+1, len(cnames)):
            if cvals[i] <= 0 or cvals[j] <= 0: continue
            ratio = cvals[j] / cvals[i]
            octaves = np.log2(ratio)
            
            best_name, best_val = min(targets.items(), key=lambda x: abs(x[1] - octaves))
            err = abs(best_val - octaves) / best_val * 100
            marker = f"≈ {best_name} ({err:.3f}%)" if err < 5 else ""
            
            print(f"  {cnames[i]:>6s}→{cnames[j]:<6s}  {ratio:8.5f}  {octaves:8.5f}  {marker}")
    
    # ---- 1/alpha vs z2 position ----
    if len(zeros) >= 2:
        z2_r = zeros[1][1]
        z2_bit = z2_r / bit_width
        log2_alpha_inv = np.log2(137.036)
        print(f"\n  1/α in octaves: {log2_alpha_inv:.4f}")
        print(f"  z₂ bit position: {z2_bit:.4f}")
        print(f"  Difference: {abs(log2_alpha_inv - z2_bit):.4f} bits ({abs(log2_alpha_inv - z2_bit)/8*100:.2f}% of byte)")
    
    return {
        'run_id': run_id,
        'grid': grid, 'dx': dx, 'scale': scale,
        'octaves': octaves, 'fold_R': fold_R,
        'n_nodes_sw': len(nodes),
        'const_nodes': const_nodes,
        'nodes': nodes,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Run ID (e.g. 0740)')
    p.add_argument('--run2', default=None, help='Second run ID to compare')
    args = p.parse_args()
    
    r1 = analyze(args.run)
    
    if args.run2:
        r2 = analyze(args.run2)
        
        print(f"\n{'='*80}")
        print(f"COMPARISON: {args.run} (grid {r1['grid']}) vs {args.run2} (grid {r2['grid']})")
        print(f"{'='*80}")
        print(f"Resolution: {r1['octaves']:.2f} vs {r2['octaves']:.2f} octaves")
        print(f"SW nodes: {r1['n_nodes_sw']} vs {r2['n_nodes_sw']}")
        
        # Compare constant node positions
        print(f"\n  {'const':>8s}  {'R₁':>8s}  {'R₂':>8s}  {'ΔR':>8s}  {'converging?':>12s}")
        print("  " + "-"*50)
        for name in r1['const_nodes']:
            if name in r2['const_nodes']:
                r1v = r1['const_nodes'][name]
                r2v = r2['const_nodes'][name]
                dr = r1v - r2v
                consts_true = {'hbar': 1.054572, 'phi': 1.618034, 'e': 2.718282, 
                               'c': 2.997925, 'pi': 3.141593}
                true_v = consts_true[name]
                err1 = abs(r1v - true_v)
                err2 = abs(r2v - true_v)
                conv = "YES" if err1 < err2 else ("same" if abs(err1-err2) < 0.001 else "no")
                print(f"  {name:>8s}  {r1v:8.4f}  {r2v:8.4f}  {dr:+8.4f}  {conv:>12s}")
