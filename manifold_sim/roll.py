"""Roll the engine along the critical line. Each lap starts where the last ended.
No external time scalars. Position recorded in field units (beat periods)."""

import subprocess, json, math, sys, os

R_MAX = math.sqrt(3) * 10.0
ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_emergent.py")


def run_lap(t_offset):
    result = subprocess.run(
        [sys.executable, ENGINE, "--bifurcation", "zeta", "--auto",
         "--t-offset", f"{t_offset:.6f}"],
        capture_output=True, text=True, timeout=600)

    for line in result.stdout.split('\n'):
        if 'Run:' in line and '→' in line:
            run_dir = line.split('→')[1].strip()
            break
    else:
        return None

    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)

    fold_R = None
    for line in reversed(result.stdout.split('\n')):
        line = line.strip()
        if '|' in line and 'fold_R' not in line and 'Step' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    fold_R = float(parts[6])
                    break
                except ValueError:
                    continue

    scale = meta['scale']
    t_end = t_offset + R_MAX * scale
    bp = meta.get('auto_beat_period', 1)

    return {
        't_start': t_offset,
        't_end': t_end,
        'scale': scale,
        'grid': meta['grid_size'],
        'periods': meta['total_steps'] // bp if bp else None,
        'fold_R': fold_R,
        'curv_ratio': meta['field_stats']['curvature_ratio'],
        'char_len': meta['field_stats']['char_length'],
        'omega_rms': meta['field_stats']['omega_rms'],
        'run_id': meta['run_id'],
    }


def main():
    n_laps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t_start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roll_log.jsonl")

    print(f"Rolling {n_laps} laps from t={t_start:.2f}")
    print(f"{'lap':>4} {'t_start':>12} {'t_end':>12} {'scale':>8} {'grid':>5} "
          f"{'periods':>8} {'fold_R':>7} {'cr':>6} {'cl':>6}")
    print("-" * 80)

    t = t_start
    total_periods = 0
    for lap in range(n_laps):
        try:
            r = run_lap(t)
        except Exception as e:
            print(f"  LAP {lap}: FAILED at t={t:.2f} — {e}")
            break

        if r is None:
            print(f"  LAP {lap}: NO OUTPUT at t={t:.2f}")
            break

        total_periods += r['periods'] or 0

        print(f"{lap:>4} {r['t_start']:>12.2f} {r['t_end']:>12.2f} {r['scale']:>8.4f} {r['grid']:>5} "
              f"{r['periods']:>8} {r['fold_R']:>7.3f} {r['curv_ratio']:>6.3f} {r['char_len']:>6.3f}")

        with open(log_path, 'a') as f:
            f.write(json.dumps(r) + '\n')

        t = r['t_end']

    print(f"\nRolled to t={t:.2f} in {total_periods} total beat periods. Log: {log_path}")


if __name__ == '__main__':
    main()
