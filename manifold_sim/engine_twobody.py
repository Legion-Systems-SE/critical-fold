"""
Resonant Field Engine — Two-Body Experimental v0.1
===================================================
Two zeta-seeded bodies on a shared grid. Interact through the Laplacian.

Adds to engine_emergent.py:
  - Dual injection at ±separation/2 along X
  - Per-node field identity (body A vs B)
  - Cluster-based bulk measurement (COM, velocity, momentum)
  - Quantized tensor rotator modes (emergent, pi, laplacian)

Output dir: runs_twobody/NNNN/

Usage:
    python3 manifold_sim/engine_twobody.py --separation 10 --steps 2000
    python3 manifold_sim/engine_twobody.py --separation 8 --rotator pi --auto
    python3 manifold_sim/engine_twobody.py --separation 12 --rotator laplacian --grid 65

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import torch
import argparse
import math
import os
import json
import numpy as np
import time as _time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_emergent import (
    discrete_laplacian,
    compute_membrane,
    kappa_H,
    build_metric_from_field,
    compute_field_rates,
    compute_auto_scale,
    compute_auto_grid,
    EnergyTracker,
)


RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "runs_twobody")


def next_run_id():
    os.makedirs(RUNS_DIR, exist_ok=True)
    existing = []
    for name in os.listdir(RUNS_DIR):
        try:
            existing.append(int(name))
        except ValueError:
            pass
    return max(existing) + 1 if existing else 1


def setup_run_dir(run_id):
    run_dir = os.path.join(RUNS_DIR, f"{run_id:04d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def update_latest(run_id):
    with open(os.path.join(RUNS_DIR, "latest.txt"), 'w') as f:
        f.write(f"{run_id:04d}\n")


# =============================================================================
# ABSORBING BOUNDARY — Hann-tapered edge damping
# =============================================================================

def build_absorb_mask(grid_size, width, device):
    if width <= 0:
        return None
    m = torch.ones(grid_size, device=device)
    for i in range(width):
        x = (i + 1.0) / (width + 1.0)
        t = 0.5 * (1.0 - math.cos(math.pi * x))
        m[i] = t
        m[grid_size - 1 - i] = t
    return m[:, None, None] * m[None, :, None] * m[None, None, :]


# =============================================================================
# QUANTIZED TENSOR ROTATOR
# =============================================================================

def quantize_tensors(T_vec, mag_T, mode):
    if mode == 'emergent':
        return T_vec

    if mode == 'pi':
        signs = torch.sign(T_vec)
        n_active = (T_vec.abs() > 1e-10).float().sum(dim=-1, keepdim=True).clamp(min=1)
        return signs * mag_T.unsqueeze(-1) / torch.sqrt(n_active)

    if mode == 'laplacian':
        k = 8
        step = math.pi / k
        r = mag_T.unsqueeze(-1) + 1e-10
        T_unit = T_vec / r
        theta = torch.acos(T_unit[..., 2].clamp(-1, 1))
        phi = torch.atan2(T_unit[..., 1], T_unit[..., 0])
        theta_q = torch.round(theta / step) * step
        phi_q = torch.round(phi / step) * step
        sin_tq = torch.sin(theta_q)
        return r * torch.stack([
            sin_tq * torch.cos(phi_q),
            sin_tq * torch.sin(phi_q),
            torch.cos(theta_q),
        ], dim=-1)

    raise ValueError(f"Unknown rotator mode: {mode}")


# =============================================================================
# WAVE STEP — with quantized tensor rotator
# =============================================================================

def wave_step(psi, omega, g_mu_nu, w_alpha_t, kappa, rates,
              use_laplacian=True, use_advection=False, no_pump=False,
              propagate=True, rotator='emergent'):

    psi_eff = kappa * g_mu_nu * psi

    fdx = torch.roll(psi_eff, 1, 0) - torch.roll(psi_eff, -1, 0)
    fdy = torch.roll(psi_eff, 1, 1) - torch.roll(psi_eff, -1, 1)
    fdz = torch.roll(psi_eff, 1, 2) - torch.roll(psi_eff, -1, 2)

    if use_laplacian:
        laplacian = (
            torch.roll(psi_eff, 1, 0) + torch.roll(psi_eff, -1, 0) +
            torch.roll(psi_eff, 1, 1) + torch.roll(psi_eff, -1, 1) +
            torch.roll(psi_eff, 1, 2) + torch.roll(psi_eff, -1, 2) -
            6.0 * psi_eff
        )
    else:
        laplacian = torch.zeros_like(psi_eff)

    T_vec = torch.stack([fdx, fdy, fdz], dim=-1)
    mag_T = torch.norm(T_vec, dim=-1) + 1e-8

    if rotator != 'emergent':
        T_vec = quantize_tensors(T_vec, mag_T, rotator)

    T_nbr = (
        torch.roll(T_vec, 1, 0) + torch.roll(T_vec, -1, 0) +
        torch.roll(T_vec, 1, 1) + torch.roll(T_vec, -1, 1) +
        torch.roll(T_vec, 1, 2) + torch.roll(T_vec, -1, 2)
    ) / 6.0
    mag_N = torch.norm(T_nbr, dim=-1) + 1e-8
    cos_theta = torch.sum(T_vec * T_nbr, dim=-1) / (mag_T * mag_N)

    lap_omega = discrete_laplacian(omega)
    lap_abs = lap_omega.abs()
    lap_mean = lap_abs.mean() + 1e-10
    curvature_local = torch.tanh(lap_abs / lap_mean)

    local_diffusion = rates['diffusion'] * (1.0 - 0.5 * curvature_local)
    local_tension_decay = rates['tension_decay'] * (1.0 + curvature_local)

    routing_pressure = 1.0 - cos_theta
    if no_pump:
        latent_transfer = torch.zeros_like(omega)
    else:
        latent_transfer = w_alpha_t * routing_pressure * rates['coupling'] * kappa
    new_omega = torch.relu(
        omega + latent_transfer - (mag_T * local_tension_decay * cos_theta)
    )

    if not propagate:
        psi_update = laplacian * local_diffusion
    else:
        psi_update = (
            (T_vec[..., 0] + T_vec[..., 1] + T_vec[..., 2])
            * rates['propagation'] * new_omega / g_mu_nu +
            laplacian * local_diffusion
        )
        if not no_pump:
            psi_update = psi_update + w_alpha_t * rates['coupling'] * kappa

    if use_advection:
        dpsi_x = torch.roll(psi, 1, 0) - torch.roll(psi, -1, 0)
        dpsi_y = torch.roll(psi, 1, 1) - torch.roll(psi, -1, 1)
        dpsi_z = torch.roll(psi, 1, 2) - torch.roll(psi, -1, 2)
        v_unit = T_vec / mag_T.unsqueeze(-1)
        advection = (v_unit[..., 0] * dpsi_x +
                     v_unit[..., 1] * dpsi_y +
                     v_unit[..., 2] * dpsi_z)
        psi_update = psi_update - rates['advection'] * advection

    new_psi = psi + psi_update
    return new_psi, new_omega


# =============================================================================
# TWO-BODY INJECTION
# =============================================================================

def inject_twobody(grid_size, device, dx, severity, separation=10.0,
                   scale=3.0, t_offset=0.0, sigma=0.5):
    try:
        from mpmath import mp, zeta as mpzeta
    except ImportError:
        raise ImportError("mpmath required: pip install mpmath")

    print(f"\n--- BIFURCATION: TWO-BODY ZETA ---")
    print(f"  Grid: {grid_size}³ | Scale: {scale} | σ={sigma}")
    print(f"  Separation: {separation} | Centers: (±{separation/2:.1f}, 0, 0)")

    mp.dps = 25

    coords = torch.linspace(-10, 10, grid_size)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

    cx = separation / 2.0
    R_A = torch.sqrt((X + cx)**2 + Y**2 + Z**2 + dx**2)
    R_B = torch.sqrt((X - cx)**2 + Y**2 + Z**2 + dx**2)

    r_max = max(float(R_A.max()), float(R_B.max()))
    n_samples = 1000
    r_samples = np.linspace(0, r_max, n_samples)

    print(f"  Evaluating ζ({sigma}+it) on {n_samples} radial samples "
          f"(r_max={r_max:.2f})...")
    zeta_vals = np.zeros(n_samples, dtype=complex)
    for k, r in enumerate(r_samples):
        t = float(r * scale + t_offset)
        zeta_vals[k] = complex(mpzeta(sigma + 1j * t))

    zeta_abs = np.abs(zeta_vals)
    z_max = zeta_abs.max()
    w_profile = zeta_abs / (z_max + 1e-10)
    zeta_phase = np.angle(zeta_vals)

    from scipy.interpolate import interp1d
    interp_w = interp1d(r_samples, w_profile, kind='cubic',
                        fill_value='extrapolate')
    interp_ph = interp1d(r_samples, zeta_phase, kind='cubic',
                         fill_value='extrapolate')

    R_A_np = R_A.numpy()
    w_A = torch.from_numpy(np.clip(interp_w(R_A_np), 0, 1).astype(np.float32))
    ph_A = torch.from_numpy(interp_ph(R_A_np).astype(np.float32))

    R_B_np = R_B.numpy()
    w_B = torch.from_numpy(np.clip(interp_w(R_B_np), 0, 1).astype(np.float32))
    ph_B = torch.from_numpy(interp_ph(R_B_np).astype(np.float32))

    omega1 = w_A + w_B
    omega2 = torch.relu(1.0 - omega1)

    def _grad_mag(field):
        gx = torch.roll(field, 1, 0) - torch.roll(field, -1, 0)
        gy = torch.roll(field, 1, 1) - torch.roll(field, -1, 1)
        gz = torch.roll(field, 1, 2) - torch.roll(field, -1, 2)
        return torch.sqrt(gx**2 + gy**2 + gz**2)

    grad_A = _grad_mag(w_A)
    grad_B = _grad_mag(w_B)
    grad_combined = _grad_mag(omega1)

    w_alpha_base = grad_combined * severity

    n_eff = float(
        (grad_combined.sum()) ** 2 / ((grad_combined ** 2).sum() + 1e-10))

    grad_mean = grad_combined.mean()
    node_mask = grad_combined > grad_mean
    node_positions = torch.nonzero(node_mask)

    field_id_grid = (grad_B > grad_A).int()
    field_ids_tensor = field_id_grid[
        node_positions[:, 0], node_positions[:, 1], node_positions[:, 2]]

    sort_idx = torch.argsort(field_ids_tensor)
    node_positions = node_positions[sort_idx]
    field_ids_tensor = field_ids_tensor[sort_idx]
    field_ids = field_ids_tensor.numpy().astype(np.int32)

    n_A = int((field_ids == 0).sum())
    n_B = int((field_ids == 1).sum())

    phase_combined = (ph_A * grad_A + ph_B * grad_B) / (grad_A + grad_B + 1e-10)
    phase_at_nodes = phase_combined[
        node_positions[:, 0], node_positions[:, 1], node_positions[:, 2]
    ].numpy().astype(np.float32)

    registry_np = node_positions.numpy().astype(np.int32)

    zr = zeta_vals.real
    sign_changes = int(np.sum(np.diff(np.sign(zr)) != 0))

    print(f"  ζ sign changes in domain: {sign_changes}")
    print(f"  Participation ratio: {n_eff:.1f}")
    print(f"  Registry: {len(registry_np):,} nodes "
          f"(A: {n_A:,} | B: {n_B:,})")

    omega1 = omega1.to(device)
    omega2 = omega2.to(device)
    w_alpha_base = w_alpha_base.to(device)
    node_positions = node_positions.to(device)

    del X, Y, Z, coords, grad_A, grad_B, grad_combined
    return (omega1, omega2, w_alpha_base, node_positions,
            registry_np, phase_at_nodes, field_ids, n_eff, n_A, n_B)


# =============================================================================
# CLUSTER RECORDER — lightweight bulk dynamics per body
# =============================================================================

class ClusterRecorder:
    def __init__(self, grid_size, dx):
        self.grid_size = grid_size
        self.dx = dx
        self.steps = []
        self.com_A = []
        self.com_B = []
        self.mass_A = []
        self.mass_B = []
        self.n_active_A = []
        self.n_active_B = []
        self.coherence_A = []
        self.coherence_B = []
        self.separation = []
        self.boundary_omega = []

    def record(self, step, omega1, psi1, node_positions, n_A,
               active_threshold, grid_size, device):
        self.steps.append(step)

        idx = node_positions
        values = omega1[idx[:, 0], idx[:, 1], idx[:, 2]]
        active = values > active_threshold

        phys = (idx.float() - grid_size / 2.0) * (20.0 / grid_size)

        active_A = active.clone()
        active_A[n_A:] = False
        na_A = int(active_A.sum().item())
        self.n_active_A.append(na_A)

        if na_A > 0:
            w_A = values[active_A]
            p_A = phys[active_A]
            w_sum = w_A.sum().item()
            com = (p_A * w_A.unsqueeze(-1)).sum(dim=0) / w_sum
            self.com_A.append(com.cpu().numpy())
            self.mass_A.append(w_sum)
        else:
            self.com_A.append(np.zeros(3))
            self.mass_A.append(0.)

        active_B = active.clone()
        active_B[:n_A] = False
        na_B = int(active_B.sum().item())
        self.n_active_B.append(na_B)

        if na_B > 0:
            w_B = values[active_B]
            p_B = phys[active_B]
            w_sum = w_B.sum().item()
            com = (p_B * w_B.unsqueeze(-1)).sum(dim=0) / w_sum
            self.com_B.append(com.cpu().numpy())
            self.mass_B.append(w_sum)
        else:
            self.com_B.append(np.zeros(3))
            self.mass_B.append(0.)

        psi_vals = psi1[idx[:, 0], idx[:, 1], idx[:, 2]]
        self.coherence_A.append(
            float(psi_vals[active_A].sign().mean().abs().item())
            if na_A > 0 else 0.)
        self.coherence_B.append(
            float(psi_vals[active_B].sign().mean().abs().item())
            if na_B > 0 else 0.)

        if na_A > 0 and na_B > 0:
            self.separation.append(
                float(np.linalg.norm(self.com_A[-1] - self.com_B[-1])))
        else:
            self.separation.append(0.)

        mid_band = phys[:, 0].abs() < self.dx * 5
        boundary = mid_band & active
        self.boundary_omega.append(
            float(values[boundary].sum().item()) if boundary.any() else 0.)

    def save(self, run_dir):
        com_A = np.array(self.com_A)
        com_B = np.array(self.com_B)
        steps = np.array(self.steps, dtype=float)

        data = {
            'steps': steps,
            'com_A': com_A, 'com_B': com_B,
            'mass_A': np.array(self.mass_A),
            'mass_B': np.array(self.mass_B),
            'n_active_A': np.array(self.n_active_A),
            'n_active_B': np.array(self.n_active_B),
            'coherence_A': np.array(self.coherence_A),
            'coherence_B': np.array(self.coherence_B),
            'separation': np.array(self.separation),
            'boundary_omega': np.array(self.boundary_omega),
        }

        if len(steps) > 1:
            dt = np.diff(steps)
            vel_A = np.diff(com_A, axis=0) / dt[:, np.newaxis]
            vel_B = np.diff(com_B, axis=0) / dt[:, np.newaxis]
            data['vel_A'] = vel_A
            data['vel_B'] = vel_B
            mid_mass_A = (data['mass_A'][:-1] + data['mass_A'][1:]) / 2.0
            mid_mass_B = (data['mass_B'][:-1] + data['mass_B'][1:]) / 2.0
            data['momentum_A'] = vel_A * mid_mass_A[:, np.newaxis]
            data['momentum_B'] = vel_B * mid_mass_B[:, np.newaxis]
            if len(dt) > 1:
                dt2 = (dt[:-1] + dt[1:]) / 2.0
                data['acc_A'] = np.diff(vel_A, axis=0) / dt2[:, np.newaxis]
                data['acc_B'] = np.diff(vel_B, axis=0) / dt2[:, np.newaxis]

        np.savez(os.path.join(run_dir, 'clusters.npz'), **data)


# =============================================================================
# CLOUD RECORDER — extended with per-body active counts
# =============================================================================

class CloudRecorder:
    def __init__(self):
        self.data = {}

    def record(self, step, omega1, omega2, psi1, node_positions, grid_size,
               g_mu_nu, kappa, device, n_A,
               active_threshold=0.5,
               membrane=None, L1=None, L2=None, R=None):
        idx = node_positions
        values = omega1[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
        active_total = int((values > active_threshold).sum())
        active_A = int((values[:n_A] > active_threshold).sum())
        active_B = int((values[n_A:] > active_threshold).sum())
        mean_t = float(values.mean())

        psi_eff = kappa * g_mu_nu * psi1
        tdx = torch.roll(psi_eff, 1, 0) - torch.roll(psi_eff, -1, 0)
        tdy = torch.roll(psi_eff, 1, 1) - torch.roll(psi_eff, -1, 1)
        tdz = torch.roll(psi_eff, 1, 2) - torch.roll(psi_eff, -1, 2)
        T_vec = torch.stack([tdx, tdy, tdz], dim=-1)

        tvec_at_nodes = T_vec[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
        mag = np.linalg.norm(tvec_at_nodes, axis=1) + 1e-8
        unit = tvec_at_nodes / mag[:, np.newaxis]

        prefix = f"s{step:04d}"
        self.data[f"{prefix}_values"] = values.astype(np.float32)
        self.data[f"{prefix}_active"] = np.array(
            [active_total, active_A, active_B], dtype=np.int32)
        self.data[f"{prefix}_mean_t"] = np.array([mean_t], dtype=np.float32)
        self.data[f"{prefix}_tvec_unit"] = unit.astype(np.float32)
        self.data[f"{prefix}_tvec_mag"] = mag.astype(np.float32)

        if membrane is not None:
            mem = membrane[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
            self.data[f"{prefix}_membrane"] = mem.astype(np.float32)
        if L1 is not None:
            self.data[f"{prefix}_L1_rms"] = np.array(
                [float(torch.sqrt((L1**2).mean()).item())], dtype=np.float32)
        if L2 is not None:
            self.data[f"{prefix}_L2_rms"] = np.array(
                [float(torch.sqrt((L2**2).mean()).item())], dtype=np.float32)
        if membrane is not None and R is not None:
            m_sum = membrane.sum().item()
            fold_r = float(
                (membrane * R).sum().item() / m_sum) if m_sum > 0 else 0.0
            self.data[f"{prefix}_fold_radius"] = np.array(
                [fold_r], dtype=np.float32)

    def save(self, run_dir):
        if self.data:
            np.savez(os.path.join(run_dir, 'clouds.npz'), **self.data)


# =============================================================================
# MAIN SIMULATION — TWO-BODY
# =============================================================================

def run_twobody_simulation(
        steps, grid_size, separation,
        scale, t_offset, sigma_val,
        beat_detune, no_beat,
        use_laplacian, use_exchange, use_advection,
        no_pump, single_shot,
        probe_interval,
        rotator='emergent',
        auto=False,
        time_sig=(4, 4),
        cross_rhythm=None,
        force_device=None,
        ternary=False,
        quant_levels=None,
        save_fields=False,
        perturb=0.0,
        absorb_width=0):

    run_id = next_run_id()
    run_dir = setup_run_dir(run_id)
    if force_device:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if auto:
        scale = compute_auto_scale(t_offset=t_offset)
        grid_size = compute_auto_grid(scale, t_offset=t_offset)
        print(f"\n  AUTO-SCALE: {scale:.6f} | AUTO-GRID: {grid_size}")

    if grid_size % 2 == 0:
        grid_size -= 1

    dx = 20.0 / (grid_size - 1)

    print(f"\n=== Resonant Field Engine — TWO-BODY v0.1 ===")
    print(f"Run: {run_id:04d} → {run_dir}")
    print(f"Device: {device} | Grid: {grid_size}³ | Steps: {steps}")
    print(f"Separation: {separation} | Rotator: {rotator}")
    if absorb_width > 0:
        print(f"Boundary: absorbing (Hann taper, width={absorb_width})")
    else:
        print(f"Boundary: periodic (torus)")
    print(f"All constants derived from field")

    c = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(c, c, c, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    absorb_mask = build_absorb_mask(grid_size, absorb_width, device)

    (omega1, omega2, w_alpha_base, node_positions,
     registry_np, phase_at_nodes, field_ids, n_eff, n_A, n_B) = \
        inject_twobody(grid_size, device, dx, 1.0, separation,
                       scale, t_offset, sigma_val)

    n_registry = len(registry_np)

    rates = compute_field_rates(omega1, dx, n_eff)
    w_alpha_base = w_alpha_base * rates['severity']
    sigma_mem = rates['sigma']

    print(f"\n--- FIELD-DERIVED SCALES ---")
    print(f"  natural_freq:    {rates['_natural_freq']:.4f}")
    print(f"  curvature_ratio: {rates['_curvature_ratio']:.4f}")
    print(f"  base (CFL):      {rates['_base']:.6f}")
    print(f"  propagation:     {rates['propagation']:.6f}")
    print(f"  diffusion:       {rates['diffusion']:.6f}")
    print(f"  severity:        {rates['severity']:.4f}")
    print(f"  active_thresh:   {rates['active_threshold']:.4f}")

    if auto:
        beat_detune = rates['_base']
        beats_per_measure, beat_unit = time_sig
        beat_detune_eff = beat_detune * beats_per_measure / beat_unit
        cross_detune = None
        if cross_rhythm:
            cr_beats, cr_unit = cross_rhythm
            cross_detune = beat_detune * cr_beats / cr_unit
        beat_period = round(1.0 / beat_detune_eff)
        no_beat = False
        sig_str = f"{beats_per_measure}/{beat_unit}"
        print(f"\n--- AUTO-DERIVED TIMING ---")
        print(f"  time_signature:  {sig_str}")
        print(f"  beat_period:     {beat_period} steps")
    else:
        beat_detune_eff = beat_detune
        cross_detune = None
        beat_period = None

    if perturb > 0:
        amp = perturb * float(omega1.mean().item())
        omega1 = omega1 + amp * (Z / (R + dx))
        omega1 = torch.relu(omega1)
        print(f"  PERTURB: dipole, amplitude {amp:.6f}")

    g_mu_nu = build_metric_from_field(omega1, omega2, rates['metric_floor'])

    gx = torch.roll(omega1, 1, 0) - torch.roll(omega1, -1, 0)
    gy = torch.roll(omega1, 1, 1) - torch.roll(omega1, -1, 1)
    gz = torch.roll(omega1, 1, 2) - torch.roll(omega1, -1, 2)
    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2)
    grad_sum = grad_mag.sum()
    if grad_sum > 0:
        psi1 = grad_mag * (omega1.sum() / grad_sum)
    else:
        psi1 = torch.ones_like(omega1) * 0.01
    psi2 = psi1.clone()
    del gx, gy, gz, grad_mag

    meta = {
        'engine_version':    'twobody-0.1',
        'run_id':            run_id,
        'grid_size':         grid_size,
        'dx':                dx,
        'separation':        separation,
        'rotator':           rotator,
        'bifurcation':       'twobody-zeta',
        'scale':             scale,
        'sigma':             sigma_mem,
        't_offset':          t_offset,
        'auto':              auto,
        'auto_grid':         grid_size if auto else None,
        'auto_beat_period':  beat_period if auto else None,
        'time_sig':          f"{time_sig[0]}/{time_sig[1]}",
        'cross_rhythm':      f"{cross_rhythm[0]}/{cross_rhythm[1]}"
                             if cross_rhythm else None,
        'constants_type':    'all_field_derived',
        'rates':             {k: v for k, v in rates.items()
                              if not k.startswith('_')},
        'field_stats':       {
            'natural_freq':    rates['_natural_freq'],
            'curvature_ratio': rates['_curvature_ratio'],
            'omega_rms':       rates['_omega_rms'],
            'grad_rms':        rates['_grad_rms'],
            'lap_rms':         rates['_lap_rms'],
        },
        'n_eff':             n_eff,
        'n_registry':        n_registry,
        'n_body_A':          n_A,
        'n_body_B':          n_B,
        'use_laplacian':     use_laplacian,
        'use_exchange':      use_exchange,
        'use_advection':     use_advection,
        'no_pump':           no_pump,
        'single_shot':       single_shot,
        'beat_detune':       beat_detune,
        'no_beat':           no_beat,
        'probe_interval':    probe_interval,
        'ternary_exchange':  ternary,
        'quant_levels':      quant_levels,
        'absorb_width':      absorb_width,
        'boundary':          'absorbing' if absorb_width > 0 else 'periodic',
        'collapse_step':     None,
        'total_steps':       0,
        'wall_seconds':      0,
    }

    np.save(os.path.join(run_dir, 'registry.npy'), registry_np)
    np.save(os.path.join(run_dir, 'phase.npy'), phase_at_nodes)
    np.save(os.path.join(run_dir, 'field_ids.npy'), field_ids)

    tracker = EnergyTracker()
    clouds = CloudRecorder()
    clusters = ClusterRecorder(grid_size, dx)

    del c

    print(f"\n--- RUNNING --- "
          f"(A: {n_A:,} | B: {n_B:,} | rotator: {rotator})")
    print(f"{'Step':>6} | {'Sigma_o1':>10} | {'Sigma_o2':>10} | "
          f"{'flux_o':>10} | {'fold_R':>7} | "
          f"{'A_act':>6} | {'B_act':>6} | {'sep':>7}")
    print("-" * 82)

    wall_start = _time.time()
    final_step = 0
    membrane = None
    L1 = L2 = None
    last_sep = separation

    max_steps = steps if not auto else steps * 10
    period_flux = []
    prev_period_mean = None
    stable_count = 0
    auto_stopped = False

    i = 0
    while i < max_steps:
        kappa1 = kappa_H(omega1)
        kappa2 = kappa_H(omega2)

        do_propagate = not single_shot or i == 0

        if single_shot and i > 0:
            w_alpha_t = w_alpha_base * 0.0
        elif no_beat:
            w_alpha_t = w_alpha_base
        else:
            mod = math.cos(2.0 * math.pi * beat_detune_eff * i)
            if cross_detune:
                mod *= math.cos(2.0 * math.pi * cross_detune * i)
            w_alpha_t = w_alpha_base * mod

        psi1, omega1 = wave_step(
            psi1, omega1, g_mu_nu, w_alpha_t, kappa1, rates,
            use_laplacian=use_laplacian, use_advection=use_advection,
            no_pump=no_pump, propagate=do_propagate, rotator=rotator)
        psi2, omega2 = wave_step(
            psi2, omega2, g_mu_nu, w_alpha_t, kappa2, rates,
            use_laplacian=use_laplacian, use_advection=use_advection,
            no_pump=no_pump, propagate=do_propagate, rotator=rotator)

        if absorb_mask is not None:
            psi1 = psi1 * absorb_mask
            psi2 = psi2 * absorb_mask
            omega1 = torch.relu(omega1 * absorb_mask)
            omega2 = torch.relu(omega2 * absorb_mask)

        flux_omega = 0.0
        flux_psi = 0.0
        fold_r = 0.0

        if use_exchange:
            membrane, L1, L2 = compute_membrane(omega1, omega2, sigma_mem)

            if quant_levels is not None:
                m_gate = (membrane > 0.5).float()
                if quant_levels <= 3:
                    q_omega = torch.sign(omega1 - omega2)
                    q_psi = torch.sign(psi1 - psi2)
                else:
                    half_n = (quant_levels - 1) // 2
                    raw_o = omega1 - omega2
                    raw_p = psi1 - psi2
                    o_scale = raw_o.abs().max() + 1e-10
                    p_scale = raw_p.abs().max() + 1e-10
                    q_omega = torch.round(
                        raw_o / o_scale * half_n
                    ).clamp(-half_n, half_n) / half_n
                    q_psi = torch.round(
                        raw_p / p_scale * half_n
                    ).clamp(-half_n, half_n) / half_n
                delta_omega = m_gate * rates['_base'] * q_omega
                delta_psi = m_gate * rates['_base'] * q_psi
            else:
                delta_omega = membrane * rates['_base'] * (omega1 - omega2)
                delta_psi = membrane * rates['_base'] * (psi1 - psi2)

            omega1 = torch.relu(omega1 - delta_omega)
            omega2 = torch.relu(omega2 + delta_omega)
            psi1 = psi1 - delta_psi
            psi2 = psi2 + delta_psi

            flux_omega = float(delta_omega.abs().sum().item())
            flux_psi = float(delta_psi.abs().sum().item())

            m_sum = membrane.sum().item()
            if m_sum > 0:
                fold_r = float((membrane * R).sum().item() / m_sum)

        g_mu_nu = build_metric_from_field(omega1, omega2, rates['metric_floor'])
        final_step = i

        if i % probe_interval == 0:
            tracker.record(i, omega1, omega2, psi1, psi2,
                           flux_psi, flux_omega,
                           X=X, Y=Y, Z=Z, R=R, membrane=membrane)
            clusters.record(i, omega1, psi1, node_positions, n_A,
                            rates['active_threshold'], grid_size, device)
            if clusters.separation:
                last_sep = clusters.separation[-1]

        is_last = (not auto and i == steps - 1)
        record_cloud = ((i < 200 and i % 10 == 0) or
                        (i >= 200 and i % 50 == 0) or
                        is_last)

        if record_cloud:
            clouds.record(i, omega1, omega2, psi1, node_positions,
                          grid_size, g_mu_nu, kappa1, device, n_A,
                          active_threshold=rates['active_threshold'],
                          membrane=membrane, L1=L1, L2=L2, R=R)

        if i % max(1, 50) == 0 or i < 5 or is_last:
            with torch.no_grad():
                vals_A = omega1[node_positions[:n_A, 0],
                                node_positions[:n_A, 1],
                                node_positions[:n_A, 2]]
                vals_B = omega1[node_positions[n_A:, 0],
                                node_positions[n_A:, 1],
                                node_positions[n_A:, 2]]
                act_A = int(
                    (vals_A > rates['active_threshold']).sum().item())
                act_B = int(
                    (vals_B > rates['active_threshold']).sum().item())
            print(f"{i:>6} | {omega1.sum().item():>10.3e} | "
                  f"{omega2.sum().item():>10.3e} | "
                  f"{flux_omega:>10.3e} | {fold_r:>7.3f} | "
                  f"{act_A:>6d} | {act_B:>6d} | {last_sep:>7.2f}")

        if torch.isnan(omega1).any() or torch.isinf(omega1).any():
            print(f"\n  COLLAPSE (field 1) at step {i}")
            meta['collapse_step'] = i
            break
        if torch.isnan(omega2).any() or torch.isinf(omega2).any():
            print(f"\n  COLLAPSE (field 2) at step {i}")
            meta['collapse_step'] = i
            break

        if auto and beat_period and use_exchange:
            period_flux.append(flux_omega)
            if (i + 1) % beat_period == 0 and i >= 3 * beat_period:
                current_mean = sum(
                    period_flux[-beat_period:]) / beat_period
                if prev_period_mean is not None:
                    rel_change = abs(
                        current_mean - prev_period_mean
                    ) / max(prev_period_mean, 1e-10)
                    if rel_change < rates['_base']:
                        stable_count += 1
                        if stable_count >= 2:
                            print(f"\n  AUTO-STOP at step {i+1}: "
                                  f"exchange stable")
                            auto_stopped = True
                            clouds.record(
                                i, omega1, omega2, psi1,
                                node_positions, grid_size,
                                g_mu_nu, kappa1, device, n_A,
                                active_threshold=
                                    rates['active_threshold'],
                                membrane=membrane,
                                L1=L1, L2=L2, R=R)
                            final_step = i
                            break
                    else:
                        stable_count = 0
                prev_period_mean = current_mean

        if not auto and i >= steps - 1:
            break
        i += 1

    wall_elapsed = _time.time() - wall_start
    meta['total_steps'] = final_step + 1
    meta['auto_stopped'] = auto_stopped if auto else None
    meta['wall_seconds'] = round(wall_elapsed, 1)

    if save_fields:
        np.savez(os.path.join(run_dir, 'fields.npz'),
                 omega1=omega1.cpu().numpy(),
                 omega2=omega2.cpu().numpy(),
                 psi1=psi1.cpu().numpy(),
                 psi2=psi2.cpu().numpy(),
                 membrane=membrane.cpu().numpy()
                     if membrane is not None else np.array([]))

    tracker.save(run_dir)
    clouds.save(run_dir)
    clusters.save(run_dir)
    with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    update_latest(run_id)

    if len(tracker.steps) > 1:
        e0 = tracker.total_omega_1[0] + tracker.total_omega_2[0]
        ef = tracker.total_omega_1[-1] + tracker.total_omega_2[-1]
        drift = (ef - e0) / max(abs(e0), 1e-12)
        n_periods = (final_step + 1) / beat_period if beat_period else 0

        print(f"\n=== Run {run_id:04d} complete ===")
        print(f"Steps: {final_step + 1}" +
              (f" ({n_periods:.1f} beat periods)" if beat_period else ""))
        print(f"Body A: {n_A:,} nodes → "
              f"{clusters.n_active_A[-1]} active")
        print(f"Body B: {n_B:,} nodes → "
              f"{clusters.n_active_B[-1]} active")
        print(f"Final separation: {clusters.separation[-1]:.3f}")
        print(f"Σω drift: {drift*100:+.3f}%")
        print(f"Wall time: {wall_elapsed:.1f}s")
        print(f"Output: {run_dir}")

    return run_dir


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Resonant Field Engine — Two-Body v0.1')

    p.add_argument('--grid', type=int, default=89)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--separation', type=float, default=10.0,
                   help='Distance between body centers (default: 10.0)')
    p.add_argument('--rotator', type=str, default='emergent',
                   choices=['emergent', 'pi', 'laplacian'],
                   help='Tensor quantization mode')

    p.add_argument('--scale', type=float, default=3.0)
    p.add_argument('--t-offset', type=float, default=0.0)
    p.add_argument('--sigma', type=float, default=0.5)

    p.add_argument('--beat', type=float, default=0.0)
    p.add_argument('--no-beat', action='store_true')
    p.add_argument('--no-laplacian', action='store_true')
    p.add_argument('--no-exchange', action='store_true')
    p.add_argument('--advection', action='store_true')
    p.add_argument('--no-pump', action='store_true')
    p.add_argument('--single-shot', action='store_true')
    p.add_argument('--probe-interval', type=int, default=1)

    p.add_argument('--auto', action='store_true',
                   help='Field determines grid, beat, and step count')
    p.add_argument('--time-sig', type=str, default='4/4')
    p.add_argument('--cross-rhythm', type=str, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--ternary', action='store_true')
    p.add_argument('--quant', type=int, default=None)
    p.add_argument('--save-fields', action='store_true')
    p.add_argument('--perturb', type=float, default=0.0)
    p.add_argument('--absorb', type=int, default=0,
                   help='Absorbing boundary layer width in cells (0=periodic)')

    args = p.parse_args()

    run_twobody_simulation(
        steps          = args.steps,
        grid_size      = args.grid,
        separation     = args.separation,
        scale          = args.scale,
        t_offset       = args.t_offset,
        sigma_val      = args.sigma,
        beat_detune    = args.beat,
        no_beat        = args.no_beat or args.beat == 0.0,
        use_laplacian  = not args.no_laplacian,
        use_exchange   = not args.no_exchange,
        use_advection  = args.advection,
        no_pump        = args.no_pump,
        single_shot    = args.single_shot,
        probe_interval = args.probe_interval,
        rotator        = args.rotator,
        auto           = args.auto,
        time_sig       = tuple(int(x) for x in args.time_sig.split('/')),
        cross_rhythm   = tuple(int(x) for x in args.cross_rhythm.split('/'))
                         if args.cross_rhythm else None,
        force_device   = args.device,
        ternary        = args.ternary,
        quant_levels   = args.quant if args.quant else
                         (3 if args.ternary else None),
        save_fields    = args.save_fields,
        perturb        = args.perturb,
        absorb_width   = args.absorb,
    )
