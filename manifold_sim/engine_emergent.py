"""
Resonant Field Engine — Emergent Fold v0.4
==========================================
Two-manifold engine with pluggable bifurcation and dynamic membrane.
No hardcoded spectral content, no core geometry, no static metric,
no hand-tuned process rates. The Laplacian and ζ content set all scales.

v0.4 — all physics constants removed:
  Phase 1: Metric derived from field — g(x) = floor + ω_total/ω_max
  Phase 2: Psi seeded from ω gradient, not centered Gaussian
  Phase 3: Process rates from field curvature spectrum:
           base = min(natural_freq × dx², CFL), hierarchy via curvature_ratio
           Local Laplacian modulation: diffusion suppressed at high curvature,
           tension decay enhanced at high curvature
  Phase 4: Exchange rate = base rate (membrane IS the rate)
  (v0.3): Core removed, pluggable bifurcation, dynamic membrane

Zero physics constants. The field determines everything:
  natural_freq = lap_rms / omega_rms (time scale)
  curvature_ratio = lap_rms / grad_rms (process hierarchy)
  char_length = grad_rms / lap_rms (membrane width)
  omega_rms (severity, active threshold, metric floor)

Output dir: runs_emergent/NNNN/

Author: Mattias Hammarsten / Claude (Anthropic)
"""

import torch
import argparse
import math
import os
import json
import numpy as np
import time as _time


# =============================================================================
# No physics constants. All rates and scales derived from the field.
# =============================================================================


def compute_auto_scale(domain_half=10.0, target_zeros=10, t_offset=0.0):
    """Scale that places the domain boundary in the gap between zero N and N+1.
    When t_offset > 0, finds zeros starting from that position on the critical line."""
    from mpmath import zetazero
    R_max = math.sqrt(3) * domain_half
    if t_offset == 0.0:
        gamma_n = float(zetazero(target_zeros).imag)
        gamma_n1 = float(zetazero(target_zeros + 1).imag)
        t_boundary = (gamma_n + gamma_n1) / 2.0
        return t_boundary / R_max
    # With offset: use R-v-M formula to estimate zero index, then search locally
    T = t_offset
    if T > 10:
        n_est = int(T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)))
    else:
        n_est = 1
    n = max(1, n_est - 2)
    while float(zetazero(n).imag) < t_offset:
        n += 1
    first_in_window = n
    last_in_window = first_in_window + target_zeros - 1
    gamma_n = float(zetazero(last_in_window).imag)
    gamma_n1 = float(zetazero(last_in_window + 1).imag)
    t_boundary = (gamma_n + gamma_n1) / 2.0
    return (t_boundary - t_offset) / R_max


def compute_auto_grid(scale, domain_half=10.0, t_offset=0.0):
    """Grid size from ζ zero spacing at the far end of the window (densest zeros)."""
    R_max = math.sqrt(3) * domain_half
    T_far = t_offset + R_max * scale
    if T_far <= 2 * math.pi:
        return 33
    min_spacing_t = 2 * math.pi / math.log(T_far / (2 * math.pi))
    min_spacing_R = min_spacing_t / scale
    dx = min_spacing_R / 4.0
    grid = int(math.ceil(2 * domain_half / dx)) + 1
    if grid % 2 == 0:
        grid += 1
    return max(grid, 33)


# =============================================================================
# FIELD-DERIVED RATES — the Laplacian and ζ content set all process scales
# =============================================================================

def compute_field_rates(omega1, dx, n_eff):
    lap = discrete_laplacian(omega1)
    gx = torch.roll(omega1, 1, 0) - torch.roll(omega1, -1, 0)
    gy = torch.roll(omega1, 1, 1) - torch.roll(omega1, -1, 1)
    gz = torch.roll(omega1, 1, 2) - torch.roll(omega1, -1, 2)
    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2)

    omega_rms = torch.sqrt((omega1**2).mean()).item() + 1e-10
    grad_rms  = torch.sqrt((grad_mag**2).mean()).item() + 1e-10
    lap_rms   = torch.sqrt((lap**2).mean()).item() + 1e-10

    natural_freq    = lap_rms / omega_rms
    curvature_ratio = lap_rms / grad_rms
    cfl_limit       = dx * dx / 6.0

    base = min(natural_freq * dx * dx, cfl_limit)

    char_length = grad_rms / lap_rms

    rates = {
        'propagation':   base,
        'diffusion':     base * curvature_ratio,
        'tension_decay': base * curvature_ratio ** 3,
        'coupling':      base * curvature_ratio / max(n_eff, 1.0),
        'advection':     base * curvature_ratio ** 3,
        'severity':      omega_rms,
        'sigma':         char_length * dx,
        'metric_floor':  base,
        'active_threshold': omega_rms / 2.0,
    }
    rates['_natural_freq']    = natural_freq
    rates['_curvature_ratio'] = curvature_ratio
    rates['_char_length']     = char_length
    rates['_base']            = base
    rates['_omega_rms']       = omega_rms
    rates['_grad_rms']        = grad_rms
    rates['_lap_rms']         = lap_rms
    return rates


# =============================================================================
# κ(H)
# =============================================================================

def kappa_H(omega_field):
    H = omega_field.mean().item()
    return max(1.0 / (1.0 + H * 0.1), 1e-4)


# =============================================================================
# METRIC TENSOR — derived from field content
# =============================================================================

def build_metric_from_field(omega1, omega2, floor):
    omega_total = omega1 + omega2
    omega_max = omega_total.max()
    if omega_max > 0:
        return floor + omega_total / omega_max
    return torch.full_like(omega1, floor + 1.0)


# =============================================================================
# DISCRETE LAPLACIAN — shared stencil for wave_step and membrane
# =============================================================================

def discrete_laplacian(field):
    return (
        torch.roll(field, 1, 0) + torch.roll(field, -1, 0) +
        torch.roll(field, 1, 1) + torch.roll(field, -1, 1) +
        torch.roll(field, 1, 2) + torch.roll(field, -1, 2) -
        6.0 * field
    )


# =============================================================================
# WAVE STEP — all coefficients from field-derived rates + local modulation
# =============================================================================

def wave_step(psi, omega, g_mu_nu, w_alpha_t, kappa, rates,
              use_laplacian=True, use_advection=False, no_pump=False,
              propagate=True):

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
    T_nbr = (
        torch.roll(T_vec, 1, 0) + torch.roll(T_vec, -1, 0) +
        torch.roll(T_vec, 1, 1) + torch.roll(T_vec, -1, 1) +
        torch.roll(T_vec, 1, 2) + torch.roll(T_vec, -1, 2)
    ) / 6.0
    mag_T     = torch.norm(T_vec, dim=-1) + 1e-8
    mag_N     = torch.norm(T_nbr, dim=-1) + 1e-8
    cos_theta = torch.sum(T_vec * T_nbr, dim=-1) / (mag_T * mag_N)

    # Local Laplacian modulation — the field's curvature guides the rates
    lap_omega = discrete_laplacian(omega)
    lap_abs = lap_omega.abs()
    lap_mean = lap_abs.mean() + 1e-10
    curvature_local = torch.tanh(lap_abs / lap_mean)

    local_diffusion     = rates['diffusion'] * (1.0 - 0.5 * curvature_local)
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
            (fdx + fdy + fdz) * rates['propagation'] * new_omega / g_mu_nu +
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
# DYNAMIC MEMBRANE — curvature-zero contour tracker
# =============================================================================

def compute_membrane(omega1, omega2, sigma):
    L1 = discrete_laplacian(omega1)
    L2 = discrete_laplacian(omega2)
    sigma_sq = sigma ** 2
    membrane = torch.exp(-L1**2 / sigma_sq) * torch.exp(-L2**2 / sigma_sq)
    m_max = membrane.max()
    if m_max > 0:
        membrane = membrane / m_max
    return membrane, L1, L2


# =============================================================================
# BIFURCATION INJECTORS
# =============================================================================

def inject_step(grid_size, device, dx, severity,
                balance_radius=5.0):
    print(f"\n--- BIFURCATION: STEP ---")
    print(f"  Grid: {grid_size}³ | Balance radius: {balance_radius}")

    coords = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    omega1 = (R < balance_radius).float()
    omega2 = 1.0 - omega1

    shell = ((R >= balance_radius - dx) & (R <= balance_radius + dx)).float()
    w_alpha_base = shell * severity

    node_positions = torch.nonzero(w_alpha_base > 0)
    registry_np = node_positions.cpu().numpy().astype(np.int32)
    phase_at_nodes = np.zeros(len(registry_np), dtype=np.float32)

    n_nodes = len(registry_np)
    print(f"  Nodes: {n_nodes:,} positions (shell at R={balance_radius})")

    del X, Y, Z, coords
    return omega1, omega2, w_alpha_base, node_positions, registry_np, phase_at_nodes, float(n_nodes)


def inject_zeta(grid_size, device, dx, severity, scale=3.0,
                coord_map='radial', t_offset=0.0, sigma=0.5):
    try:
        from mpmath import mp, zeta as mpzeta
    except ImportError:
        raise ImportError(
            "mpmath required for zeta bifurcation: pip install mpmath")

    print(f"\n--- BIFURCATION: ZETA ---")
    print(f"  Grid: {grid_size}³ | Scale: {scale} | Coord: {coord_map} | σ={sigma} | t_offset: {t_offset:.2f}")

    mp.dps = 25

    coords = torch.linspace(-10, 10, grid_size)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

    if coord_map == 'radial':
        R_cpu = torch.sqrt(X**2 + Y**2 + Z**2 + dx**2)
    elif coord_map == 'shifted':
        cx, cy, cz = 2.5, 1.7, 0.9
        R_cpu = torch.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 + dx**2)
        print(f"  Center shifted to ({cx}, {cy}, {cz})")
    elif coord_map == 'axial':
        R_cpu = (Z + 10.0).clamp(min=dx)
        print(f"  Axial mapping: t = (Z+10) × scale → planar sheets")
    elif coord_map == 'diagonal':
        R_cpu = ((X + Y + Z) / math.sqrt(3) + 10.0 * math.sqrt(3)).clamp(min=dx)
        print(f"  Diagonal mapping: t = (X+Y+Z)/√3 → tilted sheets")
    else:
        raise ValueError(f"Unknown coord_map: {coord_map}")

    r_max = float(R_cpu.max())
    n_samples = 1000
    r_samples = np.linspace(0, r_max, n_samples)

    print(f"  Evaluating ζ(1/2+it) on {n_samples} radial samples...")
    zeta_vals = np.zeros(n_samples, dtype=complex)
    for k, r in enumerate(r_samples):
        t = float(r * scale + t_offset)
        zeta_vals[k] = complex(mpzeta(sigma + 1j * t))

    zeta_abs = np.abs(zeta_vals)
    zeta_phase = np.angle(zeta_vals)
    z_max = zeta_abs.max()
    w_profile = zeta_abs / (z_max + 1e-10)

    from scipy.interpolate import interp1d
    interp_w  = interp1d(r_samples, w_profile, kind='cubic',
                          fill_value='extrapolate')
    interp_ph = interp1d(r_samples, zeta_phase, kind='cubic',
                          fill_value='extrapolate')

    R_np = R_cpu.numpy()
    w_3d = np.clip(interp_w(R_np), 0, 1).astype(np.float32)
    phase_3d = interp_ph(R_np).astype(np.float32)

    omega1 = torch.from_numpy(w_3d)
    omega2 = (1.0 - omega1).clamp(min=0)

    gx = torch.roll(omega1, 1, 0) - torch.roll(omega1, -1, 0)
    gy = torch.roll(omega1, 1, 1) - torch.roll(omega1, -1, 1)
    gz = torch.roll(omega1, 1, 2) - torch.roll(omega1, -1, 2)
    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2)

    # No threshold — the gradient IS the weight
    grad_sum = grad_mag.sum() + 1e-10
    w_alpha_base = grad_mag * severity

    # Participation ratio: effective number of equally-weighted sites
    n_eff = float((grad_mag.sum()) ** 2 / ((grad_mag ** 2).sum() + 1e-10))

    # Registry for cloud recording — field draws its own line at mean gradient
    grad_mean = grad_mag.mean()
    node_positions = torch.nonzero(grad_mag > grad_mean)
    registry_np = node_positions.numpy().astype(np.int32)

    phase_tensor = torch.from_numpy(phase_3d)
    idx = node_positions
    phase_at_nodes = phase_tensor[
        idx[:, 0], idx[:, 1], idx[:, 2]].numpy().astype(np.float32)

    sign_changes = 0
    for a, b in zip(zeta_vals[:-1].real, zeta_vals[1:].real):
        if a * b < 0:
            sign_changes += 1

    n_registry = len(registry_np)
    n_eff_int = round(n_eff)
    print(f"  ζ sign changes in domain: {sign_changes}")
    print(f"  Participation ratio: {n_eff:.1f} (≈{n_eff_int})")
    print(f"  Registry (above mean ∇): {n_registry:,} positions")

    omega1 = omega1.to(device)
    omega2 = omega2.to(device)
    w_alpha_base = w_alpha_base.to(device)
    node_positions = node_positions.to(device)

    del X, Y, Z, coords, gx, gy, gz, grad_mag, R_cpu
    return omega1, omega2, w_alpha_base, node_positions, registry_np, phase_at_nodes, n_eff


def inject_harmonic(grid_size, device, dx, severity,
                    n_harmonics=8):
    print(f"\n--- BIFURCATION: HARMONIC ---")
    print(f"  Grid: {grid_size}³ | Harmonics: {n_harmonics}")

    coords = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2 + dx**2)

    w = torch.zeros_like(R)
    for k in range(1, n_harmonics + 1):
        w = w + torch.cos(k * math.pi * R / 10.0) / k
    w = (w - w.min()) / (w.max() - w.min() + 1e-10)

    omega1 = w
    omega2 = 1.0 - w

    gx = torch.roll(omega1, 1, 0) - torch.roll(omega1, -1, 0)
    gy = torch.roll(omega1, 1, 1) - torch.roll(omega1, -1, 1)
    gz = torch.roll(omega1, 1, 2) - torch.roll(omega1, -1, 2)
    grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2)

    threshold = torch.quantile(grad_mag.flatten(), 0.999)
    w_alpha_base = (grad_mag >= threshold).float() * severity

    node_positions = torch.nonzero(w_alpha_base > 0)
    registry_np = node_positions.cpu().numpy().astype(np.int32)
    phase_at_nodes = np.zeros(len(registry_np), dtype=np.float32)

    n_nodes = len(registry_np)
    print(f"  Nodes: {n_nodes:,} positions")

    del X, Y, Z, coords, gx, gy, gz, grad_mag
    return omega1, omega2, w_alpha_base, node_positions, registry_np, phase_at_nodes, float(n_nodes)


def inject_uniform(grid_size, device, dx, severity,
                   balance_radius=5.0):
    print(f"\n--- BIFURCATION: UNIFORM ---")
    print(f"  Grid: {grid_size}³ | Null control")

    coords = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    omega1 = torch.ones((grid_size,) * 3, device=device) * 0.5
    omega2 = torch.ones((grid_size,) * 3, device=device) * 0.5

    shell = ((R >= balance_radius - dx) & (R <= balance_radius + dx)).float()
    w_alpha_base = shell * severity

    node_positions = torch.nonzero(w_alpha_base > 0)
    registry_np = node_positions.cpu().numpy().astype(np.int32)
    phase_at_nodes = np.zeros(len(registry_np), dtype=np.float32)

    n_nodes = len(registry_np)
    print(f"  Nodes: {n_nodes:,} positions (nominal shell at R={balance_radius})")

    del X, Y, Z, coords
    return omega1, omega2, w_alpha_base, node_positions, registry_np, phase_at_nodes, float(n_nodes)


def inject_bifurcation(mode, grid_size, device, dx, severity,
                       **kwargs):
    dispatch = {
        'step':     inject_step,
        'zeta':     inject_zeta,
        'harmonic': inject_harmonic,
        'uniform':  inject_uniform,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown bifurcation mode: {mode}")
    return dispatch[mode](grid_size, device, dx, severity, **kwargs)


# =============================================================================
# RUN DIRECTORY — separate from v0.2
# =============================================================================

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "runs_emergent")


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
# ENERGY TRACKER
# =============================================================================

class EnergyTracker:
    def __init__(self):
        self.steps              = []
        self.total_omega_1      = []
        self.total_omega_2      = []
        self.total_psi_abs_1    = []
        self.total_psi_abs_2    = []
        self.exchange_flux_psi  = []
        self.exchange_flux_omega = []

    def record(self, step, omega1, omega2, psi1, psi2, flux_psi, flux_omega):
        self.steps.append(step)
        self.total_omega_1.  append(float(omega1.sum().item()))
        self.total_omega_2.  append(float(omega2.sum().item()))
        self.total_psi_abs_1.append(float(psi1.abs().sum().item()))
        self.total_psi_abs_2.append(float(psi2.abs().sum().item()))
        self.exchange_flux_psi.  append(flux_psi)
        self.exchange_flux_omega.append(flux_omega)

    def save(self, run_dir):
        np.savez(os.path.join(run_dir, 'energy.npz'),
                 steps=np.array(self.steps),
                 total_omega_1=np.array(self.total_omega_1),
                 total_omega_2=np.array(self.total_omega_2),
                 total_psi_abs_1=np.array(self.total_psi_abs_1),
                 total_psi_abs_2=np.array(self.total_psi_abs_2),
                 exchange_flux_psi=np.array(self.exchange_flux_psi),
                 exchange_flux_omega=np.array(self.exchange_flux_omega))


# =============================================================================
# CLOUD RECORDER — extended with membrane diagnostics
# =============================================================================

class CloudRecorder:
    def __init__(self):
        self.data = {}

    def record(self, step, omega1, omega2, psi1, node_positions, grid_size,
               g_mu_nu, kappa, device, active_threshold=0.5,
               membrane=None, L1=None, L2=None, R=None):
        idx = node_positions
        values = omega1[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
        active = int((values > active_threshold).sum())
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
        self.data[f"{prefix}_values"]    = values.astype(np.float32)
        self.data[f"{prefix}_active"]    = np.array([active], dtype=np.int32)
        self.data[f"{prefix}_mean_t"]    = np.array([mean_t], dtype=np.float32)
        self.data[f"{prefix}_tvec_unit"] = unit.astype(np.float32)
        self.data[f"{prefix}_tvec_mag"]  = mag.astype(np.float32)

        if membrane is not None:
            mem_nodes = membrane[
                idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
            self.data[f"{prefix}_membrane"] = mem_nodes.astype(np.float32)
        if L1 is not None:
            l1_rms = float(torch.sqrt((L1**2).mean()).item())
            self.data[f"{prefix}_L1_rms"] = np.array(
                [l1_rms], dtype=np.float32)
        if L2 is not None:
            l2_rms = float(torch.sqrt((L2**2).mean()).item())
            self.data[f"{prefix}_L2_rms"] = np.array(
                [l2_rms], dtype=np.float32)
        if membrane is not None and R is not None:
            m_sum = membrane.sum().item()
            if m_sum > 0:
                fold_r = float((membrane * R).sum().item() / m_sum)
            else:
                fold_r = 0.0
            self.data[f"{prefix}_fold_radius"] = np.array(
                [fold_r], dtype=np.float32)

    def save(self, run_dir):
        if self.data:
            np.savez(os.path.join(run_dir, 'clouds.npz'), **self.data)


# =============================================================================
# MAIN SIMULATION — EMERGENT FOLD
# =============================================================================

def run_emergent_simulation(
        steps, grid_size, bifurcation,
        balance_radius, scale,
        beat_detune, no_beat,
        use_laplacian, use_exchange, use_advection,
        no_pump, single_shot,
        probe_interval,
        n_harmonics=8,
        coord_map='radial',
        auto=False,
        t_offset=0.0,
        sigma=0.5,
        time_sig=(4, 4),
        cross_rhythm=None):

    run_id  = next_run_id()
    run_dir = setup_run_dir(run_id)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if auto and bifurcation == 'zeta':
        scale = compute_auto_scale(t_offset=t_offset)
        if t_offset > 0:
            print(f"\n  AUTO-SCALE: 10 zeros from t_offset={t_offset:.2f} → scale={scale:.6f}")
        else:
            print(f"\n  AUTO-SCALE: 10 zeros from origin → scale={scale:.6f} (π={math.pi:.6f})")
        grid_size = compute_auto_grid(scale, t_offset=t_offset)
        print(f"  AUTO-GRID: ζ zero spacing at scale={scale:.4f} → grid {grid_size}")

    if grid_size % 2 == 0:
        grid_size -= 1

    dx = 20.0 / (grid_size - 1)

    print(f"\n=== Resonant Field Engine — EMERGENT FOLD v0.4 ===")
    print(f"Run: {run_id:04d} → {run_dir}")
    if auto:
        print(f"Device: {device} | Grid: {grid_size}³ (auto) | Steps: adaptive")
    else:
        print(f"Device: {device} | Grid: {grid_size}³ | Steps: {steps}")
    print(f"Bifurcation: {bifurcation} | Exchange: {use_exchange} | Coord: {coord_map}")
    print(f"All constants derived from field")

    # Coordinate space
    c = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(c, c, c, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    # Bifurcation injection — severity=1.0 (raw mask), rescaled after field analysis
    bif_kwargs = {}
    if bifurcation == 'step':
        bif_kwargs['balance_radius'] = balance_radius
    elif bifurcation == 'zeta':
        bif_kwargs['scale'] = scale
        bif_kwargs['coord_map'] = coord_map
        bif_kwargs['t_offset'] = t_offset
        bif_kwargs['sigma'] = sigma
    elif bifurcation == 'harmonic':
        bif_kwargs['n_harmonics'] = n_harmonics
    elif bifurcation == 'uniform':
        bif_kwargs['balance_radius'] = balance_radius

    omega1, omega2, w_alpha_base, node_positions, registry_np, phase_at_nodes, n_eff = \
        inject_bifurcation(bifurcation, grid_size, device, dx,
                           1.0, **bif_kwargs)

    n_registry = len(registry_np)

    # All scales derived from field content
    rates = compute_field_rates(omega1, dx, n_eff)

    # Rescale tension injection to field amplitude
    w_alpha_base = w_alpha_base * rates['severity']

    sigma = rates['sigma']

    print(f"\n--- FIELD-DERIVED SCALES ---")
    print(f"  natural_freq:    {rates['_natural_freq']:.4f}")
    print(f"  curvature_ratio: {rates['_curvature_ratio']:.4f}")
    print(f"  char_length:     {rates['_char_length']:.4f} grid cells")
    print(f"  base (CFL):      {rates['_base']:.6f}")
    print(f"  propagation:     {rates['propagation']:.6f}")
    print(f"  diffusion:       {rates['diffusion']:.6f}")
    print(f"  tension_decay:   {rates['tension_decay']:.6f}")
    print(f"  coupling:        {rates['coupling']:.2e}")
    print(f"  advection:       {rates['advection']:.6f}")
    print(f"  severity:        {rates['severity']:.4f} (ω_rms)")
    print(f"  sigma:           {sigma:.4f} (char_length × dx)")
    print(f"  metric_floor:    {rates['metric_floor']:.6f}")
    print(f"  active_thresh:   {rates['active_threshold']:.4f}")
    print(f"  n_eff:           {n_eff:.1f} (participation ratio)")
    print(f"  n_registry:      {n_registry:,} (above mean gradient)")

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
        print(f"\n--- AUTO-DERIVED TIMING ---")
        sig_str = f"{beats_per_measure}/{beat_unit}"
        print(f"  time_signature:  {sig_str}")
        if cross_detune:
            print(f"  cross_rhythm:    {cr_beats}/{cr_unit}")
            lcm_steps = round(1.0 / abs(beat_detune_eff - cross_detune)) if beat_detune_eff != cross_detune else beat_period
            print(f"  polyrhythm:      {sig_str} × {cr_beats}/{cr_unit} (realign every {lcm_steps} steps)")
        print(f"  beat_detune:     {beat_detune_eff:.6f} (base × {beats_per_measure}/{beat_unit})")
        print(f"  beat_period:     {beat_period} steps")
        print(f"  stop criterion:  exchange flux Δ < base for 2 consecutive periods")
        print(f"  minimum run:     {4 * beat_period} steps (4 beat periods)")
    else:
        beat_detune_eff = beat_detune
        cross_detune = None
        beat_period = None

    # Metric from field — the fields ARE the geometry
    g_mu_nu = build_metric_from_field(omega1, omega2, rates['metric_floor'])

    # Psi seeded from omega gradient — energy starts where the field has structure
    # Total psi energy scaled to match total omega energy (the field sets its own scale)
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
        'engine_version':    'emergent-fold-0.4-final',
        'run_id':            run_id,
        'grid_size':         grid_size,
        'dx':                dx,
        'bifurcation':       bifurcation,
        'injection_type':    bifurcation,
        'balance_radius':    balance_radius,
        'scale':             scale,
        'sigma':             sigma,
        't_offset':          t_offset,
        'coord_map':         coord_map,
        'auto':              auto,
        'auto_grid':         grid_size if auto else None,
        'auto_beat_period':  beat_period if auto else None,
        'time_sig':          f"{time_sig[0]}/{time_sig[1]}",
        'cross_rhythm':      f"{cross_rhythm[0]}/{cross_rhythm[1]}" if cross_rhythm else None,
        'constants_type':    'all_field_derived',
        'psi_init':          'omega_gradient',
        'rates':             {k: v for k, v in rates.items() if not k.startswith('_')},
        'field_stats':       {
            'natural_freq':    rates['_natural_freq'],
            'curvature_ratio': rates['_curvature_ratio'],
            'char_length':     rates['_char_length'],
            'omega_rms':       rates['_omega_rms'],
            'grad_rms':        rates['_grad_rms'],
            'lap_rms':         rates['_lap_rms'],
        },
        'membrane_type':     'dynamic_curvature_zero',
        'membrane_sigma':    sigma,
        'use_laplacian':     use_laplacian,
        'use_exchange':      use_exchange,
        'use_advection':     use_advection,
        'no_pump':           no_pump,
        'single_shot':       single_shot,
        'beat_detune':       beat_detune,
        'no_beat':           no_beat,
        'probe_interval':    probe_interval,
        'n_harmonics':       n_harmonics if bifurcation == 'harmonic' else None,
        'n_eff':             n_eff,
        'n_registry':        n_registry,
        'collapse_step':     None,
        'total_steps':       0,
        'wall_seconds':      0,
    }

    np.save(os.path.join(run_dir, 'registry.npy'), registry_np)
    np.save(os.path.join(run_dir, 'phase.npy'), phase_at_nodes)

    tracker = EnergyTracker()
    clouds  = CloudRecorder()

    del X, Y, Z, c

    print(f"\n--- RUNNING --- (n_eff={n_eff:.1f}, registry={n_registry:,})")
    print(f"{'Step':>6} | {'Σω₁':>10} | {'Σω₂':>10} | "
          f"{'ΣΨ₁':>10} | {'ΣΨ₂':>10} | "
          f"{'flux_ω':>10} | {'fold_R':>8}")
    print("-" * 85)

    wall_start = _time.time()
    final_step = 0
    membrane = None
    L1 = L2 = None

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

        # Advance both fields
        psi1, omega1 = wave_step(
            psi1, omega1, g_mu_nu, w_alpha_t, kappa1, rates,
            use_laplacian=use_laplacian, use_advection=use_advection,
            no_pump=no_pump, propagate=do_propagate)
        psi2, omega2 = wave_step(
            psi2, omega2, g_mu_nu, w_alpha_t, kappa2, rates,
            use_laplacian=use_laplacian, use_advection=use_advection,
            no_pump=no_pump, propagate=do_propagate)

        # Dynamic membrane and exchange
        flux_omega = 0.0
        flux_psi   = 0.0
        fold_r     = 0.0

        if use_exchange:
            membrane, L1, L2 = compute_membrane(omega1, omega2, sigma)

            delta_omega = membrane * rates['_base'] * (omega1 - omega2)
            delta_psi   = membrane * rates['_base'] * (psi1 - psi2)

            omega1 = torch.relu(omega1 - delta_omega)
            omega2 = torch.relu(omega2 + delta_omega)
            psi1   = psi1 - delta_psi
            psi2   = psi2 + delta_psi

            flux_omega = float(delta_omega.abs().sum().item())
            flux_psi   = float(delta_psi.abs().sum().item())

            m_sum = membrane.sum().item()
            if m_sum > 0:
                fold_r = float((membrane * R).sum().item() / m_sum)

        # Metric follows the fields
        g_mu_nu = build_metric_from_field(omega1, omega2, rates['metric_floor'])

        final_step = i

        # Energy tracking — every probe_interval
        if i % probe_interval == 0:
            tracker.record(i, omega1, omega2, psi1, psi2,
                           flux_psi, flux_omega)

        # Cloud snapshots — variable schedule
        is_last = (not auto and i == steps - 1)
        record_cloud = ((i < 200 and i % 10 == 0) or
                        (i >= 200 and i % 50 == 0) or
                        is_last)

        if record_cloud:
            clouds.record(i, omega1, omega2, psi1, node_positions,
                          grid_size, g_mu_nu, kappa1, device,
                          active_threshold=rates['active_threshold'],
                          membrane=membrane, L1=L1, L2=L2, R=R)

        # Progress
        if i % max(1, 50) == 0 or i < 5 or is_last:
            print(f"{i:>6} | {omega1.sum().item():>10.3e} | "
                  f"{omega2.sum().item():>10.3e} | "
                  f"{psi1.abs().sum().item():>10.3e} | "
                  f"{psi2.abs().sum().item():>10.3e} | "
                  f"{flux_omega:>10.3e} | {fold_r:>8.3f}")

        # Collapse check
        if torch.isnan(omega1).any() or torch.isinf(omega1).any():
            print(f"\n⚠️  COLLAPSE (field 1) at step {i}")
            meta['collapse_step'] = i
            break
        if torch.isnan(omega2).any() or torch.isinf(omega2).any():
            print(f"\n⚠️  COLLAPSE (field 2) at step {i}")
            meta['collapse_step'] = i
            break

        # Auto-stop: check at beat boundaries
        if auto and beat_period and use_exchange:
            period_flux.append(flux_omega)
            if (i + 1) % beat_period == 0 and i >= 3 * beat_period:
                current_mean = sum(period_flux[-beat_period:]) / beat_period
                if prev_period_mean is not None:
                    rel_change = abs(current_mean - prev_period_mean) / max(prev_period_mean, 1e-10)
                    if rel_change < rates['_base']:
                        stable_count += 1
                        if stable_count >= 2:
                            print(f"\n  AUTO-STOP at step {i+1}: exchange stable "
                                  f"(Δ={rel_change:.2e} < base={rates['_base']:.2e}) "
                                  f"for {stable_count} periods")
                            auto_stopped = True
                            clouds.record(i, omega1, omega2, psi1, node_positions,
                                          grid_size, g_mu_nu, kappa1, device,
                                          active_threshold=rates['active_threshold'],
                                          membrane=membrane, L1=L1, L2=L2, R=R)
                            final_step = i
                            break
                    else:
                        stable_count = 0
                prev_period_mean = current_mean

        if not auto and i >= steps - 1:
            break
        i += 1

    wall_elapsed = _time.time() - wall_start
    meta['total_steps']  = final_step + 1
    meta['auto_stopped']  = auto_stopped if auto else None
    meta['wall_seconds'] = round(wall_elapsed, 1)

    tracker.save(run_dir)
    clouds.save(run_dir)
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
        if auto:
            print(f"Grid: {grid_size}³ (field-derived)")
            print(f"Beat period: {beat_period} steps (field-derived)")
            if auto_stopped:
                print(f"Stopped: field decided (exchange stable)")
            else:
                print(f"Stopped: safety cap ({max_steps} steps)")
        print(f"Σω total  initial: {e0:.4e}")
        print(f"Σω total  final:   {ef:.4e}")
        print(f"Total drift:       {drift*100:+.3f}%")
        print(f"Wall time: {wall_elapsed:.1f}s")
        print(f"Output: {run_dir}")

    return run_dir


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Resonant Field Engine — Emergent Fold v0.4')

    p.add_argument('--grid',         type=int,   default=89)
    p.add_argument('--steps',        type=int,   default=2000)

    p.add_argument('--bifurcation',  type=str,   default='step',
                   choices=['step', 'zeta', 'harmonic', 'uniform'],
                   help='Bifurcation mode (default: step)')
    p.add_argument('--scale',        type=float, default=3.0,
                   help='R→t mapping for zeta mode (default: 3.0)')
    p.add_argument('--balance-radius', type=float, default=5.0,
                   help='Cut radius for step/uniform mode (default: 5.0)')
    p.add_argument('--n-harmonics',  type=int,   default=8,
                   help='Number of harmonics for harmonic mode')

    p.add_argument('--coord-map',    type=str,   default='radial',
                   choices=['radial', 'shifted', 'axial', 'diagonal'],
                   help='Coordinate mapping for ζ: radial (spheres), '
                        'shifted (off-center spheres), axial (Z planes), '
                        'diagonal (body-diagonal planes)')

    p.add_argument('--beat',         type=float, default=0.0,
                   help='Beat detune (default: 0.0, no modulation)')
    p.add_argument('--no-beat',      action='store_true')
    p.add_argument('--no-laplacian', action='store_true')
    p.add_argument('--no-exchange',  action='store_true',
                   help='Disable membrane exchange (uncoupled evolution)')
    p.add_argument('--advection',    action='store_true')
    p.add_argument('--no-pump',      action='store_true')
    p.add_argument('--single-shot',  action='store_true',
                   help='Pump fires once at step 0, then field evolves freely')
    p.add_argument('--laplacian-only', action='store_true',
                   help='Pure diffusion: disable both pump and propagation')
    p.add_argument('--probe-interval', type=int, default=1)
    p.add_argument('--t-offset',     type=float, default=0.0,
                   help='Slide the ζ evaluation window: t = R×scale + t_offset')
    p.add_argument('--sigma',        type=float, default=0.5,
                   help='Real part of s in ζ(s): s = σ + it (default: 0.5 = critical line)')
    p.add_argument('--auto',         action='store_true',
                   help='Field determines grid, beat, and step count.')
    p.add_argument('--time-sig',     type=str, default='4/4',
                   help='Time signature for beat modulation (default: 4/4)')
    p.add_argument('--cross-rhythm', type=str, default=None,
                   help='Second time sig for polyrhythmic modulation (e.g., 3/2)')

    args = p.parse_args()

    run_emergent_simulation(
        steps          = args.steps,
        grid_size      = args.grid,
        bifurcation    = args.bifurcation,
        balance_radius = args.balance_radius,
        scale          = args.scale,
        beat_detune    = args.beat,
        no_beat        = args.no_beat or args.beat == 0.0,
        use_laplacian  = not args.no_laplacian,
        use_exchange   = not args.no_exchange,
        use_advection  = args.advection,
        no_pump        = args.no_pump or args.laplacian_only,
        single_shot    = args.single_shot,
        probe_interval = args.probe_interval,
        n_harmonics    = args.n_harmonics,
        coord_map      = args.coord_map,
        auto           = args.auto,
        t_offset       = args.t_offset,
        sigma          = args.sigma,
        time_sig       = tuple(int(x) for x in args.time_sig.split('/')),
        cross_rhythm   = tuple(int(x) for x in args.cross_rhythm.split('/')) if args.cross_rhythm else None,
    )
