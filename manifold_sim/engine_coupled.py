"""
Resonant Field Engine — Coupled Variant v0.2
=============================================
Two-manifold coupled engine for testing the conservation hypothesis.

v0.2 changes (Claude Opus 4.6 rebuild):
  - Default exchange signs fixed to +1.0 (conservative: total energy
    preserved across both fields). Previous -1.0 default caused double
    absorption with symmetric fields — a systematic drain, not exchange.
  - Added --mirror-init: field 2 starts empty (psi=0, omega=1). The
    exchange populates it naturally from field 1's core absorption.
    This breaks the symmetry degeneracy that made the exchange inert.
  - Added cloud data output (clouds.npz) — per-step spatial snapshots
    at node positions, compatible with analyze.py's full tool suite.
  - Added --conjugate-init: field 2 starts with psi = -psi1 (phase
    conjugate). Alternative to mirror-init for PT-symmetric studies.

Physics note on exchange signs:
  sign=+1 (conservative): what field 1 loses at core, field 2 gains.
    Total psi and omega are preserved across both fields. The core
    acts as a portal, not a sink. Only the drain removes energy.
  sign=-1 (draining): what field 1 loses at core, field 2 ALSO loses.
    With symmetric fields this doubles the absorption rate. Use only
    for anti-symmetric coupling studies.

Architecture note: built deliberately parallel to engine.py v2.4.0.
Run isolation, output format, analyzer compatibility all preserved.
Output dir: runs_coupled/NNNN/ (separate from original).

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
# PHYSICAL CONSTANTS — identical defaults to engine.py v2.4.0
# =============================================================================

CORE_RADIUS             = 0.0
CORE_SURFACE_THICKNESS  = 0.3
DRAIN_RADIUS            = 0.15
CORE_ABSORPTION         = 0.1

WEIGHT_THRESHOLD        = 3.0
WEIGHT_SINK_RATE        = 0.05
METRIC_CORE_VALUE       = 3.0
METRIC_OUTER_VALUE      = 1.0
METRIC_DECAY            = 0.3

PSI_PROPAGATION         = 0.01
LATENT_COUPLING         = 1e-5
DIFFUSION_COEFF         = 0.005
TENSION_DECAY           = 0.001

BEAT_SEVERITY_DEFAULT   = 5.0
BEAT_SEVERITY_DISASTER  = 1e6

ACTIVE_THRESHOLD        = 0.5
SOFT_ALPHA              = 1.0

# New for coupled variant
ADVECTION_COEFF         = 0.001   # small by design; transport should not
                                  # overwhelm the diffusive/wave behavior
FOLD_COUPLING           = 0.01    # coupling rate through the fold membrane
                                  # small to avoid instability; the fold
                                  # transfers, not teleports


# =============================================================================
# ZETA ZEROS
# =============================================================================

def get_zeta_zeros(count=50):
    zeros = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425274, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846
    ]
    return zeros[:count]


# =============================================================================
# κ(H)
# =============================================================================

def kappa_H(omega_field, core_mask):
    active = (1.0 - core_mask).bool()
    H = omega_field[active].mean().item() if active.any() else 1.0
    return max(1.0 / (1.0 + H * 0.1), 1e-4)


# =============================================================================
# METRIC TENSOR
# =============================================================================

def build_metric(R, core_radius):
    return METRIC_OUTER_VALUE + (METRIC_CORE_VALUE - METRIC_OUTER_VALUE) * \
        torch.exp(-METRIC_DECAY * torch.clamp(R - core_radius, min=0.0))


# =============================================================================
# WAVE STEP — extended with optional advection, returns absorbed quantities
# =============================================================================
#
# Key changes from engine.py:
#   1. `use_advection` flag adds a v·∇psi transport term (fluid-like)
#   2. `return_absorbed=True` returns (new_psi, new_omega, absorbed_psi,
#      absorbed_omega) — the quantities that WOULD have been removed by
#      core absorption, so we can deliver them to the mirror field instead.
#      With return_absorbed=True the core absorption is NOT applied here;
#      the caller is responsible for applying exchange + any residual sink.
#
def wave_step_ex(psi, omega, g_mu_nu, w_alpha_t, core_mask, drain_mask,
                 surface_mask, kappa, absorption, use_laplacian=True,
                 use_advection=False, return_absorbed=False, no_pump=False,
                 propagate=True):

    psi_eff = kappa * g_mu_nu * psi

    dx = torch.roll(psi_eff, 1, 0) - torch.roll(psi_eff, -1, 0)
    dy = torch.roll(psi_eff, 1, 1) - torch.roll(psi_eff, -1, 1)
    dz = torch.roll(psi_eff, 1, 2) - torch.roll(psi_eff, -1, 2)

    if use_laplacian:
        laplacian = (
            torch.roll(psi_eff, 1, 0) + torch.roll(psi_eff, -1, 0) +
            torch.roll(psi_eff, 1, 1) + torch.roll(psi_eff, -1, 1) +
            torch.roll(psi_eff, 1, 2) + torch.roll(psi_eff, -1, 2) -
            6.0 * psi_eff
        )
    else:
        laplacian = torch.zeros_like(psi_eff)

    T_vec = torch.stack([dx, dy, dz], dim=-1)
    T_nbr = (
        torch.roll(T_vec, 1, 0) + torch.roll(T_vec, -1, 0) +
        torch.roll(T_vec, 1, 1) + torch.roll(T_vec, -1, 1) +
        torch.roll(T_vec, 1, 2) + torch.roll(T_vec, -1, 2)
    ) / 6.0
    mag_T     = torch.norm(T_vec, dim=-1) + 1e-8
    mag_N     = torch.norm(T_nbr, dim=-1) + 1e-8
    cos_theta = torch.sum(T_vec * T_nbr, dim=-1) / (mag_T * mag_N)

    routing_pressure = 1.0 - cos_theta
    if no_pump:
        latent_transfer = torch.zeros_like(omega)
    else:
        latent_transfer = w_alpha_t * routing_pressure * LATENT_COUPLING * kappa
    new_omega = torch.relu(
        omega + latent_transfer - (mag_T * TENSION_DECAY * cos_theta)
    )

    field_mean = new_omega[new_omega > 0].mean().clamp(min=1e-6)
    too_heavy  = (new_omega / field_mean > WEIGHT_THRESHOLD).float()
    new_omega  = new_omega * (1.0 - surface_mask * too_heavy * WEIGHT_SINK_RATE)

    # Base propagation
    if no_pump == 'laplacian_only' or not propagate:
        psi_update = laplacian * DIFFUSION_COEFF
    else:
        psi_update = (
            (dx + dy + dz) * PSI_PROPAGATION * new_omega / g_mu_nu +
            laplacian * DIFFUSION_COEFF
        )
        if not no_pump:
            psi_update = psi_update + w_alpha_t * LATENT_COUPLING * kappa

    # Advection term: v · ∇psi where v ∝ T_vec / mag_T (unit tension direction).
    # Upwind-biased via centered difference on psi; small coefficient.
    if use_advection:
        # Use raw psi here (not psi_eff) so advection is distinct from
        # propagation's metric-coupled gradient.
        dpsi_x = torch.roll(psi, 1, 0) - torch.roll(psi, -1, 0)
        dpsi_y = torch.roll(psi, 1, 1) - torch.roll(psi, -1, 1)
        dpsi_z = torch.roll(psi, 1, 2) - torch.roll(psi, -1, 2)
        v_unit = T_vec / mag_T.unsqueeze(-1)  # unit vector field
        advection = (v_unit[..., 0] * dpsi_x +
                     v_unit[..., 1] * dpsi_y +
                     v_unit[..., 2] * dpsi_z)
        psi_update = psi_update - ADVECTION_COEFF * advection

    new_psi = psi + psi_update

    # Compute what core would absorb (before applying it)
    absorbed_psi   = new_psi   * core_mask * absorption
    absorbed_omega = new_omega * core_mask * absorption

    if not return_absorbed:
        # Classical behavior: absorb at core, drain at center
        new_omega = new_omega * (1.0 - core_mask * absorption)
        new_psi   = new_psi   * (1.0 - core_mask * absorption)
        new_omega = new_omega * (1.0 - drain_mask)
        new_psi   = new_psi   * (1.0 - drain_mask)
        return new_psi, new_omega

    # Coupled behavior: caller will handle core via exchange.
    # We still apply drain (pure sink, separate from core).
    # Subtract what we're handing off.
    new_omega = new_omega - absorbed_omega
    new_psi   = new_psi   - absorbed_psi
    new_omega = new_omega * (1.0 - drain_mask)
    new_psi   = new_psi   * (1.0 - drain_mask)

    return new_psi, new_omega, absorbed_psi, absorbed_omega


# =============================================================================
# 4D LATENT INJECTION — unchanged from engine.py
# =============================================================================

def inject_dual_sheet(grid_size, device, use_zeta, tune_scalar,
                      beat_detune, use_hann, severity,
                      n_zeros=50, no_carrier=False, inject_test=None,
                      soft_alpha=SOFT_ALPHA, extra_freq=None):

    print(f"\n--- 4D LATENT INJECTION ---")
    print(f"  Grid: {grid_size}³ | Severity: {severity:.2e} | "
          f"Carrier: {tune_scalar}x | Beat: {beat_detune}")
    print(f"  Zeta: {'ON' if use_zeta else 'OFF'} | "
          f"Hann: {'ON' if use_hann else 'OFF'}")

    dx = 20.0 / (grid_size - 1)
    coords = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    R_soft = torch.sqrt(X**2 + Y**2 + Z**2 + (soft_alpha * dx)**2)
    del X, Y, Z, coords

    amplitude = 1.0 / torch.sqrt(R_soft)
    log_R     = torch.log(R_soft)
    complex_field = torch.zeros_like(R_soft, dtype=torch.complex64)

    if inject_test is not None:
        freqs = inject_test
        n     = len(freqs)
        for i, gamma in enumerate(freqs):
            w     = (0.5*(1.0+math.cos(math.pi*i/n))) if use_hann else 1.0
            phase = gamma * log_R
            complex_field += (amplitude * torch.exp( 1j*phase) -
                              amplitude * torch.exp(-1j*phase)) * w
    elif use_zeta:
        zeros = get_zeta_zeros(n_zeros)
        if extra_freq:
            zeros = zeros + extra_freq
        n     = len(zeros)
        for i, gamma in enumerate(zeros):
            w     = (0.5*(1.0+math.cos(math.pi*i/n))) if use_hann else 1.0
            phase = gamma * log_R
            complex_field += (amplitude * torch.exp( 1j*phase) -
                              amplitude * torch.exp(-1j*phase)) * w
    elif not no_carrier:
        complex_field = amplitude * torch.exp(1j * log_R)
    else:
        complex_field = torch.zeros_like(R_soft, dtype=torch.complex64)

    base_freq     = (2.0 * torch.pi / 20.0) * tune_scalar
    beat_envelope = (torch.exp(1j * base_freq * R_soft) +
                     torch.exp(1j * (base_freq + beat_detune) * R_soft))
    complex_field = complex_field * beat_envelope.real

    tension      = torch.abs(complex_field)
    t_min, t_max = tension.min(), tension.max()
    tension      = (tension - t_min) / (t_max - t_min + 1e-8)

    threshold    = torch.quantile(tension.flatten(), 0.999)
    w_alpha_base = (tension >= threshold).float() * severity

    node_positions = torch.nonzero(w_alpha_base > 0)
    coords_sim = (node_positions.float() - grid_size/2.0) * (20.0/grid_size)
    R_nodes = torch.sqrt((coords_sim**2).sum(dim=1))
    node_positions = node_positions[R_nodes >= DRAIN_RADIUS]
    n_nodes        = len(node_positions)
    registry_np    = node_positions.cpu().numpy().astype(np.int32)

    # Preserve complex phase at node positions (not just magnitude)
    idx = node_positions
    phase_at_nodes = torch.angle(
        complex_field[idx[:, 0], idx[:, 1], idx[:, 2]]
    ).cpu().numpy().astype(np.float32)

    print(f"  Nodes: {n_nodes:,} positions")

    del complex_field, tension, amplitude, log_R, R_soft
    return w_alpha_base, node_positions, registry_np, phase_at_nodes


# =============================================================================
# RUN DIRECTORY — separate from engine.py's runs/
# =============================================================================

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "runs_coupled")


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
    """
    Records, per step:
      - total_omega_1, total_omega_2
      - total_psi_abs_1, total_psi_abs_2  (L1 norms)
      - core_flux_psi, core_flux_omega     (what transferred this step)
    """
    def __init__(self):
        self.steps              = []
        self.total_omega_1      = []
        self.total_omega_2      = []
        self.total_psi_abs_1    = []
        self.total_psi_abs_2    = []
        self.core_flux_psi      = []
        self.core_flux_omega    = []

    def record(self, step, omega1, omega2, psi1, psi2, flux_psi, flux_omega):
        self.steps.append(step)
        self.total_omega_1.  append(float(omega1.sum().item()))
        self.total_omega_2.  append(float(omega2.sum().item()))
        self.total_psi_abs_1.append(float(psi1.abs().sum().item()))
        self.total_psi_abs_2.append(float(psi2.abs().sum().item()))
        self.core_flux_psi.  append(flux_psi)
        self.core_flux_omega.append(flux_omega)

    def save(self, run_dir):
        np.savez(os.path.join(run_dir, 'energy.npz'),
                 steps=np.array(self.steps),
                 total_omega_1=np.array(self.total_omega_1),
                 total_omega_2=np.array(self.total_omega_2),
                 total_psi_abs_1=np.array(self.total_psi_abs_1),
                 total_psi_abs_2=np.array(self.total_psi_abs_2),
                 core_flux_psi=np.array(self.core_flux_psi),
                 core_flux_omega=np.array(self.core_flux_omega))


# =============================================================================
# CLOUD RECORDER — per-step spatial snapshots at node positions
# =============================================================================
#
# Compatible with analyze.py's load_cloud_at_step() format.
# Saves omega values at registered node positions, active count,
# mean tension, and tension vectors for the primary field.
#

class CloudRecorder:
    """
    Records per-step spatial snapshots at node positions.
    Output format matches analyze.py expectations:
      s{step:04d}_values   — omega at each node
      s{step:04d}_active   — [count of nodes with omega > threshold]
      s{step:04d}_mean_t   — [mean tension magnitude across nodes]
      s{step:04d}_tvec_unit — unit tension vectors (N×3)
      s{step:04d}_tvec_mag  — tension magnitudes (N,)
    """
    def __init__(self):
        self.data = {}

    def record(self, step, omega, psi, node_positions, grid_size, g_mu_nu,
               kappa, device):
        """Extract and store cloud data for one timestep."""
        # omega values at node positions
        idx = node_positions
        values = omega[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
        active = int((values > ACTIVE_THRESHOLD).sum())
        mean_t = float(values.mean())

        # Tension vectors at node positions
        psi_eff = kappa * g_mu_nu * psi
        dx = torch.roll(psi_eff, 1, 0) - torch.roll(psi_eff, -1, 0)
        dy = torch.roll(psi_eff, 1, 1) - torch.roll(psi_eff, -1, 1)
        dz = torch.roll(psi_eff, 1, 2) - torch.roll(psi_eff, -1, 2)
        T_vec = torch.stack([dx, dy, dz], dim=-1)

        tvec_at_nodes = T_vec[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
        mag = np.linalg.norm(tvec_at_nodes, axis=1) + 1e-8
        unit = tvec_at_nodes / mag[:, np.newaxis]

        prefix = f"s{step:04d}"
        self.data[f"{prefix}_values"]    = values.astype(np.float32)
        self.data[f"{prefix}_active"]    = np.array([active], dtype=np.int32)
        self.data[f"{prefix}_mean_t"]    = np.array([mean_t], dtype=np.float32)
        self.data[f"{prefix}_tvec_unit"] = unit.astype(np.float32)
        self.data[f"{prefix}_tvec_mag"]  = mag.astype(np.float32)

    def save(self, run_dir):
        if self.data:
            np.savez(os.path.join(run_dir, 'clouds.npz'), **self.data)


# =============================================================================
# MAIN SIMULATION — COUPLED
# =============================================================================

def run_coupled_simulation(
        steps, grid_size, use_zeta, tune_scalar, beat_detune,
        use_hann, use_laplacian, severity, core_radius, absorption,
        n_zeros=50, no_carrier=False, no_beat=False, no_core=False,
        inject_test=None, soft_alpha=SOFT_ALPHA, extra_freq=None,
        dense_phase=50, probe_interval=1, slice_interval=5,
        # coupled-specific
        use_dual=True, use_exchange=True, use_advection=False,
        mirror_sign_psi=1.0, mirror_sign_omega=1.0,
        mirror_init=False, conjugate_init=False,
        no_pump=False,
        single_shot=False,
        force_device=None):

    run_id  = next_run_id()
    run_dir = setup_run_dir(run_id)
    if force_device:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if grid_size % 2 == 0:
        grid_size -= 1

    dx = 20.0 / (grid_size - 1)

    print(f"\n=== Resonant Field Engine — COUPLED v0.2 ===")
    print(f"Run: {run_id:04d} → {run_dir}")
    print(f"Device: {device} | Grid: {grid_size}³ | Steps: {steps}")
    print(f"Dual: {use_dual} | Exchange: {use_exchange} | Advection: {use_advection}")
    print(f"Mirror sign flip: psi={mirror_sign_psi}, omega={mirror_sign_omega}")
    if mirror_init:
        print(f"Init mode: MIRROR (field 2 starts empty, populated by exchange)")
    elif conjugate_init:
        print(f"Init mode: CONJUGATE (field 2 = -field 1)")
    else:
        print(f"Init mode: SYMMETRIC (both fields identical)")
    if not use_dual and use_exchange:
        print("WARN: exchange requires dual fields; disabling exchange.")
        use_exchange = False

    meta = {
        'engine_version':    'coupled-0.2',
        'run_id':            run_id,
        'grid_size':         grid_size,
        'dx':                dx,
        'use_zeta':          use_zeta,
        'use_hann':          use_hann,
        'use_laplacian':     use_laplacian,
        'use_dual':          use_dual,
        'use_exchange':      use_exchange,
        'use_advection':     use_advection,
        'mirror_init':       mirror_init,
        'conjugate_init':    conjugate_init,
        'mirror_sign_psi':   mirror_sign_psi,
        'mirror_sign_omega': mirror_sign_omega,
        'no_pump':           no_pump,
        'single_shot':       single_shot,
        'tune_scalar':       tune_scalar,
        'beat_detune':       beat_detune,
        'severity':          severity,
        'core_radius':       core_radius,
        'absorption':        absorption,
        'n_zeros':           n_zeros,
        'no_carrier':        no_carrier,
        'no_beat':           no_beat,
        'no_core':           no_core,
        'soft_alpha':        soft_alpha,
        'extra_freq':        extra_freq,
        'dense_phase':       dense_phase,
        'probe_interval':    probe_interval,
        'slice_interval':    slice_interval,
        'n_nodes':           0,
        'collapse_step':     None,
        'total_steps':       0,
    }

    # Coordinate space & masks
    c = torch.linspace(-10, 10, grid_size, device=device)
    X, Y, Z = torch.meshgrid(c, c, c, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2)

    core_mask    = (R <= core_radius).float()
    drain_mask   = (R <= DRAIN_RADIUS).float()
    surface_mask = ((R > core_radius) &
                    (R <= core_radius + CORE_SURFACE_THICKNESS)).float()

    if no_core:
        core_mask    = torch.zeros_like(core_mask)
        surface_mask = torch.zeros_like(surface_mask)

    g_mu_nu = build_metric(R, core_radius)

    # Fold membrane: Hann-windowed shell at core radius.
    # This is the literal fold in space — a smooth region where
    # the two manifolds touch and energy can transfer.
    # Active only when no_core=True (coreless fold-in-space mode).
    fold_distance = torch.abs(R - core_radius) / max(CORE_SURFACE_THICKNESS, 1e-6)
    fold_mask = 0.5 * (1.0 + torch.cos(
        math.pi * torch.clamp(fold_distance, 0.0, 1.0)
    )) * (fold_distance <= 1.0).float()

    R_soft_init = torch.sqrt(X**2 + Y**2 + Z**2 + (soft_alpha * dx)**2)

    # Primary field
    psi1   = torch.zeros((grid_size, grid_size, grid_size), device=device)
    omega1 = torch.ones( (grid_size, grid_size, grid_size), device=device)
    psi1  += 1000.0 * torch.exp(-R_soft_init**2 / 0.5)

    # Mirror field — initialization depends on mode
    if use_dual:
        if mirror_init:
            # Empty start: exchange populates field 2 from field 1's core
            psi2   = torch.zeros_like(psi1)
            omega2 = torch.ones_like(omega1)
            # No Gaussian pulse — field 2 is born from the exchange
        elif conjugate_init:
            # Phase conjugate: psi2 = -psi1 (PT-symmetric pair)
            psi2   = -psi1.clone()
            omega2 = torch.ones_like(omega1)
        else:
            # Symmetric (legacy behavior — exchange is degenerate)
            psi2   = psi1.clone()
            omega2 = torch.ones_like(omega1)
    else:
        psi2, omega2 = None, None

    del X, Y, Z, c, R, R_soft_init

    # Inject — same registry for both fields
    w_alpha_base, node_positions, registry_np, phase_at_nodes = inject_dual_sheet(
        grid_size, device, use_zeta, tune_scalar,
        beat_detune, use_hann, severity,
        n_zeros=n_zeros, no_carrier=no_carrier,
        inject_test=inject_test, soft_alpha=soft_alpha,
        extra_freq=extra_freq
    )
    n_nodes         = len(registry_np)
    meta['n_nodes'] = n_nodes
    np.save(os.path.join(run_dir, 'registry.npy'), registry_np)
    np.save(os.path.join(run_dir, 'phase.npy'), phase_at_nodes)

    tracker = EnergyTracker()
    clouds  = CloudRecorder()

    print(f"\n--- RUNNING --- ({n_nodes:,} nodes)")
    print(f"{'Step':>6} | {'Σω₁':>10} | {'Σω₂':>10} | {'ΣΨ₁':>10} | "
          f"{'ΣΨ₂':>10} | {'flux_ω':>10} | {'flux_Ψ':>10}")
    print("-" * 90)

    wall_start = _time.time()
    final_step = 0

    for i in range(steps):
        kappa1 = kappa_H(omega1, core_mask)
        kappa2 = kappa_H(omega2, core_mask) if use_dual else 1.0

        do_propagate = not single_shot or i == 0

        if single_shot and i > 0:
            w_alpha_t = w_alpha_base * 0.0
        elif no_beat:
            w_alpha_t = w_alpha_base
        else:
            w_alpha_t = w_alpha_base * math.cos(2.0 * math.pi * beat_detune * i)

        # Primary advance
        if use_exchange and use_dual and not no_core:
            # CORE EXCHANGE MODE: absorption at core creates flux between fields
            psi1, omega1, abs_psi1, abs_omega1 = wave_step_ex(
                psi1, omega1, g_mu_nu, w_alpha_t,
                core_mask, drain_mask, surface_mask,
                kappa1, absorption,
                use_laplacian=use_laplacian,
                use_advection=use_advection,
                return_absorbed=True, no_pump=no_pump,
                propagate=do_propagate
            )
            psi2, omega2, abs_psi2, abs_omega2 = wave_step_ex(
                psi2, omega2, g_mu_nu, w_alpha_t,
                core_mask, drain_mask, surface_mask,
                kappa2, absorption,
                use_laplacian=use_laplacian,
                use_advection=use_advection,
                return_absorbed=True, no_pump=no_pump,
                propagate=do_propagate
            )
            # Conservative exchange (sign=+1): what field 1 absorbed at core
            # is delivered to field 2, and vice versa. Total energy preserved.
            # With sign=-1: anti-symmetric (draining) coupling instead.
            # relu keeps omega non-negative.
            psi1   = psi1   + mirror_sign_psi   * abs_psi2
            omega1 = torch.relu(omega1 + mirror_sign_omega * abs_omega2)
            psi2   = psi2   + mirror_sign_psi   * abs_psi1
            omega2 = torch.relu(omega2 + mirror_sign_omega * abs_omega1)

            flux_omega = float(abs_omega1.sum().item() + abs_omega2.sum().item())
            flux_psi   = float(abs_psi1.abs().sum().item() +
                                abs_psi2.abs().sum().item())

        elif use_exchange and use_dual and no_core:
            # FOLD COUPLING MODE: no hard core mass. The Hann-windowed
            # fold membrane replaces the core as the coupling surface.
            # It absorbs smoothly (permeability, not gravity) and the
            # absorbed quantities get exchanged between fields.
            # The fold_mask acts as a smooth core_mask — same physics,
            # but shaped by the Hann window instead of a hard sphere.
            psi1, omega1, abs_psi1, abs_omega1 = wave_step_ex(
                psi1, omega1, g_mu_nu, w_alpha_t,
                fold_mask, drain_mask, surface_mask,
                kappa1, absorption,
                use_laplacian=use_laplacian,
                use_advection=use_advection,
                return_absorbed=True, no_pump=no_pump,
                propagate=do_propagate
            )
            psi2, omega2, abs_psi2, abs_omega2 = wave_step_ex(
                psi2, omega2, g_mu_nu, w_alpha_t,
                fold_mask, drain_mask, surface_mask,
                kappa2, absorption,
                use_laplacian=use_laplacian,
                use_advection=use_advection,
                return_absorbed=True, no_pump=no_pump,
                propagate=do_propagate
            )
            # Conservative exchange through the fold
            psi1   = psi1   + mirror_sign_psi   * abs_psi2
            omega1 = torch.relu(omega1 + mirror_sign_omega * abs_omega2)
            psi2   = psi2   + mirror_sign_psi   * abs_psi1
            omega2 = torch.relu(omega2 + mirror_sign_omega * abs_omega1)

            flux_psi   = float(abs_psi1.abs().sum().item() +
                                abs_psi2.abs().sum().item())
            flux_omega = float(abs_omega1.sum().item() + abs_omega2.sum().item())

        else:
            # Classical single or dual-uncoupled evolution
            psi1, omega1 = wave_step_ex(
                psi1, omega1, g_mu_nu, w_alpha_t,
                core_mask, drain_mask, surface_mask,
                kappa1, absorption,
                use_laplacian=use_laplacian,
                use_advection=use_advection,
                return_absorbed=False, no_pump=no_pump,
                propagate=do_propagate
            )
            if use_dual:
                psi2, omega2 = wave_step_ex(
                    psi2, omega2, g_mu_nu, w_alpha_t,
                    core_mask, drain_mask, surface_mask,
                    kappa2, absorption,
                    use_laplacian=use_laplacian,
                    use_advection=use_advection,
                    return_absorbed=False, no_pump=no_pump,
                    propagate=do_propagate
                )
            flux_omega = 0.0
            flux_psi   = 0.0

        final_step = i

        if i % probe_interval == 0 or i < dense_phase:
            psi2_log   = psi2   if use_dual else torch.zeros_like(psi1)
            omega2_log = omega2 if use_dual else torch.zeros_like(omega1)
            tracker.record(i, omega1, omega2_log, psi1, psi2_log,
                           flux_psi, flux_omega)

            # Spatial snapshots at node positions (for analyze.py)
            clouds.record(i, omega1, psi1, node_positions, grid_size,
                          g_mu_nu, kappa1, device)

            if i % max(1, (probe_interval * 10)) == 0 or i < 5:
                print(f"{i:>6} | {omega1.sum().item():>10.3e} | "
                      f"{omega2_log.sum().item():>10.3e} | "
                      f"{psi1.abs().sum().item():>10.3e} | "
                      f"{psi2_log.abs().sum().item():>10.3e} | "
                      f"{flux_omega:>10.3e} | {flux_psi:>10.3e}")

        if torch.isnan(omega1).any() or torch.isinf(omega1).any():
            print(f"\n⚠️  COLLAPSE (field 1) at step {i}")
            meta['collapse_step'] = i
            break
        if use_dual and (torch.isnan(omega2).any() or torch.isinf(omega2).any()):
            print(f"\n⚠️  COLLAPSE (field 2) at step {i}")
            meta['collapse_step'] = i
            break

    wall_elapsed = _time.time() - wall_start
    meta['total_steps']  = final_step + 1
    meta['wall_seconds'] = round(wall_elapsed, 1)

    tracker.save(run_dir)
    clouds.save(run_dir)
    with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    update_latest(run_id)

    # Quick summary
    if len(tracker.steps) > 1:
        e0 = tracker.total_omega_1[0] + tracker.total_omega_2[0]
        ef = tracker.total_omega_1[-1] + tracker.total_omega_2[-1]
        drift = (ef - e0) / max(abs(e0), 1e-12)
        print(f"\n=== Run {run_id:04d} complete ===")
        print(f"Steps: {final_step + 1}")
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
    p = argparse.ArgumentParser(description='Resonant Field Engine — Coupled')
    p.add_argument('--grid',         type=int,   default=89)
    p.add_argument('--steps',        type=int,   default=2000)
    p.add_argument('--no-zeta',      action='store_true')
    p.add_argument('--hann',         action='store_true')
    p.add_argument('--no-laplacian', action='store_true')
    p.add_argument('--tune',         type=float, default=1.0)
    p.add_argument('--beat',         type=float, default=0.1)
    p.add_argument('--severity',     type=float, default=BEAT_SEVERITY_DEFAULT)
    p.add_argument('--core-radius',  type=float, default=CORE_RADIUS)
    p.add_argument('--absorption',   type=float, default=CORE_ABSORPTION)
    p.add_argument('--n-zeros',      type=int,   default=50)
    p.add_argument('--no-carrier',   action='store_true')
    p.add_argument('--no-beat',      action='store_true')
    p.add_argument('--no-core',      action='store_true')
    p.add_argument('--soft-alpha',   type=float, default=SOFT_ALPHA)
    p.add_argument('--extra-freq',   type=str,   default=None)
    p.add_argument('--dense-phase',  type=int,   default=50)
    p.add_argument('--probe-interval', type=int, default=1)
    p.add_argument('--slice-interval', type=int, default=5)

    # Coupled-specific flags
    p.add_argument('--no-dual',      action='store_true',
                   help='Disable mirror field (classical single-field mode)')
    p.add_argument('--no-exchange',  action='store_true',
                   help='Disable core exchange (dual fields evolve independently)')
    p.add_argument('--advection',    action='store_true',
                   help='Enable v·∇psi advection term')
    p.add_argument('--sign-psi',     type=float, default=1.0,
                   help='Sign on psi at exchange (+1 = conservative, -1 = drain)')
    p.add_argument('--sign-omega',   type=float, default=1.0,
                   help='Sign on omega at exchange (+1 = conservative, -1 = drain)')
    p.add_argument('--mirror-init',  action='store_true',
                   help='Field 2 starts empty (populated by exchange)')
    p.add_argument('--conjugate-init', action='store_true',
                   help='Field 2 starts as phase conjugate (psi2 = -psi1)')
    p.add_argument('--no-pump', action='store_true',
                   help='Disable per-step energy injection (keeps propagation)')
    p.add_argument('--single-shot', action='store_true',
                   help='Pump fires once at step 0, then field evolves freely')
    p.add_argument('--laplacian-only', action='store_true',
                   help='Pure diffusion: disable both pump AND propagation')
    p.add_argument('--device', type=str, default=None,
                   help='Force device: cpu or cuda (default: auto-detect)')

    args = p.parse_args()

    extra_freq = None
    if args.extra_freq:
        extra_freq = [float(x.strip()) for x in args.extra_freq.split(',')][:3]

    run_coupled_simulation(
        steps            = args.steps,
        grid_size        = args.grid,
        use_zeta         = not args.no_zeta,
        use_hann         = args.hann,
        use_laplacian    = not args.no_laplacian,
        tune_scalar      = args.tune,
        beat_detune      = args.beat,
        severity         = args.severity,
        core_radius      = args.core_radius,
        absorption       = args.absorption,
        n_zeros          = args.n_zeros,
        no_carrier       = args.no_carrier,
        no_beat          = args.no_beat,
        no_core          = args.no_core,
        soft_alpha       = args.soft_alpha,
        extra_freq       = extra_freq,
        dense_phase      = args.dense_phase,
        probe_interval   = args.probe_interval,
        slice_interval   = args.slice_interval,
        use_dual         = not args.no_dual,
        use_exchange     = not args.no_exchange,
        use_advection    = args.advection,
        mirror_sign_psi  = args.sign_psi,
        mirror_sign_omega= args.sign_omega,
        mirror_init      = args.mirror_init,
        conjugate_init   = args.conjugate_init,
        no_pump          = 'laplacian_only' if args.laplacian_only else args.no_pump,
        single_shot      = args.single_shot,
        force_device     = args.device,
    )
