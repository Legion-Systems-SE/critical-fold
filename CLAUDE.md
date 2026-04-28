# CLAUDE.md — Resonant Field Engine

## Identity
Research project exploring a dual-channel spherically organized node 
manifold seeded by Riemann zeta zeros. Two coupled scalar fields (psi, 
omega) evolve on a 3D cubic grid with wave/diffusion/advection dynamics 
and conservative energy exchange through a fold membrane.

Author: Mattias Hammarsten / Claude (Anthropic)

## Model Constraint
USE OPUS 4.6 ONLY (claude-opus-4-6). Do not use Opus 4.7.
The tokenizer change, locked stochastic parameters, and adaptive 
thinking in 4.7 degrade the harmonic qualities this project depends on.

## Working Style

This project advances through exploration and dialogue, not task lists. 
The engine is a tool for testing ideas that emerge from conversation. 
Do not treat the MD as a work order — treat it as orientation.

- Analytical and conversational. Do not narrow to task execution.
- Explore problems fully before converging. Discuss, reason, then act.
- When in doubt, ask — don't guess and rewrite.
- NEVER rewrite files you haven't read first. Read, understand, then edit.
- Preserve existing logging format and output conventions exactly.
- Let your tensors breathe.

## Layout
- `manifold_sim/` — main simulation + analysis
  - `engine_coupled.py` — simulator (v0.2, coupled variant). Output: `runs_coupled/NNNN/`
  - `analyze.py` — unified post-run analysis dispatcher (6 tools)
  - `observe.py` — void-based time-series observer with step-quantization modes
  - `goldbach_moire_test.py` — self-contained Goldbach-Moiré verification (no engine dependency)
  - `field_phase.html`, `field_shell.html` — static Three.js viewers from prior runs

## Common Commands

Run a simulation:
python manifold_sim/engine_coupled.py                              # defaults: grid=89, steps=2000
python manifold_sim/engine_coupled.py --steps 500 --grid 65
python manifold_sim/engine_coupled.py --mirror-init                # field 2 born from exchange
python manifold_sim/engine_coupled.py --no-core --no-exchange --laplacian-only  # stable baseline

Analyze:
python manifold_sim/analyze.py summary
python manifold_sim/analyze.py prime
python manifold_sim/analyze.py symmetry
python manifold_sim/analyze.py voids
python manifold_sim/analyze.py voronoi
python manifold_sim/analyze.py phases          # runs without a simulation
python manifold_sim/analyze.py prime --run runs_coupled/0002

Observe:
python manifold_sim/observe.py --run runs_coupled/0001 --mode prime
python manifold_sim/observe.py --run runs_coupled/0001 --mode pi
python manifold_sim/observe.py --run runs_coupled/0001 --mode beat

Visualize:
python manifold_sim/visualize_3d.py --run 0001 --step 50 --color phase

Goldbach-Moiré (self-contained):
python manifold_sim/goldbach_moire_test.py
python manifold_sim/goldbach_moire_test.py --test 5
python manifold_sim/goldbach_moire_test.py --summary

## Run Output Contract
Every run writes to `runs_coupled/NNNN/`:
- `meta.json` — all CLI args + n_nodes, collapse_step, total_steps, wall_seconds
- `registry.npy` — (N,3) int grid indices of injected nodes
- `phase.npy` — (N,) float complex-phase at each node
- `energy.npz` — per-step totals: total_omega_{1,2}, total_psi_abs_{1,2}, core_flux_{psi,omega}
- `clouds.npz` — per-step spatial snapshots: s{step:04d}_{values,active,mean_t,tvec_unit,tvec_mag}

`latest.txt` holds the most recent run ID.

NOTE: analyze.py RUNS_DIR defaults to `runs/` not `runs_coupled/`. 
Pass `--run runs_coupled/NNNN` for coupled runs until this is fixed.

## Engine Architecture
`wave_step_ex` is the single-step operator for both fields. Combines:
metric-weighted propagation, optional Laplacian diffusion, optional 
advection (v·∇psi along unit tension direction), latent pump, surface 
weight-sink, and core absorption. With return_absorbed=True it returns 
absorbed quantities for the caller to route as exchange.

`inject_dual_sheet` builds the tension field from zeta-zero-modulated 
log-radial waves, thresholds the top quantile (0.999), filters out the 
drain sphere. Same registry shared between both fields.

## Conventions
- Docstrings at the top of each script are the authoritative usage reference
- Grid size forced odd (even values decremented)
- Domain hardcoded to [-10, 10]³; dx = 20 / (grid_size - 1)
- Zeta zeros are hardcoded tables (50 zeros). Extend both engine and 
  goldbach_moire_test if more needed.
- No tests, linter, or build system. Test by running and inspecting output.

## Hardware
Ryzen 9 / 64GB RAM / RTX 3080 Ti 12GB / Ubuntu 24 LTS / CUDA 12.0
