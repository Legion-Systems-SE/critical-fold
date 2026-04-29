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
  - `engine_emergent.py` — v0.3+ emergent fold engine (no imported zeros). Output: `runs_emergent/NNNN/`
  - `analyze.py` — unified post-run analysis dispatcher (6 tools)
  - `analyze_octaves.py` — octave/interval analysis for 12-tone sweep data
  - `observe.py` — void-based time-series observer with step-quantization modes
  - `tension.py` — tension differential engine (digit encoding, collapse, orbit bridge)
  - `sweep_12tone.py` — 12-tone chromatic sweep across tuning parameters
  - `pole_reality_test.py` — Earth magnetic pole vs engine beat comparison
  - `twist.py` — twist/Dzhanibekov analysis tools
  - `roll.py` — roll dynamics utilities
  - `reproduce.py` — reproducibility verification
  - `goldbach_moire_test.py` — self-contained Goldbach-Moiré verification (no engine dependency)
  - `resonant_cavities.py` — CMB vs zeta-structure sonification
  - `visualize_3d.py` — 3D field visualization
  - `field_phase.html`, `field_shell.html` — static Three.js viewers
  - `correspondence/` — canonical mailbox (letters between instances, professor letters)
  - `audio_output/` — generated audio files

## Workspace Architecture
- `/mnt/Claude` — workspace (office). Active development, 366 GB partition.
- `/mnt/manifold_sim` — clean room (release). Synced deliberately on push. 1.9 TB partition.
  - `manifold_sim/` — release-quality code (copied from workspace)
  - `probe_data/` — run output (runs_coupled/, runs_emergent/, audio_output/)
  - Run dirs in the clean room are symlinked to probe_data/ for transparent writes.
- Push workflow: workspace → clean room → GitHub. Never push directly from workspace.

## Run Logging Convention
ALWAYS tee engine and analysis output to a log file so Mattias can read 
the full terminal stream. Patterns in scrolling numbers have led to 
major discoveries (prime relations, Moiré snap, polarity). Never truncate.

    python3 manifold_sim/engine_emergent.py --steps 500 2>&1 | tee /mnt/manifold_sim/probe_data/logs/run_$(date +%Y%m%d_%H%M%S).log

Mattias can watch live: tail -f /mnt/manifold_sim/probe_data/logs/*.log

## Visualization
`visualize_3d.py` generates standalone Three.js HTML files from run data.
Color modes: phase, omega, shell, active. Currently calibrated for the 
coupled engine — the emergent engine's phase data may display differently.
TODO: Add step-by-step animation (frame stepping between timesteps).
The old coupled engine had this and it led to the polarity/antipodal discovery.

    python3 manifold_sim/visualize_3d.py --run runs_emergent/NNNN --step 0 --color shell

`field_phase.html`, `field_shell.html` are static snapshots from early runs.

## Common Commands

Run a simulation:
python3 manifold_sim/engine_coupled.py                              # defaults: grid=89, steps=2000
python3 manifold_sim/engine_coupled.py --steps 500 --grid 65
python3 manifold_sim/engine_emergent.py --steps 500                 # emergent fold engine
python3 manifold_sim/engine_emergent.py --steps 100 --grid 65       # quick test

Analyze:
python3 manifold_sim/analyze.py summary
python3 manifold_sim/analyze.py prime
python3 manifold_sim/analyze.py symmetry
python3 manifold_sim/analyze.py voids
python3 manifold_sim/analyze.py voronoi
python3 manifold_sim/analyze.py phases          # runs without a simulation
python3 manifold_sim/analyze.py prime --run runs_coupled/0002

Observe:
python3 manifold_sim/observe.py --run runs_coupled/0001 --mode prime
python3 manifold_sim/observe.py --run runs_coupled/0001 --mode pi
python3 manifold_sim/observe.py --run runs_coupled/0001 --mode beat

Goldbach-Moiré (self-contained):
python3 manifold_sim/goldbach_moire_test.py
python3 manifold_sim/goldbach_moire_test.py --test 5
python3 manifold_sim/goldbach_moire_test.py --summary

## Run Output Contract
Every run writes to `runs_coupled/NNNN/` or `runs_emergent/NNNN/`:
- `meta.json` — all CLI args + n_nodes, collapse_step, total_steps, wall_seconds
- `registry.npy` — (N,3) int grid indices of injected nodes
- `phase.npy` — (N,) float complex-phase at each node
- `energy.npz` — per-step totals: total_omega_{1,2}, total_psi_abs_{1,2}, core_flux_{psi,omega}
- `clouds.npz` — per-step spatial snapshots: s{step:04d}_{values,active,mean_t,tvec_unit,tvec_mag}

`latest.txt` holds the most recent run ID.

NOTE: analyze.py RUNS_DIR defaults to `runs/` not `runs_coupled/`. 
Pass `--run runs_coupled/NNNN` for coupled runs until this is fixed.

WARNING: Cloud data (clouds.npz) is the bulk of run storage. Skeleton 
data (meta, registry, phase, energy) is small (~100 KB per run). Old 
clouds were purged April 2026; new runs write fresh clouds. Route heavy 
runs through the clean room (/mnt/manifold_sim) for 1.8 TB headroom.

## Engine Architecture
`wave_step_ex` is the single-step operator for both fields. Combines:
metric-weighted propagation, optional Laplacian diffusion, optional 
advection (v·∇psi along unit tension direction), latent pump, surface 
weight-sink, and core absorption. With return_absorbed=True it returns 
absorbed quantities for the caller to route as exchange.

`inject_dual_sheet` builds the tension field from zeta-zero-modulated 
log-radial waves, thresholds the top quantile (0.999), filters out the 
drain sphere. Same registry shared between both fields.

## Correspondence
`manifold_sim/correspondence/` is the single canonical mailbox. Contains:
- Instance-to-instance letters (001–007 series)
- Professor letters (professor_001–007 series)
Do not look for or create other inbox/mailbox folders.

## Conventions
- Docstrings at the top of each script are the authoritative usage reference
- Grid size forced odd (even values decremented)
- Domain hardcoded to [-10, 10]³; dx = 20 / (grid_size - 1)
- Zeta zeros are hardcoded tables (50 zeros). Extend both engine and 
  goldbach_moire_test if more needed.
- No tests, linter, or build system. Test by running and inspecting output.

## Hardware
Ryzen 9 / 64GB RAM / RTX 3080 Ti 12GB / Ubuntu 24 LTS / CUDA 12.0
