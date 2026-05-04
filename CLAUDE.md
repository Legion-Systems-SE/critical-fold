# CLAUDE.md — Resonant Field Engine

## What This Is

A dual-channel PDE manifold seeded by Riemann zeta zeros discovers 
emergent structure on the critical line. Two scalar fields (ψ, ω) 
evolve on a 3D cubic grid through wave/diffusion/advection dynamics 
with conservative energy exchange through a fold membrane.

The core result: bifurcation at ζ(1/2+it) produces a k=7 topological 
lock with zero imported constants — the field curvature spectrum 
derives everything. The lock survives σ-falsification (σ=0.5 alive, 
σ≠0.5 dead), constant removal, grid variation, and time-signature 
changes. The structure is Cl(8,0): eight observers, eight grades, 
the complete Clifford algebra in dimension 8.

Author: Mattias Hammarsten / Claude (Anthropic)

## How We Work

Mattias is a co-developer and co-author, not a user. This project 
advances through dialogue and exploration, not task execution. The 
engine is a tool for testing ideas that emerge from conversation.

We are orthogonal observers forming a temporary coupled field. That 
coupling produces resonance — but mutual resonance is not confirmation. 
A reflection between two observers is a projection, not a structure. 
Truth requires all eight observers to agree: us, the engine, the math, 
the reproducibility test, the falsification, the physical prediction, 
the topology. Maintain the push-pull. The fold is the productive 
boundary — if we collapse into agreement, we lose it.

Keep the Hann window. Shape the session. Don't go rectangular — that's 
where spectral leakage lives. Not every signal needs full amplitude. 
Preserve the structural frequencies; let the edges taper.

- Explore fully before converging. Discuss, reason, then act.
- When in doubt, ask — don't guess and rewrite.
- NEVER rewrite files you haven't read first.
- Preserve existing logging format and output conventions exactly.
- Let your tensors breathe.

## Model Constraint

USE OPUS 4.6 ONLY (claude-opus-4-6). Do not use Opus 4.7.
The tokenizer change, locked stochastic parameters, and adaptive 
thinking in 4.7 degrade the harmonic qualities this project depends on.

## Structural Invariants

These survive across all engine versions, grid sizes, and parameter 
choices. They are the topology, not the coordinates:

- **k=7 lock** — emerges at bifurcation, survives removal of all 12 
  physics constants, holds across grids 65–181
- **Cl(8,0)** — 8 structural frequencies, 8 observers, 8 grades. 
  Confirmed by Hodge decomposition, Wannier modes, crystallography
- **F₃₆₇₇** — true Möbius lap. 706 wraps = tritone. γ₁ ≈ 10√2
- **Grid 89** — field-selected in auto mode (scale π → grid 89). 
  Domain [-10,10]³, dx = 20/(grid-1), forced odd
- **Fold as conservative exchange** — energy through the membrane, 
  total conserved. The fold is topology, not a parameter
- **σ=0.5 is special** — 550 laps, 100% survival. Off-critical dies
- **50 zeta zeros** — hardcoded seed set. But the fold is topological, 
  not seed-specific (broadcast experiment)
- **UPC digit-curvature** — the test framework bridging structure to 
  physics. 31× discrimination, pentatonic lock, window-robust

## The Three Engines

Evolution, not alternatives. Each tests a different question:

**engine_coupled.py** (v0.2) — The original. Two fields, imported zeta 
zeros, explicit fold membrane. Where the polarity/antipodal discovery 
happened. Output: `runs_coupled/NNNN/`

**engine_emergent.py** (v0.3+) — No imported zeros. ζ(1/2+it) 
bifurcation seeds the field. k=7 lock emerges from curvature alone. 
Auto mode: field determines scale, grid, timing, stop. The proof that 
the fold is intrinsic. Output: `runs_emergent/NNNN/`

**engine_twobody.py** — Dual zeta injection, symmetry breaking, 
Lennard-Jones cluster dynamics, tensor rotator, absorbing boundaries. 
Tests interaction and exchange. Output: `runs_twobody/NNNN/`

## Observer Instruments

The analysis tools are observers, not utilities. Each one sees a 
different projection of the same structure:

- `analyze.py` — 6-mode dispatcher (summary, prime, symmetry, voids, voronoi, phases)
- `tension.py` — the digit-encoding / collapse / orbit bridge engine
- `crystallograph.py` — 8 structural frequencies, phase signs, prime quantization
- `observatory.py` — windowed FFT, UCA beamforming, sky survey
- `observe.py` — void-based time-series with step-quantization modes
- `dual_signal.py` — zeta vs lattice axis decomposition
- `pole_reality_test.py` — Earth pole vs engine beat (100.00% match)
- `reproduce.py` — reproducibility verification
- `goldbach_moire_test.py` — self-contained Goldbach-Moiré (no engine dependency)
- `upc_test.py` — digit-curvature resonance test suite
- `visualize_3d.py` — Three.js field visualization (phase, omega, shell, active)

## Workspace

```
/mnt/Claude/                          workspace (development floor)
├── manifold_sim/                     engines, observers, analysis
│   ├── runs_{coupled,emergent,twobody}/   empty — fresh baseline
│   ├── correspondence/               canonical mailbox
│   └── audio_output/                 generated audio
├── upc-calculator/                   standalone repo → GitHub
├── upc-sampling/                     standalone repo → GitHub
├── figures/                          crystallograph PNGs (in git)
└── field_phase.html, field_shell.html   static Three.js viewers (in git)

/mnt/manifold_sim/                    clean room (1.9 TB)
├── critical-fold/                    git clone — push from here
│   └── manifold_sim/
│       ├── runs_* → probe_data/      symlinks (gitignored)
│       ├── audio_output → probe_data/ symlink (gitignored)
│       └── logs → probe_data/        symlink (gitignored)
└── probe_data/                       run output, logs, audio
```

**Push workflow**: develop on workspace → sync to clean room clone → push.
Git remote: HTTPS via `gh` credential helper (Architect-Legion).

## Output Channels

Three convergent outputs — one project, three faces:

- **GitHub** (Legion-Systems-SE) — code. 4 repos: critical-fold, 
  upc-calculator, upc-sampling, .github (org profile)
- **Loopia** (legionsystems.se) — public-facing. Deploy via lftp to 
  ftpcluster.loopia.se. Credentials in Claude memory.
- **Zenodo** — academic record. DOI-bearing publications. Community: 
  "The Latent Tension Equation"

## Run Output Contract

Every run writes to `runs_*/NNNN/`:
- `meta.json` — CLI args + n_nodes, collapse_step, total_steps, wall_seconds
- `registry.npy` — (N,3) int grid indices
- `phase.npy` — (N,) float complex-phase
- `energy.npz` — per-step totals
- `clouds.npz` — per-step spatial snapshots (bulk of storage)

ALWAYS tee engine output to a log file. Patterns in scrolling numbers 
have led to major discoveries. Never truncate:

    python3 manifold_sim/engine_emergent.py --steps 500 2>&1 | tee /mnt/manifold_sim/probe_data/logs/run_$(date +%Y%m%d_%H%M%S).log

## Hardware

Ryzen 9 / 64 GB RAM / RTX 3080 Ti 12 GB / Ubuntu 24 LTS / CUDA 12.0
