# HfNbTaTiZr shock MD — input scripts and post-processing

LAMMPS input files and Python post-processing scripts used for the MD shock
simulations of single-crystal BCC HfNbTaTiZr described in [paper title /
DOI once accepted]. Three loading orientations ([001], [110], [111]),
piston velocities of 0.8, 1.0 and 1.2 km/s, plus a secondary set of runs
with a pre-existing dislocation network introduced into the [001] crystal
before shock loading.

Potential: Wu et al. EAM/alloy potential for HfNbTaTiZr. (DOI: 10.1016/j.ijplas.2026.104626)
Structure built in Atomsk, simulations run in LAMMPS.

## What's in here

**lammps/**

- `in_equilibrate.lmp` — three-stage minimization (CG → FIRE → CG) followed
  by 40 ps NPT equilibration at 300 K, 0 bar. Run this first on the raw
  Atomsk-generated data file.
- `in_shock.lmp` — the actual shock run. Uses a moving wall/piston (momentum
  mirror) along z, loops over up = 0.8/1.0/1.2 km/s automatically. Bins the
  cell into 10 Å slabs along z and dumps per-bin stress, density, velocity,
  Voronoi volume via `fix ave/chunk` — this is the `profile_*.txt` output
  that both Python scripts read.
- `in_pre_existing_disloc.lmp` — tension-compression-relaxation (TCR)
  protocol: strains the equilibrated [001] cell 25% along z at a set strain
  rate, then relaxes back to zero stress under anisotropic NPT. This is what
  generates the pre-dislocated starting configurations (Case A / Case B in
  the paper) before running `in_shock.lmp` on the resulting restart file.
