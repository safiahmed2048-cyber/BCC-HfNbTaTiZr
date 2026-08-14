`in_shock.lmp` additionally needs `runid` and reads whatever data/restart
file you point `readfile` at. `in_pre_existing_disloc.lmp` needs
`restartfile`, `strain_rate`, `target_strain`, `Ttarget`, `runid`.

**analysis/**

- `stress_profile.py` — reads the `profile_*.txt` chunk files and plots
  normal stress (σ_zz) and resolved shear stress vs. position along the
  shock direction, at a chosen timestep. Also does the pristine-vs-defect
  and velocity-sweep-vs-defect comparison plots used for the pre-dislocated
  [001] results.
- `spall_strength.py` — full spall-strength pipeline: free-surface
  velocity history → u_max / pull-back velocity → spall strength via
  ρ₀·C_L·Δu/2, using direction-dependent longitudinal modulus from the
  measured C11/C12/C44. Also computes Hugoniot pressure from the shocked
  chunk window and compares against Rankine-Hugoniot theory (P = ρ₀·Us·up)
  using manually-measured Us values (`MANUAL_US_KM_S` — these came from the
  x-t diagrams, not an automatic fit, though the script does have
  diagnostic-only two-point and linear-fit Us estimators for
  cross-checking). Writes everything to `spall_us_summary.csv` and
  generates the spall-vs-up, Us-vs-up, pressure-vs-up, pressure-vs-V/V0 and
  free-surface-velocity-trace figures.
- `x_t_diagram.py` — turns the same chunk files into position-time (x-t)
  contour plots of density and σ_zz, which is how the shock/release wave
  structure figures (Fig. 4-type plots) were made.
