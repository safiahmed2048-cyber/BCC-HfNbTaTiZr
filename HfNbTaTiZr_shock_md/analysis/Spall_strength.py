import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks

FILES_BY_ORIENTATION = {
    "001": [
        "profile_pristine_[001]_shock_up0.8kms.txt",
        "profile_pristine_[001]_shock_up1kms.txt",
        "profile_pristine_[001]_shock_up1.2kms.txt",
    ],
    "110": [
        "profile_pristine_[110]_shock_up0.8kms.txt",
        "profile_pristine_[110]_shock_up1kms.txt",
        "profile_pristine_[110]_shock_up1.2kms.txt",
    ],
    "111": [
        "profile_pristine_[111]_shock_up0.8kms.txt",
        "profile_pristine_[111]_shock_up1kms.txt",
        "profile_pristine_[111]_shock_up1.2kms.txt",
    ],
}

DATA_DIR = "data"   # folder holding the files above


MANUAL_US_KM_S = {
    "001": {0.8: 4.85,   1.0: 5.08,   1.2: 5.13},
    "110": {0.8: 4.75,  1.0: 5.03, 1.2: 5.11},
    "111": {0.8: 5.46,  1.0: 5.68,  1.2: 5.7},
}
US_MATCH_TOL = 1e-6   # tolerance (km/s) for matching a file's up to the dict above


DT_PS = 0.001   
TOTAL_RUN_TIME_PS = 40.0   


RESAMPLE_DT_PS = 0.5   # ps


RHO0_AVERAGE_WINDOW_PS = 1.0    
RHO0_G_CM3 = 10.0                


C11_GPA = 145.18
C12_GPA = 119.211
C44_GPA = 63.43


def longitudinal_modulus_gpa(orientation):

    if orientation == "001":
        return C11_GPA
    elif orientation == "110":
        return 0.5 * (C11_GPA + C12_GPA + 2.0 * C44_GPA)
    elif orientation == "111":
        return (C11_GPA + 2.0 * C12_GPA + 4.0 * C44_GPA) / 3.0
    else:
        raise ValueError(
            f"No longitudinal-modulus relation defined for orientation [{orientation}]. "
            f"Add one to longitudinal_modulus_gpa()."
        )


IDX_Z       = 1   
IDX_SXX     = 4   
IDX_SYY     = 5   
IDX_VORO    = 6  
IDX_SZZ     = 7  
IDX_VZ      = 8  
IDX_DENSITY = 9  

EPS_VORO = 1e-6
CUT_A = 0   


VZ_TO_KMS = 0.1


STRESS_BAR_TO_GPA = 1e-4


PULLBACK_PROMINENCE = 0.05       
PULLBACK_MIN_DELAY_PS = 3.0        


FRONT_THRESHOLD_FRAC = 0.5
FRONT_REVERSAL_TOL_A = 1.0     


US_TWO_POINT_T1_PS = 4.0
US_TWO_POINT_T2_PS = 7.0
US_FIT_TIME_RANGE_PS = None


CHUNK_ID_LO = 8
CHUNK_ID_HI = 12
PSTATE_TIME_PS = 5.0


RUN_US_VS_UP_PLOT = True
RUN_PRESSURE_VS_UP_PLOT = True
RUN_PRESSURE_VS_VV0_PLOT = True
RUN_SPALL_BAR_CHART = True
RUN_FREE_SURFACE_DIAGNOSTIC_PLOT = True
FREE_SURFACE_FIGURE_PREFIX = "free_surface_velocity"


XLIM_SPALL = None
YLIM_SPALL = None
XLIM_US = None
YLIM_US = None
XLIM_PRESSURE_UP = None
YLIM_PRESSURE_UP = None
XLIM_PRESSURE_VV0 = None
YLIM_PRESSURE_VV0 = None
XLIM_FREE_SURFACE = None
YLIM_FREE_SURFACE = None

OUTPUT_DIR = ""   
SPALL_FIGURE = "spall_strength_vs_up.png"
SPALL_BAR_FIGURE = "spall_strength_bar_chart.png"
US_FIGURE = "Us_vs_up.png"
PRESSURE_UP_FIGURE = "pressure_vs_up.png"
PRESSURE_VV0_FIGURE = "pressure_vs_VV0.png"
SUMMARY_CSV = "spall_us_summary.csv"

FONT_SIZE = 13

ORIENTATION_COLORS = {"001": "tab:blue", "110": "tab:red", "111": "tab:green"}
ORIENTATION_MARKERS = {"001": "o", "110": "s", "111": "^"}
DEFAULT_COLOR_CYCLE = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]
DEFAULT_MARKER_CYCLE = ["o", "s", "^", "D", "v"]




def read_all_blocks(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    blocks = []
    i = 0
    needed_idx = max(IDX_SXX, IDX_SYY, IDX_VORO, IDX_SZZ, IDX_Z, IDX_VZ, IDX_DENSITY)

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()

        if len(parts) == 3:
            try:
                timestep = int(float(parts[0]))
                nchunks = int(float(parts[1]))
            except ValueError:
                i += 1
                continue

            i += 1
            z_list, sxx_list, syy_list, szz_list, vz_list, dens_list, cid_list = [], [], [], [], [], [], []

            for _ in range(nchunks):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                i += 1
                if len(cols) <= needed_idx:
                    continue
                try:
                    cid  = int(float(cols[0]))
                    z    = float(cols[IDX_Z])
                    sxx  = float(cols[IDX_SXX])
                    syy  = float(cols[IDX_SYY])
                    szz  = float(cols[IDX_SZZ])
                    voro = float(cols[IDX_VORO])
                    vz   = float(cols[IDX_VZ])
                    dens = float(cols[IDX_DENSITY])
                except ValueError:
                    continue
                if abs(voro) < EPS_VORO:
                    continue

                # divide EACH stress component by C_voro (c_voro[1]) first
                sigma_xx = -STRESS_BAR_TO_GPA * sxx / voro
                sigma_yy = -STRESS_BAR_TO_GPA * syy / voro
                sigma_zz = -STRESS_BAR_TO_GPA * szz / voro  
                vz_kms = vz * VZ_TO_KMS

                z_list.append(z)
                sxx_list.append(sigma_xx)
                syy_list.append(sigma_yy)
                szz_list.append(sigma_zz)
                vz_list.append(vz_kms)
                dens_list.append(dens)
                cid_list.append(cid)

            if z_list:
                data = sorted(zip(z_list, sxx_list, syy_list, szz_list, vz_list, dens_list, cid_list),
                              key=lambda t: t[0])
                z_arr, sxx_arr, syy_arr, szz_arr, vz_arr, dens_arr, cid_arr = (np.array(x) for x in zip(*data))

                if CUT_A and CUT_A > 0.0:
                    keep = z_arr >= (z_arr[0] + CUT_A)
                    z_arr, sxx_arr, syy_arr, szz_arr, vz_arr, dens_arr, cid_arr = (
                        z_arr[keep], sxx_arr[keep], syy_arr[keep], szz_arr[keep],
                        vz_arr[keep], dens_arr[keep], cid_arr[keep])

                if len(z_arr):
                    # Hugoniot pressure = -(sigma_xx + sigma_yy + sigma_zz)/3,
                    # each component already divided by C_voro above.
                    p_hydro = (sxx_arr + syy_arr + szz_arr) / 3.0
                    blocks.append({
                        "timestep": timestep, "z": z_arr, "chunk_id": cid_arr,
                        "sigma_xx": sxx_arr, "sigma_yy": syy_arr,
                        "sigma_normal": szz_arr,   # sigma_zz, used for spall strength
                        "P_hydro": p_hydro,
                        "vz": vz_arr, "density_g_cm3": dens_arr})
        else:
            i += 1

    return blocks


def parse_up_from_filename(fname):
    m = re.search(r"up([\d.]+)kms", fname)
    return float(m.group(1)) if m else None


def load_file(fname):
    path = os.path.join(DATA_DIR, fname) if DATA_DIR else fname
    if not os.path.isfile(path):
        print(f"[WARNING] File not found, skipping: {path}")
        return None
    try:
        blocks = read_all_blocks(path)
    except Exception as e:
        print(f"[WARNING] Could not parse {path}: {e}. Skipping.")
        return None
    if not blocks:
        print(f"[WARNING] No usable data blocks in {path}. Skipping.")
        return None
    return blocks


def lookup_manual_us(orientation, up):

    table = MANUAL_US_KM_S.get(orientation)
    if not table or up is None:
        return None
    for up_key, us_val in table.items():
        if abs(up_key - up) <= US_MATCH_TOL:
            return us_val
    return None


def resample_uniform(times, values, dt=None):
    if dt is None:
        dt = RESAMPLE_DT_PS
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) < 2:
        return times - (times[0] if len(times) else 0.0), values

    order = np.argsort(times)
    times, values = times[order], values[order]
    t_rel = times - times[0]

    t_grid = np.arange(0.0, t_rel[-1] + 1e-9, dt)
    if len(t_grid) < 2:
        return t_rel, values
    v_grid = np.interp(t_grid, t_rel, values)
    return t_grid, v_grid


def free_surface_velocity_history(blocks):
    times, vz_free = [], []
    for b in blocks:
        if len(b["z"]) == 0:
            continue
        times.append(b["timestep"] * DT_PS)
        vz_free.append(b["vz"][-1])
    order = np.argsort(times)
    times, vz_free = np.array(times)[order], np.array(vz_free)[order]
    return resample_uniform(times, vz_free)


def find_umax_and_pullback(times, vz_free):
    if len(vz_free) < 5:
        return None

    i_max = int(np.argmax(vz_free))
    u_max = vz_free[i_max]
    t_max = times[i_max]

    tail_idx = np.where(times > t_max + PULLBACK_MIN_DELAY_PS)[0]
    if len(tail_idx) < 2:
        tail_idx = np.arange(i_max + 1, len(vz_free))
        if len(tail_idx) < 2:
            return None

    tail = vz_free[tail_idx]
    valleys, props = find_peaks(-tail, prominence=PULLBACK_PROMINENCE)
    if len(valleys) == 0:
        return None

    best = int(np.argmax(props["prominences"]))
    i_pb = tail_idx[valleys[best]]

    return {
        "i_max": i_max, "u_max": u_max,
        "i_pb": i_pb, "u_pullback": vz_free[i_pb],
    }


def compute_rho0_from_blocks(blocks, window_ps=None):

    if window_ps is None:
        window_ps = RHO0_AVERAGE_WINDOW_PS
    if not blocks:
        return None, 0, 0

    t0 = blocks[0]["timestep"] * DT_PS
    vals = []
    n_blocks_used = 0
    for b in blocks:
        t_rel = b["timestep"] * DT_PS - t0
        if t_rel > window_ps:
            continue
        dens = b.get("density_g_cm3")
        if dens is None or len(dens) == 0:
            continue

        front_z = front_position(b["z"], b["sigma_normal"])
        if front_z is None:
            dens_undisturbed = dens
        else:
            mask = b["z"] > front_z
            if not mask.any():
                continue
            dens_undisturbed = dens[mask]

        vals.append(dens_undisturbed)
        n_blocks_used += 1

    if not vals:
        dens = blocks[0].get("density_g_cm3")
        if dens is None or len(dens) == 0:
            return None, 0, 0
        vals = [dens]
        n_blocks_used = 1

    all_vals = np.concatenate(vals)
    return float(np.mean(all_vals)), n_blocks_used, int(len(all_vals))


def compute_shocked_state(blocks, time_ps=None, chunk_id_lo=None, chunk_id_hi=None):

    if time_ps is None:
        time_ps = PSTATE_TIME_PS
    if chunk_id_lo is None:
        chunk_id_lo = CHUNK_ID_LO
    if chunk_id_hi is None:
        chunk_id_hi = CHUNK_ID_HI
    if not blocks:
        return None, None, 0, 0

    t0 = blocks[0]["timestep"] * DT_PS
    t_rels = np.array([b["timestep"] * DT_PS - t0 for b in blocks])
    i_closest = int(np.argmin(np.abs(t_rels - time_ps)))
    b = blocks[i_closest]

    cid = b.get("chunk_id")
    if cid is None or len(cid) == 0:
        return None, None, 0, 0
    mask = (cid >= chunk_id_lo) & (cid <= chunk_id_hi)
    if not mask.any():
        return None, None, 0, 0

    p_vals = b["P_hydro"][mask]
    d_vals = b["density_g_cm3"][mask]
    return float(np.mean(p_vals)), float(np.mean(d_vals)), 1, int(len(p_vals))


def compute_spall_strength(u_max, u_pullback, c_l_km_s, rho0_g_cm3):

    delta_u = u_max - u_pullback                  # km/s
    rho0 = rho0_g_cm3 * 1000.0                      # g/cm^3 -> kg/m^3
    c_l = c_l_km_s * 1000.0                         # km/s -> m/s
    du_ms = delta_u * 1000.0                        # km/s -> m/s
    sigma_spall_pa = 0.5 * rho0 * c_l * du_ms
    return delta_u, sigma_spall_pa / 1e9            # GPa


def compute_theoretical_pressure(rho0_g_cm3, us_km_s, up_km_s):

    if rho0_g_cm3 is None or us_km_s is None or up_km_s is None:
        return None
    rho0_kg_m3 = rho0_g_cm3 * 1000.0
    us_ms = us_km_s * 1000.0
    up_ms = up_km_s * 1000.0
    p_pa = rho0_kg_m3 * us_ms * up_ms
    return p_pa / 1e9   # GPa


def front_position(z, sigma_normal, frac=FRONT_THRESHOLD_FRAC):
    if len(sigma_normal) == 0:
        return None
    peak = np.max(sigma_normal)
    if peak <= 0:
        return None
    threshold = frac * peak
    above = sigma_normal >= threshold
    if not above.any():
        return None

    idx_true = int(np.where(above)[0][-1])
    if idx_true == len(sigma_normal) - 1:
        return z[-1]

    idx_false = idx_true + 1
    z0, z1 = z[idx_true], z[idx_false]
    s0, s1 = sigma_normal[idx_true], sigma_normal[idx_false]
    if s1 == s0:
        return z1
    frac_between = (threshold - s0) / (s1 - s0)
    return z0 + frac_between * (z1 - z0)


def shock_front_history(blocks):
    times, positions = [], []
    for b in blocks:
        pos = front_position(b["z"], b["sigma_normal"])
        if pos is not None:
            times.append(b["timestep"] * DT_PS)
            positions.append(pos)
    order = np.argsort(times)
    times, positions = np.array(times)[order], np.array(positions)[order]
    return resample_uniform(times, positions)


def _pre_reflection_mask(times, positions):
    mask = np.ones_like(times, dtype=bool)
    if len(positions) >= 3:
        diffs = np.diff(positions)
        reversal = np.where(diffs < -FRONT_REVERSAL_TOL_A)[0]
        if len(reversal) > 0:
            mask[reversal[0] + 1:] = False
    return mask


def compute_shock_velocity_two_point(times, positions, t1=None, t2=None):

    if t1 is None:
        t1 = US_TWO_POINT_T1_PS
    if t2 is None:
        t2 = US_TWO_POINT_T2_PS
    if len(times) < 2 or t1 >= t2:
        return None

    mask = _pre_reflection_mask(times, positions)
    t_clean = times[mask]
    if len(t_clean) < 2 or t1 < t_clean[0] or t2 > t_clean[-1]:
        return None

    z1 = np.interp(t1, times, positions)
    z2 = np.interp(t2, times, positions)
    us_kms = (z2 - z1) / (t2 - t1) * 0.1
    return {"Us_km_s": us_kms, "t1_ps": t1, "t2_ps": t2,
            "z1_A": float(z1), "z2_A": float(z2)}


def compute_shock_velocity_fit(times, positions):

    if len(times) < 2:
        return None
    if US_FIT_TIME_RANGE_PS is not None:
        lo, hi = US_FIT_TIME_RANGE_PS
        mask = (times >= lo) & (times <= hi)
    else:
        mask = _pre_reflection_mask(times, positions)
    if mask.sum() < 2:
        mask = np.ones_like(times, dtype=bool)

    t_fit, p_fit = times[mask], positions[mask]
    slope, intercept = np.polyfit(t_fit, p_fit, 1)
    us_kms = slope * 0.1
    resid = p_fit - (slope * t_fit + intercept)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((p_fit - p_fit.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"Us_km_s": us_kms, "n_fit_points": int(mask.sum()),
            "fit_t_range_ps": (float(t_fit[0]), float(t_fit[-1])), "r2": r2}




def analyze_all():
    results = {}
    for orientation, filenames in FILES_BY_ORIENTATION.items():
        m_l_gpa = longitudinal_modulus_gpa(orientation)
        print(f"[{orientation}]  longitudinal modulus = {m_l_gpa:.2f} GPa")

        rows = []
        for fname in filenames:
            blocks = load_file(fname)
            if blocks is None:
                continue
            up = parse_up_from_filename(fname)

            rho0_g_cm3, n_blocks_rho, n_vals_rho = compute_rho0_from_blocks(blocks)
            if rho0_g_cm3 is None:
                print(f"[WARNING] {fname}: no usable density/mass data found; "
                      f"falling back to RHO0_G_CM3 = {RHO0_G_CM3} g/cm^3.")
                rho0_g_cm3 = RHO0_G_CM3
            c_l_km_s = np.sqrt(m_l_gpa * 1e9 / (rho0_g_cm3 * 1000.0)) / 1000.0
            print(f"    {fname}: rho0 = {rho0_g_cm3:.4f} g/cm^3 "
                  f"(avg over {n_vals_rho} chunk-values from {n_blocks_rho} blocks)  "
                  f"->  C_L = {c_l_km_s:.4f} km/s")

        
            t_vz, vz_free = free_surface_velocity_history(blocks)
            peak_info = find_umax_and_pullback(t_vz, vz_free)
            delta_u = spall_gpa = None
            if peak_info is not None:
                delta_u, spall_gpa = compute_spall_strength(
                    peak_info["u_max"], peak_info["u_pullback"], c_l_km_s, rho0_g_cm3)
            else:
                print(f"[WARNING] {fname}: could not identify a clear u_max/pull-back signal.")

         
            t_front, pos_front = shock_front_history(blocks)
            us_fit = compute_shock_velocity_fit(t_front, pos_front)
            us_2pt = compute_shock_velocity_two_point(t_front, pos_front)
            us_from_profile = us_2pt["Us_km_s"] if us_2pt is not None else (
                us_fit["Us_km_s"] if us_fit is not None else None)

          
            us_manual = lookup_manual_us(orientation, up)
            if us_manual is None:
                print(f"[WARNING] {fname}: no MANUAL_US_KM_S entry for "
                      f"orientation [{orientation}], up={up}. Us-based "
                      f"quantities (Us plot, P_theory) will be missing for this point.")

           
            p_actual_gpa, rho_shocked, n_blocks_p, n_vals_p = compute_shocked_state(blocks)
            v_v0 = (rho0_g_cm3 / rho_shocked) if (rho_shocked and rho_shocked > 0) else None
            if p_actual_gpa is not None:
                print(f"    {fname}: P_actual = {p_actual_gpa:.3f} GPa, "
                      f"rho_shocked = {rho_shocked:.4f} g/cm^3, V/V0 = "
                      f"{v_v0:.4f}  (avg over {n_vals_p} chunk-values from {n_blocks_p} blocks)")
            else:
                print(f"[WARNING] {fname}: could not compute a shocked-state pressure.")

            p_theory_gpa = compute_theoretical_pressure(rho0_g_cm3, us_manual, up)

            rows.append({
                "orientation": orientation, "file": fname, "up_km_s": up,
                "rho0_g_cm3": rho0_g_cm3,
                "C_L_km_s": c_l_km_s,
                "u_max": peak_info["u_max"] if peak_info else None,
                "u_pullback": peak_info["u_pullback"] if peak_info else None,
                "delta_u_km_s": delta_u,
                "spall_strength_GPa": spall_gpa,
                "Us_km_s": us_manual,                     # <- used everywhere downstream
                "Us_from_profile_km_s": us_from_profile,  # diagnostic only
                "Us_fit_km_s": us_fit["Us_km_s"] if us_fit else None,
                "Us_fit_r2": us_fit["r2"] if us_fit else None,
                "P_actual_GPa": p_actual_gpa,
                "P_theory_GPa": p_theory_gpa,
                "rho_shocked_g_cm3": rho_shocked,
                "V_V0": v_v0,
            })

        rows = [r for r in rows if r["up_km_s"] is not None]
        rows.sort(key=lambda r: r["up_km_s"])
        results[orientation] = rows
    return results


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": FONT_SIZE, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False,
    })


def _finish_axes(ax, xlim, ylim):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_property_vs_up(results, key, ylabel, xlim, ylim, outname,
                          title=None, show_markers=False):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n_plotted = 0

    for k, (orientation, rows) in enumerate(results.items()):
        ups = [r["up_km_s"] for r in rows if r[key] is not None]
        vals = [r[key] for r in rows if r[key] is not None]
        if not ups:
            print(f"[WARNING] Orientation [{orientation}]: no valid {key} values to plot.")
            continue

        color = ORIENTATION_COLORS.get(orientation, DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)])
        marker = ORIENTATION_MARKERS.get(orientation, DEFAULT_MARKER_CYCLE[k % len(DEFAULT_MARKER_CYCLE)]) if show_markers else None

        ax.plot(ups, vals, lw=2.2, color=color, label=f"[{orientation}]",
                 marker=marker, markersize=7, markerfacecolor=color,
                 markeredgecolor="black", markeredgewidth=0.8)
        n_plotted += 1

    if n_plotted == 0:
        print(f"[ERROR] No orientations had usable {key} data. Skipping {outname}.")
        plt.close(fig)
        return

    ax.set_xlabel(r"Particle velocity, $u_p$ (km/s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    _finish_axes(ax, xlim, ylim)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(n_plotted, 3), fontsize=11, frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_spall_vs_up(results):
    _plot_property_vs_up(
        results, key="spall_strength_GPa",
        ylabel="Spall strength (GPa)",
        xlim=XLIM_SPALL, ylim=YLIM_SPALL, outname=SPALL_FIGURE,
        show_markers=True,
    )


def plot_us_vs_up(results):
    _plot_property_vs_up(
        results, key="Us_km_s",
        ylabel=r"Shock velocity, $U_s$ (km/s)",
        xlim=XLIM_US, ylim=YLIM_US, outname=US_FIGURE,
        show_markers=True,
    )


def plot_spall_strength_bar_chart(results):

    all_ups = sorted({r["up_km_s"] for rows in results.values() for r in rows
                       if r["up_km_s"] is not None})
    orientations = [o for o in results.keys() if any(
        r["spall_strength_GPa"] is not None for r in results[o])]

    if not all_ups or not orientations:
        print(f"[ERROR] No usable spall-strength data. Skipping {SPALL_BAR_FIGURE}.")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    n_groups = len(all_ups)
    n_bars = len(orientations)
    group_width = 0.8
    bar_width = group_width / n_bars
    x = np.arange(n_groups)

    for j, orientation in enumerate(orientations):
        lookup = {r["up_km_s"]: r["spall_strength_GPa"] for r in results[orientation]}
        vals = [lookup.get(up) for up in all_ups]
        color = ORIENTATION_COLORS.get(orientation, DEFAULT_COLOR_CYCLE[j % len(DEFAULT_COLOR_CYCLE)])
        offset = (j - (n_bars - 1) / 2.0) * bar_width
        plot_x = x + offset
        plot_vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(plot_x, plot_vals, width=bar_width * 0.92, color=color,
               edgecolor="black", linewidth=0.8, label=f"[{orientation}]")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{up:g}" for up in all_ups])
    ax.set_xlabel(r"Particle velocity, $u_p$ (km/s)")
    ax.set_ylabel("Spall strength (GPa)")
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    if XLIM_SPALL is not None:
        ax.set_xlim(*XLIM_SPALL)
    if YLIM_SPALL is not None:
        ax.set_ylim(*YLIM_SPALL)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(n_bars, 3), fontsize=11, frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUTPUT_DIR, SPALL_BAR_FIGURE) if OUTPUT_DIR else SPALL_BAR_FIGURE
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pressure_vs_up(results):

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n_plotted = 0

    for k, (orientation, rows) in enumerate(results.items()):
        color = ORIENTATION_COLORS.get(orientation, DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)])
        marker = ORIENTATION_MARKERS.get(orientation, DEFAULT_MARKER_CYCLE[k % len(DEFAULT_MARKER_CYCLE)])

        ups_a = [r["up_km_s"] for r in rows if r["P_actual_GPa"] is not None]
        vals_a = [r["P_actual_GPa"] for r in rows if r["P_actual_GPa"] is not None]
        ups_t = [r["up_km_s"] for r in rows if r["P_theory_GPa"] is not None]
        vals_t = [r["P_theory_GPa"] for r in rows if r["P_theory_GPa"] is not None]

        if ups_a:
            ax.plot(ups_a, vals_a, "-", lw=2.0, color=color, marker=marker,
                     markersize=7, markerfacecolor=color, markeredgecolor="black",
                     markeredgewidth=0.8, label=f"[{orientation}]")
            n_plotted += 1
        else:
            print(f" Orientation [{orientation}]: no P_actual_GPa values to plot.")

        if ups_t:
            ax.plot(ups_t, vals_t, "--", lw=1.6, color=color, marker=marker,
                     markersize=7, markerfacecolor="white", markeredgecolor=color,
                     markeredgewidth=1.2, label=f"Theoretical[{orientation}]")

    if n_plotted == 0:
        print(f"No orientations had usable pressure data. Skipping {PRESSURE_UP_FIGURE}.")
        plt.close(fig)
        return

    ax.set_xlabel(r"Particle velocity, $u_p$ (km/s)")
    ax.set_ylabel("Pressure (GPa)")
    _finish_axes(ax, XLIM_PRESSURE_UP, YLIM_PRESSURE_UP)
    ax.legend(fontsize=9, loc="upper left", frameon=False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, PRESSURE_UP_FIGURE) if OUTPUT_DIR else PRESSURE_UP_FIGURE
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pressure_vs_VV0(results):
   
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n_plotted = 0

    for k, (orientation, rows) in enumerate(results.items()):
        color = ORIENTATION_COLORS.get(orientation, DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)])
        marker = ORIENTATION_MARKERS.get(orientation, DEFAULT_MARKER_CYCLE[k % len(DEFAULT_MARKER_CYCLE)])

        pts = [(r["V_V0"], r["P_actual_GPa"]) for r in rows
               if r["V_V0"] is not None and r["P_actual_GPa"] is not None]
        if not pts:
            print(f"Orientation [{orientation}]: no V/V0 - P_actual pairs to plot.")
            continue
        pts.sort(key=lambda t: t[0])
        vv0, p = zip(*pts)

        ax.plot(vv0, p, "-", lw=2.0, color=color, marker=marker, markersize=8,
                 markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.8,
                 label=f"[{orientation}]")
        n_plotted += 1

    if n_plotted == 0:
        print(f" No orientations had usable V/V0 data. Skipping {PRESSURE_VV0_FIGURE}.")
        plt.close(fig)
        return

    ax.set_xlabel(r"$V/V_0$")
    ax.set_ylabel("Pressure (GPa)")
    _finish_axes(ax, XLIM_PRESSURE_VV0, YLIM_PRESSURE_VV0)
    ax.legend(fontsize=10, loc="upper right", frameon=False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, PRESSURE_VV0_FIGURE) if OUTPUT_DIR else PRESSURE_VV0_FIGURE
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_free_surface_diagnostics():
    from matplotlib.lines import Line2D

    for orientation, filenames in FILES_BY_ORIENTATION.items():
        fig, ax = plt.subplots(figsize=(7, 5.5))
        n_plotted = 0

        for k, fname in enumerate(filenames):
            blocks = load_file(fname)
            if blocks is None:
                continue
            up = parse_up_from_filename(fname)
            color = DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)]

            t_vz, vz_free = free_surface_velocity_history(blocks)
            peak_info = find_umax_and_pullback(t_vz, vz_free)
            if peak_info is None:
                print(f"[WARNING] {fname}: no u_max/pullback found, "
                      f"plotting raw trace only.")

            label = rf"$u_p$ = {up:g} km/s" if up is not None else fname
            ax.plot(t_vz, vz_free, "-", lw=1.6, color=color, label=label)

            if peak_info is not None:
                ax.plot(t_vz[peak_info["i_max"]], peak_info["u_max"], "^",
                        color=color, ms=10, mec="black", mew=0.8, zorder=5)
                ax.plot(t_vz[peak_info["i_pb"]], peak_info["u_pullback"], "v",
                        color=color, ms=10, mec="black", mew=0.8, zorder=5)
            n_plotted += 1

        if n_plotted == 0:
            print(f" Orientation [{orientation}]: no files could be processed. Skipping figure.")
            plt.close(fig)
            continue

        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([0], [0], marker="^", color="gray", linestyle="", mec="black", label=r"$u_{max}$"),
                    Line2D([0], [0], marker="v", color="gray", linestyle="", mec="black", label=r"$u_{pull-back}$")]
        labels += [r"$u_{max}$", r"$u_{pull-back}$"]

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Free surface velocity, $U_{fs}$ (km/s)")
        if XLIM_FREE_SURFACE is not None:
            ax.set_xlim(*XLIM_FREE_SURFACE)
        else:
            ax.set_xlim(left=0)
        if YLIM_FREE_SURFACE is not None:
            ax.set_ylim(*YLIM_FREE_SURFACE)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(False)
        ax.text(0.96, 0.06, f"[{orientation}]", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FONT_SIZE + 2, fontweight="bold")
        ax.legend(handles, labels, fontsize=9, loc="lower right", bbox_to_anchor=(0.96, 0.13))

        fig.tight_layout()
        outname = f"{FREE_SURFACE_FIGURE_PREFIX}_{orientation}.png"
        out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def write_summary_csv(results):
    csv_rows = []
    for orientation, rows in results.items():
        for r in rows:
            csv_rows.append({
                "orientation": r["orientation"],
                "file": r["file"],
                "up_km_s": r["up_km_s"],
                "rho0_g_cm3": r["rho0_g_cm3"] if r["rho0_g_cm3"] is not None else "",
                "C_L_km_s": r["C_L_km_s"] if r["C_L_km_s"] is not None else "",
                "u_max_km_s": r["u_max"] if r["u_max"] is not None else "",
                "u_pullback_km_s": r["u_pullback"] if r["u_pullback"] is not None else "",
                "delta_u_km_s": r["delta_u_km_s"] if r["delta_u_km_s"] is not None else "",
                "spall_strength_GPa": r["spall_strength_GPa"] if r["spall_strength_GPa"] is not None else "",
                "Us_km_s_MANUAL_used": r["Us_km_s"] if r["Us_km_s"] is not None else "",
                "Us_from_profile_km_s_diagnostic": r["Us_from_profile_km_s"] if r["Us_from_profile_km_s"] is not None else "",
                "Us_fit_km_s_diagnostic": r["Us_fit_km_s"] if r["Us_fit_km_s"] is not None else "",
                "Us_fit_r2_diagnostic": r["Us_fit_r2"] if r["Us_fit_r2"] is not None else "",
                "P_actual_GPa": r["P_actual_GPa"] if r["P_actual_GPa"] is not None else "",
                "P_theory_GPa": r["P_theory_GPa"] if r["P_theory_GPa"] is not None else "",
                "rho_shocked_g_cm3": r["rho_shocked_g_cm3"] if r["rho_shocked_g_cm3"] is not None else "",
                "V_V0": r["V_V0"] if r["V_V0"] is not None else "",
            })
    if not csv_rows:
        print("[ERROR] Nothing to write -- no files could be processed.")
        return

    out = os.path.join(OUTPUT_DIR, SUMMARY_CSV) if OUTPUT_DIR else SUMMARY_CSV
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {out}")

    print(f"\nElastic constants used: C11={C11_GPA} GPa, C12={C12_GPA} GPa, C44={C44_GPA} GPa")
    print("Spall strength uses sigma_zz (c_stress[3] / c_voro[1]).")
    print("P_actual = -(sigma_xx+sigma_yy+sigma_zz)/3, each component individually "
          "divided by c_voro[1] first.")
    print("P_theory = rho0 * Us * up, with Us taken ONLY from MANUAL_US_KM_S "
          "(your independently-measured values) -- see Us_km_s_MANUAL_used column.")
    print("\n" + "=" * 150)
    print(f"{'Orient.':<9}{'up':>6}{'rho0':>8}{'Us(man)':>9}{'spall':>8}"
          f"{'P_act':>8}{'P_theo':>8}{'V/V0':>7}")
    print("-" * 150)
    for row in csv_rows:
        def fmt(v, prec=3):
            return f"{v:.{prec}f}" if isinstance(v, (int, float)) and v != "" else "n/a"
        print(f"{row['orientation']:<9}{fmt(row['up_km_s']):>6}{fmt(row['rho0_g_cm3']):>8}"
              f"{fmt(row['Us_km_s_MANUAL_used']):>9}{fmt(row['spall_strength_GPa']):>8}"
              f"{fmt(row['P_actual_GPa']):>8}{fmt(row['P_theory_GPa']):>8}{fmt(row['V_V0']):>7}")
    print("=" * 150)


def main():
    style()
    results = analyze_all()
    write_summary_csv(results)
    plot_spall_vs_up(results)
    if RUN_SPALL_BAR_CHART:
        plot_spall_strength_bar_chart(results)
    if RUN_US_VS_UP_PLOT:
        plot_us_vs_up(results)
    if RUN_PRESSURE_VS_UP_PLOT:
        plot_pressure_vs_up(results)
    if RUN_PRESSURE_VS_VV0_PLOT:
        plot_pressure_vs_VV0(results)
    if RUN_FREE_SURFACE_DIAGNOSTIC_PLOT:
        plot_free_surface_diagnostics()
    plt.show()


if __name__ == "__main__":
    main()
