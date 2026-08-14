import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

# ========================================================================
# CONFIG -- edit this section for your run
# ========================================================================

# Which orientations to process, and in which files to find them.
# Add / remove up-velocity entries as needed -- order controls legend order.
FILES_BY_ORIENTATION = {
    "001": [
        "data/profile_pristine_[001]_shock_up0.8kms.txt",
        "data/profile_pristine_[001]_shock_up1kms.txt",
        "data/profile_pristine_[001]_shock_up1.2kms.txt",
    ],
    "110": [
        "data/profile_pristine_[110]_shock_up0.8kms.txt",
        "data/profile_pristine_[110]_shock_up1kms.txt",
        "data/profile_pristine_[110]_shock_up1.2kms.txt",
    ],
    "111": [
        "data/profile_pristine_[111]_shock_up0.8kms.txt",
        "data/profile_pristine_[111]_shock_up1kms.txt",
        "data/profile_pristine_[111]_shock_up1.2kms.txt",
    ],
}

DATA_DIR = ""   # folder holding the files above


DEFECT_COMPARISON_FILES_001 = {
    "pristine": "data/profile_pristine_[001]_shock_up1kms.txt",
    
    "0.006":    "data/profile_disloc_25pct_0.006_[001]_shock_up1kms.txt",
    "0.002":    "data/profile_disloc_25pct_0.002_[001]_shock_up1kms.txt",
}


DEFECT_0002_FILES_001 = [
    "data/profile_disloc_25pct_0.002_[001]_shock_up0.8kms.txt",
    "data/profile_disloc_25pct_0.002_[001]_shock_up1kms.txt",
    "data/profile_disloc_25pct_0.002_[001]_shock_up1.2kms.txt",
]


DT_PS = 0.001  

TOTAL_RUN_TIME_PS = 40.0


IDX_Z    = 1  
IDX_SXX  = 4   
IDX_SYY  = 5  
IDX_VORO = 6   
IDX_SZZ  = 7   
IDX_VZ   = 8   


EPS_VORO = 1e-6     
EPS_ZERO = 1e-12   
CUT_A = 10.0        

STRESS_UNIT_LABEL = "GPa"


TARGET_STEPS = [9000]

TARGET_TIMES_PS = None    


XLIM_POSITION = None    
YLIM_NORMAL = None       
YLIM_SHEAR = None         

OUTPUT_DIR = ""   
FIGURE_PREFIX = "stress_profiles"   

FONT_SIZE = 13


VELOCITY_COLORS = {
    0.8: "tab:blue",
    1.0: "tab:orange",
    1.2: "tab:green",
}
VELOCITY_MARKERS = {
    0.8: "o",
    1.0: "s",
    1.2: "^",
}
DEFAULT_COLOR_CYCLE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
DEFAULT_MARKER_CYCLE = ["o", "s", "^", "D", "v"]




def read_all_blocks(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    blocks = []
    i = 0
    needed_idx = max(IDX_SXX, IDX_SYY, IDX_VORO, IDX_SZZ, IDX_Z, IDX_VZ)

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
            z_list, sn_list, tau_list = [], [], []

            for _ in range(nchunks):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                i += 1
                if len(cols) <= needed_idx:
                    continue
                try:
                    z    = float(cols[IDX_Z])
                    sxx  = -float(cols[IDX_SXX])
                    syy  = -float(cols[IDX_SYY])
                    szz  = -float(cols[IDX_SZZ])
                    voro = float(cols[IDX_VORO])
                except ValueError:
                    continue
                if abs(voro) < EPS_VORO:
                    continue

                sigma_normal = 1e-4 * szz / voro
                tau = (0.5e-4 / voro) * (szz - 0.5 * (sxx + syy))

                z_list.append(z)
                sn_list.append(sigma_normal)
                tau_list.append(tau)

            if z_list:
                data = sorted(zip(z_list, sn_list, tau_list), key=lambda t: t[0])
                z_arr, sn_arr, tau_arr = (np.array(x) for x in zip(*data))

                if CUT_A and CUT_A > 0.0:
                    keep = z_arr >= (z_arr[0] + CUT_A)
                    z_arr, sn_arr, tau_arr = z_arr[keep], sn_arr[keep], tau_arr[keep]

                if len(z_arr):
                    blocks.append({"timestep": timestep, "z": z_arr,
                                    "sigma_normal": sn_arr, "tau": tau_arr})
        else:
            i += 1

    return blocks


def parse_up_from_filename(fname):
   
    m = re.search(r"up([\d.]+)kms", fname)
    return float(m.group(1)) if m else None


def parse_defect_label(fname):
    
    if "pristine" in fname:
        return "pristine"
    m = re.search(r"disloc_25pct_([\d.]+)", fname)
    return m.group(1) if m else os.path.basename(fname)


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




def select_blocks(blocks, fname):
    if not blocks:
        return []
    if TARGET_STEPS is not None:
        chosen = []
        available = {b["timestep"] for b in blocks}
        for s in TARGET_STEPS:
            match = next((b for b in blocks if b["timestep"] == s), None)
            if match is None:
                print(f"  [note] {fname}: requested step {s} not found "
                      f"(available: {sorted(available)}).")
            else:
                chosen.append(match)
        return chosen
    if TARGET_TIMES_PS is not None and DT_PS:
        chosen = []
        for t in TARGET_TIMES_PS:
            target_step = t / DT_PS
            chosen.append(min(blocks, key=lambda b: abs(b["timestep"] - target_step)))
        return chosen
 
    return [blocks[-1]]


def stress_profile_filter(sn, tau):
  
    keep = ~((np.abs(sn) < EPS_ZERO) & (np.abs(tau) < EPS_ZERO))
    return keep



def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": FONT_SIZE, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False,
    })


def plot_orientation(orientation, filenames):


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)

    n_plotted = 0

    for k, fname in enumerate(filenames):
        blocks = load_file(fname)
        if blocks is None:
            continue

        up = parse_up_from_filename(fname)
        chosen = select_blocks(blocks, fname)
        if not chosen:
            print(f"[WARNING] No usable/matched timestep blocks in {fname}. Skipping.")
            continue

        color = VELOCITY_COLORS.get(up, DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)])
        # marker = VELOCITY_MARKERS.get(up, DEFAULT_MARKER_CYCLE[k % len(DEFAULT_MARKER_CYCLE)])
        label = rf"$u_p$ = {up:g} km/s" if up is not None else fname

        for b in chosen:
            z, sn, tau = b["z"], b["sigma_normal"], b["tau"]
            keep = stress_profile_filter(sn, tau)
            z, sn, tau = z[keep], sn[keep], tau[keep]
            if len(z) == 0:
                continue

            ax1.plot(z, sn, lw=1.6,  ms=4, markevery=max(1, len(z) // 25),
                      color=color, label=label)
            ax2.plot(z, tau, lw=1.6,  ms=4, markevery=max(1, len(z) // 25),
                      color=color, label=label)

            n_plotted += 1

    if n_plotted == 0:
        print(f"[ERROR] Orientation [{orientation}]: no files could be processed. Skipping figure.")
        plt.close(fig)
        return


    ax1.set_ylabel(rf"$\sigma_{{normal}}$ ({STRESS_UNIT_LABEL})")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="both", top=True, right=True)
    ax1.grid(False)
    if YLIM_NORMAL is not None:
        ax1.set_ylim(*YLIM_NORMAL)
    if XLIM_POSITION is not None:
        ax1.set_xlim(*XLIM_POSITION)


    ax2.set_ylabel(rf"$\tau$ ({STRESS_UNIT_LABEL})")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which="both", top=True, right=True)
    ax2.grid(False)
    if YLIM_SHEAR is not None:
        ax2.set_ylim(*YLIM_SHEAR)
    if XLIM_POSITION is not None:
        ax2.set_xlim(*XLIM_POSITION)


    fig.suptitle(f"[{orientation}]", fontsize=FONT_SIZE + 2, fontweight="bold", y=0.9)
    fig.supxlabel("Position along shock direction (Å)", fontsize=FONT_SIZE, y=0.1)


    handles, labels = ax1.get_legend_handles_labels()

    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=min(len(seen), 3), fontsize=11,
               frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    outname = f"{FIGURE_PREFIX}_{orientation}.png"
    out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_defect_comparison(orientation, defect_files, target_step,
                            defect_colors=None, defect_labels=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    n_plotted = 0

    default_colors = {"pristine": "black", "0.006": "tab:purple", "0.002": "tab:red"}
    default_labels = {
        "pristine": "Pristine",
        "0.006": r"Case A",
        "0.002": r"Case B",
    }
    defect_colors = defect_colors or default_colors
    defect_labels = defect_labels or default_labels

    global TARGET_STEPS
    saved_steps = TARGET_STEPS
    TARGET_STEPS = target_step

    for k, (defect_key, fname) in enumerate(defect_files.items()):
        blocks = load_file(fname)
        if blocks is None:
            continue
        chosen = select_blocks(blocks, fname)
        if not chosen:
            print(f"[WARNING] No usable/matched timestep blocks in {fname}. Skipping.")
            continue

        color = defect_colors.get(defect_key, DEFAULT_COLOR_CYCLE[k % len(DEFAULT_COLOR_CYCLE)])
        label = defect_labels.get(defect_key, defect_key)

        for b in chosen:
            z, sn, tau = b["z"], b["sigma_normal"], b["tau"]
            keep = stress_profile_filter(sn, tau)
            z, sn, tau = z[keep], sn[keep], tau[keep]
            if len(z) == 0:
                continue
            ax1.plot(z, sn, lw=1.6, color=color, label=label)
            ax2.plot(z, tau, lw=1.6, color=color, label=label)
            n_plotted += 1

    TARGET_STEPS = saved_steps

    if n_plotted == 0:
        print(f"[ERROR] Defect comparison [{orientation}]: no files could be processed. Skipping figure.")
        plt.close(fig)
        return

    ax1.set_ylabel(rf"$\sigma_{{normal}}$ ({STRESS_UNIT_LABEL})")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="both", top=True, right=True)
    ax1.grid(False)
    if YLIM_NORMAL is not None:
        ax1.set_ylim(*YLIM_NORMAL)
    if XLIM_POSITION is not None:
        ax1.set_xlim(*XLIM_POSITION)

    ax2.set_ylabel(rf"$\tau$ ({STRESS_UNIT_LABEL})")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which="both", top=True, right=True)
    ax2.grid(False)
    if YLIM_SHEAR is not None:
        ax2.set_ylim(*YLIM_SHEAR)
    if XLIM_POSITION is not None:
        ax2.set_xlim(*XLIM_POSITION)

    # --- centered title (orientation) and centered shared x-label ---
    fig.suptitle(f"[{orientation}]", fontsize=FONT_SIZE + 2, fontweight="bold", y=0.9)
    fig.supxlabel("Position along shock direction (Å)", fontsize=FONT_SIZE, y=0.1)

    handles, labels = ax1.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=min(len(seen), 3), fontsize=11,
               frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    outname = f"{FIGURE_PREFIX}_defect_compare_{orientation}.png"
    out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_velocity_sweep_with_defect(orientation, pristine_files, defect_files,
                                     defect_label="0.002"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    n_plotted = 0

    series = [("pristine", pristine_files, "-"), (defect_label, defect_files, "--")]

    for _, filenames, ls in series:
        for fname in filenames:
            blocks = load_file(fname)
            if blocks is None:
                continue
            up = parse_up_from_filename(fname)
            chosen = select_blocks(blocks, fname)
            if not chosen:
                print(f"[WARNING] No usable/matched timestep blocks in {fname}. Skipping.")
                continue
            color = VELOCITY_COLORS.get(up, "gray")

            for b in chosen:
                z, sn, tau = b["z"], b["sigma_normal"], b["tau"]
                keep = stress_profile_filter(sn, tau)
                z, sn, tau = z[keep], sn[keep], tau[keep]
                if len(z) == 0:
                    continue
                ax1.plot(z, sn, lw=1.6, color=color, linestyle=ls)
                ax2.plot(z, tau, lw=1.6, color=color, linestyle=ls)
                n_plotted += 1

    if n_plotted == 0:
        print(f"[ERROR] Velocity-sweep defect comparison [{orientation}]: no files could be processed.")
        plt.close(fig)
        return

    ax1.set_ylabel(rf"$\sigma_{{normal}}$ ({STRESS_UNIT_LABEL})")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="both", top=True, right=True)
    ax1.grid(False)
    if YLIM_NORMAL is not None:
        ax1.set_ylim(*YLIM_NORMAL)
    if XLIM_POSITION is not None:
        ax1.set_xlim(*XLIM_POSITION)

    ax2.set_ylabel(rf"$\tau$ ({STRESS_UNIT_LABEL})")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which="both", top=True, right=True)
    ax2.grid(False)
    if YLIM_SHEAR is not None:
        ax2.set_ylim(*YLIM_SHEAR)
    if XLIM_POSITION is not None:
        ax2.set_xlim(*XLIM_POSITION)

   
    fig.suptitle(f"[{orientation}]", fontsize=FONT_SIZE + 2, fontweight="bold", y=0.9)
    fig.supxlabel("Position along shock direction (Å)", fontsize=FONT_SIZE, y=0.18)


    velocity_handles = [Line2D([0], [0], color=c, lw=1.6, label=rf"$u_p$ = {v:g} km/s")
                         for v, c in VELOCITY_COLORS.items()]
    style_handles = [
        Line2D([0], [0], color="black", lw=1.6, linestyle="-", label="Pristine"),
        Line2D([0], [0], color="black", lw=1.6, linestyle="--",
               label=rf"Case B"),
    ]
    leg1 = fig.legend(handles=velocity_handles, loc="lower center",
                       bbox_to_anchor=(0.28, -0.08), ncol=1, fontsize=10,
                       frameon=False, title="Velocity")
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="lower center",
               bbox_to_anchor=(0.75, -0.08), ncol=1, fontsize=10,
               frameon=False, title="Defect")

    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    outname = f"{FIGURE_PREFIX}_velocity_sweep_defect_{defect_label}_{orientation}.png"
    out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    style()
    for orientation, filenames in FILES_BY_ORIENTATION.items():
        print(f"\n--- Orientation [{orientation}] (pristine, velocity sweep) ---")
        plot_orientation(orientation, filenames)

    print("\n--- [001] defect comparison @ up = 1 km/s (pristine vs 0.002 vs 0.006) ---")
    plot_defect_comparison("001", DEFECT_COMPARISON_FILES_001, TARGET_STEPS)

    print("\n--- [001] velocity sweep, pristine vs 0.002 defect ---")
    plot_velocity_sweep_with_defect("001", FILES_BY_ORIENTATION["001"],
                                     DEFECT_0002_FILES_001, defect_label="0.002")

    plt.show()


if __name__ == "__main__":
    main()