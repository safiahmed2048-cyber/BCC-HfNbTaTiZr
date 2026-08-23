import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap



XT_FILES = [
    "profile_pristine_[110]_shock_up1kms.txt",
    
]

DATA_DIR = "data"   
OUTPUT_DIR = ""    


DT_PS = 0.001       
Z_ANGSTROM_TO_NM = 0.1   


IDX_Z       = 1   # Coord1
IDX_VORO    = 6   # c_voro[1]
IDX_SZZ     = 7   # c_stress[3]  normal to shock direction
IDX_DENSITY = 9   # density/mass 

EPS_VORO = 1e-6


DENSITY_CLIM = (0, None)   
STRESS_CLIM = None       
CMAP = "jet"                

FONT_SIZE = 12




def read_all_blocks(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    blocks = []
    i = 0
    needed_idx = max(IDX_Z, IDX_VORO, IDX_SZZ, IDX_DENSITY)

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
            z_list, dens_list, szz_list = [], [], []

            for _ in range(nchunks):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                i += 1
                if len(cols) <= needed_idx:
                    continue
                try:
                    z    = float(cols[IDX_Z])
                    voro = float(cols[IDX_VORO])
                    szz  = -float(cols[IDX_SZZ])    
                    dens = float(cols[IDX_DENSITY])
                except ValueError:
                    continue
                if abs(voro) < EPS_VORO:
                    continue

                sigma_zz_gpa = 1e-4 * szz / voro

                z_list.append(z)
                dens_list.append(dens)
                szz_list.append(sigma_zz_gpa)

            if z_list:
                data = sorted(zip(z_list, dens_list, szz_list), key=lambda t: t[0])
                z_arr, dens_arr, szz_arr = (np.array(x) for x in zip(*data))
                blocks.append({"timestep": timestep, "z": z_arr,
                                "density_g_cm3": dens_arr, "sigma_zz_gpa": szz_arr})
        else:
            i += 1

    return blocks


def load_file(fname):
    path = os.path.join(DATA_DIR, fname) if DATA_DIR else fname
    if not os.path.isfile(path):
        print(f"[WARNING] File not found, skipping: {path}")
        return None
    try:
        blocks = read_all_blocks(path)
    except Exception as e:
        print(f" Could not parse {path}: {e}. Skipping.")
        return None
    if not blocks:
        print(f" No usable data blocks in {path}. Skipping.")
        return None
    return blocks



def build_xt_grids(blocks):

    t0 = blocks[0]["timestep"] * DT_PS

    z_lo = min(b["z"].min() for b in blocks)
    z_hi = max(b["z"].max() for b in blocks)

    dz = min(np.min(np.diff(b["z"])) for b in blocks if len(b["z"]) > 1)
    n_pts = int(round((z_hi - z_lo) / dz)) + 1
    z_ref = np.linspace(z_lo, z_hi, n_pts)

    times, dens_rows, szz_rows = [], [], []
    for b in blocks:
        t_rel = b["timestep"] * DT_PS - t0
        dens_interp = np.interp(z_ref, b["z"], b["density_g_cm3"],
                                 left=np.nan, right=np.nan)
        szz_interp = np.interp(z_ref, b["z"], b["sigma_zz_gpa"],
                                left=np.nan, right=np.nan)
        times.append(t_rel)
        dens_rows.append(dens_interp)
        szz_rows.append(szz_interp)

    t_ps = np.array(times)
    z_nm = z_ref * Z_ANGSTROM_TO_NM
    dens_grid = np.array(dens_rows) 
    szz_grid = np.array(szz_rows)

    order = np.argsort(t_ps)
    return t_ps[order], z_nm, dens_grid[order], szz_grid[order]




def resolve_clim(clim, data):

    if clim is None:
        return float(np.nanmin(data)), float(np.nanmax(data))
    vmin, vmax = clim
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    return vmin, vmax


def white_zero_cmap(base_cmap=CMAP):

    base = plt.get_cmap(base_cmap, 256)
    colors = base(np.linspace(0.0, 1.0, 256))
    colors[0] = [1, 1, 1, 1]      
    cmap = ListedColormap(colors, name=f"white_{base_cmap}")
    cmap.set_bad("white")        
    return cmap


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": FONT_SIZE, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })


def plot_xt_diagram(fname, t_ps, z_nm, dens_grid, szz_grid):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.5))
    dens_cmap = white_zero_cmap(CMAP)
    stress_cmap = plt.get_cmap(CMAP).copy()
    stress_cmap.set_bad("white")

    Z, T = np.meshgrid(z_nm, t_ps)

    dmin, dmax = resolve_clim(DENSITY_CLIM, dens_grid)
    dens_norm = Normalize(dmin, dmax)
    im_a = ax_a.pcolormesh(Z, T, dens_grid, shading="gouraud", cmap=dens_cmap, norm=dens_norm)
    ax_a.set_xlabel("Z coordinate (nm)")
    ax_a.set_ylabel("Time (ps)")
    ax_a.set_title("(a)  Mass density", loc="left")
    cb_a = fig.colorbar(im_a, ax=ax_a)
    cb_a.set_label(r"$\rho$ (g/cm$^3$)")

    smin, smax = resolve_clim(STRESS_CLIM, szz_grid)
    szz_norm = Normalize(smin, smax)
    im_b = ax_b.pcolormesh(Z, T, szz_grid, shading="gouraud", cmap=stress_cmap, norm=szz_norm)
    ax_b.set_xlabel("Z coordinate (nm)")
    ax_b.set_ylabel("Time (ps)")
    ax_b.set_title(r"(b)  Normal stress $\sigma_{zz}$", loc="left")
    cb_b = fig.colorbar(im_b, ax=ax_b)
    cb_b.set_label(r"$\sigma_{zz}$ (GPa)")

    for ax in (ax_a, ax_b):
        ax.tick_params(which="both", top=True, right=True)
        ax.set_xlim(0, z_nm.max()) 

    fig.tight_layout()
    base = os.path.splitext(os.path.basename(fname))[0]
    outname = f"xt_diagram_{base}.png"
    out = os.path.join(OUTPUT_DIR, outname) if OUTPUT_DIR else outname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    style()
    for fname in XT_FILES:
        blocks = load_file(fname)
        if blocks is None:
            continue
        t_ps, z_nm, dens_grid, szz_grid = build_xt_grids(blocks)
        plot_xt_diagram(fname, t_ps, z_nm, dens_grid, szz_grid)


if __name__ == "__main__":
    main()
