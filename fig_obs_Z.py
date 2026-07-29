"""
fig_obs_Z.py — Stellar metallicity vs size, colour-coded by Compton-Y, z~1.

Three suites on one panel (different marker shapes), shared log colorbar for Y_SZ.
Point: the three simulations separate in the (Z_star, r_half, Y_SZ) space
even at fixed stellar mass, driven by their different feedback prescriptions.

Usage
-----
    python fig_obs_Z.py
    python fig_obs_Z.py --tables_dir tables/obs --outdir figures

Input
-----
    tables/obs/{suite_key}_snap*_p*.parquet  (from obs_proxies.py, needs Z_star column)

Output
------
    figures/fig_obs_Z_rh_ysz.{pdf,png}
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  11,
    "legend.fontsize":  9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":       True,
    "ytick.right":     True,
    "figure.dpi":      150,
})

SUITES  = ["TNG100-1", "Eagle100-1", "Simba25-1"]
LABELS  = {"TNG100-1": "TNG100",  "Eagle100-1": "EAGLE100", "Simba25-1": "SIMBA25"}
MARKERS = {"TNG100-1": "o",       "Eagle100-1": "s",        "Simba25-1": "^"}
MS      = {"TNG100-1": 30,        "Eagle100-1": 30,         "Simba25-1": 45}

_HERE       = pathlib.Path(__file__).resolve().parent
TABLES      = _HERE / "tables" / "obs"
FIGURES     = _HERE / "figures"
MIN_N_GAS_Y = 10
Z_SUN       = 0.0127    # Asplund et al. (2009)


def load_suite(tables_dir, suite_key):
    files = sorted(pathlib.Path(tables_dir).glob(f"{suite_key}_snap*.parquet"))
    if not files:
        print(f"  [WARN] no parquet for {suite_key}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "Z_star" not in df.columns:
        print(f"  [WARN] {suite_key}: Z_star column missing — rerun obs_proxies.py")
        return None
    ok = (
        df["Y_SZ_Mpc2"].gt(0) &
        df["r_half_kpc"].gt(0) &
        df["Z_star"].gt(0) &
        df["n_gas_Y"].ge(MIN_N_GAS_Y) &
        df["logM_star"].between(10.5, 11.0) &
        df[["Y_SZ_Mpc2", "r_half_kpc", "Z_star", "logM_star"]].notna().all(axis=1)
    )
    cleaned = df[ok].copy()
    print(f"  {suite_key}: {len(cleaned):,} galaxies")
    return cleaned


def make_figure(tables_dir, outdir):
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── load all suites first to set common colorbar range ────────────────────
    data = {}
    for suite in SUITES:
        df = load_suite(tables_dir, suite)
        if df is not None and len(df) > 0:
            data[suite] = df

    if not data:
        print("No data found.")
        return

    all_Y = np.concatenate([d["Y_SZ_Mpc2"].values for d in data.values()])
    vmin  = np.nanpercentile(all_Y, 5)
    vmax  = np.nanpercentile(all_Y, 95)
    norm  = LogNorm(vmin=vmin, vmax=vmax)
    cmap  = plt.cm.plasma

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig.subplots_adjust(right=0.82)

    for suite in SUITES:
        if suite not in data:
            continue
        df = data[suite]

        Z_solar = df["Z_star"].values / Z_SUN      # Z / Z_sun
        rh      = df["r_half_kpc"].values
        Y       = df["Y_SZ_Mpc2"].values
        z_val   = float(df["z_actual"].median()) if "z_actual" in df.columns else float("nan")

        sc = ax.scatter(
            np.log10(Z_solar), rh,
            c=Y, cmap=cmap, norm=norm,
            marker=MARKERS[suite], s=MS[suite],
            alpha=0.75, linewidths=0.4, edgecolors="k",
            label=f"{LABELS[suite]}  (z={z_val:.2f}, N={len(df)})",
            zorder=3,
        )

    # ── colorbar ──────────────────────────────────────────────────────────────
    cax = fig.add_axes([0.84, 0.15, 0.03, 0.70])
    cb  = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(
        r"$Y \cdot D_A^2$  [Mpc$^2$]",
        fontsize=10)

    ax.set_xlabel(r"$\log(Z_\star / Z_\odot)$", fontsize=12)
    ax.set_ylabel(r"3D stellar half-mass radius  $r_{1/2}$  [kpc]", fontsize=12)
    ax.set_title(
        r"$\log(M_\star/M_\odot) \in [10.5,\,11.0]$,  $z \approx 1$",
        fontsize=11, pad=6)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_obs_Z_rh_ysz.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fig: Z_star vs r_half coloured by Y_SZ, z~1")
    parser.add_argument("--tables_dir", default=str(TABLES))
    parser.add_argument("--outdir",     default=str(FIGURES))
    args = parser.parse_args()
    make_figure(args.tables_dir, args.outdir)


if __name__ == "__main__":
    main()
