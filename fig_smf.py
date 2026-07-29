"""
fig_smf.py — Stellar mass function comparison at z=0, three suites.

Shows dn/dlogM [Mpc⁻³ dex⁻¹] with Poisson error bands.
The point: the three simulations are calibrated to reproduce the same galaxy
population, validating the comparison in fig_obs_rh_ysz.

Usage (runs locally on downloaded parquets)
-------------------------------------------
    python fig_smf.py
    python fig_smf.py --tables_dir tables/smf --outdir figures

Input
-----
    tables/smf/{suite_key}_z0.parquet  (from smf.py)

Output
------
    figures/fig_smf.{pdf,png}
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

SUITES  = ["TNG100-1", "Eagle100-1", "Simba100-1"]
COLORS  = {"TNG100-1": "#1f77b4", "Eagle100-1": "#d62728", "Simba100-1": "#2ca02c"}
LABELS  = {"TNG100-1": "TNG100",  "Eagle100-1": "EAGLE100", "Simba100-1": "SIMBA100"}
LS      = {"TNG100-1": "-",       "Eagle100-1": "--",       "Simba100-1": "-."}

_HERE   = pathlib.Path(__file__).resolve().parent
TABLES  = _HERE / "tables" / "smf"
FIGURES = _HERE / "figures"

# highlight the stellar mass range used in the obs_proxies comparison
OBS_LOGM_LO, OBS_LOGM_HI = 10.5, 11.0

# minimum counts to show a bin (avoid plotting noisy empty bins)
MIN_N = 3


def load_smf(tables_dir, suite_key):
    f = pathlib.Path(tables_dir) / f"{suite_key}_z0.parquet"
    if not f.exists():
        print(f"  [WARN] {f} not found")
        return None
    return pd.read_parquet(f)


def make_figure(tables_dir, outdir):
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for suite in SUITES:
        df = load_smf(tables_dir, suite)
        if df is None:
            continue

        ok  = df["N"] >= MIN_N
        x   = df.loc[ok, "logM_cen"].values
        phi = df.loc[ok, "phi"].values
        err = df.loc[ok, "phi_err"].values
        col = COLORS[suite]

        ax.plot(x, phi, color=col, ls=LS[suite], lw=2.0,
                label=LABELS[suite], zorder=3)
        ax.fill_between(x, phi - err, phi + err,
                        color=col, alpha=0.15, zorder=2)

    # shade the comparison mass range from fig_obs
    ax.axvspan(OBS_LOGM_LO, OBS_LOGM_HI,
               color="0.85", zorder=0,
               label=rf"$\log M_\star \in [{OBS_LOGM_LO},{OBS_LOGM_HI}]$ (Fig. obs)")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log(M_\star / M_\odot)$", fontsize=12)
    ax.set_ylabel(r"$\phi$ [$\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}$]", fontsize=12)
    ax.set_title("Stellar mass function,  $z = 0$", fontsize=11, pad=6)
    ax.set_xlim(9.0, 12.3)
    ax.legend(frameon=False, fontsize=9)

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_smf.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SMF comparison figure, z=0")
    parser.add_argument("--tables_dir", default=str(TABLES))
    parser.add_argument("--outdir",     default=str(FIGURES))
    args = parser.parse_args()
    make_figure(args.tables_dir, args.outdir)


if __name__ == "__main__":
    main()
