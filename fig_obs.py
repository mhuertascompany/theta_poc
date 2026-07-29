"""
fig_obs.py — Figure: 3D half-mass radius vs integrated Compton-Y at fixed M_star, z~1.

Three suites (TNG100-1, Eagle100-1, Simba25-1) overlaid.
Individual galaxies as scatter + per-suite median with 16th/84th error bars.

Intended divergence signal
--------------------------
  SIMBA's kinetic AGN feedback evacuates CGM gas → lower Y_SZ at fixed M_star.
  TNG retains more CGM → higher Y_SZ.
  Galaxy sizes (r_half) encode the wind-reinjection geometry differences.
  The (Y_SZ, r_half) plane is a two-dimensional discriminator between models.

Usage (runs locally on downloaded parquets)
-------------------------------------------
    python fig_obs.py
    python fig_obs.py --tables_dir tables/obs --outdir figures

Input
-----
    tables/obs/{suite_key}_snap*_p*.parquet  (from obs_proxies.py)

Output
------
    figures/fig_obs_rh_ysz.{pdf,png}
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── style (matches compare.py / sbi_pilot.py) ─────────────────────────────────
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
COLORS  = {"TNG100-1": "#1f77b4", "Eagle100-1": "#d62728", "Simba25-1": "#2ca02c"}
LABELS  = {"TNG100-1": "TNG100",  "Eagle100-1": "EAGLE100", "Simba25-1": "SIMBA25"}
MARKERS = {"TNG100-1": "o",       "Eagle100-1": "s",        "Simba25-1": "^"}

_HERE   = pathlib.Path(__file__).resolve().parent
TABLES  = _HERE / "tables" / "obs"
FIGURES = _HERE / "figures"

MIN_N_GAS_Y = 10    # minimum non-SF gas particles inside R_200c for Y_SZ to be reliable


def load_suite(tables_dir, suite_key):
    """
    Load and clean the most recent parquet for a suite.

    Keeps only rows with:
      - positive Y_SZ and r_half
      - n_gas_Y >= MIN_N_GAS_Y  (enough particles for reliable Y_SZ)
      - logM_star in the extraction range [10.5, 11.0]
    """
    files = sorted(pathlib.Path(tables_dir).glob(f"{suite_key}_snap*.parquet"))
    if not files:
        print(f"  [WARN] no parquet found for {suite_key} in {tables_dir}")
        return None

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    ok = (
        df["Y_SZ_Mpc2"].gt(0) &
        df["r_half_kpc"].gt(0) &
        df["n_gas_Y"].ge(MIN_N_GAS_Y) &
        df["logM_star"].between(10.5, 11.0) &
        df[["Y_SZ_Mpc2", "r_half_kpc", "logM_star"]].notna().all(axis=1)
    )
    cleaned = df[ok].copy()
    print(f"  {suite_key}: {len(cleaned):,} galaxies  (raw {len(df):,})")
    return cleaned


def make_figure(tables_dir, outdir):
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    annotations = []

    for suite in SUITES:
        df = load_suite(tables_dir, suite)
        if df is None or len(df) == 0:
            continue

        z_val = float(df["z_actual"].median()) if "z_actual" in df.columns else float("nan")
        col   = COLORS[suite]
        mk    = MARKERS[suite]

        Y  = df["Y_SZ_Mpc2"].values
        rh = df["r_half_kpc"].values

        # individual galaxies (rasterized; fine for dense suites)
        ax.scatter(Y, rh,
                   c=col, s=10, alpha=0.25, linewidths=0,
                   rasterized=True, zorder=2)

        # per-suite median + 16th/84th percentile bars
        Y_med,  Y_lo,  Y_hi  = np.nanpercentile(Y,  [50, 16, 84])
        rh_med, rh_lo, rh_hi = np.nanpercentile(rh, [50, 16, 84])

        ax.errorbar(
            Y_med, rh_med,
            xerr=[[Y_med - Y_lo], [Y_hi - Y_med]],
            yerr=[[rh_med - rh_lo], [rh_hi - rh_med]],
            fmt=mk, color=col, ms=10, markeredgecolor="k", markeredgewidth=0.6,
            elinewidth=1.8, capsize=4, capthick=1.4,
            zorder=5, label=LABELS[suite],
        )

        annotations.append(f"{LABELS[suite]}: z={z_val:.2f}, N={len(df)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Integrated Compton-$Y \cdot D_A^2$ within $R_{200c}$  [Mpc$^2$]",
        fontsize=12)
    ax.set_ylabel(
        r"3D stellar half-mass radius  $r_{1/2}$  [kpc]",
        fontsize=12)
    ax.set_title(
        r"$\log(M_\star/M_\odot) \in [10.5,\,11.0]$,  $z \approx 1$",
        fontsize=11, pad=6)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    # per-suite metadata annotation (bottom-right)
    for i, txt in enumerate(annotations):
        ax.annotate(
            txt,
            xy=(0.97, 0.05 + 0.055 * (len(annotations) - 1 - i)),
            xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
        )

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_obs_rh_ysz.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fig: r_half vs Y_SZ at fixed M_star, z~1 (three suites)")
    parser.add_argument("--tables_dir", default=str(TABLES),
                        help="Directory containing obs_proxies parquets")
    parser.add_argument("--outdir",     default=str(FIGURES))
    args = parser.parse_args()

    print(f"Loading from: {args.tables_dir}")
    make_figure(args.tables_dir, args.outdir)


if __name__ == "__main__":
    main()
