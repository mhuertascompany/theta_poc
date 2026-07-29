"""
fig_obs_Z.py — Stellar metallicity vs size, colour-coded by Compton-Y, z~1.

Four suites (TNG100-1, Eagle100-1, Simba100-1, Simba25-1), one point each:
  x  = median log(Z_star / Z_sun)  ± bootstrap 68% CI
  y  = median r_half [kpc]         ± bootstrap 68% CI
  colour = median Y_SZ [Mpc²]      (shared log colorbar)

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

SUITES          = ["TNG100-1", "Eagle100-1", "Simba100-1"]
SUITES_OPTIONAL = ["Simba25-1"]   # included only if --include_simba25 is passed
COLORS  = {"TNG100-1": "#1f77b4", "Eagle100-1": "#d62728",
           "Simba100-1": "#2ca02c", "Simba25-1": "#74c476"}
LABELS  = {"TNG100-1": "TNG100",  "Eagle100-1": "EAGLE100",
           "Simba100-1": "SIMBA100", "Simba25-1": "SIMBA25 (res. check)"}
MARKERS = {"TNG100-1": "o",       "Eagle100-1": "s",
           "Simba100-1": "^",     "Simba25-1": "v"}

_HERE       = pathlib.Path(__file__).resolve().parent
TABLES      = _HERE / "tables" / "obs"
FIGURES     = _HERE / "figures"
MIN_N_GAS_Y = 10
Z_SUN       = 0.0127    # Asplund et al. (2009)

# Which metallicity column to plot on x-axis.
# "Z_gas_sfr" = SF-gas only (mass-weighted) — closest to nebular emission-line obs.
# "Z_gas"     = all-gas (mass-weighted) — fallback if SF-gas field absent.
# "Z_star"    = stellar metallicity.
Z_COL       = "Z_gas_sfr"
Z_LABEL     = r"$\log(Z_\mathrm{gas,SF} / Z_\odot)$"
Z_COL_FALLBACK = "Z_gas"


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(arr, n_boot=2000, ci=68, rng=None):
    """Bootstrap CI on the median. Returns (median, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(arr)
    boot = np.array([np.median(arr[rng.integers(0, n, size=n)])
                     for _ in range(n_boot)])
    lo_p, hi_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo, hi = np.percentile(boot, [lo_p, hi_p])
    return np.median(arr), lo, hi


# ── data loading ──────────────────────────────────────────────────────────────

def load_suite(tables_dir, suite_key):
    files = sorted(pathlib.Path(tables_dir).glob(f"{suite_key}_snap*.parquet"))
    if not files:
        print(f"  [WARN] no parquet for {suite_key}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "Z_star" not in df.columns:
        print(f"  [WARN] {suite_key}: Z_star column missing — rerun obs_proxies.py")
        return None
    # choose metallicity column: prefer Z_gas_sfr, fall back to Z_gas
    if Z_COL in df.columns and df[Z_COL].gt(0).sum() > 0:
        z_col = Z_COL
    else:
        print(f"  [INFO] {suite_key}: {Z_COL} absent/zero — using {Z_COL_FALLBACK}")
        z_col = Z_COL_FALLBACK
    if z_col not in df.columns:
        print(f"  [WARN] {suite_key}: no gas metallicity column — rerun obs_proxies.py")
        return None

    ok = (
        df["Y_SZ_Mpc2"].gt(0) &
        df["r_half_kpc"].gt(0) &
        df[z_col].gt(0) &
        df["n_gas_Y"].ge(MIN_N_GAS_Y) &
        df["logM_star"].between(10.5, 11.0) &
        df[["Y_SZ_Mpc2", "r_half_kpc", z_col, "logM_star"]].notna().all(axis=1)
    )
    cleaned = df[ok].copy()
    cleaned["_Z_col"] = z_col   # carry which column was actually used
    print(f"  {suite_key}: {len(cleaned):,} galaxies  [Z col: {z_col}]")
    return cleaned


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(tables_dir, outdir, n_boot=2000, seed=42, include_simba25=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    suites = SUITES + (SUITES_OPTIONAL if include_simba25 else [])

    # ── collect medians first to set colorbar range ───────────────────────────
    results = {}
    for suite in suites:
        df = load_suite(tables_dir, suite)
        if df is None or len(df) == 0:
            continue
        z_col = df["_Z_col"].iloc[0]
        Z_log = np.log10(df[z_col].values / Z_SUN)
        rh    = df["r_half_kpc"].values
        Y     = df["Y_SZ_Mpc2"].values
        z_val = float(df["z_actual"].median()) if "z_actual" in df.columns else float("nan")
        results[suite] = dict(
            Z_log=Z_log, rh=rh, Y=Y, z_val=z_val, N=len(df), z_col=z_col)

    if not results:
        print("No data found.")
        return

    Y_medians = [np.median(v["Y"]) for v in results.values()]
    norm = LogNorm(vmin=min(Y_medians) * 0.5, vmax=max(Y_medians) * 2.0)
    cmap = plt.cm.plasma

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig.subplots_adjust(right=0.80)

    for suite in suites:
        if suite not in results:
            continue
        v   = results[suite]
        col = COLORS[suite]
        mk  = MARKERS[suite]

        Z_med, Z_lo, Z_hi   = bootstrap_ci(v["Z_log"], n_boot=n_boot, rng=rng)
        rh_med, rh_lo, rh_hi = bootstrap_ci(v["rh"],   n_boot=n_boot, rng=rng)
        Y_med = float(np.median(v["Y"]))

        marker_color = cmap(norm(Y_med))

        ax.errorbar(
            Z_med, rh_med,
            xerr=[[Z_med - Z_lo], [Z_hi - Z_med]],
            yerr=[[rh_med - rh_lo], [rh_hi - rh_med]],
            fmt=mk, color=marker_color,
            ms=11, markeredgecolor=col, markeredgewidth=1.5,
            elinewidth=2.0, capsize=5, capthick=1.6,
            ecolor=col, zorder=5,
            label=f"{LABELS[suite]}  (z={v['z_val']:.2f}, N={v['N']})",
        )

    # ── colorbar ──────────────────────────────────────────────────────────────
    cax = fig.add_axes([0.82, 0.15, 0.03, 0.70])
    cb  = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(
        r"median $Y \cdot D_A^2$  [Mpc$^2$]", fontsize=10)

    ax.set_xlabel(Z_LABEL, fontsize=12)
    ax.set_ylabel(r"3D stellar half-mass radius  $r_{1/2}$  [kpc]", fontsize=12)
    ax.set_title(
        r"$\log(M_\star/M_\odot) \in [10.5,\,11.0]$,  $z \approx 1$"
        "\nMedians ± 68% bootstrap CI",
        fontsize=10, pad=6)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_obs_Z_rh_ysz.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fig: Z_star vs r_half coloured by Y_SZ, bootstrap medians")
    parser.add_argument("--tables_dir", default=str(TABLES))
    parser.add_argument("--outdir",     default=str(FIGURES))
    parser.add_argument("--n_boot",         type=int,  default=2000)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--include_simba25", action="store_true",
                        help="Add Simba25-1 as a resolution-check point")
    args = parser.parse_args()
    make_figure(args.tables_dir, args.outdir, n_boot=args.n_boot,
                seed=args.seed, include_simba25=args.include_simba25)


if __name__ == "__main__":
    main()
