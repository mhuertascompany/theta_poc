"""
fig_obs.py — Figure: 3D half-mass radius vs integrated Compton-Y at fixed M_star, z~1.

Three suites (TNG100-1, Eagle100-1, Simba25-1) overlaid.
Per-suite median with bootstrap 68% confidence intervals (2000 resamples).
No individual scatter — the medians + CIs are the intended comparison.

Intended divergence signal
--------------------------
  SIMBA's kinetic AGN feedback evacuates CGM gas → lower Y_SZ at fixed M_star.
  TNG retains more CGM → higher Y_SZ.
  Galaxy sizes (r_half) encode wind-reinjection geometry differences.
  The (Y_SZ, r_half) plane is a two-dimensional discriminator between models.

Usage (runs locally on downloaded parquets)
-------------------------------------------
    python fig_obs.py
    python fig_obs.py --tables_dir tables/obs --outdir figures
    python fig_obs.py --n_boot 5000 --seed 7

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

_HERE       = pathlib.Path(__file__).resolve().parent
TABLES      = _HERE / "tables" / "obs"
FIGURES     = _HERE / "figures"
MIN_N_GAS_Y = 10


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_median_ci(x, y, n_boot=2000, ci=68, rng=None):
    """
    Bootstrap 68% CI on the median of x and y independently.

    Returns (x_med, x_lo, x_hi, y_med, y_lo, y_hi) where lo/hi are the
    (50-ci/2)th and (50+ci/2)th percentiles of the bootstrap distribution
    of medians.  Asymmetric because we work in linear space (log axes handled
    by matplotlib).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    boot_x = np.empty(n_boot)
    boot_y = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_x[i] = np.median(x[idx])
        boot_y[i] = np.median(y[idx])
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p
    x_lo, x_hi = np.percentile(boot_x, [lo_p, hi_p])
    y_lo, y_hi = np.percentile(boot_y, [lo_p, hi_p])
    return np.median(x), x_lo, x_hi, np.median(y), y_lo, y_hi


# ── data loading ──────────────────────────────────────────────────────────────

def load_suite(tables_dir, suite_key):
    """
    Load and clean parquets for one suite.

    Uses the snap-numbered parquet with the highest snap number if multiple
    exist (i.e. a corrected re-run replaces an earlier wrong-snap file, as
    long as the wrong-snap parquet has been deleted).
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


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(tables_dir, outdir, n_boot=2000, seed=42):
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    annotations = []

    for suite in SUITES:
        df = load_suite(tables_dir, suite)
        if df is None or len(df) == 0:
            continue

        z_val = float(df["z_actual"].median()) if "z_actual" in df.columns else float("nan")
        col   = COLORS[suite]
        mk    = MARKERS[suite]
        N     = len(df)

        Y  = df["Y_SZ_Mpc2"].values
        rh = df["r_half_kpc"].values

        Y_med, Y_lo, Y_hi, rh_med, rh_lo, rh_hi = bootstrap_median_ci(
            Y, rh, n_boot=n_boot, rng=rng)

        ax.errorbar(
            Y_med, rh_med,
            xerr=[[Y_med - Y_lo], [Y_hi - Y_med]],
            yerr=[[rh_med - rh_lo], [rh_hi - rh_med]],
            fmt=mk, color=col,
            ms=11, markeredgecolor="k", markeredgewidth=0.7,
            elinewidth=2.0, capsize=5, capthick=1.6,
            zorder=5, label=LABELS[suite],
        )

        annotations.append(f"{LABELS[suite]}: z={z_val:.2f},  N={N}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Integrated Compton-$Y \cdot D_A^2$ within $R_{200c}$  [Mpc$^2$]",
        fontsize=12)
    ax.set_ylabel(
        r"3D stellar half-mass radius  $r_{1/2}$  [kpc]",
        fontsize=12)
    ax.set_title(
        r"$\log(M_\star/M_\odot) \in [10.5,\,11.0]$,  $z \approx 1$"
        "\nMedians ± 68% bootstrap CI",
        fontsize=10, pad=6)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

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
        description="Fig: r_half vs Y_SZ at fixed M_star, z~1 (bootstrap medians)")
    parser.add_argument("--tables_dir", default=str(TABLES))
    parser.add_argument("--outdir",     default=str(FIGURES))
    parser.add_argument("--n_boot",     type=int, default=2000,
                        help="Bootstrap resamples (default 2000)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print(f"Loading from: {args.tables_dir}")
    make_figure(args.tables_dir, args.outdir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
