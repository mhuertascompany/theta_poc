"""
fig_erc.py — three-panel ERC B1 overview figure.

  a · models agree      stellar mass function at z=0
  b · feedback differs  f_hot (hot-gas fraction descriptor) vs halo mass
  c · data can separate galaxy size vs metallicity, coloured by Y_SZ

Panel b requires run_suite.py output from the cluster:
    tables/{suite_key}_z0.0_p*.parquet   (columns: logM200c, f_hot, ...)
A "data pending" placeholder is shown if those files are absent.

Runs locally on downloaded parquets — no cluster access needed.

Usage
-----
    python fig_erc.py
    python fig_erc.py --tables_dir tables --outdir figures
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# ── palette (consistent across all panels) ────────────────────────────────────
SUITES = ["TNG100-1", "Eagle100-1", "Simba100-1"]
COLORS = {"TNG100-1": "#1f77b4", "Eagle100-1": "#d62728", "Simba100-1": "#2ca02c"}
LABELS = {"TNG100-1": "TNG100",  "Eagle100-1": "EAGLE100", "Simba100-1": "SIMBA100"}
LS     = {"TNG100-1": "-",       "Eagle100-1": "--",        "Simba100-1": "-."}
MARK   = {"TNG100-1": "o",       "Eagle100-1": "s",         "Simba100-1": "^"}

Z_SUN       = 0.0127
MIN_N_GAS_Y = 10

mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def boot_med(arr, n_boot=1000, rng=None):
    """Bootstrap 68% CI on the median. Returns (median, lo, hi)."""
    rng = rng or np.random.default_rng(42)
    n   = len(arr)
    b   = np.array([np.median(arr[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(np.median(arr)), *np.percentile(b, [16, 84])


def load_obs(tables_dir, suite_key):
    files = sorted(pathlib.Path(tables_dir, "obs").glob(f"{suite_key}_snap*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ok = (
        df["Y_SZ_Mpc2"].gt(0) & df["r_half_kpc"].gt(0) &
        df["n_gas_Y"].ge(MIN_N_GAS_Y) & df["logM200c"].notna()
    )
    return df[ok].copy()


# ── observed SMF: Baldry+2012 double Schechter (Chabrier IMF, z≈0) ───────────

def _baldry2012(logm):
    logMs, phi1, a1, phi2, a2 = 10.66, 3.96e-3, -0.35, 0.79e-3, -1.47
    x = 10 ** (logm - logMs)
    return np.log(10) * np.exp(-x) * (phi1 * x**(a1 + 1) + phi2 * x**(a2 + 1))


# ── panel a: stellar mass function ────────────────────────────────────────────

def panel_smf(ax, tables_dir):
    # observations
    logm_obs = np.linspace(8.8, 11.9, 80)
    ax.plot(logm_obs, _baldry2012(logm_obs),
            color="0.3", lw=1.2, zorder=1)
    logm_pts = np.array([9.1, 9.6, 10.1, 10.6, 10.95, 11.2])
    ax.scatter(logm_pts, _baldry2012(logm_pts),
               color="0.2", s=30, zorder=4, label="Baldry+12 (obs)")

    # simulations
    for suite in SUITES:
        f = pathlib.Path(tables_dir, "smf", f"{suite}_z0.parquet")
        if not f.exists():
            continue
        df  = pd.read_parquet(f)
        ok  = df["N"] >= 3
        x   = df.loc[ok, "logM_cen"].values
        phi = df.loc[ok, "phi"].values
        err = df.loc[ok, "phi_err"].values
        ax.plot(x, phi, color=COLORS[suite], ls=LS[suite], lw=2.0,
                label=LABELS[suite], zorder=3)
        ax.fill_between(x, phi - err, phi + err,
                        color=COLORS[suite], alpha=0.13, zorder=2)

    # shade the mass range used in panels b & c
    ax.axvspan(10.5, 11.0, color="0.88", zorder=0,
               label=r"target mass range")

    ax.set_yscale("log")
    ax.set_xlim(9.0, 12.2)
    ax.set_ylim(5e-7, 0.05)
    ax.set_xlabel(r"$\log(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$")
    ax.legend(frameon=False, loc="lower left", fontsize=8)


# ── panel b: f_hot descriptor vs halo mass ───────────────────────────────────
# Loads run_suite.py output: tables/{suite}_z*_p*.parquet
# Shows "data pending" placeholder if files are absent.

def panel_fhot(ax, tables_dir, rng):
    any_found = False

    for suite in SUITES:
        files = sorted(pathlib.Path(tables_dir).glob(f"{suite}_z*_p*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "f_hot" not in df.columns or "logM200c" not in df.columns:
            continue
        ok = df["f_hot"].notna() & df["logM200c"].notna() & df["f_hot"].gt(0)
        df = df[ok]
        if len(df) < 10:
            continue

        lm = df["logM200c"].values
        fh = df["f_hot"].values

        lo, hi = np.percentile(lm, 5), np.percentile(lm, 95)
        edges = np.linspace(lo, hi, 8)   # 7 bins across the halo mass range
        cen, meds, ci_lo, ci_hi = [], [], [], []
        for i in range(len(edges) - 1):
            mask = (lm >= edges[i]) & (lm < edges[i + 1])
            if mask.sum() < 5:
                continue
            med, l, h = boot_med(fh[mask], rng=rng)
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(med); ci_lo.append(l); ci_hi.append(h)
        if not cen:
            continue
        cen, meds, ci_lo, ci_hi = map(np.array, (cen, meds, ci_lo, ci_hi))

        ax.plot(cen, meds, color=COLORS[suite], ls=LS[suite], lw=2.0,
                label=LABELS[suite], zorder=3)
        ax.fill_between(cen, ci_lo, ci_hi,
                        color=COLORS[suite], alpha=0.18, zorder=2)
        any_found = True

    if any_found:
        ax.axvline(12.2, color="0.5", ls=":", lw=1.0, zorder=1)
        ylo, yhi = ax.get_ylim()
        ax.text(12.25, ylo + 0.02 * (yhi - ylo), "MW",
                fontsize=8, color="0.5", va="bottom")
        ax.legend(frameon=False, loc="upper left")
    else:
        ax.set_xlim(11.5, 13.5)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5,
                "data pending\n(download run_suite parquets\nfrom cluster)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.55", style="italic")
        ax.axvline(12.2, color="0.5", ls=":", lw=1.0, zorder=1)
        ax.text(12.25, 0.03, "MW", fontsize=8, color="0.5", va="bottom")

    ax.set_xlabel(r"$\log(M_{200c} / M_\odot)$")
    ax.set_ylabel(r"$f_\mathrm{hot} = M_\mathrm{gas,hot} / M_\mathrm{gas}$")


# ── panel c: size vs metallicity, colour = Y_SZ ───────────────────────────────

def panel_scatter(ax, tables_dir, rng):
    results = {}
    for suite in SUITES:
        df = load_obs(tables_dir, suite)
        if df is None:
            continue
        ok = df["logM_star"].between(10.5, 11.0) & df["Z_star"].gt(0)
        df = df[ok]
        if len(df) < 5:
            continue
        results[suite] = dict(
            rh    = df["r_half_kpc"].values,
            Z_log = np.log10(df["Z_star"].values / Z_SUN),
            Y     = df["Y_SZ_Mpc2"].values,
            z_val = float(df["z_actual"].iloc[0]),
            N     = len(df),
        )

    if not results:
        return

    Y_meds = [np.median(v["Y"]) for v in results.values()]
    # TwoSlopeNorm: centres on the middle value so the 3 points map to
    # blue / white / red — maximally distinct even though the range is narrow.
    Y_sorted = sorted(Y_meds)
    norm = TwoSlopeNorm(vmin=Y_sorted[0], vcenter=Y_sorted[1], vmax=Y_sorted[2])
    cmap = plt.cm.RdBu_r   # blue=low Y (less hot gas), red=high Y (more hot gas)

    for suite, v in results.items():
        rh_med, rh_lo, rh_hi = boot_med(v["rh"],    rng=rng)
        Z_med,  Z_lo,  Z_hi  = boot_med(v["Z_log"], rng=rng)
        Y_med = float(np.median(v["Y"]))
        col   = COLORS[suite]

        ax.errorbar(
            Z_med, rh_med,
            xerr=[[Z_med - Z_lo], [Z_hi - Z_med]],
            yerr=[[rh_med - rh_lo], [rh_hi - rh_med]],
            fmt=MARK[suite], color=cmap(norm(Y_med)),
            ms=14, markeredgecolor=col, markeredgewidth=2.0,
            ecolor=col, elinewidth=2.0, capsize=5, capthick=1.5,
            zorder=5,
        )
        # dummy handle for legend
        ax.plot([], [], MARK[suite], color=cmap(norm(Y_med)),
                markeredgecolor=col, markeredgewidth=1.4,
                ms=9, label=LABELS[suite])

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log(Z_\star / Z_\odot)$")
    ax.set_ylabel(r"$r_{1/2}\ [\mathrm{kpc}]$")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    # slim colorbar for Y_SZ
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$Y \cdot D_A^2\ [\mathrm{Mpc}^2]$", fontsize=9)


# ── assemble ──────────────────────────────────────────────────────────────────

def make_figure(tables_dir, outdir):
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.subplots_adjust(wspace=0.42)

    panel_titles = [
        r"$\mathbf{a}$  ·  models agree",
        r"$\mathbf{b}$  ·  feedback differs",
        r"$\mathbf{c}$  ·  data can separate",
    ]
    subtitles = [
        "galaxies per volume  (z = 0)",
        r"hot-gas fraction $f_\mathrm{hot}$ vs halo mass",
        r"size  ×  metallicity  ×  hot-gas signal  (z ~ 1)",
    ]
    for ax, pt, st in zip(axes, panel_titles, subtitles):
        ax.set_title(f"{pt}\n", loc="left", fontsize=11, pad=4)
        ax.text(0.02, 1.01, st, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="0.45",
                style="italic")

    panel_smf(axes[0], tables_dir)
    panel_fhot(axes[1], tables_dir, rng)
    panel_scatter(axes[2], tables_dir, rng)

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_erc.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="ERC B1 three-panel overview figure")
    p.add_argument("--tables_dir", default="tables")
    p.add_argument("--outdir",     default="figures")
    args = p.parse_args()
    make_figure(args.tables_dir, args.outdir)


if __name__ == "__main__":
    main()
