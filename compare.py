"""
compare.py — Figure 1: θ_feedback descriptor loci vs halo/BH mass.

Three suites overlaid (TNG100-1, Eagle100-1, Simba25-1), z=0.
Binned medians + 16th/84th percentile shading.

Usage:
    python compare.py
    python compare.py --hash 6121e6cc   # override protocol hash
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as C

# ── paths ─────────────────────────────────────────────────────────────────────
TABLES  = pathlib.Path(__file__).parent / "tables"
FIGURES = pathlib.Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

SUITES = ["TNG100-1", "Eagle100-1", "Simba25-1"]

COLORS = {"TNG100-1": "#1f77b4", "Eagle100-1": "#d62728", "Simba25-1": "#2ca02c"}
LS     = {"TNG100-1": "-",       "Eagle100-1": "--",       "Simba25-1": "-."}
LABELS = {"TNG100-1": "TNG100-1","Eagle100-1": "EAGLE100", "Simba25-1": "SIMBA25"}

LAM_THR_MODE = 1e-3   # Eddington filter for mode_balance panel
MIN_N        = 10     # minimum halos per bin to show

# ── style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "figure.dpi":       150,
})


# ── binning utilities ─────────────────────────────────────────────────────────

def bin_stats(x, y, bins, min_n=MIN_N):
    """Binned median + 16th/84th percentile. NaN where N < min_n."""
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    med  = np.full(len(ctrs), np.nan)
    lo   = np.full(len(ctrs), np.nan)
    hi   = np.full(len(ctrs), np.nan)
    nn   = np.zeros(len(ctrs), dtype=int)
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        nn[i] = m.sum()
        if nn[i] >= min_n:
            v = y[m]
            med[i], lo[i], hi[i] = (np.nanmedian(v),
                                     np.nanpercentile(v, 16),
                                     np.nanpercentile(v, 84))
    return ctrs, med, lo, hi, nn


def bin_active_fraction(x, lam_max, bins, lam_thr, min_n=MIN_N):
    """Fraction of halos with lambda_edd_max > lam_thr per bin."""
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    frac = np.full(len(ctrs), np.nan)
    nn   = np.zeros(len(ctrs), dtype=int)
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (x >= b0) & (x < b1) & np.isfinite(x)
        nn[i] = m.sum()
        if nn[i] >= min_n:
            frac[i] = (lam_max[m] > lam_thr).mean()
    return ctrs, frac, nn


def draw(ax, ctrs, med, lo, hi, suite):
    ok = np.isfinite(med)
    if ok.sum() == 0:
        return
    ax.plot(ctrs[ok], med[ok], color=COLORS[suite], ls=LS[suite],
            lw=1.8, label=LABELS[suite])
    ax.fill_between(ctrs[ok], lo[ok], hi[ok],
                    color=COLORS[suite], alpha=0.15)


# ── main ──────────────────────────────────────────────────────────────────────

def make_figure1(hash_):
    # load parquets
    dfs = {}
    for suite in SUITES:
        p = TABLES / f"{suite}_z0.0_p{hash_}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        dfs[suite] = pd.read_parquet(p)
        print(f"  {suite}: {len(dfs[suite]):,} rows")

    MBINS  = C.HALO_BINS   # logM200c
    BHBINS = C.BH_BINS     # logM_BH

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.40, wspace=0.32)

    # ── 1. f_hot ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for s in SUITES:
        df = dfs[s]
        c, med, lo, hi, _ = bin_stats(df["logM200c"].values,
                                       df["f_hot"].values, MBINS)
        draw(ax, c, med, lo, hi, s)
    ax.set_ylabel(r"$f_{\rm hot}$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.set_title("Hot gas fraction")

    # ── 2. η_M ────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for s in SUITES:
        df = dfs[s]
        y  = df["eta_M"].values.copy()
        y[y <= 0] = np.nan
        c, med, lo, hi, _ = bin_stats(df["logM200c"].values, y, MBINS)
        draw(ax, c, med, lo, hi, s)
    ax.set_ylabel(r"$\eta_M = \dot{M}_{\rm hot,out}/\dot{M}_\star$")
    ax.set_yscale("log")
    ax.set_title(r"Hot-phase mass loading $\eta_M$  [lead panel]")

    # ── 3. f_duty (population duty cycle vs M_BH) ─────────────────────────────
    # The mode transition tracks M_BH, not halo mass — use logM_BH per SPEC.
    # Local bins start at 6.0 (TNG seeds at ~10^5.5, grown to ~10^6 by z=0).
    ax = axes[0, 2]
    BHBINS_LOCAL = np.arange(6.0, 9.51, 0.25)
    for s in SUITES:
        df  = dfs[s]
        lam = df["lambda_edd_max"].fillna(0).values
        c, frac, _ = bin_active_fraction(df["logM_BH"].values, lam,
                                          BHBINS_LOCAL,
                                          C.PROTOCOL["lambda_edd_thr"])
        ok = np.isfinite(frac)
        if ok.sum() > 0:
            ax.plot(c[ok], frac[ok], color=COLORS[s], ls=LS[s],
                    lw=1.8, label=LABELS[s])
    ax.set_xlabel(r"$\log M_{\rm BH}\ [M_\odot]$")
    ax.set_ylabel(r"$f_{\rm duty}\ (\lambda > \lambda_{\rm thr})$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.set_title(r"AGN duty cycle  ($\lambda_{\rm thr}=" +
                 f"{C.PROTOCOL['lambda_edd_thr']:.0e}$)")

    # ── 4. mode_balance (Eddington-filtered) ──────────────────────────────────
    ax = axes[1, 0]
    for s in SUITES:
        df   = dfs[s]
        lam  = df["lambda_edd_max"].fillna(0).values
        y    = df["mode_balance"].values.copy()
        filt = (lam >= LAM_THR_MODE) & (y > 0)
        c, med, lo, hi, nn = bin_stats(df["logM200c"].values[filt],
                                        y[filt], MBINS)
        draw(ax, c, med, lo, hi, s)
        print(f"  mode_balance {s}: {filt.sum()} halos pass λ > {LAM_THR_MODE:.0e}")
    ax.set_ylabel(r"$P_{\rm kin}/L_{\rm rad}$  (proxy)")
    ax.set_yscale("log")
    ax.set_title(r"Mode-balance proxy  ($\lambda_{\rm Edd}>10^{-3}$)")
    ax.text(0.97, 0.05,
            "EAGLE RM≡0 ground-truth\nzero anchor → Fig. 2",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="lightgray", alpha=0.85))

    # ── 5. ε_ff ────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    for s in SUITES:
        df = dfs[s]
        c, med, lo, hi, _ = bin_stats(df["logM200c"].values,
                                       df["eps_ff"].values, MBINS)
        draw(ax, c, med, lo, hi, s)
    ax.set_ylabel(r"$\varepsilon_{\rm ff}$")
    ax.set_title(r"SF efficiency per free-fall time $\varepsilon_{\rm ff}$")
    ax.text(0.97, 0.97,
            "TNG/EAGLE: recovers KS input\nSIMBA: H$_2$-based SF law",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="lightgray", alpha=0.85))

    # ── 6. p★/m★ ───────────────────────────────────────────────────────────────
    ax = axes[1, 2]
    for s in SUITES:
        df = dfs[s]
        y  = df["p_star"].values.copy()
        y[y <= 0] = np.nan
        c, med, lo, hi, _ = bin_stats(df["logM200c"].values, y, MBINS)
        draw(ax, c, med, lo, hi, s)
    ax.set_ylabel(r"$p_\star/m_\star\ [\rm km\,s^{-1}]$")
    ax.set_yscale("log")
    ax.set_title(r"Momentum per stellar mass $p_\star/m_\star$")
    ax.text(0.97, 0.97,
            "AGN-driven shell contamination\nat high $M_{200c}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="lightgray", alpha=0.85))

    # ── shared x-label and axes tweaks ────────────────────────────────────────
    for i, ax in enumerate(axes.flat):
        if ax.get_xlabel() == "":
            ax.set_xlabel(r"$\log M_{200c}\ [M_\odot]$")
        # panel 3 (f_duty) has logM_BH x-axis — set its own limits
        if i == 2:
            ax.set_xlim(6.0, 9.6)
        else:
            ax.set_xlim(11.4, 14.1)

    fig.suptitle(
        r"$\theta_{\rm feedback}$ descriptors — "
        r"TNG100-1 · EAGLE100 · SIMBA25  ($z = 0$)",
        fontsize=13, y=1.002,
    )

    out_pdf = FIGURES / f"figure1_descriptors_p{hash_}.pdf"
    out_png = FIGURES / f"figure1_descriptors_p{hash_}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", default="6121e6cc",
                        help="Protocol hash from run_suite (default: 6121e6cc)")
    args = parser.parse_args()
    make_figure1(args.hash)


if __name__ == "__main__":
    main()
