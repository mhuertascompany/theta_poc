"""
fig_p2.py — Figure P2: descriptor-space transferability test.

Scatter of all 1P sims (TNG + SIMBA, fixed cosmology Omega0=0.3, sigma8=0.8)
in the (eta_M, f_hot) descriptor plane, colored by catalog observables.

Scientific question: at matched descriptor values, do TNG and SIMBA produce
similar catalog observables (SMF, f_gas, sSFR)?
  - YES → descriptors span a common physical space (supports MANIFOLD)
  - NO  → current 2-descriptor set is insufficient; identifies the gap

Both suites use identical protocol and fixed cosmology.
x-axis / y-axis are MEASURED descriptors, not input parameters.
Colors are catalog observables from groups_090.hdf5 only.

Usage:
    python fig_p2.py
    python fig_p2.py --outdir figures
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config as C

_HERE       = pathlib.Path(__file__).resolve().parent
FIGURES     = _HERE / "figures"
FIGURES.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":       True,
    "ytick.right":     True,
    "figure.dpi":      150,
})

SUITE_STYLES = {
    "camels_tng_1p":   dict(marker="o", ms=60,  label="IllustrisTNG", zorder=3),
    "camels_simba_1p": dict(marker="s", ms=60,  label="SIMBA",        zorder=4),
}

PARAM_LABELS = {
    "A_SN1":  "A_SN1",
    "A_AGN1": "A_AGN1",
    "A_SN2":  "A_SN2",
    "A_AGN2": "A_AGN2",
}

# catalog observables to color by: (column, label, colormap)
COLOR_VARS = [
    ("smf_2",   r"$\log_{10}(N_{M_* \in [9,9.5]}\!+\!1)$",    "viridis"),
    ("fgas_1",  r"Median $f_{\rm gas}$  ($\log M_*\!\in\![9.5,10]$)", "plasma"),
    ("ssfr_1",  r"Median $\log\,{\rm sSFR}$  ($\log M_*\!\in\![9.5,10]$)", "RdYlBu_r"),
]


def load_data(parquet_root):
    dfs = []
    for suite_key in SUITE_STYLES:
        p = pathlib.Path(parquet_root) / suite_key / "catalog_x_1p.parquet"
        if not p.exists():
            print(f"  [WARN] {p} not found — skipping {suite_key}")
            print(f"         Run on Binder: python extract_catalog_1p.py {suite_key}")
            continue
        df = pd.read_parquet(p)
        df["suite_key"] = suite_key
        dfs.append(df)
        print(f"  {suite_key}: {len(df)} sims  "
              f"eta_M=[{df.eta_M_median.min():.1f},{df.eta_M_median.max():.1f}]  "
              f"f_hot=[{df.f_hot_median.min():.2f},{df.f_hot_median.max():.2f}]")
    if not dfs:
        raise FileNotFoundError(
            "No catalog_x_1p.parquet found. "
            "Run extract_catalog_1p.py on the Binder first.")
    return pd.concat(dfs, ignore_index=True)


def make_figure(df, outdir, proto_hash=""):
    n_color = len(COLOR_VARS)
    fig, axes = plt.subplots(1, n_color, figsize=(4.2 * n_color, 4.5),
                             constrained_layout=True)
    if n_color == 1:
        axes = [axes]

    for ax, (col, clabel, cmap) in zip(axes, COLOR_VARS):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col}\nnot in data",
                    transform=ax.transAxes, ha="center", va="center", color="0.6")
            continue

        # shared color range across both suites
        valid = df[col].dropna()
        vmin, vmax = float(valid.quantile(0.05)), float(valid.quantile(0.95))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for suite_key, sty in SUITE_STYLES.items():
            sub = df[(df["suite_key"] == suite_key) & df[col].notna() &
                     df["eta_M_median"].notna() & df["f_hot_median"].notna()]
            if sub.empty:
                continue

            sc = ax.scatter(
                sub["eta_M_median"], sub["f_hot_median"],
                c=sub[col], cmap=cmap, norm=norm,
                s=sty["ms"], marker=sty["marker"],
                edgecolors="k", linewidths=0.6,
                zorder=sty["zorder"],
                label=sty["label"],
            )

        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, shrink=0.85, pad=0.02,
        )
        cb.set_label(clabel, fontsize=8)

        ax.set_xlabel(r"$\eta_M$   (hot mass-loading, pivot median)", fontsize=10)
        ax.set_ylabel(r"$f_{\rm hot}$   (hot gas fraction, pivot median)", fontsize=10)
        ax.set_title(
            "Fixed cosmology  ($\\Omega_0=0.3,\\,\\sigma_8=0.8$)\n"
            "TNG-1P ●  SIMBA-1P ■   color = catalog observable",
            fontsize=8, pad=4,
        )

        # legend (suite markers only, first panel)
        if ax is axes[0]:
            handles = [
                mpl.lines.Line2D([0], [0], marker=sty["marker"], color="w",
                                 markerfacecolor="0.5", markeredgecolor="k",
                                 markersize=8, label=sty["label"])
                for suite_key, sty in SUITE_STYLES.items()
            ]
            ax.legend(handles=handles, fontsize=8, frameon=True, loc="best")

        ax.grid(alpha=0.2, linewidth=0.5)

    # ── annotation: what agreement/disagreement means ─────────────────────────
    fig.suptitle(
        "Descriptor-space transferability:  same $(\\eta_M,\\,f_{\\rm hot})$ "
        "$\\rightarrow$ same color?",
        fontsize=9, y=1.02,
    )

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"figure_p2_p{proto_hash}" if proto_hash else "figure_p2"
    for ext in ("pdf", "png"):
        out = outdir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Figure P2: descriptor transferability")
    parser.add_argument("--parquet_root", default=str(_HERE / "outputs" / "camels"))
    parser.add_argument("--outdir",       default=str(FIGURES))
    args = parser.parse_args()

    print("Loading 1P catalog parquets (fixed cosmology) ...")
    df = load_data(args.parquet_root)

    proto_hash = ""
    if "protocol_hash" in df.columns:
        h = df["protocol_hash"].dropna().unique()
        if len(h) == 1:
            proto_hash = str(h[0])

    print(f"\nTotal sims: {len(df)}  ({df.suite_key.value_counts().to_dict()})")
    print("Building Figure P2 ...")
    make_figure(df, outdir=args.outdir, proto_hash=proto_hash)
    print("Done.")


if __name__ == "__main__":
    main()
