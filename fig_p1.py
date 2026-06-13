"""
fig_p1.py — Figure P1: feedback descriptor loci across IllustrisTNG and SIMBA.

Stage B script — runs locally on parquets downloaded from the Binder.

Figure layout: 2 rows (η_M, f_hot) × 4 columns (A_SN1, A_AGN1, A_SN2, A_AGN2).
Each panel shows the median descriptor in the pivot bin (logM200c = 12.0 ± 0.25)
vs the feedback input multiplier, with 16–84th percentile halo-to-halo bands.
TNG and SIMBA overlaid.  Same x-axis (nominal multiplier), different curves —
demonstrating input non-commensurability.

Usage:
    python fig_p1.py
    python fig_p1.py --parquet_root outputs/camels --outdir figures
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config as C

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE        = pathlib.Path(__file__).resolve().parent
PARQUET_ROOT = _HERE / "outputs" / "camels"
FIGURES      = _HERE / "figures"
FIGURES.mkdir(exist_ok=True)

# ── style (matches compare.py) ────────────────────────────────────────────────

mpl.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":    9,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "ytick.right":      True,
    "figure.dpi":       150,
})

# ── suite definitions ─────────────────────────────────────────────────────────

SUITES = {
    "camels_tng_1p":   dict(color="#2166ac", label="IllustrisTNG (AREPO)",
                            marker="o", ms=6, lw=1.5, zorder=3),
    "camels_simba_1p": dict(color="#d6604d", label="SIMBA (GIZMO)",
                            marker="s", ms=6, lw=1.5, zorder=2),
}

# ── parameter layout ──────────────────────────────────────────────────────────
# Each entry: (varied_param value in parquet, short header, per-suite physics note)
PARAMS = [
    ("A_SN1",
     "A_SN1  (p3)",
     "TNG: wind energy\nSIMBA: wind mass-loading"),
    ("A_AGN1",
     "A_AGN1  (p4)",
     "TNG: kinetic-mode BH energy\nSIMBA: QSO/jet momentum flux"),
    ("A_SN2",
     "A_SN2  (p5)",
     "TNG: wind speed\nSIMBA: wind speed"),
    ("A_AGN2",
     "A_AGN2  (p6)",
     "TNG: BH reorientation\nSIMBA: jet speed"),
]

# ── descriptor layout ─────────────────────────────────────────────────────────
# Each entry: (parquet column, y-axis label)
DESCRIPTORS = [
    ("eta_M", r"$\eta_M$   [hot mass-loading]"),
    ("f_hot", r"$f_{\rm hot}$   [hot gas fraction]"),
]

PIVOT_LO = C.PIVOT["logM200_lo"]
PIVOT_HI = C.PIVOT["logM200_hi"]

# Nominal multiplier grid (same for all parameters and both suites)
MULT_TICKS  = [0.5, 0.75, 1.0, 1.5, 2.0]
MULT_LABELS = ["×0.5", "×0.75", "×1", "×1.5", "×2"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_parquets(parquet_root):
    """Load all per-sim parquets into a single DataFrame."""
    parquet_root = pathlib.Path(parquet_root)
    dfs = []
    for suite_key in SUITES:
        d = parquet_root / suite_key
        if not d.exists():
            print(f"  [WARN] {d} not found — skipping {suite_key}")
            continue
        files = sorted(d.glob("*.parquet"))
        if not files:
            print(f"  [WARN] no parquets in {d}")
            continue
        suite_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        suite_df["suite_key"] = suite_key
        dfs.append(suite_df)
        n_piv = ((suite_df.logM200c >= PIVOT_LO) &
                 (suite_df.logM200c <  PIVOT_HI)).sum()
        print(f"  {suite_key}: {len(files)} sims, {len(suite_df):,} halos"
              f", {n_piv:,} in pivot bin")
    if not dfs:
        raise FileNotFoundError(f"No parquets found under {parquet_root}")
    return pd.concat(dfs, ignore_index=True)


# ── per-panel statistics ──────────────────────────────────────────────────────

def panel_stats(df, suite_key, param_col, desc_col):
    """
    Compute per-sim median + 16/84th percentile of desc_col in the pivot bin,
    for one suite and one feedback parameter sweep.

    x-axis: nominal multiplier = param_value / fiducial_param_value
            (fiducial = value in the _0 sim of this sweep)

    Returns DataFrame sorted by multiplier; columns:
        sim_id, param_value, multiplier, median, p16, p84, n_halos
    Returns None if data is missing or insufficient.
    """
    sel = (
        (df["suite_key"]    == suite_key) &
        (df["varied_param"] == param_col) &
        (df["logM200c"]     >= PIVOT_LO)  &
        (df["logM200c"]     <  PIVOT_HI)  &
        np.isfinite(df[desc_col])
    )
    sub = df[sel].copy()
    if sub.empty:
        return None

    # fiducial value: param_col value in the _0 sim (e.g. 1P_p3_0)
    fid_rows = sub[sub["sim_id"].str.endswith("_0")]
    if fid_rows.empty:
        return None
    fid_val = float(fid_rows[param_col].iloc[0])
    if fid_val == 0:
        return None

    rows = []
    for sim_id, grp in sub.groupby("sim_id"):
        vals = grp[desc_col].dropna().values
        if len(vals) < 3:
            continue
        pv = float(grp[param_col].iloc[0])
        rows.append(dict(
            sim_id      = sim_id,
            param_value = pv,
            multiplier  = pv / fid_val,
            median      = float(np.median(vals)),
            p16         = float(np.percentile(vals, 16)),
            p84         = float(np.percentile(vals, 84)),
            n_halos     = int(len(vals)),
        ))

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("multiplier").reset_index(drop=True)


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(df, outdir, proto_hash=""):
    n_rows = len(DESCRIPTORS)
    n_cols = len(PARAMS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.3 * n_cols, 3.0 * n_rows),
        sharex="all",
        sharey="row",
        constrained_layout=True,
    )
    # ensure 2-D indexing even for single row/col
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # ── column headers ────────────────────────────────────────────────────────
    for col_i, (param_col, col_title, subtitle) in enumerate(PARAMS):
        axes[0, col_i].set_title(
            f"$\\bf{{{col_title.replace(' ', r'\,')}}}$\n{subtitle}",
            fontsize=8.5, pad=5, linespacing=1.5,
        )

    # ── row y-labels ──────────────────────────────────────────────────────────
    for row_i, (desc_col, ylabel) in enumerate(DESCRIPTORS):
        axes[row_i, 0].set_ylabel(ylabel, fontsize=11)

    # ── fill panels ───────────────────────────────────────────────────────────
    for col_i, (param_col, *_) in enumerate(PARAMS):
        for row_i, (desc_col, _) in enumerate(DESCRIPTORS):
            ax = axes[row_i, col_i]

            any_data = False
            for suite_key, sty in SUITES.items():
                st = panel_stats(df, suite_key, param_col, desc_col)
                if st is None or st.empty:
                    continue
                any_data = True

                x  = st["multiplier"].values
                y  = st["median"].values
                lo = st["p16"].values
                hi = st["p84"].values

                ax.fill_between(x, lo, hi,
                                color=sty["color"], alpha=0.15,
                                zorder=sty["zorder"] - 1)
                ax.plot(x, y,
                        color=sty["color"],
                        marker=sty["marker"],
                        markersize=sty["ms"],
                        linewidth=sty["lw"],
                        label=sty["label"],
                        zorder=sty["zorder"])

            if not any_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="0.6", fontsize=9)

            # fiducial dotted line at multiplier = 1
            ax.axvline(1.0, color="0.45", linestyle=":", linewidth=1.1, zorder=1)

            ax.set_xscale("log")
            ax.set_xticks(MULT_TICKS)
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(MULT_LABELS))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # ── f_hot y-limits: it's a fraction ──────────────────────────────────────
    fhot_row = next(i for i, (col, _) in enumerate(DESCRIPTORS) if col == "f_hot")
    axes[fhot_row, 0].set_ylim(0, 1)

    # ── x-axis labels (bottom row only) ──────────────────────────────────────
    for col_i in range(n_cols):
        axes[-1, col_i].set_xlabel("Input multiplier", fontsize=10)

    # ── shared legend ─────────────────────────────────────────────────────────
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    if handles:
        fig.legend(handles, labels,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.04),
                   ncol=len(SUITES),
                   fontsize=9,
                   frameon=True,
                   handlelength=2.0)

    # ── super-title ───────────────────────────────────────────────────────────
    pivot_ctr = (PIVOT_LO + PIVOT_HI) / 2
    fig.suptitle(
        rf"CAMELS 1P  |  $\log M_{{200c}} = {pivot_ctr:.2f} \pm 0.25$  |  $z = 0$"
        + (f"  |  protocol {proto_hash}" if proto_hash else ""),
        fontsize=9, y=1.08,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"figure_p1_p{proto_hash}" if proto_hash else "figure_p1"
    for ext in ("pdf", "png"):
        out = outdir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Figure P1")
    parser.add_argument("--parquet_root", default=str(PARQUET_ROOT),
                        help="Root dir of parquets (default: outputs/camels)")
    parser.add_argument("--outdir", default=str(FIGURES),
                        help="Output dir for figures (default: figures/)")
    args = parser.parse_args()

    print("Loading parquets ...")
    df = load_parquets(args.parquet_root)

    # extract protocol hash for the filename
    proto_hash = ""
    if "protocol_hash" in df.columns:
        hashes = df["protocol_hash"].dropna().unique()
        if len(hashes) == 1:
            proto_hash = str(hashes[0])
        elif len(hashes) > 1:
            print(f"  [WARN] multiple protocol hashes: {hashes} — using first")
            proto_hash = str(hashes[0])

    n_piv = ((df.logM200c >= PIVOT_LO) & (df.logM200c < PIVOT_HI)).sum()
    print(f"Total halos: {len(df):,}  |  pivot bin: {n_piv:,}"
          f"  |  protocol: {proto_hash}")

    print("Building figure ...")
    make_figure(df, outdir=args.outdir, proto_hash=proto_hash)
    print("Done.")


if __name__ == "__main__":
    main()
