"""
sbi_pilot.py — Stage B: NPE on CAMELS-LH, SBC diagnostics, Figure P2.

Runs locally on parquets downloaded from the Binder.

Pipeline
--------
1. Load LH parquets → build (θ, x) arrays.
2. Filter sims with insufficient pivot halos or NaN θ.
3. Split 80/20 train/test.
4. Standardize x; train NPE (p(θ|x), θ = η_M + f_hot at pivot).
5. SBC: for each test sim, rank of true θ among posterior samples.
   Uniform ranks → calibrated posterior.
6. Figure P2: corner plot for one test sim (truth crosshairs)
   + SBC rank histograms.

Dependencies: pandas, numpy, matplotlib, torch, sbi (>=0.22)
    pip install sbi

Usage:
    python sbi_pilot.py
    python sbi_pilot.py --parquet_root outputs/camels/camels_tng_lh
                        --outdir figures
                        --min_pivot 5
                        --n_sbc 50
"""

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as C

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE        = pathlib.Path(__file__).resolve().parent
PARQUET_ROOT = _HERE / "outputs" / "camels" / "camels_tng_lh"
FIGURES      = _HERE / "figures"
FIGURES.mkdir(exist_ok=True)

# ── style (matches compare.py) ────────────────────────────────────────────────

mpl.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":       True,
    "ytick.right":     True,
    "figure.dpi":      150,
})

# ── column definitions ────────────────────────────────────────────────────────

THETA_COLS = ["eta_M_median", "f_hot_median"]
THETA_LABELS = [r"$\eta_M$  (pivot median)", r"$f_{\rm hot}$  (pivot median)"]

X_COLS = (
    ["Omega0", "sigma8"] +            # cosmological context (known from CMB/BAO)
    [f"smf_{i}"  for i in range(5)] + # stellar mass function counts (5 bins)
    [f"fgas_{i}" for i in range(3)] + # median gas fraction (3 M* bins)
    [f"ssfr_{i}" for i in range(3)]   # median log sSFR (3 M* bins)
)   # 13-dimensional: catalog observables conditioned on cosmology


# ── data loading ──────────────────────────────────────────────────────────────

def load_lh_parquets(parquet_root, min_pivot=5):
    """
    Load all per-sim LH parquets.  Each parquet has one row.

    Filters:
    - n_pivot >= min_pivot  (enough pivot-bin halos for reliable θ)
    - all θ columns finite
    - all x columns finite (drops sims with empty catalog bins)

    Returns DataFrame with one row per valid sim.
    """
    root = pathlib.Path(parquet_root)
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets in {root}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} sims from {len(files)} parquets")

    # filter
    mask = df["n_pivot"] >= min_pivot
    df = df[mask].copy()
    print(f"  after n_pivot >= {min_pivot}: {len(df)} sims")

    mask = df[THETA_COLS].notna().all(axis=1)
    df = df[mask].copy()
    print(f"  after finite θ:               {len(df)} sims")

    # x: drop sims where ANY x column is NaN (empty catalog bins)
    x_available = [c for c in X_COLS if c in df.columns]
    mask = df[x_available].notna().all(axis=1)
    df = df[mask].copy()
    print(f"  after finite x ({len(x_available)}-dim): {len(df)} sims")

    return df, x_available


# ── NPE training ──────────────────────────────────────────────────────────────

def train_npe(theta_train, x_train):
    """
    Train a Neural Posterior Estimator p(θ|x).

    θ: (N, 2)  — eta_M_median, f_hot_median  (standardized)
    x: (N, D)  — catalog summary vector       (standardized)

    Returns (posterior, theta_scaler, x_scaler).
    """
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    # uniform prior over empirical θ range (10% padding)
    t_min = theta_train.min(axis=0)
    t_max = theta_train.max(axis=0)
    pad   = 0.1 * (t_max - t_min)
    prior = BoxUniform(
        low  = torch.tensor((t_min - pad).astype(np.float32)),
        high = torch.tensor((t_max + pad).astype(np.float32)),
    )

    theta_t = torch.tensor(theta_train.astype(np.float32))
    x_t     = torch.tensor(x_train.astype(np.float32))

    inference = NPE(prior=prior)
    inference.append_simulations(theta_t, x_t)

    print("  Training NPE ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        density_estimator = inference.train(show_train_summary=False)

    posterior = inference.build_posterior(density_estimator)
    print("  Training done.")
    return posterior


# ── SBC ───────────────────────────────────────────────────────────────────────

def run_sbc(posterior, theta_test, x_test, num_samples=1000):
    """
    Simulation-Based Calibration.

    For each test sim:
      - sample num_samples from posterior | x_test[i]
      - rank of true θ[i] among samples (per dimension)

    Returns ranks array (N_test, D_theta).
    Under a calibrated posterior, ranks are uniform over [0, num_samples].
    """
    import torch

    ranks = []
    for i in range(len(theta_test)):
        x_obs = torch.tensor(x_test[i:i+1].astype(np.float32))
        with torch.no_grad():
            samples = posterior.sample(
                (num_samples,), x=x_obs, show_progress_bars=False)
        samples_np = samples.numpy()   # (num_samples, D)
        true_np    = theta_test[i]     # (D,)
        # rank = number of samples below the true value
        r = (samples_np < true_np[np.newaxis, :]).sum(axis=0)
        ranks.append(r)
        if (i + 1) % 10 == 0:
            print(f"    SBC: {i+1}/{len(theta_test)}")

    return np.array(ranks)   # (N_test, D_theta)


# ── Figure P2 ─────────────────────────────────────────────────────────────────

def make_figure_p2(posterior, theta_test, x_test, ranks,
                   test_idx=0, num_posterior_samples=5000,
                   outdir=FIGURES, proto_hash=""):
    """
    Figure P2: corner plot for one test sim + SBC rank histograms.

    Layout: 3 columns
      col 0-1: corner plot (η_M marginal | joint scatter / f_hot marginal)
      col 2:   SBC rank histograms for η_M (top) and f_hot (bottom)
    """
    import torch

    D = len(THETA_COLS)

    # posterior samples for the chosen test sim
    x_obs = torch.tensor(x_test[test_idx:test_idx+1].astype(np.float32))
    with torch.no_grad():
        post_samples = posterior.sample(
            (num_posterior_samples,), x=x_obs, show_progress_bars=False
        ).numpy()   # (N, D)
    truth = theta_test[test_idx]   # (D,)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], wspace=0.35)

    ax_eta  = fig.add_subplot(gs[0, 0])   # η_M marginal
    ax_fhot = fig.add_subplot(gs[1, 1])   # f_hot marginal
    ax_joint = fig.add_subplot(gs[1, 0])  # joint (η_M vs f_hot)
    ax_sbc0  = fig.add_subplot(gs[0, 2])  # SBC rank — η_M
    ax_sbc1  = fig.add_subplot(gs[1, 2])  # SBC rank — f_hot
    ax_eta.sharex(ax_joint)               # share η_M axis

    corner_color = "#2166ac"
    alpha_fill   = 0.25

    # ── corner: η_M marginal ──────────────────────────────────────────────────
    ax_eta.hist(post_samples[:, 0], bins=40, density=True,
                color=corner_color, alpha=alpha_fill, edgecolor="none")
    ax_eta.axvline(truth[0], color="k", lw=1.5, ls="--", label="truth")
    ax_eta.set_ylabel("density", fontsize=9)
    ax_eta.set_title(THETA_LABELS[0], fontsize=9, pad=3)
    ax_eta.tick_params(labelbottom=False)
    ax_eta.legend(fontsize=8, frameon=False)

    # ── corner: f_hot marginal ────────────────────────────────────────────────
    ax_fhot.hist(post_samples[:, 1], bins=40, density=True,
                 color=corner_color, alpha=alpha_fill, edgecolor="none")
    ax_fhot.axvline(truth[1], color="k", lw=1.5, ls="--")
    ax_fhot.set_xlabel(THETA_LABELS[1], fontsize=9)
    ax_fhot.set_ylabel("density", fontsize=9)

    # ── corner: joint scatter ─────────────────────────────────────────────────
    ax_joint.scatter(post_samples[:, 0], post_samples[:, 1],
                     s=2, c=corner_color, alpha=0.08, rasterized=True)
    ax_joint.scatter(*truth, marker="+", s=120, c="k", zorder=5,
                     linewidths=1.5, label="truth")
    ax_joint.set_xlabel(THETA_LABELS[0], fontsize=9)
    ax_joint.set_ylabel(THETA_LABELS[1], fontsize=9)

    # ── SBC rank histograms ───────────────────────────────────────────────────
    n_sbc = ranks.shape[0]
    n_sam = int(ranks.max()) + 1   # approximate num_samples used in SBC

    for ax, dim, lbl in zip([ax_sbc0, ax_sbc1], [0, 1], THETA_LABELS):
        r = ranks[:, dim]
        ax.hist(r, bins=20, range=(0, n_sam),
                color="#d6604d", alpha=0.6, edgecolor="none", density=True)
        # expected uniform density
        ax.axhline(1.0 / n_sam * (n_sam / 20),
                   color="0.4", lw=1.0, ls=":", label="uniform")
        ax.set_xlabel(f"rank  [{lbl}]", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"SBC  (n={n_sbc})", fontsize=8, pad=3)
        ax.legend(fontsize=7, frameon=False)

    # ── suptitle ──────────────────────────────────────────────────────────────
    pivot_ctr = (C.PIVOT["logM200_lo"] + C.PIVOT["logM200_hi"]) / 2
    fig.suptitle(
        rf"NPE: $p(\eta_M, f_{{\rm hot}}\,|\,\mathbf{{x}}_\mathrm{{cat}})$  "
        rf"|  pivot $\log M_{{200c}} = {pivot_ctr:.2f} \pm 0.25$  "
        rf"|  test sim #{test_idx}",
        fontsize=9, y=1.02,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"figure_p2_p{proto_hash}" if proto_hash else "figure_p2"
    for ext in ("pdf", "png"):
        out = outdir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NPE on CAMELS-LH (Stage B, Exp 2)")
    parser.add_argument("--parquet_root", default=str(PARQUET_ROOT))
    parser.add_argument("--outdir",       default=str(FIGURES))
    parser.add_argument("--min_pivot",    type=int,   default=5,
                        help="Min pivot-bin halos for a sim to be included")
    parser.add_argument("--n_sbc",        type=int,   default=50,
                        help="Number of test sims for SBC (default: 50)")
    parser.add_argument("--test_idx",     type=int,   default=0,
                        help="Index of test sim for corner plot (default: 0)")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── load ─────────────────────────────────────────────────────────────────
    print("\n── Loading parquets ──────────────────────────────────────────")
    df, x_cols = load_lh_parquets(args.parquet_root, min_pivot=args.min_pivot)

    n_valid = len(df)
    if n_valid < 50:
        raise RuntimeError(
            f"Only {n_valid} valid sims — need ≥50 for a meaningful pilot. "
            "Check n_pivot or run more LH sims.")
    print(f"\n  Valid sims for training: {n_valid}")
    print(f"  θ range:  η_M  [{df.eta_M_median.min():.2f}, {df.eta_M_median.max():.2f}]"
          f"   f_hot [{df.f_hot_median.min():.2f}, {df.f_hot_median.max():.2f}]")

    proto_hash = ""
    if "protocol_hash" in df.columns:
        h = df["protocol_hash"].dropna().unique()
        if len(h) == 1:
            proto_hash = str(h[0])

    # ── build arrays ─────────────────────────────────────────────────────────
    theta = df[THETA_COLS].values.astype(np.float64)   # (N, 2)
    x     = df[x_cols].values.astype(np.float64)       # (N, D)

    # standardize θ and x
    theta_mean, theta_std = theta.mean(0), theta.std(0)
    x_mean,     x_std     = x.mean(0),     x.std(0)
    x_std = np.where(x_std > 0, x_std, 1.0)   # avoid div-by-zero

    theta_z = (theta - theta_mean) / theta_std
    x_z     = (x     - x_mean)     / x_std

    # ── train/test split ──────────────────────────────────────────────────────
    n_test  = min(args.n_sbc, max(10, int(0.2 * n_valid)))
    n_train = n_valid - n_test
    idx     = np.random.permutation(n_valid)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    theta_train, theta_test = theta_z[train_idx], theta_z[test_idx]
    x_train,     x_test     = x_z[train_idx],     x_z[test_idx]

    # keep unstandardized truth for plotting
    theta_test_raw = theta[test_idx]

    print(f"\n  Train: {n_train}  |  Test: {n_test}")

    # ── train NPE ─────────────────────────────────────────────────────────────
    print("\n── Training NPE ──────────────────────────────────────────────")
    try:
        posterior = train_npe(theta_train, x_train)
    except ImportError:
        raise ImportError(
            "sbi not installed.  Run:  pip install sbi\n"
            "Needs torch (CPU is fine) and sbi >= 0.22")

    # ── SBC ───────────────────────────────────────────────────────────────────
    print("\n── Running SBC ───────────────────────────────────────────────")
    print(f"  {n_test} test sims × 1000 posterior samples each ...")
    ranks = run_sbc(posterior, theta_test, x_test, num_samples=1000)

    # quick calibration check: KS test against uniform
    from scipy import stats as sp_stats
    for d, lbl in enumerate(THETA_LABELS):
        r_norm = ranks[:, d] / 1000.0
        ks_stat, ks_p = sp_stats.kstest(r_norm, "uniform")
        flag = "[OK]" if ks_p > 0.05 else "[NON-UNIFORM]"
        print(f"  {flag}  {lbl}: KS p={ks_p:.3f}")

    # ── Figure P2 ─────────────────────────────────────────────────────────────
    print("\n── Building Figure P2 ────────────────────────────────────────")
    ti = min(args.test_idx, n_test - 1)
    make_figure_p2(
        posterior, theta_test_raw, x_test,
        ranks=ranks,
        test_idx=ti,
        outdir=args.outdir,
        proto_hash=proto_hash,
    )

    # ── optional: cross-code robustness test ─────────────────────────────────
    simba_catalog = (pathlib.Path(args.parquet_root).parent
                     / "camels_simba_1p" / "catalog_x_1p.parquet")
    if simba_catalog.exists():
        print("\n── Cross-code robustness test (SIMBA-1P) ────────────────────")
        cross_code_test(posterior, simba_catalog,
                        theta_mean, theta_std, x_mean, x_std, x_cols,
                        outdir=args.outdir, proto_hash=proto_hash)
    else:
        print(f"\n[INFO] Cross-code test skipped — {simba_catalog} not found.")
        print("       Run on Binder:  python extract_catalog_1p.py camels_simba_1p")

    print("\nDone.")


def cross_code_test(posterior, simba_path,
                    theta_mean, theta_std, x_mean, x_std, x_cols,
                    outdir, proto_hash=""):
    """
    Apply TNG-trained NPE to SIMBA-1P catalog summaries.

    For each SIMBA 1P sim:
      - x_simba (catalog observables) → posterior p(θ|x_simba) from TNG NPE
      - θ_simba (measured by frozen protocol on SIMBA snapshot)
      - Compare predicted vs measured

    If descriptors are cross-suite robust: measured θ_simba should fall within
    the predicted posterior.  If only input params were robust, this would fail.
    """
    import torch

    df = pd.read_parquet(simba_path)
    # filter finite θ and x
    x_avail = [c for c in x_cols if c in df.columns]
    ok = df[["eta_M_median", "f_hot_median"]].notna().all(axis=1)
    ok &= df[x_avail].notna().all(axis=1)
    df = df[ok].copy()

    if len(df) < 3:
        print("  [WARN] too few valid SIMBA sims for cross-code test")
        return

    print(f"  {len(df)} SIMBA-1P sims with finite θ and x")

    x_simba     = df[x_avail].values.astype(np.float64)
    theta_simba = df[["eta_M_median", "f_hot_median"]].values.astype(np.float64)

    # standardize using TNG training statistics
    x_simba_z     = (x_simba - x_mean) / np.where(x_std > 0, x_std, 1.0)
    theta_simba_z = (theta_simba - theta_mean) / theta_std

    # posterior coverage: for each SIMBA sim, what fraction of TNG posterior
    # samples lie closer to the origin than the true SIMBA θ?
    # (coverage fraction near 0.5 → well-calibrated; near 0 or 1 → biased)
    n_samples = 2000
    coverage  = []
    pred_medians = []

    for i in range(len(df)):
        x_obs = torch.tensor(x_simba_z[i:i+1].astype(np.float32))
        with torch.no_grad():
            samp = posterior.sample(
                (n_samples,), x=x_obs, show_progress_bars=False).numpy()
        # posterior predictive median (in standardized space → back to physical)
        pred_med = samp.mean(axis=0) * theta_std + theta_mean
        pred_medians.append(pred_med)
        # coverage: rank of true θ among samples
        true_z = theta_simba_z[i]
        dist_samples = np.linalg.norm(samp, axis=1)
        dist_true    = np.linalg.norm(true_z)
        coverage.append(float((dist_samples < dist_true).mean()))

    pred_medians = np.array(pred_medians)
    coverage     = np.array(coverage)

    print(f"  Mean coverage: {coverage.mean():.2f}  (0.5 = perfectly calibrated)")
    print(f"  Predicted η_M range: [{pred_medians[:,0].min():.2f},"
          f" {pred_medians[:,0].max():.2f}]")
    print(f"  Measured  η_M range: [{theta_simba[:,0].min():.2f},"
          f" {theta_simba[:,0].max():.2f}]")

    # ── Figure: predicted vs measured θ for SIMBA ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    varied = df["varied_param"].values
    colors = {"A_SN1": "#1f77b4", "A_AGN1": "#d62728",
              "A_SN2": "#2ca02c", "A_AGN2": "#ff7f0e"}

    for desc_i, (ax, lbl) in enumerate(zip(axes, THETA_LABELS)):
        meas  = theta_simba[:, desc_i]
        pred  = pred_medians[:, desc_i]
        for vp, color in colors.items():
            m = varied == vp
            ax.scatter(meas[m], pred[m], c=color, s=30, alpha=0.8,
                       label=vp, zorder=3)
        lo = min(meas.min(), pred.min()) * 0.9
        hi = max(meas.max(), pred.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlabel(f"Measured {lbl}  (SIMBA)", fontsize=10)
        ax.set_ylabel(f"Predicted {lbl}  (TNG NPE)", fontsize=10)
        ax.set_title("Cross-code transfer", fontsize=10)
        if desc_i == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle(
        "TNG-trained NPE evaluated on SIMBA-1P\n"
        "Descriptor robustness: if calibrated → descriptors span a common space",
        fontsize=9,
    )

    stem = f"figure_crosscode_p{proto_hash}" if proto_hash else "figure_crosscode"
    for ext in ("pdf", "png"):
        out = pathlib.Path(outdir) / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
