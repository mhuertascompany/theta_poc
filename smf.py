"""
smf.py — Stellar mass function extraction at z=0.

Runs on cluster (needs illustris_python).  Loads the subhalo catalog only
(no particle data) and saves a binned SMF to tables/smf/.

All subhalos (centrals + satellites) with M_star > 0 are included — the SMF
is a galaxy-population statistic, not just centrals.
Stellar mass = SubhaloMassType[:,4] (includes wind PT4 in TNG; negligible
effect on the integrated SMF at logM_star > 9).

Volume normalization
--------------------
Each suite is divided by its own comoving box volume [Mpc³].  The three
boxes have very different sizes (TNG100≈1.4e6, EAGLE100≈3.2e6, SIMBA25≈5e4 Mpc³)
so the SMF traces are just number densities — they can be compared directly.
Poisson error = sqrt(N) / V / dlogM per bin.

Usage (on cluster)
------------------
    python smf.py TNG100-1
    python smf.py Eagle100-1
    python smf.py Simba25-1

Output
------
    tables/smf/{suite_key}_z0.parquet
    Columns: logM_lo, logM_hi, logM_cen, N, phi, phi_err, suite, V_Mpc3, z_actual
"""

import sys
import argparse
import pathlib

import numpy as np
import pandas as pd
import h5py

import registry as R
import units as U

TABLES = pathlib.Path(__file__).parent / "tables" / "smf"
TABLES.mkdir(parents=True, exist_ok=True)

LOGM_BINS = np.arange(9.0, 12.51, 0.25)   # 14 bins, 0.25 dex wide
LOGM_MIN  = 9.0                            # exclude unresolved dwarfs


def _il():
    import illustris_python as il
    return il


def run(suite_key):
    reg  = R.get_suite(suite_key)
    path = reg["path"]
    snap = R.snapnum_for_z(suite_key, 0.0)

    out_path = TABLES / f"{suite_key}_z0.parquet"
    print(f"Suite  : {suite_key}")
    print(f"Snap   : {snap}  (z=0)")
    print(f"Output : {out_path}")

    # ── header ────────────────────────────────────────────────────────────────
    snap_file = _il().snapshot.snapPath(path, snap)
    with h5py.File(snap_file, "r") as f:
        hdr = dict(f["Header"].attrs)

    h        = float(hdr["HubbleParam"])
    a        = float(hdr["Time"])
    box_ckph = float(hdr["BoxSize"])          # ckpc/h
    z_actual = 1.0 / a - 1.0

    # comoving box side [Mpc] and volume [Mpc³]
    box_cMpc = box_ckph / 1e3 / h
    V_Mpc3   = box_cMpc ** 3
    print(f"h={h:.4f}  a={a:.4f}  z={z_actual:.4f}")
    print(f"Box = {box_cMpc:.1f} cMpc  →  V = {V_Mpc3:.3e} Mpc³")

    # ── stellar masses ─────────────────────────────────────────────────────────
    sub = _il().groupcat.loadSubhalos(path, snap, fields=["SubhaloMassType"])
    m_star = U.code_mass_to_msun(
        sub["SubhaloMassType"][:, 4].astype(np.float64), h)   # Msun

    valid = m_star > 10**LOGM_MIN
    m_star = m_star[valid]
    print(f"Subhalos with logM_star > {LOGM_MIN}: {len(m_star):,}")

    # ── binned SMF ─────────────────────────────────────────────────────────────
    logm = np.log10(m_star)
    dlogm = float(np.diff(LOGM_BINS)[0])

    counts, _ = np.histogram(logm, bins=LOGM_BINS)
    phi     = counts / V_Mpc3 / dlogm            # [Mpc⁻³ dex⁻¹]
    phi_err = np.sqrt(counts) / V_Mpc3 / dlogm  # Poisson

    df = pd.DataFrame({
        "logM_lo"  : LOGM_BINS[:-1],
        "logM_hi"  : LOGM_BINS[1:],
        "logM_cen" : 0.5 * (LOGM_BINS[:-1] + LOGM_BINS[1:]),
        "N"        : counts,
        "phi"      : phi,
        "phi_err"  : phi_err,
        "suite"    : suite_key,
        "V_Mpc3"   : V_Mpc3,
        "z_actual" : z_actual,
    })

    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df)} rows → {out_path}")
    print(df[["logM_cen", "N", "phi"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="SMF extraction at z=0")
    parser.add_argument("suite_key", help="e.g. TNG100-1, Eagle100-1, Simba25-1")
    args = parser.parse_args()
    run(args.suite_key)


if __name__ == "__main__":
    main()
