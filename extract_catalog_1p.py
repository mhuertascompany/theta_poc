#!/usr/bin/env python3
"""
extract_catalog_1p.py — catalog-only x summaries for the 1P sims.

Runs on the Binder.  Reads ONLY the groups_090.hdf5 catalog (no snapshot).
Appends x = (SMF, f_gas, sSFR) to the existing 1P parquets and writes
a combined table: outputs/camels/<suite_key>/catalog_x_1p.parquet

This is the missing piece for the cross-code robustness test:
  train NPE on LH-TNG (theta + x),
  evaluate on SIMBA-1P (theta from run_camels.py + x from this script),
  compare predicted vs measured theta.

Usage (on Binder):
    python extract_catalog_1p.py camels_tng_1p
    python extract_catalog_1p.py camels_simba_1p
"""

import pathlib
import sys

import h5py
import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import registry as R
import units as U

try:
    import hdf5plugin
except ImportError:
    pass

ROOT        = pathlib.Path("/home/jovyan/Data")
OUTPUT_ROOT = _HERE / "outputs" / "camels"

_SMF_BINS  = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5]
_STAT_BINS = [9.0, 9.5, 10.0, 10.5]


def compute_x(cat_path, h):
    with h5py.File(cat_path, "r") as cf:
        mass_type = cf["Subhalo/SubhaloMassType"][:]
        sfr_raw   = cf["Subhalo/SubhaloSFR"][:].astype(np.float64)

    m_gas  = U.code_mass_to_msun(mass_type[:, 0].astype(np.float64), h)
    m_star = U.code_mass_to_msun(mass_type[:, 4].astype(np.float64), h)

    ok     = m_star > 10**7.5
    m_gas  = m_gas[ok];  m_star = m_star[ok];  sfr = sfr_raw[ok]

    with np.errstate(divide="ignore", invalid="ignore"):
        logm  = np.log10(m_star)
        f_gas = np.where((m_gas + m_star) > 0, m_gas / (m_gas + m_star), np.nan)
        ssfr  = np.where(m_star > 0, sfr / m_star, np.nan)

    counts, _ = np.histogram(logm, bins=_SMF_BINS)
    smf_vals  = np.log10(counts.astype(float) + 1.0)

    fgas_meds, ssfr_meds = [], []
    for lo, hi in zip(_STAT_BINS[:-1], _STAT_BINS[1:]):
        b = (logm >= lo) & (logm < hi)
        if b.sum() >= 3:
            fgas_meds.append(float(np.nanmedian(f_gas[b])))
            sv = ssfr[b]; sv = sv[np.isfinite(sv) & (sv > 0)]
            ssfr_meds.append(float(np.log10(np.median(sv))) if len(sv) >= 3 else np.nan)
        else:
            fgas_meds.append(np.nan); ssfr_meds.append(np.nan)

    result = {}
    for i, v in enumerate(smf_vals):    result[f"smf_{i}"]  = float(v)
    for i, v in enumerate(fgas_meds):   result[f"fgas_{i}"] = float(v)
    for i, v in enumerate(ssfr_meds):   result[f"ssfr_{i}"] = float(v)
    return result


def main():
    suite_key = sys.argv[1] if len(sys.argv) > 1 else "camels_tng_1p"
    reg       = R.get_suite(suite_key)
    suite_dir = reg["suite_dir"]
    set_name  = reg["set_name"]

    # load existing 1P parquets (theta + provenance)
    parquet_dir = OUTPUT_ROOT / suite_key
    parquets    = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquets found in {parquet_dir}. Run run_camels.py first.")
        sys.exit(1)

    existing = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    # keep one row per sim (median over pivot-bin halos already in parquet)
    per_sim = (existing.groupby("sim_id")
               .agg(
                   varied_param = ("varied_param", "first"),
                   param_index  = ("param_index",  "first"),
                   Omega0       = ("Omega0",        "first"),
                   sigma8       = ("sigma8",        "first"),
                   A_SN1        = ("A_SN1",         "first"),
                   A_AGN1       = ("A_AGN1",        "first"),
                   A_SN2        = ("A_SN2",         "first"),
                   A_AGN2       = ("A_AGN2",        "first"),
                   protocol_hash= ("protocol_hash", "first"),
                   eta_M_median = ("eta_M",         "median"),
                   f_hot_median = ("f_hot",         "median"),
                   n_pivot      = ("logM200c",      "count"),
               )
               .reset_index())

    print(f"{suite_key}: {len(per_sim)} sims from parquets")

    rows = []
    for _, row in per_sim.iterrows():
        sim_id   = row["sim_id"]
        cat_path = ROOT / "Sims" / suite_dir / set_name / sim_id / "groups_090.hdf5"
        if not cat_path.exists():
            print(f"  [WARN] {sim_id}: catalog missing")
            continue

        with h5py.File(ROOT / "Sims" / suite_dir / set_name / sim_id
                       / "snapshot_090.hdf5", "r") as sf:
            h = float(sf["Header"].attrs["HubbleParam"])

        x = compute_x(cat_path, h)
        combined = {**row.to_dict(), **x, "suite_key": suite_key}
        rows.append(combined)
        print(f"  [OK] {sim_id}  smf_2={x['smf_2']:.2f}"
              f"  fgas_1={x.get('fgas_1', float('nan')):.2f}")

    out = OUTPUT_ROOT / suite_key / "catalog_x_1p.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nWrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
