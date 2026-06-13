#!/usr/bin/env python3
"""
extract_lh_camels.py — Stage A: extract θ + x for CAMELS-LH sims.

Runs on the Binder (binder.flatironinstitute.org, CAMELS_PUBLIC).

Per sim, produces ONE ROW containing:
  θ  = eta_M_median, f_hot_median  in the pivot bin  (from snapshot)
  x  = catalog-level observables: SMF counts (5), median f_gas (3),
        median log sSFR (3)  — 11-dim, catalog-only, snapshot-free
  provenance = Omega0, sigma8, A_SN1..A_AGN2, n_pivot, protocol_hash, git

One parquet per sim in outputs/camels/camels_tng_lh/.
RESUMABLE: skips sims whose parquet already exists.

Usage (on Binder):
    python extract_lh_camels.py                  # first 200 LH sims
    python extract_lh_camels.py --max_sims 500   # full set
    python extract_lh_camels.py --max_sims 5     # quick test

Note: hdf5plugin must be importable (pre-installed on Binder).
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

try:
    import hdf5plugin
except ImportError:
    print("[WARN] hdf5plugin not available")

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import registry as R
import loaders
import selection
import config as C
import units as U
from descriptors import eta_M, f_hot

# ── constants ─────────────────────────────────────────────────────────────────

ROOT         = pathlib.Path("/home/jovyan/Data")
OUTPUT_ROOT  = _HERE / "outputs" / "camels"
CAMELS_RELEASE = "CAMELS_PUBLIC_DR3"

# Pivot-only selection (faster: ~20-30 halos vs 255)
_PIVOT_SEL = {
    **C.CAMELS_HALO_SELECT,
    "logM200_min": C.PIVOT["logM200_lo"],
    "logM200_max": C.PIVOT["logM200_hi"],
}

# x summary bins
_SMF_BINS  = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5]   # log M* [Msun], 5 bins
_STAT_BINS = [9.0, 9.5, 10.0, 10.5]               # log M* for f_gas/sSFR, 3 bins


# ── helpers ───────────────────────────────────────────────────────────────────

def _protocol_hash():
    s = json.dumps({**C.PROTOCOL, **_PIVOT_SEL}, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _snap_path(suite_dir, set_name, sim_id):
    return ROOT / "Sims" / suite_dir / set_name / sim_id / "snapshot_090.hdf5"


def _cat_path(suite_dir, set_name, sim_id):
    return ROOT / "Sims" / suite_dir / set_name / sim_id / "groups_090.hdf5"


def _parse_lh_params(suite_dir, set_name):
    """
    Read CosmoAstroSeed_<suite>_L25n256_<set>.txt.
    Returns dict: sim_name → {Omega0, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2}.
    """
    candidates = [
        ROOT / "Sims" / suite_dir / set_name /
            f"CosmoAstroSeed_{suite_dir}_L25n256_{set_name}.txt",
        ROOT / "Sims" / suite_dir /
            f"CosmoAstroSeed_{suite_dir}_L25n256_{set_name}.txt",
        ROOT / f"CosmoAstroSeed_{suite_dir}_L25n256_{set_name}.txt",
    ]
    fpath = next((p for p in candidates if p.exists()), None)
    if fpath is None:
        print(f"[WARN] CosmoAstroSeed file not found for {suite_dir}/{set_name}"
              " — provenance params will be NaN")
        return {}

    result = {}
    for line in fpath.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        name = tokens[0]
        try:
            nums = [float(t) for t in tokens[1:]]
        except ValueError:
            continue
        result[name] = dict(
            Omega0=nums[0], sigma8=nums[1],
            A_SN1=nums[2], A_AGN1=nums[3], A_SN2=nums[4], A_AGN2=nums[5],
        )
    return result


# ── catalog-only x summaries ──────────────────────────────────────────────────

def compute_x(cat_path, h):
    """
    Compute 11-dim catalog-level observable summaries from groups_090.hdf5.

    Fields used (SubFind subhalo catalog):
      SubhaloMassType[:, 0]  — gas mass  (code units → Msun)
      SubhaloMassType[:, 4]  — stellar mass (code units → Msun)
      SubhaloSFR             — SFR in Msun/yr

    Returns dict: smf_0..4 (log10 counts), fgas_0..2 (medians), ssfr_0..2 (log10 medians).
    NaN for bins with < 3 galaxies.
    """
    with h5py.File(cat_path, "r") as cf:
        mass_type = cf["Subhalo/SubhaloMassType"][:]   # (N, 6)
        sfr_raw   = cf["Subhalo/SubhaloSFR"][:].astype(np.float64)

    m_gas  = U.code_mass_to_msun(mass_type[:, 0].astype(np.float64), h)
    m_star = U.code_mass_to_msun(mass_type[:, 4].astype(np.float64), h)

    # keep only subhalos with M* > 10^7.5 Msun (remove resolution noise)
    ok = m_star > 10**7.5
    m_gas  = m_gas[ok]
    m_star = m_star[ok]
    sfr    = sfr_raw[ok]

    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log10(m_star)
        f_gas = np.where((m_gas + m_star) > 0,
                         m_gas / (m_gas + m_star), np.nan)
        ssfr  = np.where(m_star > 0, sfr / m_star, np.nan)

    # SMF: log10(count + 1) per bin — handles empty bins without -inf
    counts, _ = np.histogram(logm, bins=_SMF_BINS)
    smf_vals  = np.log10(counts.astype(float) + 1.0)

    # f_gas and log sSFR medians in 3 bins
    fgas_meds = []
    ssfr_meds = []
    for lo, hi in zip(_STAT_BINS[:-1], _STAT_BINS[1:]):
        in_bin = (logm >= lo) & (logm < hi)
        if in_bin.sum() >= 3:
            fgas_meds.append(float(np.nanmedian(f_gas[in_bin])))
            sv = ssfr[in_bin]
            sv = sv[np.isfinite(sv) & (sv > 0)]
            ssfr_meds.append(float(np.log10(np.median(sv))) if len(sv) >= 3
                              else np.nan)
        else:
            fgas_meds.append(np.nan)
            ssfr_meds.append(np.nan)

    result = {}
    for i, v in enumerate(smf_vals):
        result[f"smf_{i}"] = float(v)
    for i, v in enumerate(fgas_meds):
        result[f"fgas_{i}"] = float(v)
    for i, v in enumerate(ssfr_meds):
        result[f"ssfr_{i}"] = float(v)
    return result


# ── snapshot-based θ extraction ───────────────────────────────────────────────

def compute_theta(snap_path, cat_path, gfm_recon):
    """
    Extract η_M and f_hot medians in pivot bin from snapshot.

    Returns dict: eta_M_median, eta_M_p16, eta_M_p84, f_hot_median,
                  f_hot_p16, f_hot_p84, n_pivot, n_valid.
    """
    raw = loaders.load_snapshot_camels(snap_path, cat_path)

    with h5py.File(snap_path, "r") as sf:
        hdr = dict(sf["Header"].attrs)
        h   = float(hdr["HubbleParam"])

    group_ids = loaders.select_centrals_from_snapshot(raw, _PIVOT_SEL)
    n_pivot   = int(len(group_ids))

    eta_vals, fhot_vals = [], []
    n_valid = 0

    for hid in group_ids:
        try:
            halo = loaders.extract_halo_from_snapshot(raw, int(hid), gfm_recon)
            halo["gas"] = selection.to_halo_frame(
                halo["gas"], halo["subhalo"], halo["meta"]["box_kpc"])
            r_eta  = eta_M.compute(halo, C.PROTOCOL)
            r_fhot = f_hot.compute(halo, C.PROTOCOL)
            v_eta  = r_eta.get("value", np.nan)
            v_fhot = r_fhot.get("value", np.nan)
            if np.isfinite(v_eta):
                eta_vals.append(float(v_eta))
            if np.isfinite(v_fhot):
                fhot_vals.append(float(v_fhot))
            n_valid += 1
        except Exception:
            pass

    def _stats(vals):
        if len(vals) >= 3:
            return (float(np.nanmedian(vals)),
                    float(np.nanpercentile(vals, 16)),
                    float(np.nanpercentile(vals, 84)))
        elif vals:
            return float(np.nanmedian(vals)), np.nan, np.nan
        return np.nan, np.nan, np.nan

    e_med, e_lo, e_hi = _stats(eta_vals)
    f_med, f_lo, f_hi = _stats(fhot_vals)

    return dict(
        n_pivot        = n_pivot,
        n_valid        = n_valid,
        eta_M_median   = e_med,
        eta_M_p16      = e_lo,
        eta_M_p84      = e_hi,
        f_hot_median   = f_med,
        f_hot_p16      = f_lo,
        f_hot_p84      = f_hi,
    )


# ── per-sim runner ────────────────────────────────────────────────────────────

def run_sim(sim_id, suite_dir, set_name, params, reg, proto_hash, git_commit, out_dir):
    """
    Extract θ + x for one LH sim.  Returns True if successful, False if skipped.
    """
    out_path = out_dir / f"{sim_id}_p{proto_hash}.parquet"
    tmp_path = out_dir / f"{sim_id}_p{proto_hash}.tmp.parquet"

    if out_path.exists():
        return "skip"

    if tmp_path.exists():
        tmp_path.unlink()

    snap = _snap_path(suite_dir, set_name, sim_id)
    cat  = _cat_path(suite_dir, set_name, sim_id)

    if not snap.exists() or not cat.exists():
        print(f"  [WARN] {sim_id}: snapshot or catalog missing")
        return "missing"

    gfm_recon = reg["gfm_initial_mass_reconstructed"]
    t0 = time.time()

    # catalog x (fast, no snapshot)
    with h5py.File(snap, "r") as sf:
        h = float(sf["Header"].attrs["HubbleParam"])
    x_dict = compute_x(cat, h)

    # snapshot θ (pivot-only)
    theta_dict = compute_theta(snap, cat, gfm_recon)

    row = dict(
        sim_id         = sim_id,
        suite_key      = reg.get("suite_key", "camels_tng_lh"),
        **params,
        **theta_dict,
        **x_dict,
        protocol_hash  = proto_hash,
        git_commit     = git_commit,
        camels_release = CAMELS_RELEASE,
    )

    df = pd.DataFrame([row])
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(out_path)

    elapsed = time.time() - t0
    print(f"  [DONE] {sim_id}  n_pivot={theta_dict['n_pivot']:3d}"
          f"  eta_M={theta_dict['eta_M_median']:.3f}"
          f"  f_hot={theta_dict['f_hot_median']:.3f}"
          f"  {elapsed:.1f}s")
    return "done"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract θ+x for CAMELS LH sims (Stage A, Exp 2)")
    parser.add_argument("--suite_key", default="camels_tng_lh",
                        help="Registry key (default: camels_tng_lh)")
    parser.add_argument("--max_sims", type=int, default=200,
                        help="Max number of LH sims to process (default: 200)")
    args = parser.parse_args()

    reg       = R.get_suite(args.suite_key)
    suite_dir = reg["suite_dir"]
    set_name  = reg["set_name"]

    proto_hash = _protocol_hash()
    git_commit = _git_commit()

    print(f"Suite      : {args.suite_key}  ({suite_dir}/{set_name})")
    print(f"Proto hash : {proto_hash}")
    print(f"Git commit : {git_commit}")
    print(f"Max sims   : {args.max_sims}")

    # output dir
    out_dir = OUTPUT_ROOT / args.suite_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}\n")

    # read parameter file
    lh_params = _parse_lh_params(suite_dir, set_name)
    if lh_params:
        print(f"Parameter file: {len(lh_params)} sims found")
    else:
        print("[WARN] No parameter file — provenance params will be NaN")

    # discover sim directories
    lh_base = ROOT / "Sims" / suite_dir / set_name
    sim_dirs = sorted(
        d.name for d in lh_base.iterdir()
        if d.is_dir() and d.name.startswith(set_name + "_")
    )[:args.max_sims]
    print(f"Sims to run: {len(sim_dirs)}\n")

    reg["suite_key"] = args.suite_key

    n_done = n_skip = n_miss = n_err = 0
    t_total = time.time()

    for sim_id in sim_dirs:
        params = lh_params.get(sim_id, dict(
            Omega0=np.nan, sigma8=np.nan,
            A_SN1=np.nan, A_AGN1=np.nan, A_SN2=np.nan, A_AGN2=np.nan,
        ))
        try:
            status = run_sim(
                sim_id, suite_dir, set_name, params,
                reg, proto_hash, git_commit, out_dir,
            )
            if status == "done":    n_done += 1
            elif status == "skip":  n_skip += 1
            else:                   n_miss += 1
        except Exception as e:
            n_err += 1
            print(f"  [ERR] {sim_id}: {e}")

    elapsed = time.time() - t_total
    print(f"\nDone. extracted={n_done}  skipped={n_skip}"
          f"  missing={n_miss}  errors={n_err}"
          f"  wall-time={elapsed/60:.1f} min")
    print(f"Parquets: {out_dir}")


if __name__ == "__main__":
    main()
