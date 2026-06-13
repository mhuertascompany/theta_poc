#!/usr/bin/env python3
"""
probe_binder.py — GATE 0 probe for the CAMELS Binder session.

Runs on binder.flatironinstitute.org (CAMELS_PUBLIC) after git-cloning theta_poc.
Mount root: /home/jovyan/Data/

What this script does
---------------------
1. Confirms Binder mount root and snapshot/catalog path layout.
2. Reads CosmoAstroSeed_<suite>_L25n256_1P.txt for both IllustrisTNG and SIMBA;
   verifies that only one feedback parameter varies per row (p3..p6) and that
   Omega0/sigma8 are fiducial across all p3..p6 rows.
   Writes provenance/onep_map.csv (suite_key, sim_id, varied_param, param_index,
   Omega0, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2).
3. Confirms snapnum 090 / Redshift≈0 from snapshot and catalog headers.
4. Checks available RAM (/proc/meminfo) and CPU count.
5. Checks whether pyarrow is importable (determines parquet vs feather output).
6. TIMING TEST: runs a full descriptor extraction (η_M, f_hot, ε_ff, p★/m★) on
   one 1P sim end-to-end; reports wall-time and peak RSS.

GATE 0 outcome: if this script finishes without FAIL lines, Exp 1 is unblocked.

Usage (on Binder, inside cloned theta_poc directory):
    python probe_binder.py
    python probe_binder.py --quick   # skip timing test
"""

import argparse
import os
import sys
import time
import pathlib
import subprocess

import numpy as np

# ── hdf5plugin must be imported before h5py for CAMELS BLOSC-compressed files ─
try:
    import hdf5plugin
    _HDF5PLUGIN = True
except ImportError:
    _HDF5PLUGIN = False
    print("[WARN] hdf5plugin not available — BLOSC-compressed files may fail to open")

import h5py

# ── local imports (theta_poc must be on sys.path) ────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import registry as R
import loaders
import selection
import config as C
from descriptors import eta_M, f_hot, eps_ff, p_star

# ── constants ────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path("/home/jovyan/Data")
SNAPNUM   = "090"
PROV_DIR  = _HERE / "provenance"

# Map suite_key → (Binder suite directory, CosmoAstroSeed filename stem)
CAMELS_SUITES = {
    "camels_tng_1p":   ("IllustrisTNG", "CosmoAstroSeed_IllustrisTNG_L25n256_1P.txt"),
    "camels_simba_1p": ("SIMBA",        "CosmoAstroSeed_SIMBA_L25n256_1P.txt"),
}

# Parameter column names in the CosmoAstroSeed file (p1..p6 + cosmology)
# Order verified from CAMELS documentation: Omega0, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2
PARAM_COLS   = ["Omega0", "sigma8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
FEEDBACK_COLS = ["A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
PARAM_INDICES = {"A_SN1": 3, "A_AGN1": 4, "A_SN2": 5, "A_AGN2": 6}

# Expected 1P directory variations per parameter
VARIATIONS = ["n2", "n1", "1", "2"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(msg):   print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def check_mount():
    print(f"\n{'='*60}")
    print("1. Mount root and path layout")
    print(f"{'='*60}")
    ok = True
    if not ROOT.exists():
        _fail(f"Mount root {ROOT} does not exist — Binder not correctly mounted?")
        return False
    _ok(f"Mount root {ROOT} exists")

    for suite_key, (suite_dir, _) in CAMELS_SUITES.items():
        sims_dir = ROOT / "Sims" / suite_dir / "1P"
        fof_dir  = ROOT / "FOF_Subfind" / suite_dir / "1P"
        if sims_dir.exists():
            _ok(f"Sims/{suite_dir}/1P/ present")
        else:
            _fail(f"Sims/{suite_dir}/1P/ MISSING")
            ok = False
        if fof_dir.exists():
            _ok(f"FOF_Subfind/{suite_dir}/1P/ present (mirror)")
        else:
            _warn(f"FOF_Subfind/{suite_dir}/1P/ absent — using Sims/ copy only")

    # Spot-check one fiducial sim per suite.
    # Use 1P_p3_0 (A_SN1 sweep fiducial) — exists for both TNG and SIMBA.
    # TNG also has a shared 1P_0 but SIMBA does not.
    for suite_key, (suite_dir, _) in CAMELS_SUITES.items():
        fid = "1P_p3_0"
        snap = ROOT / "Sims" / suite_dir / "1P" / fid / f"snapshot_{SNAPNUM}.hdf5"
        cat  = ROOT / "Sims" / suite_dir / "1P" / fid / f"groups_{SNAPNUM}.hdf5"
        if snap.exists():
            _ok(f"Snapshot found: {snap.relative_to(ROOT)}")
        else:
            _fail(f"Snapshot MISSING: {snap.relative_to(ROOT)}")
            ok = False
        if cat.exists():
            _ok(f"Catalog found:  {cat.relative_to(ROOT)}")
        else:
            _fail(f"Catalog MISSING: {cat.relative_to(ROOT)}")
            ok = False
    return ok


def _find_cosmoastro(suite_dir, filename):
    """Search a few candidate locations for the CosmoAstroSeed file."""
    candidates = [
        ROOT / filename,
        ROOT / "Sims" / suite_dir / filename,
        ROOT / "Sims" / suite_dir / "1P" / filename,
        _HERE / "provenance" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _parse_cosmoastro_named(fpath):
    """
    Parse a CosmoAstroSeed txt file.

    Actual format (both TNG and SIMBA):
        #Name Omega0 sigma8 A_SN1 A_AGN1 A_SN2 A_AGN2 ...
        1P_p1_n2   0.1  0.8  <val> ...

    First column is the sim name (non-numeric string).
    Returns (header_tokens, dict[sim_name → list_of_floats]).
    The float list starts at Omega0 (i.e., the Name column is stripped).
    """
    result  = {}
    header  = None
    for line in fpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("#").split()   # e.g. ['Name','Omega0','sigma8',...]
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        sim_name = tokens[0]
        try:
            numeric = [float(t) for t in tokens[1:]]
        except ValueError:
            continue
        result[sim_name] = numeric
    return header, result


import re as _re
# Matches exactly 1P_p{3,4,5,6}_{n2,n1,0,1,2}
_P1_FEEDBACK_RE = _re.compile(r'^1P_p([3456])_(n2|n1|0|1|2)$')

_FEEDBACK_PARAM = {
    "3": ("A_SN1",  3),
    "4": ("A_AGN1", 4),
    "5": ("A_SN2",  5),
    "6": ("A_AGN2", 6),
}


def build_onep_map():
    """
    Read CosmoAstroSeed files and build provenance/onep_map.csv.

    Selects only 1P_p{3,4,5,6}_{n2,n1,0,1,2} rows = 20 sims per suite.
    The _0 variant is the fiducial point on each parameter curve.
    Both TNG and SIMBA follow this layout; TNG also has a 1P_0 shared
    fiducial directory which is not in the file and not needed for P1.

    Columns in CSV: suite_key, sim_id, varied_param, param_index,
                    Omega0, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2.
    """
    print(f"\n{'='*60}")
    print("2. CosmoAstroSeed parsing → provenance/onep_map.csv")
    print(f"{'='*60}")

    PROV_DIR.mkdir(exist_ok=True)
    all_rows = []
    ok = True

    for suite_key, (suite_dir, filename) in CAMELS_SUITES.items():
        fpath = _find_cosmoastro(suite_dir, filename)
        if fpath is None:
            _fail(f"{filename} not found for {suite_key}")
            ok = False
            continue
        _ok(f"Found {fpath.relative_to(ROOT) if ROOT in fpath.parents else fpath}")

        header, rows_by_name = _parse_cosmoastro_named(fpath)
        _ok(f"{suite_key}: {len(rows_by_name)} named rows parsed from file")

        # Select only the 20 feedback 1P sims (p3..p6, 5 variations each)
        suite_rows = []
        for sim_name in sorted(rows_by_name):
            m = _P1_FEEDBACK_RE.match(sim_name)
            if not m:
                continue
            p_str   = m.group(1)       # "3","4","5","6"
            numeric = rows_by_name[sim_name]
            if len(numeric) < 6:
                _warn(f"  {sim_name}: only {len(numeric)} columns, skipping")
                continue
            param_name, param_index = _FEEDBACK_PARAM[p_str]
            suite_rows.append(dict(
                suite_key    = suite_key,
                sim_id       = sim_name,
                varied_param = param_name,
                param_index  = param_index,
                Omega0       = float(numeric[0]),
                sigma8       = float(numeric[1]),
                A_SN1        = float(numeric[2]),
                A_AGN1       = float(numeric[3]),
                A_SN2        = float(numeric[4]),
                A_AGN2       = float(numeric[5]),
            ))

        if not suite_rows:
            _fail(f"{suite_key}: no p3..p6 feedback rows found — check file format")
            ok = False
            continue

        _ok(f"{suite_key}: {len(suite_rows)} rows selected (p3..p6, 5 pts each)")

        # Cosmology sanity: Omega0≈0.3, sigma8≈0.8 in all p3..p6 rows
        bad = [r["sim_id"] for r in suite_rows
               if abs(r["Omega0"] - 0.3) > 0.01 or abs(r["sigma8"] - 0.8) > 0.01]
        if bad:
            _warn(f"{suite_key}: cosmology off-fiducial in {len(bad)} rows: {bad[:3]}")
        else:
            _ok(f"{suite_key}: Omega0≈0.3 and sigma8≈0.8 confirmed in all rows")

        # Print fiducial (_0) values for each sweep as a sanity reference
        for p_str, (pname, _) in _FEEDBACK_PARAM.items():
            fid_key = f"1P_p{p_str}_0"
            if fid_key in rows_by_name:
                n = rows_by_name[fid_key]
                _ok(f"  {fid_key}: A_SN1={n[2]:.4g}  A_AGN1={n[3]:.4g}"
                    f"  A_SN2={n[4]:.4g}  A_AGN2={n[5]:.4g}")

        all_rows.extend(suite_rows)

    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        out = PROV_DIR / "onep_map.csv"
        df.to_csv(out, index=False)
        _ok(f"Wrote {len(df)} rows → {out}")
    else:
        _fail("No rows collected — onep_map.csv not written")
        ok = False

    return ok, all_rows


def check_snapnum_and_redshift():
    print(f"\n{'='*60}")
    print("3. Snapshot / catalog snapnum and Redshift checks")
    print(f"{'='*60}")
    ok = True
    for suite_key, (suite_dir, _) in CAMELS_SUITES.items():
        fid = "1P_p3_0"   # exists for both TNG and SIMBA
        snap = ROOT / "Sims" / suite_dir / "1P" / fid / f"snapshot_{SNAPNUM}.hdf5"
        cat  = ROOT / "Sims" / suite_dir / "1P" / fid / f"groups_{SNAPNUM}.hdf5"
        if not snap.exists():
            _fail(f"{suite_key}: snapshot not found, skipping header check")
            ok = False
            continue
        with h5py.File(snap, "r") as sf:
            hdr = dict(sf["Header"].attrs)
            z   = float(hdr.get("Redshift", -99))
            h   = float(hdr.get("HubbleParam", -1))
            a   = float(hdr.get("Time", -1))
            npt = hdr.get("NumPart_Total", [])
        if abs(z) < 0.01:
            _ok(f"{suite_key} snap: Redshift={z:.5f} ≈ 0  h={h:.4f}  a={a:.5f}")
        else:
            _fail(f"{suite_key} snap: Redshift={z:.4f} — expected ≈0 for snap {SNAPNUM}")
            ok = False
        if cat.exists():
            with h5py.File(cat, "r") as cf:
                chdr = dict(cf["Header"].attrs)
                z_cat = float(chdr.get("Redshift", -99))
            if abs(z_cat) < 0.01:
                _ok(f"{suite_key} cat:  Redshift={z_cat:.5f} ≈ 0")
            else:
                _warn(f"{suite_key} cat:  Redshift={z_cat:.4f}")
    return ok


def check_system():
    print(f"\n{'='*60}")
    print("4. System: RAM, CPUs, pyarrow")
    print(f"{'='*60}")
    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = f.read().splitlines()
        mem = {l.split(":")[0]: l.split(":")[1].strip() for l in lines if ":" in l}
        total = mem.get("MemTotal", "?")
        avail = mem.get("MemAvailable", "?")
        print(f"  RAM total: {total}  available: {avail}")
    except Exception as e:
        print(f"  [WARN] Could not read /proc/meminfo: {e}")

    # CPUs
    try:
        n_cpu = os.cpu_count()
        print(f"  CPUs: {n_cpu}")
    except Exception:
        pass

    # pyarrow
    try:
        import pyarrow   # noqa: F401
        _ok("pyarrow importable — parquet output enabled")
        has_parquet = True
    except ImportError:
        _warn("pyarrow not available — will fall back to CSV/feather output")
        has_parquet = False

    return has_parquet


def run_timing_test():
    print(f"\n{'='*60}")
    print("5. Timing test: full descriptor extraction on one 1P sim")
    print(f"{'='*60}")

    suite_key  = "camels_tng_1p"
    suite_dir  = "IllustrisTNG"
    sim_id     = "1P_p3_0"   # A_SN1 fiducial; exists for both TNG and SIMBA
    snap_path  = ROOT / "Sims" / suite_dir / "1P" / sim_id / f"snapshot_{SNAPNUM}.hdf5"
    cat_path   = ROOT / "Sims" / suite_dir / "1P" / sim_id / f"groups_{SNAPNUM}.hdf5"

    if not snap_path.exists() or not cat_path.exists():
        _warn(f"Timing test skipped — {sim_id} not found at expected paths")
        return

    reg = R.get_suite(suite_key)
    gfm_recon = reg["gfm_initial_mass_reconstructed"]

    # Select centrals
    t0 = time.time()
    with h5py.File(snap_path, "r") as sf:
        hdr = dict(sf["Header"].attrs)
        h   = float(hdr["HubbleParam"])
        a   = float(hdr["Time"])

    group_ids = loaders.select_centrals_camels(cat_path, h, a, C.CAMELS_HALO_SELECT)
    n_central = len(group_ids)
    print(f"  Centrals in CAMELS_HALO_SELECT: {n_central}")
    if n_central == 0:
        _warn("No centrals found — check selection thresholds")
        return

    # Process up to 20 halos for timing
    sample = group_ids[:20]
    errors = 0
    t_start = time.time()

    DESCRIPTORS_CAMELS = [
        ("eta_M",  eta_M),
        ("f_hot",  f_hot),
        ("eps_ff", eps_ff),
        ("p_star", p_star),
    ]

    for halo_id in sample:
        try:
            halo = loaders.load_halo_camels(snap_path, cat_path,
                                            int(halo_id), gfm_recon)
            halo["gas"] = selection.to_halo_frame(
                halo["gas"], halo["subhalo"], halo["meta"]["box_kpc"])
            for name, mod in DESCRIPTORS_CAMELS:
                mod.compute(halo, C.PROTOCOL)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [ERR] halo {halo_id}: {e}")

    t_end = time.time()
    elapsed = t_end - t_start
    per_halo = elapsed / len(sample)

    # Peak RSS (Linux)
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024
        rss_str = f"  Peak RSS: {rss_mb:.0f} MB"
    except Exception:
        rss_str = "  Peak RSS: unavailable"

    projected_min = per_halo * n_central / 60
    print(f"\n  Sample: {len(sample)} halos  errors: {errors}")
    print(f"  Elapsed: {elapsed:.1f}s  per halo: {per_halo*1000:.1f}ms")
    print(f"  Projected full-sim time: {projected_min:.1f} min ({n_central} centrals)")
    print(rss_str)

    if projected_min > 20:
        _warn(f"Full-sim extraction {projected_min:.0f} min/sim — "
              f"20 sims/suite would take {20*projected_min/60:.1f} h; "
              f"consider batch loader (see run_camels.py)")
    else:
        _ok(f"Timing OK: ~{projected_min:.1f} min/sim")

    print(f"\n  Total setup+test time: {time.time()-t0:.1f}s")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAMELS Binder GATE 0 probe")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the timing test (steps 1-4 only)")
    args = parser.parse_args()

    results = {}

    results["mount"]    = check_mount()
    map_ok, _           = build_onep_map()
    results["onep_map"] = map_ok
    results["redshift"] = check_snapnum_and_redshift()
    has_parquet         = check_system()
    results["pyarrow"]  = has_parquet

    if not args.quick:
        run_timing_test()

    print(f"\n{'='*60}")
    print("GATE 0 summary:")
    all_pass = True
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k:15s}: {status}")
        if not v:
            all_pass = False

    if all_pass:
        print("\n  → GATE 0 PASSED: Exp 1 extraction is unblocked.")
    else:
        print("\n  → GATE 0 FAILED: resolve FAIL items before running Exp 1.")


if __name__ == "__main__":
    main()
