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

    # Spot-check one fiducial sim
    for suite_key, (suite_dir, _) in CAMELS_SUITES.items():
        snap = ROOT / "Sims" / suite_dir / "1P" / "1P_0" / f"snapshot_{SNAPNUM}.hdf5"
        cat  = ROOT / "Sims" / suite_dir / "1P" / "1P_0" / f"groups_{SNAPNUM}.hdf5"
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


def _parse_cosmoastro(fpath):
    """
    Parse a CosmoAstroSeed txt file.  Returns (header_cols, data_array).
    Skips lines starting with '#'; treats first non-comment line as data.
    Tries to detect a header line (first token non-numeric).
    """
    lines = fpath.read_text().splitlines()
    header_cols = None
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # might be a header comment: "# Omega_m sigma_8 ..."
            tokens = line.lstrip("#").split()
            if tokens and not _is_numeric(tokens[0]):
                header_cols = tokens
            continue
        tokens = line.split()
        if not _is_numeric(tokens[0]):
            # header row embedded in data
            header_cols = tokens
            continue
        rows.append([float(t) for t in tokens])
    return header_cols, np.array(rows)


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _sim_ids_for_suite(suite_dir):
    """Return sorted list of 1P sim directory names that exist on disk."""
    base = ROOT / "Sims" / suite_dir / "1P"
    if not base.exists():
        return []
    dirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return dirs


def build_onep_map():
    """
    Read CosmoAstroSeed files and build provenance/onep_map.csv.

    For each suite, identifies:
    - which row corresponds to the fiducial (1P_0)
    - which feedback parameter varies in each row (p3..p6)
    - the exact parameter values from the file

    Returns list of row dicts; writes CSV.
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
            _fail(f"{filename} not found in any candidate location for {suite_key}")
            ok = False
            continue
        _ok(f"Found {fpath.relative_to(ROOT) if ROOT in fpath.parents else fpath}")

        header_cols, data = _parse_cosmoastro(fpath)

        # We need at least 6 columns (p1..p6)
        if data.shape[1] < 6:
            _fail(f"{suite_key}: only {data.shape[1]} columns — expected ≥6")
            ok = False
            continue

        # Extract only the first 6 parameter columns
        params = data[:, :6]   # Omega0, sigma8, A_SN1, A_AGN1, A_SN2, A_AGN2

        # Identify the fiducial row: all feedback params (cols 2-5) closest to
        # each other in relative variance — simplest: Omega0≈0.3, sigma8≈0.8
        # OR row where all feedback multipliers equal (variation = 0 across cols 2-5).
        # Most robust: pick row where variance of feedback cols is minimal.
        feedback_vals = params[:, 2:]   # (N, 4)
        fid_candidates = np.where(
            np.all(np.abs(feedback_vals - feedback_vals[0]) < 1e-6, axis=1)
        )[0]
        if len(fid_candidates) == 0:
            # Fall back: row with minimum std across feedback cols
            fid_idx = int(np.argmin(feedback_vals.std(axis=1)))
            _warn(f"{suite_key}: no identical-fiducial row; using row {fid_idx} as fiducial")
        else:
            fid_idx = int(fid_candidates[0])
        fid_vals = params[fid_idx, :]
        _ok(f"{suite_key}: fiducial row {fid_idx}  "
            f"Ω0={fid_vals[0]:.3f} σ8={fid_vals[1]:.3f} "
            f"A_SN1={fid_vals[2]:.3f} A_AGN1={fid_vals[3]:.3f} "
            f"A_SN2={fid_vals[4]:.3f} A_AGN2={fid_vals[5]:.3f}")

        # Check cosmology fixed in all rows
        cos_ok = True
        for i, row in enumerate(params):
            if abs(row[0] - fid_vals[0]) > 1e-4 or abs(row[1] - fid_vals[1]) > 1e-4:
                if i != fid_idx:
                    _warn(f"  row {i}: Omega0={row[0]:.4f} sigma8={row[1]:.4f}"
                          " deviates from fiducial — may be p1/p2 row, skip")
                    cos_ok = False
        if cos_ok:
            _ok(f"{suite_key}: Omega0 and sigma8 are fiducial across all rows")

        # Get available sim directories
        sim_dirs = _sim_ids_for_suite(suite_dir)
        _ok(f"{suite_key}: {len(sim_dirs)} sim directories found: {sim_dirs}")

        # Map each row to a sim directory.
        # Row order in the file matches directory ordering for the 1P set:
        #   row 0 → 1P_0 (fiducial)
        #   rows 1..4  → 1P_p3_n2, 1P_p3_n1, 1P_p3_1, 1P_p3_2
        #   rows 5..8  → 1P_p4_n2, ...
        #   rows 9..12 → 1P_p5_n2, ...
        #   rows 13..16→ 1P_p6_n2, ...
        # Total = 17 rows.  Build the expected ID list.
        expected_ids = ["1P_0"]
        for pidx in [3, 4, 5, 6]:
            for var in VARIATIONS:
                expected_ids.append(f"1P_p{pidx}_{var}")

        if len(params) != len(expected_ids):
            _warn(f"{suite_key}: {len(params)} rows but expected {len(expected_ids)}"
                  " — mapping by order; check if extra rows present")

        for row_i, (sim_id, row) in enumerate(zip(expected_ids, params)):
            fb = row[2:]   # A_SN1, A_AGN1, A_SN2, A_AGN2
            fid_fb = fid_vals[2:]
            diff = np.abs(fb - fid_fb)
            n_diff = int((diff > 1e-6).sum())

            if sim_id == "1P_0":
                varied_param = "fiducial"
                param_index  = ""
            elif n_diff == 1:
                col_i = int(np.argmax(diff))
                varied_param = FEEDBACK_COLS[col_i]
                param_index  = PARAM_INDICES[varied_param]
            else:
                varied_param = "AMBIGUOUS"
                param_index  = ""
                _warn(f"  {sim_id}: {n_diff} columns differ from fiducial"
                      " — possible p1/p2 row; keeping but marking AMBIGUOUS")

            all_rows.append(dict(
                suite_key   = suite_key,
                sim_id      = sim_id,
                varied_param= varied_param,
                param_index = param_index,
                Omega0      = float(row[0]),
                sigma8      = float(row[1]),
                A_SN1       = float(row[2]),
                A_AGN1      = float(row[3]),
                A_SN2       = float(row[4]),
                A_AGN2      = float(row[5]),
            ))

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
        snap = ROOT / "Sims" / suite_dir / "1P" / "1P_0" / f"snapshot_{SNAPNUM}.hdf5"
        cat  = ROOT / "Sims" / suite_dir / "1P" / "1P_0" / f"groups_{SNAPNUM}.hdf5"
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
    sim_id     = "1P_0"
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
              "17 sims/suite would take {17*projected_min/60:.1f} h; "
              "consider capping to pivot bin only")
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
