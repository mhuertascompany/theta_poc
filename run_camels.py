#!/usr/bin/env python3
"""
run_camels.py — RESUMABLE per-sim descriptor extraction for CAMELS.

Runs on the Binder (Stage A) after probe_binder.py has passed GATE 0 and
provenance/onep_map.csv has been written.

Usage
-----
    python run_camels.py camels_tng_1p
    python run_camels.py camels_simba_1p
    python run_camels.py camels_tng_1p --sim_ids 1P_0 1P_p3_n2   # specific sims
    python run_camels.py camels_tng_lh  --set_name LH --max_sims 200

If a sim's output parquet already exists it is SKIPPED — rerun safely after
a session dies mid-run.

Output
------
outputs/camels/<suite_key>/<sim_id>_p<proto_hash>.parquet
One parquet per sim; one row per central halo passing CAMELS_HALO_SELECT.

Parquet columns
---------------
All halo-level descriptors + provenance: suite_key, sim_id, varied_param,
param_index, A_SN1, A_AGN1, A_SN2, A_AGN2, Omega0, sigma8, git_commit,
protocol_hash, camels_release, n_halos_pivot.

run_camels.py is pure iteration and I/O — no physics, no formulas, no masks.
All physics lives in descriptors/ and selection.py.

AGN-channel descriptors (f_duty, mode_balance) are OFF for CAMELS:
too few massive BHs at 25 Mpc/h to compute reliable statistics.
State this in Figure P1 caption.
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

# hdf5plugin must be imported before h5py reads CAMELS files
try:
    import hdf5plugin
except ImportError:
    print("[WARN] hdf5plugin not available — BLOSC-compressed files may fail")

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import registry as R
import loaders
import selection
import config as C
import units as U
from descriptors import eta_M, f_hot, eps_ff, p_star

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_ROOT = _HERE / "outputs" / "camels"

# ── descriptors to run for CAMELS (f_duty + mode_balance OFF) ────────────────
CAMELS_DESCRIPTORS = [
    ("eta_M",  eta_M),
    ("f_hot",  f_hot),
    ("eps_ff", eps_ff),
    ("p_star", p_star),
]

CAMELS_RELEASE = "CAMELS_PUBLIC_DR3"   # update if Binder mounts a different release


# ── helpers ───────────────────────────────────────────────────────────────────

def _protocol_hash():
    s = json.dumps({**C.PROTOCOL, **C.CAMELS_HALO_SELECT}, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _snap_path(reg, sim_id):
    return (pathlib.Path(reg["snap_root"])
            / "Sims" / reg["suite_dir"] / reg["set_name"]
            / sim_id / "snapshot_090.hdf5")


def _cat_path(reg, sim_id):
    return (pathlib.Path(reg["snap_root"])
            / "Sims" / reg["suite_dir"] / reg["set_name"]
            / sim_id / "groups_090.hdf5")


# ── per-sim runner ────────────────────────────────────────────────────────────

def run_sim(suite_key, sim_row, reg, proto_hash, git_commit, out_dir):
    """
    Extract descriptors for all central halos in one sim.

    Reads the snapshot ONCE into memory (load_snapshot_camels), then does
    all sphere-masks in RAM (extract_halo_from_snapshot).  This is ~50-100×
    faster than re-opening the HDF5 file per halo.

    Parquet is written atomically: rows go to <sim_id>.tmp.parquet first,
    then renamed to the final name.  A session dying mid-sim leaves a .tmp
    file that gets cleaned up on the next run.

    Returns (n_halos, n_pivot, n_errors) or (None, None, None) if skipped.
    """
    sim_id    = str(sim_row["sim_id"])
    out_path  = out_dir / f"{sim_id}_p{proto_hash}.parquet"
    tmp_path  = out_dir / f"{sim_id}_p{proto_hash}.tmp.parquet"

    if out_path.exists():
        print(f"  [SKIP] {sim_id} — parquet exists")
        return None, None, None

    # clean up any leftover .tmp from a previous crashed run
    if tmp_path.exists():
        tmp_path.unlink()

    snap = _snap_path(reg, sim_id)
    cat  = _cat_path(reg, sim_id)

    if not snap.exists() or not cat.exists():
        print(f"  [WARN] {sim_id} — snapshot or catalog missing, skipping")
        return 0, 0, 1

    gfm_recon = reg["gfm_initial_mass_reconstructed"]

    # ── load snapshot once into memory ────────────────────────────────────────
    t_load = time.time()
    print(f"  [LOAD] {sim_id} ...", end="", flush=True)
    raw = loaders.load_snapshot_camels(snap, cat)
    print(f" {time.time()-t_load:.1f}s", flush=True)

    group_ids = loaders.select_centrals_from_snapshot(raw, C.CAMELS_HALO_SELECT)

    if len(group_ids) == 0:
        print(f"  [WARN] {sim_id} — no centrals in CAMELS_HALO_SELECT range")
        pd.DataFrame([]).to_parquet(out_path, index=False)
        return 0, 0, 0

    # ── halo loop (in-memory sphere masks) ────────────────────────────────────
    rows   = []
    errors = []
    t_loop = time.time()

    for halo_id in group_ids:
        try:
            halo = loaders.extract_halo_from_snapshot(raw, int(halo_id), gfm_recon)
            halo["gas"] = selection.to_halo_frame(
                halo["gas"], halo["subhalo"], halo["meta"]["box_kpc"])

            sub  = halo["subhalo"]
            bh   = halo["bh"]
            M_BH = float(bh["mass"].max()) if len(bh["mass"]) > 0 else 0.0
            SFR_halo = float(
                halo["gas"]["sfr"][
                    selection.aperture_mask(halo["gas"], sub["r200c"], C.PROTOCOL)
                ].sum()
            )

            row = dict(
                suite_key     = suite_key,
                halo_id       = int(halo_id),
                subhalo_id    = int(halo["meta"]["subhalo_id"]),
                logM200c      = float(np.log10(sub["m200c"])),
                M200c         = float(sub["m200c"]),
                R200c         = float(sub["r200c"]),
                M_BH          = M_BH,
                logM_BH       = float(np.log10(M_BH)) if M_BH > 0 else np.nan,
                SFR_halo      = SFR_halo,
                n_gas         = len(halo["gas"]["mass"]),
                sim_id        = sim_id,
                varied_param  = str(sim_row["varied_param"]),
                param_index   = str(sim_row.get("param_index", "")),
                Omega0        = float(sim_row["Omega0"]),
                sigma8        = float(sim_row["sigma8"]),
                A_SN1         = float(sim_row["A_SN1"]),
                A_AGN1        = float(sim_row["A_AGN1"]),
                A_SN2         = float(sim_row["A_SN2"]),
                A_AGN2        = float(sim_row["A_AGN2"]),
                protocol_hash = proto_hash,
                git_commit    = git_commit,
                camels_release= CAMELS_RELEASE,
            )

            for name, mod in CAMELS_DESCRIPTORS:
                res = mod.compute(halo, C.PROTOCOL)
                row[name]             = float(res.get("value", np.nan))
                row[f"{name}_n_used"] = int(res.get("n_used", 0))
                if name == "p_star":
                    row["p_star_n_recent"] = int(res.get("n_recent_stars", 0))

            rows.append(row)

        except Exception as e:
            errors.append({"halo_id": int(halo_id), "error": str(e)})
            if len(errors) <= 3:
                print(f"    [ERR] halo {halo_id}: {e}")

    df = pd.DataFrame(rows)

    pivot_mask = (
        (df["logM200c"] >= C.PIVOT["logM200_lo"]) &
        (df["logM200c"] <  C.PIVOT["logM200_hi"])
    ) if len(df) > 0 else pd.Series(dtype=bool)
    n_pivot = int(pivot_mask.sum())
    if len(df) > 0:
        df["n_halos_pivot"] = n_pivot

    # atomic write: tmp → final
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(out_path)

    elapsed_loop = time.time() - t_loop
    elapsed_total = time.time() - t_load
    print(f"  [DONE] {sim_id}  n_halos={len(df):3d}  n_pivot={n_pivot:3d}"
          f"  errors={len(errors)}"
          f"  load={time.time()-t_load-elapsed_loop:.1f}s"
          f"  loop={elapsed_loop:.1f}s"
          f"  total={elapsed_total:.1f}s"
          f"  → {out_path.name}")

    if errors:
        err_path = out_path.with_suffix(".errors.json")
        with open(err_path, "w") as ef:
            json.dump(errors, ef, indent=2)

    return len(df), n_pivot, len(errors)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAMELS resumable descriptor runner")
    parser.add_argument("suite_key",
                        help="Registry key, e.g. camels_tng_1p or camels_simba_1p")
    parser.add_argument("--onep_map", default="provenance/onep_map.csv",
                        help="Path to onep_map.csv from probe_binder.py")
    parser.add_argument("--sim_ids", nargs="*", default=None,
                        help="Restrict to specific sim_ids (default: all in onep_map)")
    parser.add_argument("--set_name", default=None,
                        help="Override set_name in registry (e.g. LH, CV)")
    parser.add_argument("--max_sims", type=int, default=None,
                        help="Cap number of sims (for testing or LH subsets)")
    args = parser.parse_args()

    reg = R.get_suite(args.suite_key)
    if args.set_name:
        reg = {**reg, "set_name": args.set_name}

    proto_hash = _protocol_hash()
    git_commit = _git_commit()

    print(f"Suite      : {args.suite_key}")
    print(f"Set        : {reg['set_name']}")
    print(f"Proto hash : {proto_hash}")
    print(f"Git commit : {git_commit}")
    print(f"CAMELS rel.: {CAMELS_RELEASE}")

    # Load onep_map
    onep_path = pathlib.Path(args.onep_map)
    if not onep_path.exists():
        # Fall back relative to script dir
        onep_path = _HERE / args.onep_map
    if not onep_path.exists():
        print(f"[FAIL] onep_map not found at {args.onep_map}")
        print("       Run probe_binder.py first to generate provenance/onep_map.csv")
        sys.exit(1)

    onep_df = pd.read_csv(onep_path)
    # Filter to this suite and set
    suite_rows = onep_df[onep_df["suite_key"] == args.suite_key].copy()
    if len(suite_rows) == 0:
        # For LH/CV sets, onep_map won't have rows — build minimal sim list
        # from directory scan instead.
        base = (pathlib.Path(reg["snap_root"])
                / "Sims" / reg["suite_dir"] / reg["set_name"])
        if not base.exists():
            print(f"[FAIL] No rows in onep_map for {args.suite_key} and "
                  f"directory {base} not found")
            sys.exit(1)
        sim_dirs = sorted(d.name for d in base.iterdir() if d.is_dir())
        print(f"[INFO] {len(sim_dirs)} sim directories found in {base}")
        suite_rows = pd.DataFrame([
            dict(suite_key=args.suite_key, sim_id=s, varied_param="",
                 param_index="", Omega0=0.3, sigma8=0.8,
                 A_SN1=np.nan, A_AGN1=np.nan, A_SN2=np.nan, A_AGN2=np.nan)
            for s in sim_dirs
        ])

    if args.sim_ids:
        suite_rows = suite_rows[suite_rows["sim_id"].isin(args.sim_ids)]
        print(f"Restricted to {len(suite_rows)} sim(s): {list(suite_rows['sim_id'])}")

    if args.max_sims is not None:
        suite_rows = suite_rows.head(args.max_sims)
        print(f"[TEST] Capped at {args.max_sims} sims")

    # Output directory
    out_dir = OUTPUT_ROOT / args.suite_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Sims to run: {len(suite_rows)}\n")

    # Main loop
    t_total = time.time()
    total_halos = total_pivot = total_errors = 0
    n_skipped = n_done = 0

    for _, sim_row in suite_rows.iterrows():
        n_h, n_p, n_e = run_sim(
            args.suite_key, sim_row, reg, proto_hash, git_commit, out_dir)
        if n_h is None:
            n_skipped += 1
        else:
            n_done += 1
            total_halos  += n_h or 0
            total_pivot  += n_p or 0
            total_errors += n_e or 0

    elapsed = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"Done.  {n_done} sims extracted, {n_skipped} skipped.")
    print(f"Total halos: {total_halos}  pivot halos: {total_pivot}")
    print(f"Total errors: {total_errors}  wall-time: {elapsed/60:.1f} min")
    print(f"Parquets in: {out_dir}")


if __name__ == "__main__":
    main()
