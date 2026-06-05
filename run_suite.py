"""
run_suite.py — iterate over central galaxies and compute all 6 descriptors.

Usage:
    python run_suite.py TNG100-1
    python run_suite.py TNG100-1 --max_halos 100   # test with subset first

Output: tables/{suite}_z{z}_p{hash}.parquet

run_suite.py is pure iteration and I/O — no physics, no formulas, no masks.
All physics lives in descriptors/ and selection.py.
"""

import sys
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import pandas as pd
import h5py

import registry as R
import loaders
import selection
import config as C
import units as U
from descriptors import eta_M, f_hot, f_duty, mode_balance, eps_ff, p_star

TABLES = pathlib.Path(__file__).parent / "tables"
TABLES.mkdir(exist_ok=True)

DESCRIPTORS = [
    ("eta_M",        eta_M),
    ("f_hot",        f_hot),
    ("f_duty",       f_duty),
    ("mode_balance", mode_balance),
    ("eps_ff",       eps_ff),
    ("p_star",       p_star),
]


def _il():
    import illustris_python as il
    return il


def protocol_hash():
    s = json.dumps({**C.PROTOCOL, **C.HALO_SELECT}, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


# ── halo selection ────────────────────────────────────────────────────────────

def select_centrals(cat, path, snap):
    """Return sorted array of group_ids that pass HALO_SELECT criteria."""
    # gas particle count per group (PT0 column)
    glen = _il().groupcat.loadHalos(path, snap, fields=["GroupLenType"])
    n_gas = (glen["GroupLenType"] if isinstance(glen, dict) else glen)[:, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log10(cat["m200c"])

    mask = (
        (cat["first_sub"] >= 0) &            # has a valid central subhalo
        np.isfinite(logm) &
        (logm >= C.HALO_SELECT["logM200_min"]) &
        (logm <= C.HALO_SELECT["logM200_max"]) &
        (n_gas >= C.HALO_SELECT["min_n_gas"])
    )
    return np.where(mask)[0]


# ── per-halo processing ───────────────────────────────────────────────────────

def process_halo(halo_id, path, snap, cat, h, a, box_ckpch,
                 bh_avail, Omega_m, Omega_L, gfm_reconstructed, proto_hash,
                 suite_key):
    """Load particles, run all descriptors, return flat dict row."""
    halo = loaders.load_halo(path, snap, halo_id, cat, h, a, box_ckpch,
                             bh_avail=bh_avail)

    # inject cosmology and suite metadata needed by p_star / p★/m★
    halo["meta"]["Omega_m"] = Omega_m
    halo["meta"]["Omega_L"] = Omega_L
    halo["meta"]["gfm_initial_mass_reconstructed"] = gfm_reconstructed

    # recentre gas once; all descriptors receive the enriched dict
    halo["gas"] = selection.to_halo_frame(
        halo["gas"], halo["subhalo"], halo["meta"]["box_kpc"])

    sub = halo["subhalo"]
    bh  = halo["bh"]

    # most massive BH in central subhalo
    M_BH = float(bh["mass"].max()) if len(bh["mass"]) > 0 else 0.0

    SFR_halo = float(
        halo["gas"]["sfr"][
            selection.aperture_mask(halo["gas"], sub["r200c"], C.PROTOCOL)
        ].sum()
    )

    row = dict(
        suite         = suite_key,
        halo_id       = int(halo_id),
        subhalo_id    = int(halo["meta"]["subhalo_id"]),
        logM200c      = float(np.log10(sub["m200c"])),
        M200c         = float(sub["m200c"]),
        R200c         = float(sub["r200c"]),
        M_BH          = M_BH,
        logM_BH       = float(np.log10(M_BH)) if M_BH > 0 else np.nan,
        SFR_halo      = SFR_halo,
        n_gas         = len(halo["gas"]["mass"]),
        protocol_hash = proto_hash,
    )

    for name, mod in DESCRIPTORS:
        res = mod.compute(halo, C.PROTOCOL)
        row[name]              = float(res.get("value", np.nan))
        row[f"{name}_n_used"]  = int(res.get("n_used", 0))

        if name == "f_duty":
            row["f_duty_n_active"]  = int(res.get("n_active", 0))
            lam = res.get("lambda_edd", [])
            row["lambda_edd_max"] = float(max(lam)) if lam else np.nan
        elif name == "mode_balance":
            row["mode_balance_defined"] = bool(res.get("mode_defined", False))
        elif name == "p_star":
            row["p_star_n_recent"] = int(res.get("n_recent_stars", 0))

    return row


# ── main runner ───────────────────────────────────────────────────────────────

def run(suite_key, max_halos=None):
    reg  = R.get_suite(suite_key)
    path = reg["path"]
    snap = R.snapnum_for_z(suite_key, C.PROTOCOL["z_target"])
    hash_ = protocol_hash()

    z_tag    = f"z{C.PROTOCOL['z_target']:.1f}"
    out_path = TABLES / f"{suite_key}_{z_tag}_p{hash_}.parquet"

    print(f"Suite      : {suite_key}")
    print(f"Snapshot   : {snap}  (z={C.PROTOCOL['z_target']})")
    print(f"Proto hash : {hash_}")
    print(f"Output     : {out_path}")

    # ── header ────────────────────────────────────────────────────────────────
    snap_file = _il().snapshot.snapPath(path, snap)
    with h5py.File(snap_file, "r") as f:
        hdr = dict(f["Header"].attrs)

    h         = float(hdr["HubbleParam"])
    a         = float(hdr["Time"])
    box_ckpch = float(hdr["BoxSize"])
    Omega_m   = float(hdr.get("Omega0",      0.3089))
    Omega_L   = float(hdr.get("OmegaLambda", 0.6911))
    gfm_recon = bool(reg.get("gfm_initial_mass_reconstructed", False))

    print(f"h={h:.4f}  a={a:.4f}  Omega_m={Omega_m:.4f}")

    # ── catalog and selection ─────────────────────────────────────────────────
    cat       = loaders.load_catalog(path, snap, h, a)
    group_ids = select_centrals(cat, path, snap)

    if max_halos is not None:
        group_ids = group_ids[:max_halos]
        print(f"[TEST MODE] capped at {max_halos} halos")

    bh_avail = loaders.snap_bh_fields(path, snap)
    print(f"Centrals   : {len(group_ids):,}  |  BH fields: {sorted(bh_avail & {'BH_CumEgyInjection_QM','BH_CumEgyInjection_RM'})}")

    # ── halo loop ─────────────────────────────────────────────────────────────
    rows   = []
    errors = []
    t0     = time.time()

    for i, halo_id in enumerate(group_ids):
        try:
            row = process_halo(
                halo_id, path, snap, cat, h, a, box_ckpch,
                bh_avail, Omega_m, Omega_L, gfm_recon, hash_, suite_key,
            )
            rows.append(row)
        except Exception as e:
            errors.append(dict(halo_id=int(halo_id), error=str(e)))
            if len(errors) <= 5:
                print(f"  [ERROR] halo {halo_id}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(group_ids):
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta_s   = (len(group_ids) - i - 1) / max(rate, 1e-9)
            print(f"  {i+1:5d}/{len(group_ids)}"
                  f"  elapsed={elapsed/60:.1f}m"
                  f"  rate={rate:.1f}/s"
                  f"  ETA={eta_s/60:.1f}m"
                  f"  errors={len(errors)}")

    # ── write output ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nWrote {len(df):,} rows → {out_path}")
    print(f"Errors: {len(errors)}  |  Total time: {elapsed/60:.1f} min")

    if len(df) > 0:
        cols = ["logM200c", "eta_M", "f_hot", "f_duty",
                "mode_balance", "eps_ff", "p_star"]
        print(f"\nDescriptor summary:\n{df[cols].describe().to_string()}")

    if errors:
        err_path = out_path.with_suffix(".errors.json")
        with open(err_path, "w") as ef:
            json.dump(errors, ef, indent=2)
        print(f"Error log : {err_path}")


def main():
    parser = argparse.ArgumentParser(description="Run θ_feedback descriptor suite")
    parser.add_argument("suite_key", help="e.g. TNG100-1, Eagle100-1, Simba25-1")
    parser.add_argument("--max_halos", type=int, default=None,
                        help="Cap number of halos (for testing)")
    args = parser.parse_args()
    run(args.suite_key, max_halos=args.max_halos)


if __name__ == "__main__":
    main()
