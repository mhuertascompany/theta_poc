"""
obs_proxies.py — Observable proxies at fixed stellar mass and redshift.

For the primary triplet (TNG100-1, Eagle100-1, Simba25-1) at z ≈ 1,
computes per central galaxy:

  r_half  [kpc, physical]  3D stellar half-mass radius from subhalo catalog
                           (SubhaloHalfmassRadType[:,4]; includes wind PT4
                           in TNG, which has a small effect at z~1)

  Z_star  [dimensionless]  Mass-weighted stellar metallicity (metal mass fraction)
                           from SubhaloStarMetallicity.  Divide by Z_sun=0.0127
                           (Asplund+09) for solar units.

  Y_SZ    [Mpc²]           Integrated Compton-Y within R_200c
    Y · D_A² = (σ_T / m_e c²) × Σ_i (m_i X_H x_{e,i} / m_p) k_B T_i
    Sum over non-SF gas only — EoS gas temperature is unphysical for tSZ.

Selection
---------
  Central subhalos only  (GroupFirstSub ≥ 0)
  logM_star ∈ [logM_star_lo, logM_star_hi]   (default 10.5–11.0)
  GroupLenType[:,0] ≥ min_n_gas               (default 50)

Snapshot lookup
---------------
  TNG100-1   : snap 50,  z ≈ 0.99  (verify: header Time ≈ 0.503)
  Eagle100-1 : snap 15,  z ≈ 1.00  (verify: header Time ≈ 0.499)
  Simba25-1  : NOT in registry — pass --snap explicitly
               (find snap where header Time ≈ 0.500, then add to registry)

Usage (on cluster via Slurm)
----------------------------
    python obs_proxies.py TNG100-1
    python obs_proxies.py Eagle100-1
    python obs_proxies.py Simba25-1  --snap <N>   # N from verified header
    python obs_proxies.py TNG100-1   --max_gals 100   # quick smoke-test

Output
------
    tables/obs/{suite_key}_snap{snap:03d}_p{hash}.parquet

    Columns: suite, group_id, sub_id, snap, z_actual,
             logM200c, logM_star, r_half_kpc, Y_SZ_Mpc2, n_gas_Y,
             n_gas_group, r200c_kpc
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
import config as C
import units as U

# ── Compton-Y constants ───────────────────────────────────────────────────────
# m_e c² derived from units.py constants to avoid a new import
_M_E_C2     = 9.10938e-28 * U._C**2   # [erg]  ≈ 8.187e-7 erg
_CM_PER_MPC = 3.0857e24               # [cm/Mpc]

# ── selection parameters ──────────────────────────────────────────────────────
OBS_SELECT = dict(
    logM_star_lo = 10.5,
    logM_star_hi = 11.0,
    min_n_gas    = 50,     # softer floor than HALO_SELECT; lower-mass halos at z~1
)

TABLES = pathlib.Path(__file__).parent / "tables" / "obs"
TABLES.mkdir(parents=True, exist_ok=True)


def _il():
    import illustris_python as il
    return il


# ── Compton-Y ─────────────────────────────────────────────────────────────────

def compton_y_mpc2(gas, r200c, cfg):
    """
    Integrated Compton-Y × D_A² [Mpc²] for non-SF gas within R_200c.

    Y [cm²] = (σ_T / m_e c²) × Σ_i  N_{e,i} · k_B · T_i
    where N_{e,i} = m_i X_H x_{e,i} / m_p  (free electrons in particle i).

    Requires gas dict enriched with 'r' by selection.to_halo_frame().
    SF gas (sfr > 0) is excluded — EoS temperature is unphysical for tSZ.

    Returns (Y_Mpc2, n_particles_used).
    """
    mask = (gas["r"] < r200c) & (gas["sfr"] == 0.0)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0

    T_K = U.gas_temperature(
        gas["u"][mask], gas["xe"][mask], cfg["gamma"], cfg["X_H"])

    # number of free electrons per particle [dimensionless]
    m_g = gas["mass"][mask].astype(np.float64) * U._M_SUN   # [g]
    N_e = m_g * cfg["X_H"] * gas["xe"][mask].astype(np.float64) / U._M_P

    Y_cm2  = float((U._SIGMA_T / _M_E_C2) * np.sum(N_e * U._K_B * T_K))
    Y_Mpc2 = Y_cm2 / _CM_PER_MPC**2

    return Y_Mpc2, n


# ── catalog loader ─────────────────────────────────────────────────────────────

def load_obs_catalog(path, snap, h, a):
    """
    Group + subhalo catalog with M_star and r_half.  All in physical units.

    SubhaloMassType[:,4] includes wind PT4 in TNG.  At z~1 the wind
    fraction is small but the caveat should be noted in the paper.
    """
    il = _il()

    grp = il.groupcat.loadHalos(path, snap, fields=[
        "Group_M_Crit200", "Group_R_Crit200", "GroupPos",
        "GroupFirstSub", "GroupLenType",
    ])
    sub = il.groupcat.loadSubhalos(path, snap, fields=[
        "SubhaloPos", "SubhaloVel",
        "SubhaloMassType", "SubhaloHalfmassRadType",
        "SubhaloStarMetallicity",
    ])

    # GroupLenType: always a 2-D array (N_groups, 6) when multiple fields loaded
    glen = grp["GroupLenType"]

    return dict(
        # group-indexed
        m200c     = U.code_mass_to_msun(grp["Group_M_Crit200"], h),
        r200c     = U.comoving_to_physical(grp["Group_R_Crit200"], a, h),
        first_sub = grp["GroupFirstSub"].astype(int),
        n_gas     = glen[:, 0].astype(int),
        # subhalo-indexed (look up via first_sub)
        m_star    = U.code_mass_to_msun(
                        sub["SubhaloMassType"][:, 4].astype(np.float64), h),
        r_half    = U.comoving_to_physical(
                        sub["SubhaloHalfmassRadType"][:, 4].astype(np.float64),
                        a, h),
        Z_star    = sub["SubhaloStarMetallicity"].astype(np.float64),
        sub_pos   = U.comoving_to_physical(sub["SubhaloPos"], a, h),
        sub_vel   = sub["SubhaloVel"].astype(np.float64),
    )


# ── selection ─────────────────────────────────────────────────────────────────

def select_galaxies(cat, sel):
    """
    Group indices for centrals with M_star in the target range.

    Maps group → central subhalo via GroupFirstSub, looks up SubhaloMassType.
    """
    fs  = cat["first_sub"]
    n_s = len(cat["m_star"])

    valid = (fs >= 0) & (fs < n_s)
    safe  = np.where(valid, fs, 0)

    m_star = np.where(valid, cat["m_star"][safe], 0.0)
    r_half = np.where(valid, cat["r_half"][safe], 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.where(m_star > 0, np.log10(m_star), np.nan)

    mask = (
        valid &
        np.isfinite(logm) &
        (logm >= sel["logM_star_lo"]) &
        (logm <= sel["logM_star_hi"]) &
        (cat["n_gas"] >= sel["min_n_gas"]) &
        (r_half > 0)    # exclude subhalos with no measured stellar extent
    )
    return np.where(mask)[0]


# ── per-galaxy processing ─────────────────────────────────────────────────────

def process_galaxy(group_id, path, snap, cat, h, a, box_ckpch, cfg):
    """
    Load gas for one FoF group, compute Y_SZ.  Pair with catalog r_half.

    Returns flat dict row, or None if the group has no gas particles.
    """
    import selection as S
    il = _il()

    fs      = int(cat["first_sub"][group_id])
    r200c   = float(cat["r200c"][group_id])
    m200c   = float(cat["m200c"][group_id])
    m_star  = float(cat["m_star"][fs])
    r_half  = float(cat["r_half"][fs])
    Z_star  = float(cat["Z_star"][fs])
    sub_pos = cat["sub_pos"][fs]
    sub_vel = cat["sub_vel"][fs]
    box_kpc = U.comoving_to_physical(box_ckpch, a, h)

    # load all PartType0 in the FoF group (group-membership offsets, not IDs)
    gas_raw = il.snapshot.loadHalo(
        path, snap, group_id, 0,
        fields=["Coordinates", "Velocities", "Masses",
                "InternalEnergy", "ElectronAbundance", "StarFormationRate"],
    )

    if not isinstance(gas_raw, dict) or len(gas_raw.get("Masses", [])) == 0:
        return None

    gas = dict(
        pos  = U.comoving_to_physical(gas_raw["Coordinates"], a, h),
        vel  = U.code_vel_to_kms(gas_raw["Velocities"], a),
        mass = U.code_mass_to_msun(gas_raw["Masses"], h),
        u    = gas_raw["InternalEnergy"].astype(np.float64),
        xe   = gas_raw["ElectronAbundance"].astype(np.float64),
        sfr  = gas_raw["StarFormationRate"].astype(np.float64),
    )

    # recentre on central subhalo; enriches gas dict with 'r'
    gas = S.to_halo_frame(gas, {"pos": sub_pos, "vel": sub_vel}, box_kpc)

    Y_Mpc2, n_gas_Y = compton_y_mpc2(gas, r200c, cfg)

    return dict(
        suite       = "",         # filled by caller
        group_id    = int(group_id),
        sub_id      = fs,
        snap        = snap,
        z_actual    = np.nan,     # filled by caller
        logM200c    = float(np.log10(m200c)) if m200c > 0 else np.nan,
        logM_star   = float(np.log10(m_star)) if m_star > 0 else np.nan,
        r_half_kpc  = r_half,
        Z_star      = Z_star,       # raw metal mass fraction (divide by 0.0127 for Z/Zsun)
        Y_SZ_Mpc2   = Y_Mpc2,
        n_gas_Y     = n_gas_Y,
        n_gas_group = int(cat["n_gas"][group_id]),
        r200c_kpc   = r200c,
    )


# ── main runner ───────────────────────────────────────────────────────────────

def run(suite_key, snap_override=None, max_gals=None):
    reg  = R.get_suite(suite_key)
    path = reg["path"]

    if snap_override is not None:
        snap = snap_override
    else:
        try:
            snap = R.snapnum_for_z(suite_key, 1.0)
        except NotImplementedError as e:
            sys.exit(
                f"[ERROR] {e}\n"
                "Pass --snap <snapnum> (verify from cluster: header Time ≈ 0.500)."
            )

    sel_str  = json.dumps(
        {**OBS_SELECT, "gamma": C.PROTOCOL["gamma"], "X_H": C.PROTOCOL["X_H"]},
        sort_keys=True)
    phash    = hashlib.md5(sel_str.encode()).hexdigest()[:8]
    out_path = TABLES / f"{suite_key}_snap{snap:03d}_p{phash}.parquet"

    print(f"Suite    : {suite_key}")
    print(f"Snapshot : {snap}")
    print(f"Hash     : {phash}")
    print(f"Output   : {out_path}")

    # ── header ────────────────────────────────────────────────────────────────
    snap_file = _il().snapshot.snapPath(path, snap)
    with h5py.File(snap_file, "r") as f:
        hdr = dict(f["Header"].attrs)

    h         = float(hdr["HubbleParam"])
    a         = float(hdr["Time"])
    box_ckpch = float(hdr["BoxSize"])
    z_actual  = 1.0 / a - 1.0

    print(f"h={h:.4f}  a={a:.5f}  z={z_actual:.4f}  box={box_ckpch:.0f} ckpc/h")

    # ── catalog and selection ─────────────────────────────────────────────────
    cat  = load_obs_catalog(path, snap, h, a)
    gids = select_galaxies(cat, OBS_SELECT)

    lo = OBS_SELECT["logM_star_lo"]
    hi = OBS_SELECT["logM_star_hi"]
    print(f"Centrals with logM_star ∈ [{lo}, {hi}]: {len(gids):,}")

    if max_gals is not None:
        gids = gids[:max_gals]
        print(f"[TEST MODE] capped at {max_gals}")

    # ── galaxy loop ───────────────────────────────────────────────────────────
    rows, errors = [], []
    t0 = time.time()

    for i, gid in enumerate(gids):
        try:
            row = process_galaxy(gid, path, snap, cat, h, a, box_ckpch, C.PROTOCOL)
            if row is not None:
                row["suite"]    = suite_key
                row["z_actual"] = z_actual
                rows.append(row)
        except Exception as e:
            errors.append(dict(group_id=int(gid), error=str(e)))
            if len(errors) <= 5:
                print(f"  [ERROR] group {gid}: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == len(gids):
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-9)
            eta_s   = (len(gids) - i - 1) / max(rate, 1e-9)
            print(f"  {i+1:5d}/{len(gids)}"
                  f"  elapsed={elapsed/60:.1f}m"
                  f"  rate={rate:.1f}/s"
                  f"  ETA={eta_s/60:.1f}m"
                  f"  errors={len(errors)}")

    # ── write ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nWrote {len(df):,} rows → {out_path}")
    print(f"Errors: {len(errors)}  |  Total time: {elapsed/60:.1f} min")

    if len(df) > 0:
        print(
            f"\nSummary:\n"
            + df[["logM_star", "r_half_kpc", "Y_SZ_Mpc2", "n_gas_Y"]]
              .describe().to_string()
        )

    if errors:
        err_path = out_path.with_suffix(".errors.json")
        with open(err_path, "w") as ef:
            json.dump(errors, ef, indent=2)
        print(f"Error log: {err_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Observable proxies (r_half, Y_SZ) at fixed M_star, z~1")
    parser.add_argument("suite_key",
                        help="e.g. TNG100-1, Eagle100-1, Simba25-1")
    parser.add_argument("--snap", type=int, default=None,
                        help="Snapshot override (verify header Time ≈ 0.500 for z≈1)")
    parser.add_argument("--max_gals", type=int, default=None,
                        help="Cap number of galaxies for quick smoke-test")
    args = parser.parse_args()
    run(args.suite_key, snap_override=args.snap, max_gals=args.max_gals)


if __name__ == "__main__":
    main()
