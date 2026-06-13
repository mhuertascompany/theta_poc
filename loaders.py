"""
loaders.py — thin illustris_python wrappers + h5py CAMELS loader.

illustris_python is imported lazily (inside functions only) so this module
is importable locally without cluster access.

All returned arrays are in physical units (post-conversion via units.py).
Never hardcode h or a — always pass from the snapshot header.

Public API
----------
load_catalog(path, snapnum, h, a)                → catalog dict (physical units)
load_halo(path, snapnum, halo_id, ...)           → halo dict (physical units)
load_halo_from_hdf5(snap, groups, halo=0)        → same format, h5py only (tests)
select_centrals_camels(cat_path, h, a, sel)      → array of group indices (h5py)
load_halo_camels(snap_path, cat_path, halo, ...) → halo dict (h5py, memory-frugal)
"""

import numpy as np
import h5py
import units as U


def _il():
    import illustris_python as il
    return il


# ── field lists ───────────────────────────────────────────────────────────────
_GAS_FIELDS  = ["Coordinates", "Velocities", "Masses", "Density",
                 "InternalEnergy", "ElectronAbundance", "StarFormationRate"]
_STAR_FIELDS = ["Coordinates", "Masses", "GFM_InitialMass",
                "GFM_StellarFormationTime"]
# BH_CumEgyInjection_* absent in SIMBA; loaded conditionally via bh_avail
_BH_FIELDS_ALWAYS = ["Coordinates", "BH_Mass", "BH_Mdot"]
_BH_FIELDS_OPT    = ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]


# ── catalog ───────────────────────────────────────────────────────────────────

def load_catalog(path, snapnum, h, a):
    """
    Load Group + Subhalo catalogs once per suite run.

    Returns physical-unit dict:
      cat["m200c"]      [Msun]       shape (N_groups,)
      cat["r200c"]      [kpc]        shape (N_groups,)
      cat["gpos"]       [kpc]        shape (N_groups, 3)
      cat["first_sub"]  [int]        shape (N_groups,)
      cat["sub_pos"]    [kpc]        shape (N_sub, 3)
      cat["sub_vel"]    [km/s]       shape (N_sub, 3)   already physical
    """
    il = _il()
    grp = il.groupcat.loadHalos(path, snapnum, fields=[
        "Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub",
    ])
    sub = il.groupcat.loadSubhalos(path, snapnum, fields=[
        "SubhaloPos", "SubhaloVel",
    ])
    return dict(
        m200c     = U.code_mass_to_msun(grp["Group_M_Crit200"], h),
        r200c     = U.comoving_to_physical(grp["Group_R_Crit200"], a, h),
        gpos      = U.comoving_to_physical(grp["GroupPos"], a, h),
        first_sub = grp["GroupFirstSub"].astype(int),
        sub_pos   = U.comoving_to_physical(sub["SubhaloPos"], a, h),
        sub_vel   = sub["SubhaloVel"].astype(np.float64),  # km/s, already physical
    )


def snap_bh_fields(path, snapnum):
    """Return the set of available PartType5 field names (h5py, no data load)."""
    snap_file = _il().snapshot.snapPath(path, snapnum)
    with h5py.File(snap_file, "r") as f:
        return set(f.get("PartType5", {}).keys())


# ── per-halo loader ───────────────────────────────────────────────────────────

def load_halo(path, snapnum, halo_id, catalog, h, a, box_ckpch, bh_avail=None):
    """
    Load gas / stars / BHs for FoF halo `halo_id`.

    Parameters
    ----------
    catalog   : dict returned by load_catalog (load once per suite run)
    bh_avail  : set of available PartType5 field names (from snap_bh_fields);
                if None, queried here (one extra h5py open per halo — slow)

    Returns
    -------
    Standardised halo dict with keys gas / stars / bh / subhalo / meta,
    all arrays in physical units.
    """
    il = _il()

    subhalo_id = catalog["first_sub"][halo_id]

    if bh_avail is None:
        bh_avail = snap_bh_fields(path, snapnum)

    bh_fields = _BH_FIELDS_ALWAYS + [f for f in _BH_FIELDS_OPT if f in bh_avail]

    # ── raw particle loads (halo-scoped) ─────────────────────────────────────
    gas_raw  = il.snapshot.loadHalo(path, snapnum, halo_id, 0, fields=_GAS_FIELDS)
    star_raw = il.snapshot.loadHalo(path, snapnum, halo_id, 4, fields=_STAR_FIELDS)
    bh_raw   = il.snapshot.loadSubhalo(path, snapnum, subhalo_id, 5,
                                       fields=bh_fields)

    # ── gas (physical units) ─────────────────────────────────────────────────
    gas = dict(
        pos     = U.comoving_to_physical(gas_raw["Coordinates"],   a, h),
        vel     = U.code_vel_to_kms(gas_raw["Velocities"],         a),
        mass    = U.code_mass_to_msun(gas_raw["Masses"],           h),
        density = U.code_density_to_msun_kpc3(gas_raw["Density"],  a, h),
        u       = gas_raw["InternalEnergy"].astype(np.float64),
        xe      = gas_raw["ElectronAbundance"].astype(np.float64),
        sfr     = gas_raw["StarFormationRate"].astype(np.float64),
    )

    # ── stars: wind particles filtered out (GFM_StellarFormationTime > 0) ───
    real = star_raw["GFM_StellarFormationTime"] > 0
    stars = dict(
        pos       = U.comoving_to_physical(star_raw["Coordinates"][real],  a, h),
        mass      = U.code_mass_to_msun(star_raw["Masses"][real],          h),
        mass_init = U.code_mass_to_msun(star_raw["GFM_InitialMass"][real], h),
        a_form    = star_raw["GFM_StellarFormationTime"][real].astype(np.float64),
    )

    # ── BHs (central subhalo; empty arrays if subhalo has no BH particles) ──
    # loadSubhalo returns an empty dict {} when the subhalo has 0 PT5 particles
    if "Coordinates" not in bh_raw or len(bh_raw.get("BH_Mass", [])) == 0:
        bh = dict(
            pos    = np.zeros((0, 3)),
            mass   = np.zeros(0),
            mdot   = np.zeros(0),
            egy_qm = np.zeros(0),
            egy_rm = np.zeros(0),
        )
    else:
        n_bh = len(bh_raw["BH_Mass"])
        bh = dict(
            pos    = U.comoving_to_physical(bh_raw["Coordinates"], a, h),
            mass   = U.code_mass_to_msun(bh_raw["BH_Mass"].astype(np.float64), h),
            mdot   = U.bh_mdot_to_msun_yr(bh_raw["BH_Mdot"].astype(np.float64)),
            egy_qm = bh_raw.get("BH_CumEgyInjection_QM", np.zeros(n_bh)),
            egy_rm = bh_raw.get("BH_CumEgyInjection_RM", np.zeros(n_bh)),
        )

    box_kpc = U.comoving_to_physical(box_ckpch, a, h)

    return dict(
        gas     = gas,
        stars   = stars,
        bh      = bh,
        subhalo = dict(
            pos   = catalog["sub_pos"][subhalo_id],
            vel   = catalog["sub_vel"][subhalo_id],
            r200c = catalog["r200c"][halo_id],
            m200c = catalog["m200c"][halo_id],
        ),
        meta = dict(
            halo_id    = int(halo_id),
            subhalo_id = int(subhalo_id),
            h          = h,
            a          = a,
            box_kpc    = box_kpc,
        ),
    )


# ── fixture loader (h5py only — no illustris_python; for local tests) ────────

def load_halo_from_hdf5(snap_path, groups_path, halo_id=0):
    """
    Build a halo dict from the toy HDF5 fixtures without illustris_python.
    Returns the same structure as load_halo; used by pytest.
    """
    with h5py.File(snap_path, "r") as sf, h5py.File(groups_path, "r") as gf:
        hdr = dict(sf["Header"].attrs)
        h   = float(hdr["HubbleParam"])
        a   = float(hdr["Time"])

        # catalog
        m200c = U.code_mass_to_msun(
            float(gf["Group/Group_M_Crit200"][halo_id]), h)
        r200c = U.comoving_to_physical(
            float(gf["Group/Group_R_Crit200"][halo_id]), a, h)
        gpos  = U.comoving_to_physical(gf["Group/GroupPos"][halo_id], a, h)
        sub_pos = U.comoving_to_physical(gf["Subhalo/SubhaloPos"][halo_id], a, h)
        sub_vel = gf["Subhalo/SubhaloVel"][halo_id].astype(np.float64)
        box_kpc = U.comoving_to_physical(float(hdr["BoxSize"]), a, h)
        Omega_m = float(hdr.get("Omega0",      0.3089))
        Omega_L = float(hdr.get("OmegaLambda", 0.6911))

        # gas
        pt0 = sf["PartType0"]
        gas = dict(
            pos     = U.comoving_to_physical(pt0["Coordinates"][:], a, h),
            vel     = U.code_vel_to_kms(pt0["Velocities"][:], a),
            mass    = U.code_mass_to_msun(pt0["Masses"][:], h),
            density = U.code_density_to_msun_kpc3(pt0["Density"][:], a, h),
            u       = pt0["InternalEnergy"][:].astype(np.float64),
            xe      = pt0["ElectronAbundance"][:].astype(np.float64),
            sfr     = pt0["StarFormationRate"][:].astype(np.float64),
        )

        # stars (wind filtered)
        pt4   = sf["PartType4"]
        gform = pt4["GFM_StellarFormationTime"][:]
        real  = gform > 0
        stars = dict(
            pos       = U.comoving_to_physical(pt4["Coordinates"][:][real], a, h),
            mass      = U.code_mass_to_msun(pt4["Masses"][:][real], h),
            mass_init = U.code_mass_to_msun(pt4["GFM_InitialMass"][:][real], h),
            a_form    = gform[real].astype(np.float64),
        )

        # BHs
        pt5 = sf["PartType5"]
        n   = len(pt5["BH_Mass"])
        bh  = dict(
            pos    = U.comoving_to_physical(pt5["Coordinates"][:], a, h),
            mass   = U.code_mass_to_msun(pt5["BH_Mass"][:].astype(np.float64), h),
            mdot   = U.bh_mdot_to_msun_yr(pt5["BH_Mdot"][:].astype(np.float64)),
            egy_qm = pt5["BH_CumEgyInjection_QM"][:].astype(np.float64)
                     if "BH_CumEgyInjection_QM" in pt5 else np.zeros(n),
            egy_rm = pt5["BH_CumEgyInjection_RM"][:].astype(np.float64)
                     if "BH_CumEgyInjection_RM" in pt5 else np.zeros(n),
        )

    return dict(
        gas     = gas,
        stars   = stars,
        bh      = bh,
        subhalo = dict(pos=sub_pos, vel=sub_vel, r200c=r200c, m200c=m200c),
        meta    = dict(halo_id=halo_id, subhalo_id=0, h=h, a=a, box_kpc=box_kpc,
                       Omega_m=Omega_m, Omega_L=Omega_L),
    )


# ── CAMELS loaders (h5py only — no illustris_python) ────────────────────────

def select_centrals_camels(cat_path, h, a, sel):
    """
    Return sorted array of group indices that pass CAMELS selection criteria.

    Parameters
    ----------
    cat_path : str | Path  groups_090.hdf5
    h, a     : from snapshot header
    sel      : dict with logM200_min, logM200_max, min_n_gas (e.g. config.CAMELS_HALO_SELECT)

    Uses h5py only; no illustris_python.
    GroupFirstSub >= 0 screens out unresolved groups.
    """
    with h5py.File(cat_path, "r") as cf:
        m200c_code = cf["Group/Group_M_Crit200"][:]          # 1e10 Msun/h
        first_sub  = cf["Group/GroupFirstSub"][:].astype(int)
        glen_type  = cf["Group/GroupLenType"][:]             # (N_groups, 6)

    m200c   = U.code_mass_to_msun(m200c_code, h)
    n_gas   = glen_type[:, 0].astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log10(m200c)

    mask = (
        (first_sub >= 0) &
        np.isfinite(logm) &
        (logm >= sel["logM200_min"]) &
        (logm <= sel["logM200_max"]) &
        (n_gas >= sel["min_n_gas"])
    )
    return np.where(mask)[0]


def load_halo_camels(snap_path, cat_path, halo_index, gfm_reconstructed=False):
    """
    Memory-frugal spatial sphere loader for CAMELS snapshots (h5py only).

    Designed to sidestep SIMBA's unsorted snapshots and duplicate ParticleIDs:
    selection is purely spatial (sphere around central SubhaloPos, radius R200c),
    never using group-membership offsets or ID matching.

    Protocol
    --------
    1. Read catalog: Group_M_Crit200, Group_R_Crit200, GroupPos, GroupFirstSub,
       SubhaloPos, SubhaloVel for the central subhalo.
    2. Read PartType0/Coordinates (full array) → build in-sphere boolean mask.
    3. Read remaining gas fields masked → peak RAM stays < ~1 GB for 256³ box.
    4. Same spatial approach for PartType4 (stars) and PartType5 (BHs).

    Parameters
    ----------
    snap_path        : path to snapshot_090.hdf5
    cat_path         : path to groups_090.hdf5
    halo_index       : FoF group index
    gfm_reconstructed: if True, fall back to Masses for star initial mass
                       (SIMBA CAMELS: GFM_InitialMass absent)

    Returns
    -------
    Same halo dict structure as load_halo / load_halo_from_hdf5.
    h, a read from snapshot header — never hardcoded.

    Note: import hdf5plugin before h5py in the calling environment so BLOSC-
    compressed CAMELS files open correctly.
    """
    with h5py.File(snap_path, "r") as sf, h5py.File(cat_path, "r") as cf:
        # ── header ────────────────────────────────────────────────────────────
        hdr       = dict(sf["Header"].attrs)
        h         = float(hdr["HubbleParam"])
        a         = float(hdr["Time"])
        box_ckpch = float(hdr["BoxSize"])           # ckpc/h
        Omega_m   = float(hdr.get("Omega0",      0.3))
        Omega_L   = float(hdr.get("OmegaLambda", 0.6911))
        box_kpc   = U.comoving_to_physical(box_ckpch, a, h)

        # ── catalog ───────────────────────────────────────────────────────────
        m200c_code = float(cf["Group/Group_M_Crit200"][halo_index])
        r200c_code = float(cf["Group/Group_R_Crit200"][halo_index])
        gpos_code  = cf["Group/GroupPos"][halo_index].astype(np.float64)
        first_sub  = int(cf["Group/GroupFirstSub"][halo_index])

        m200c   = U.code_mass_to_msun(m200c_code, h)
        r200c   = U.comoving_to_physical(r200c_code, a, h)
        gpos    = U.comoving_to_physical(gpos_code, a, h)

        sub_pos = U.comoving_to_physical(
            cf["Subhalo/SubhaloPos"][first_sub].astype(np.float64), a, h)
        sub_vel = cf["Subhalo/SubhaloVel"][first_sub].astype(np.float64)

        # ── gas: step 1 — coordinates only → sphere mask ─────────────────────
        coords_code = sf["PartType0/Coordinates"][:]            # (N, 3) ckpc/h
        coords_phys = U.comoving_to_physical(coords_code, a, h)
        dpos = coords_phys - sub_pos
        dpos -= box_kpc * np.round(dpos / box_kpc)             # min-image
        r_gas = np.linalg.norm(dpos, axis=1)
        gas_mask = r_gas < r200c

        # step 2 — remaining gas fields, read full then mask immediately
        gas = dict(
            pos     = coords_phys[gas_mask],
            vel     = U.code_vel_to_kms(
                          sf["PartType0/Velocities"][:][gas_mask], a),
            mass    = U.code_mass_to_msun(
                          sf["PartType0/Masses"][:][gas_mask], h),
            density = U.code_density_to_msun_kpc3(
                          sf["PartType0/Density"][:][gas_mask], a, h),
            u       = sf["PartType0/InternalEnergy"][:][gas_mask].astype(np.float64),
            xe      = sf["PartType0/ElectronAbundance"][:][gas_mask].astype(np.float64),
            sfr     = sf["PartType0/StarFormationRate"][:][gas_mask].astype(np.float64),
        )

        # ── stars: spatial sphere, then wind filter (GFM_StellarFormationTime > 0) ──
        pt4 = sf["PartType4"]
        scoords_code = pt4["Coordinates"][:]
        scoords_phys = U.comoving_to_physical(scoords_code, a, h)
        sdpos = scoords_phys - sub_pos
        sdpos -= box_kpc * np.round(sdpos / box_kpc)
        r_star = np.linalg.norm(sdpos, axis=1)
        star_sphere = r_star < r200c

        gform      = pt4["GFM_StellarFormationTime"][:][star_sphere]
        real_stars = gform > 0

        smass_code = pt4["Masses"][:][star_sphere]
        if gfm_reconstructed or "GFM_InitialMass" not in pt4:
            # SIMBA CAMELS: GFM_InitialMass absent → use current mass as proxy
            smass_init_code = smass_code
            if gfm_reconstructed and "GFM_InitialMass" in pt4:
                pass   # flag set explicitly; use Masses anyway
        else:
            smass_init_code = pt4["GFM_InitialMass"][:][star_sphere]

        stars = dict(
            pos       = scoords_phys[star_sphere][real_stars],
            mass      = U.code_mass_to_msun(smass_code[real_stars], h),
            mass_init = U.code_mass_to_msun(smass_init_code[real_stars], h),
            a_form    = gform[real_stars].astype(np.float64),
        )

        # ── BHs: spatial sphere around sub_pos ────────────────────────────────
        # f_duty and mode_balance are OFF for CAMELS (too few massive BHs at
        # 25 Mpc/h), so egy_qm / egy_rm are zeroed regardless of availability.
        pt5 = sf.get("PartType5")
        if pt5 is not None and "BH_Mass" in pt5 and len(pt5["BH_Mass"]) > 0:
            bcoords_code = pt5["Coordinates"][:]
            bcoords_phys = U.comoving_to_physical(bcoords_code, a, h)
            bdpos = bcoords_phys - sub_pos
            bdpos -= box_kpc * np.round(bdpos / box_kpc)
            r_bh = np.linalg.norm(bdpos, axis=1)
            bh_mask = r_bh < r200c
            n_bh = int(bh_mask.sum())
            bh = dict(
                pos    = bcoords_phys[bh_mask],
                mass   = U.code_mass_to_msun(
                             pt5["BH_Mass"][:][bh_mask].astype(np.float64), h),
                mdot   = U.bh_mdot_to_msun_yr(
                             pt5["BH_Mdot"][:][bh_mask].astype(np.float64)),
                egy_qm = np.zeros(n_bh),
                egy_rm = np.zeros(n_bh),
            )
        else:
            bh = dict(pos=np.zeros((0, 3)), mass=np.zeros(0),
                      mdot=np.zeros(0), egy_qm=np.zeros(0), egy_rm=np.zeros(0))

    return dict(
        gas     = gas,
        stars   = stars,
        bh      = bh,
        subhalo = dict(pos=sub_pos, vel=sub_vel, r200c=r200c, m200c=m200c),
        meta    = dict(
            halo_id    = int(halo_index),
            subhalo_id = int(first_sub),
            h          = h,
            a          = a,
            box_kpc    = box_kpc,
            Omega_m    = Omega_m,
            Omega_L    = Omega_L,
            gfm_initial_mass_reconstructed = gfm_reconstructed,
        ),
    )
