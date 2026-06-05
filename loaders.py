"""
loaders.py — thin illustris_python wrappers.

illustris_python is imported lazily (inside functions only) so this module
is importable locally without cluster access.

All returned arrays are in physical units (post-conversion via units.py).
Never hardcode h or a — always pass from the snapshot header.

Public API
----------
load_catalog(path, snapnum, h, a)          → catalog dict (physical units)
load_halo(path, snapnum, halo_id, ...)     → halo dict (physical units)
load_halo_from_hdf5(snap, groups, halo=0) → same format, h5py only (for tests)
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

    # ── BHs (central subhalo; zero-length if no BH present) ─────────────────
    n_bh = len(bh_raw["BH_Mass"]) if "BH_Mass" in bh_raw else 0
    bh = dict(
        pos    = U.comoving_to_physical(bh_raw["Coordinates"], a, h),
        mass   = U.code_mass_to_msun(bh_raw["BH_Mass"],        h),
        mdot   = U.bh_mdot_to_msun_yr(bh_raw["BH_Mdot"]),
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
