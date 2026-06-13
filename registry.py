"""
Per-suite registry: paths, z→snapnum maps, and capability flags.

This is the ONLY place where suite differences are encoded.
Nothing in descriptors/, selection.py, or run_suite.py may contain
suite-conditional logic — it belongs here.

Paths assume the TNG-ized layout on /virgotng; verify which subdirectory
holds output/ (snapshots + group catalogs) and use that as the
illustris_python basePath.
"""

# ── z→snapnum maps ────────────────────────────────────────────────────────────
# Only z=0 is needed for the PoC (config.py z_target=0.0).
# Full maps to be verified from dataset headers on the cluster.

_SNAP_Z0_TNG   = 99    # TNG: 100 snapshots (0–99), z=0 → snap 99
_SNAP_Z0_EAGLE = 28    # EAGLE: 29 snapshots (0–28), z=0 → snap 28
_SNAP_Z0_SIMBA = 151   # SIMBA: 152 snapshots (0–151), z=0 → snap 151

# SIMBA Simba50-1 (L50n512FP): snapshot 126 is corrupt — never load it.
_SIMBA_CORRUPT_SNAPS = {126}


def _camels_z_to_snap(z):
    """CAMELS: z=0 is snapshot 090 for all suites. Only z=0 used in pilot."""
    if z == 0.0:
        return 90
    raise NotImplementedError(f"CAMELS snapnum for z={z} not mapped")


def _tng_z_to_snap(z):
    """Placeholder: full TNG z→snapnum map (100 entries). Verify on cluster."""
    if z == 0.0:
        return _SNAP_Z0_TNG
    raise NotImplementedError(f"TNG snapnum for z={z} not yet mapped")


def _eagle_z_to_snap(z):
    """Placeholder: full EAGLE z→snapnum map (29 entries). Verify on cluster."""
    if z == 0.0:
        return _SNAP_Z0_EAGLE
    raise NotImplementedError(f"EAGLE snapnum for z={z} not yet mapped")


def _simba_z_to_snap(z, suite_key):
    """Placeholder: full SIMBA z→snapnum map (152 entries). Verify on cluster."""
    if z == 0.0:
        snap = _SNAP_Z0_SIMBA
        if suite_key == "Simba50-1" and snap in _SIMBA_CORRUPT_SNAPS:
            raise ValueError(f"Snapshot {snap} is corrupt for Simba50-1")
        return snap
    raise NotImplementedError(f"SIMBA snapnum for z={z} not yet mapped")


# ── suite definitions ─────────────────────────────────────────────────────────
# Each entry documents:
#   path         : illustris_python basePath (verify output/ location on cluster)
#   z_to_snap    : callable(z) → snapnum
#   sim_family   : "TNG" | "EAGLE" | "SIMBA"
#   m_gas_msun   : mean gas element mass [Msun] (for resolution matching)
#   box_cmpc     : comoving box size [cMpc]
#
# Capability flags (from SPEC §A — these drive descriptor behaviour, not
# runtime if-statements; they are checked once by inspect_fields.py):
#   mode_logs_qm : BH_CumEgyInjection_QM is populated
#   mode_logs_rm : BH_CumEgyInjection_RM is populated (≡0 for EAGLE by construction)
#   has_wind_pt4 : PartType4 contains wind particles (GFM_StellarFormationTime<0)
#   has_subhalo_flag : SubhaloFlag field present in group catalog
#   duplicate_particle_ids : ParticleIDs not unique (SIMBA)
#   gfm_initial_mass_reconstructed : GFM_InitialMass is derived, not native (SIMBA)

SUITES = {

    # ── TNG ──────────────────────────────────────────────────────────────────
    "TNG50-1": dict(
        path        = "/virgotng/universe/IllustrisTNG/TNG50-1/output",
        z_to_snap   = _tng_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 8.5e4,
        box_cmpc    = 35.0,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "TNG100-1": dict(
        path        = "/virgotng/universe/IllustrisTNG/TNG100-1/output",
        z_to_snap   = _tng_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 1.4e6,
        box_cmpc    = 75.0,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "TNG300-1": dict(
        path        = "/virgotng/universe/IllustrisTNG/TNG300-1/output",
        z_to_snap   = _tng_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 1.1e7,
        box_cmpc    = 205.0,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    # ── EAGLE ─────────────────────────────────────────────────────────────────
    "Eagle100-1": dict(
        path        = "/virgotng/universe/Eagle/Eagle100-1/output",
        z_to_snap   = _eagle_z_to_snap,
        sim_family  = "EAGLE",
        m_gas_msun  = 1.8e6,
        box_cmpc    = 100.0,
        # RM ≡ 0 by construction (single thermal AGN mode).
        # The emergent mode proxy returning ≈0 is the ground-truth zero anchor.
        mode_logs_qm              = True,
        mode_logs_rm              = False,
        has_wind_pt4              = False,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "RecalL0025N0752": dict(
        path        = "/virgotng/universe/Eagle/RecalL0025N0752/output",
        z_to_snap   = _eagle_z_to_snap,
        sim_family  = "EAGLE",
        m_gas_msun  = 2.3e5,
        box_cmpc    = 25.0,
        mode_logs_qm              = True,
        mode_logs_rm              = False,
        has_wind_pt4              = False,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    # ── SIMBA ─────────────────────────────────────────────────────────────────
    "Simba25-1": dict(
        path        = "/virgotng/universe/Simba/Simba25-1/output",
        z_to_snap   = lambda z: _simba_z_to_snap(z, "Simba25-1"),
        sim_family  = "SIMBA",
        m_gas_msun  = 2.3e6,
        box_cmpc    = 25.0,
        # Both CumEgyInjection fields are ABSENT (not just zero) — no logged mode ground truth.
        mode_logs_qm              = False,
        mode_logs_rm              = False,
        # Confirmed by inspect_fields: SIMBA TNG-ization DOES encode wind particles as
        # PT4 with GFM_StellarFormationTime < 0 (~326k wind vs ~12M real stars in Simba25-1).
        # SPEC §A text was inaccurate; the >0 guard is necessary here just as for TNG.
        has_wind_pt4              = True,
        has_subhalo_flag          = False,   # use GroupFirstSub for central selection
        duplicate_particle_ids    = True,
        gfm_initial_mass_reconstructed = True,
    ),

    "Simba50-1": dict(
        path        = "/virgotng/universe/Simba/Simba50-1/output",
        z_to_snap   = lambda z: _simba_z_to_snap(z, "Simba50-1"),
        sim_family  = "SIMBA",
        m_gas_msun  = 1.8e7,
        box_cmpc    = 50.0,
        corrupt_snaps             = _SIMBA_CORRUPT_SNAPS,  # snap 126 bad
        mode_logs_qm              = False,
        mode_logs_rm              = False,
        has_wind_pt4              = True,    # confirmed; same convention as TNG
        has_subhalo_flag          = False,
        duplicate_particle_ids    = True,
        gfm_initial_mass_reconstructed = True,
    ),

    "Simba100-1": dict(
        path        = "/virgotng/universe/Simba/Simba100-1/output",
        z_to_snap   = lambda z: _simba_z_to_snap(z, "Simba100-1"),
        sim_family  = "SIMBA",
        m_gas_msun  = 1.8e7,
        box_cmpc    = 100.0,
        mode_logs_qm              = False,
        mode_logs_rm              = False,
        has_wind_pt4              = True,    # confirmed; same convention as TNG
        has_subhalo_flag          = False,
        duplicate_particle_ids    = True,
        gfm_initial_mass_reconstructed = True,
    ),

    # ── CAMELS (Binder: /home/jovyan/Data/) ───────────────────────────────────
    # These entries use snap_root + suite_dir instead of path (no illustris_python).
    # Loaded by loaders.load_halo_camels and run_camels.py via h5py only.
    # snap_root/Sims/<suite_dir>/<set_name>/<sim_id>/snapshot_090.hdf5
    # snap_root/Sims/<suite_dir>/<set_name>/<sim_id>/groups_090.hdf5
    # h=0.6711, box=25 Mpc/h, 256³, m_gas≈1.27e7 Msun/h (from header at runtime).

    "camels_tng_1p": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "IllustrisTNG",
        set_name    = "1P",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,        # ≈37.3 cMpc; exact h from header
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "camels_simba_1p": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "SIMBA",
        set_name    = "1P",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "SIMBA",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        # QM/RM energy fields absent in SIMBA (not just zero).
        mode_logs_qm              = False,
        mode_logs_rm              = False,
        # Wind particles encoded as PT4 with GFM_StellarFormationTime < 0.
        has_wind_pt4              = True,
        has_subhalo_flag          = False,   # use GroupFirstSub only
        duplicate_particle_ids    = True,
        gfm_initial_mass_reconstructed = True,  # fall back to Masses
    ),

    # Optional bonus entries: Swift-EAGLE and Astrid (mounted on Binder).
    # Field quirks unknown — run inspect_fields before trusting; defer if nontrivial.
    "camels_eagle_1p": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "Swift-EAGLE",
        set_name    = "1P",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "EAGLE",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        mode_logs_qm              = True,
        mode_logs_rm              = False,   # EAGLE single thermal mode; RM≡0
        has_wind_pt4              = False,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "camels_astrid_1p": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "Astrid",
        set_name    = "1P",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "TNG",              # Astrid uses MP-Gadget, TNG-ized format
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    # LH sets for Exp 2 (SBI pilot, Stage A on Binder or JZ).
    "camels_tng_lh": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "IllustrisTNG",
        set_name    = "LH",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "camels_tng_cv0": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "IllustrisTNG",
        set_name    = "CV",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "TNG",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        mode_logs_qm              = True,
        mode_logs_rm              = True,
        has_wind_pt4              = True,
        has_subhalo_flag          = True,
        duplicate_particle_ids    = False,
        gfm_initial_mass_reconstructed = False,
    ),

    "camels_simba_cv0": dict(
        snap_root   = "/home/jovyan/Data",
        suite_dir   = "SIMBA",
        set_name    = "CV",
        snapnum     = 90,
        z_to_snap   = _camels_z_to_snap,
        sim_family  = "SIMBA",
        m_gas_msun  = 1.27e7,
        box_cmpc    = 25.0 / 0.6711,
        mode_logs_qm              = False,
        mode_logs_rm              = False,
        has_wind_pt4              = True,
        has_subhalo_flag          = False,
        duplicate_particle_ids    = True,
        gfm_initial_mass_reconstructed = True,
    ),
}

# ── primary science comparison (matched ~10⁶ Msun resolution triplet) ─────────
PRIMARY_TRIPLET = ("TNG100-1", "Eagle100-1", "Simba25-1")

# ── within-suite convergence ladders ─────────────────────────────────────────
CONVERGENCE_LADDERS = {
    "TNG":   ("TNG50-1", "TNG100-1", "TNG300-1"),
    "EAGLE": ("RecalL0025N0752", "Eagle100-1"),
    "SIMBA": ("Simba25-1", "Simba100-1"),
}


def get_suite(key):
    """Return suite dict; raise KeyError with a helpful message if unknown."""
    if key not in SUITES:
        raise KeyError(f"Unknown suite '{key}'. Available: {list(SUITES)}")
    return SUITES[key]


def snapnum_for_z(suite_key, z):
    """Resolve a redshift to a snapshot number for the given suite."""
    snap = get_suite(suite_key)["z_to_snap"](z)
    corrupt = get_suite(suite_key).get("corrupt_snaps", set())
    if snap in corrupt:
        raise ValueError(f"Snapshot {snap} of {suite_key} is corrupt — excluded")
    return snap
