"""
Build synthetic TNG-format HDF5 fixtures for local pytest runs.
Run:  python fixtures/make_fixtures.py
Requires only numpy + h5py.

Steps 1-4: Header + group catalog + gas + stars + BHs + expected.json.
"""

import json
import pathlib
import numpy as np
import h5py

HERE   = pathlib.Path(__file__).parent
SNAP   = HERE / "snap_099.hdf5"
GROUPS = HERE / "groups_099.hdf5"

# ── fixed cosmology / snapshot parameters ────────────────────────────────────
H       = 0.6774          # TNG HubbleParam
A       = 1.0             # scale factor, z=0
BOX     = 75_000.0        # BoxSize, ckpc/h  (TNG100-1 value)

# ── one toy halo ─────────────────────────────────────────────────────────────
# M200c = 1e13 Msun  →  code units = M[Msun] * h / 1e10
M200C_MSUN = 1.0e13
M200C_CODE = M200C_MSUN * H / 1.0e10          # 677.4 code units

# R200c: chosen so physical radius = 500 kpc  →  ckpc/h = 500 * h / a
R200C_KPC     = 500.0                          # physical kpc
R200C_CODE    = R200C_KPC * H / A             # ckpc/h

GROUP_POS     = np.array([BOX / 2] * 3)       # box centre, ckpc/h
SUBHALO_VEL   = np.zeros(3, dtype=np.float32) # km/s * sqrt(a); halo at rest

# ── gas particle design (Step 2) ─────────────────────────────────────────────
# Four groups of 5 particles, each testing one selection regime.
#
#   A: hot outflowing in η_M shell   r=125 kpc  v_r=+200 km/s  T~2e6 K  SFR=0
#   B: hot inflowing  interior       r=250 kpc  v_r=−50  km/s  T~2e6 K  SFR=0
#   C: SF gas         inner          r= 50 kpc  v_r=0           SFR=0.1 Msun/yr
#   D: cold non-SF    interior       r=300 kpc  v_r=0           T~1e4 K  SFR=0
#
# Particle A drives η_M and (with B) f_hot numerator.
# Particle C drives ε_ff and SFR_halo denominator of η_M.
# Particles A+B+C+D all inside aperture_frac=1.0 × R200c=500 kpc.

N_EACH   = 5
M_CODE   = 0.014        # mass per particle [code: 1e10 Msun/h] ≈ 2.07e8 Msun
SFR_EACH = 0.1          # [Msun/yr] for group C
RHO_SF   = 1e-3         # density for SF gas [code units]
RHO_HOT  = 1e-5         # density for non-SF gas [code units]

# InternalEnergy → temperature (do NOT change without recomputing expected.json)
# Hot:  T ≈ 2×10^6 K, xe=1.0 (fully ionised)
# Cold: T ≈ 1×10^4 K, xe=0.1
U_HOT, XE_HOT   = 3.916e4, 1.0
U_COLD, XE_COLD = 111.1,   0.1

# ── star / BH design (Step 3) ─────────────────────────────────────────────────
# PartType4: 5 recent real stars + 3 old real stars + 3 wind particles
#   Recent: GFM_StellarFormationTime = 0.993  (very close to a=1, clearly < 100 Myr ago)
#   Old:    GFM_StellarFormationTime = 0.500  (formed long ago, excluded from p★/m★)
#   Wind:   GFM_StellarFormationTime = -1.0   (TNG wind particle, always excluded)
A_FORM_RECENT = np.float32(0.993)
A_FORM_OLD    = np.float32(0.500)
A_FORM_WIND   = np.float32(-1.0)
M_STAR_CODE   = 0.012   # GFM_InitialMass per star [code units] ≈ 1.77e8 Msun

# PartType5: 3 BHs with hand-chosen masses/accretion rates
# BH_Mdot in code units: Mdot[Msun/yr] / 10.227
# λ_Edd thresholds: BH0 λ≈0.045 (above 0.01), BH1 λ≈0.0045 (below), BH2 λ≈0.045 (above)
# → f_duty = 2/3
BH_MASS_CODE  = np.array([6.774e-3, 6.774e-4, 3.387e-3], dtype=np.float32)  # [code]
BH_MDOT_CODE  = np.array([9.778e-3, 9.778e-5, 4.889e-3], dtype=np.float32)  # [code]
BH_EGY_QM    = np.array([10.0, 5.0, 8.0], dtype=np.float32)   # thermal/quasar [code]
BH_EGY_RM    = np.array([ 2.0, 0.0, 4.0], dtype=np.float32)   # kinetic/radio  [code]


def write_snapshot_header(f):
    hdr = f.require_group("Header")
    hdr.attrs["HubbleParam"]             = H
    hdr.attrs["Time"]                    = A
    hdr.attrs["Redshift"]                = 0.0
    hdr.attrs["BoxSize"]                 = BOX
    hdr.attrs["UnitMass_in_g"]           = 1.989e43   # 1e10 Msun in g
    hdr.attrs["UnitLength_in_cm"]        = 3.08568e21 # 1 kpc in cm
    hdr.attrs["UnitVelocity_in_cm_per_s"]= 1.0e5      # 1 km/s in cm/s
    hdr.attrs["NumFilesPerSnapshot"]     = 1
    # NumPart_Total filled with zeros now; updated when particles are added
    hdr.attrs["NumPart_Total"]           = np.zeros(6, dtype=np.int32)


def _abs_pos(r_phys_kpc, axis, sign=1):
    """Physical radius [kpc] along one axis → absolute Coords [ckpc/h]."""
    off = np.zeros(3, dtype=np.float32)
    off[axis] = sign * r_phys_kpc * H / A
    return (GROUP_POS.astype(np.float32) + off)


def write_gas_particles(f):
    """Append PartType0 to an already-open (r+) snapshot file."""
    n = N_EACH

    # positions: each group along a different axis for clean geometry
    coords = np.vstack([
        np.tile(_abs_pos(125.0,  0, +1), (n, 1)),   # A: +x, shell centre
        np.tile(_abs_pos(250.0,  1, +1), (n, 1)),   # B: +y, interior
        np.tile(_abs_pos( 50.0,  2, +1), (n, 1)),   # C: +z, inner
        np.tile(_abs_pos(300.0,  2, -1), (n, 1)),   # D: −z, interior
    ])  # (20, 3)

    # velocities [km/s]; at a=1, v_code = v_pec (no √a correction needed)
    vels = np.vstack([
        np.tile([200.,   0.,    0.], (n, 1)),   # A: outflowing +x
        np.tile([  0.,  -50.,  0.], (n, 1)),   # B: inflowing  +y
        np.zeros((n, 3)),                       # C: at rest
        np.tile([  0.,   0., -100.], (n, 1)),  # D: arbitrary
    ]).astype(np.float32)

    n_tot = 4 * n
    masses = np.full(n_tot, M_CODE,   dtype=np.float32)
    u      = np.concatenate([np.full(n, U_HOT), np.full(n, U_HOT),
                              np.full(n, U_COLD), np.full(n, U_COLD)]).astype(np.float32)
    xe     = np.concatenate([np.full(n, XE_HOT), np.full(n, XE_HOT),
                              np.full(n, XE_COLD), np.full(n, XE_COLD)]).astype(np.float32)
    rho    = np.concatenate([np.full(n, RHO_HOT), np.full(n, RHO_HOT),
                              np.full(n, RHO_SF),  np.full(n, RHO_HOT)]).astype(np.float32)
    sfr    = np.concatenate([np.zeros(n), np.zeros(n),
                              np.full(n, SFR_EACH), np.zeros(n)]).astype(np.float32)

    pt0 = f.require_group("PartType0")
    pt0.create_dataset("Coordinates",       data=coords.astype(np.float32))
    pt0.create_dataset("Velocities",        data=vels)
    pt0.create_dataset("Masses",            data=masses)
    pt0.create_dataset("Density",           data=rho)
    pt0.create_dataset("InternalEnergy",    data=u)
    pt0.create_dataset("ElectronAbundance", data=xe)
    pt0.create_dataset("StarFormationRate", data=sfr)

    npt = f["Header"].attrs["NumPart_Total"].copy()
    npt[0] = n_tot
    f["Header"].attrs["NumPart_Total"] = npt
    return n_tot


def write_star_particles(f):
    """Append PartType4: 5 recent + 3 old real stars + 3 wind particles."""
    # positions: recent along +x, old along +y, wind along +z (all well inside aperture)
    recent = np.tile(_abs_pos(30.0,  0, +1), (5, 1))
    old    = np.tile(_abs_pos(80.0,  1, +1), (3, 1))
    wind   = np.tile(_abs_pos(150.0, 2, +1), (3, 1))
    coords = np.vstack([recent, old, wind]).astype(np.float32)

    n_tot  = 11
    masses = np.full(n_tot, M_STAR_CODE, dtype=np.float32)
    ginit  = np.full(n_tot, M_STAR_CODE, dtype=np.float32)   # GFM_InitialMass = Masses here
    gform  = np.concatenate([
        np.full(5, A_FORM_RECENT),
        np.full(3, A_FORM_OLD),
        np.full(3, A_FORM_WIND),
    ]).astype(np.float32)

    pt4 = f.require_group("PartType4")
    pt4.create_dataset("Coordinates",            data=coords)
    pt4.create_dataset("Masses",                 data=masses)
    pt4.create_dataset("GFM_InitialMass",        data=ginit)
    pt4.create_dataset("GFM_StellarFormationTime", data=gform)

    npt = f["Header"].attrs["NumPart_Total"].copy()
    npt[4] = n_tot
    f["Header"].attrs["NumPart_Total"] = npt
    return n_tot


def write_bh_particles(f):
    """Append PartType5: 3 BHs at halo centre with known λ_Edd and RM/QM ratio."""
    # All three BHs sit at the halo centre (typical for the central subhalo)
    coords = np.tile(GROUP_POS.astype(np.float32), (3, 1))

    pt5 = f.require_group("PartType5")
    pt5.create_dataset("Coordinates",              data=coords)
    pt5.create_dataset("BH_Mass",                  data=BH_MASS_CODE)
    pt5.create_dataset("BH_Mdot",                  data=BH_MDOT_CODE)
    pt5.create_dataset("BH_CumEgyInjection_QM",    data=BH_EGY_QM)
    pt5.create_dataset("BH_CumEgyInjection_RM",    data=BH_EGY_RM)

    npt = f["Header"].attrs["NumPart_Total"].copy()
    npt[5] = 3
    f["Header"].attrs["NumPart_Total"] = npt


def write_group_catalog():
    with h5py.File(GROUPS, "w") as f:
        hdr = f.require_group("Header")
        hdr.attrs["HubbleParam"] = H
        hdr.attrs["Time"]        = A
        hdr.attrs["BoxSize"]     = BOX

        grp = f.require_group("Group")
        grp.create_dataset("Group_M_Crit200", data=np.array([M200C_CODE],  dtype=np.float32))
        grp.create_dataset("Group_R_Crit200", data=np.array([R200C_CODE],  dtype=np.float32))
        grp.create_dataset("GroupPos",        data=GROUP_POS[np.newaxis,:].astype(np.float32))
        grp.create_dataset("GroupFirstSub",   data=np.array([0],           dtype=np.int32))
        grp.create_dataset("GroupNsubs",      data=np.array([1],           dtype=np.int32))

        sub = f.require_group("Subhalo")
        sub.create_dataset("SubhaloPos", data=GROUP_POS[np.newaxis,:].astype(np.float32))
        sub.create_dataset("SubhaloVel", data=SUBHALO_VEL[np.newaxis,:].astype(np.float32))

    print(f"  wrote {GROUPS}")


def compute_expected():
    """Derive every expected descriptor value analytically from the fixture constants."""
    import sys, pathlib
    sys.path.insert(0, str(HERE.parent))
    import units as U

    gamma, X_H, eps_r = 5/3, 0.76, 0.10
    KPC = U._KPC; YR = U._YR; G_CGS = U._G; M_SUN = U._M_SUN

    # ── unit conversions for fixture particles ────────────────────────────────
    m_gas_msun  = U.code_mass_to_msun(M_CODE,   H)    # Msun per gas particle
    m_star_msun = U.code_mass_to_msun(M_STAR_CODE, H) # Msun per star particle

    # ── temperature of hot gas (groups A & B, xe=XE_HOT) ─────────────────────
    T_hot_gas = float(U.gas_temperature(U_HOT, XE_HOT, gamma, X_H))

    # ── gas_mass_total [Msun] ─────────────────────────────────────────────────
    gas_mass_total = 4 * N_EACH * m_gas_msun

    # ── f_hot: hot(T>T_hot_proto, SFR=0) within aperture / all within aperture ─
    # Groups A+B are hot non-SF (10 particles); all 20 are within aperture=R200c
    T_hot_proto = 10**5.5
    f_hot = (2 * N_EACH) / (4 * N_EACH)   # = 0.5

    # ── η_M: hot outflowing shell flux / SFR_halo ────────────────────────────
    # Only group A: r=125 kpc (shell centre), v_r=+200 km/s, hot, SFR=0
    shell_dr_kpc = 0.05 * R200C_KPC                   # 25 kpc
    kms_to_kpc_per_yr = 1e5 / KPC * YR                # 1.02269e-9
    mdot_hot = N_EACH * m_gas_msun * 200.0 * kms_to_kpc_per_yr / shell_dr_kpc
    sfr_halo = N_EACH * SFR_EACH
    eta_M = mdot_hot / sfr_halo

    # ── ε_ff: over SF gas (group C, rho_code=RHO_SF) ─────────────────────────
    rho_phys = U.code_density_to_msun_kpc3(RHO_SF, A, H)          # Msun/kpc³
    rho_cgs  = rho_phys * M_SUN / KPC**3
    t_ff_s   = np.sqrt(3 * np.pi / (32 * G_CGS * rho_cgs))
    t_ff_yr  = t_ff_s / YR
    eps_ff   = (N_EACH * SFR_EACH * t_ff_yr) / (N_EACH * m_gas_msun)

    # ── f_duty: fraction of BHs with λ_Edd > 0.01 ────────────────────────────
    bh_mass_msun = U.code_mass_to_msun(BH_MASS_CODE.astype(np.float64), H)
    bh_mdot_msun_yr = U.bh_mdot_to_msun_yr(BH_MDOT_CODE.astype(np.float64))
    lam = U.lambda_edd(bh_mass_msun, bh_mdot_msun_yr, eps_r)
    f_duty = float((lam > 0.01).sum()) / len(lam)

    # ── p★/m★: shell outflow momentum / recent stellar mass ──────────────────
    # Outflow+non-SF in shell: only group A
    p_out_msun_kms = N_EACH * m_gas_msun * 200.0   # Msun * km/s
    m_recent_msun  = 5 * m_star_msun               # 5 recent stars
    p_star = p_out_msun_kms / m_recent_msun         # km/s

    # ── mode_balance: no hot outflow inside inner_frac*R200c=50 kpc ──────────
    # Group A is at 125 kpc > 50 kpc → P_kin = 0 → proxy = 0
    mode_balance_proxy = 0.0

    # ── logged RM/QM ratios (for validate_modes) ─────────────────────────────
    rm_qm_logged = [
        float(BH_EGY_RM[i] / BH_EGY_QM[i]) if BH_EGY_QM[i] > 0 else 0.0
        for i in range(3)
    ]

    return dict(
        H                  = H,
        A                  = A,
        R200c_kpc          = R200C_KPC,
        gas_mass_total_msun= float(gas_mass_total),
        T_hot_gas_K        = float(T_hot_gas),
        f_hot              = float(f_hot),
        eta_M              = float(eta_M),
        eps_ff             = float(eps_ff),
        f_duty             = float(f_duty),
        lambda_edd         = [float(x) for x in lam],
        p_star_kms         = float(p_star),
        mode_balance_proxy = mode_balance_proxy,
        rm_qm_logged       = rm_qm_logged,
        n_wind_particles   = 3,
        n_recent_stars     = 5,
    )


def main():
    # Step 1: header + group catalog
    with h5py.File(SNAP, "w") as f:
        write_snapshot_header(f)
        write_gas_particles(f)
        write_star_particles(f)
        write_bh_particles(f)
    print(f"  wrote {SNAP}")

    write_group_catalog()

    # ── readback checks ───────────────────────────────────────────────────────
    with h5py.File(SNAP, "r") as f:
        assert f["Header"].attrs["HubbleParam"] == H
        assert f["Header"].attrs["Time"]        == A
        npt = f["Header"].attrs["NumPart_Total"]
        assert npt[0] == 4 * N_EACH, f"NumPart_Total[0]={npt[0]}"

        masses = f["PartType0/Masses"][:]
        assert len(masses) == 4 * N_EACH
        assert np.allclose(masses.sum(), 4 * N_EACH * M_CODE, rtol=1e-5)

        sfr = f["PartType0/StarFormationRate"][:]
        assert abs(sfr.sum() - N_EACH * SFR_EACH) < 1e-5

    with h5py.File(GROUPS, "r") as f:
        m200 = f["Group/Group_M_Crit200"][0]
        r200 = f["Group/Group_R_Crit200"][0]
        assert abs(m200 - M200C_CODE) < 1e-3, f"M200 mismatch: {m200}"
        assert abs(r200 - R200C_CODE) < 1e-3, f"R200 mismatch: {r200}"

    # Step 3 readback
    with h5py.File(SNAP, "r") as f:
        gform = f["PartType4/GFM_StellarFormationTime"][:]
        assert (gform < 0).sum() == 3,  "expected 3 wind particles"
        assert (gform > 0).sum() == 8,  "expected 8 real stars"
        assert (gform == A_FORM_RECENT).sum() == 5, "expected 5 recent stars"
        bh_rm = f["PartType5/BH_CumEgyInjection_RM"][:]
        assert bh_rm[1] == 0.0, "BH1 should have RM=0 (EAGLE-like anchor)"
        npt = f["Header"].attrs["NumPart_Total"]
        assert npt[4] == 11 and npt[5] == 3

    print(f"\n  M200c = {M200C_CODE:.2f} code  ({M200C_MSUN:.1e} Msun)")
    print(f"  R200c = {R200C_CODE:.2f} ckpc/h  ({R200C_KPC:.1f} kpc physical)")
    print(f"  gas: {4*N_EACH}  stars: 11 (5 recent + 3 old + 3 wind)  BHs: 3")
    print(f"  BH λ_Edd: above/below/above threshold → f_duty = 2/3")
    print(f"  BH RM/QM ratios: {BH_EGY_RM[0]/BH_EGY_QM[0]:.2f}, "
          f"{BH_EGY_RM[1]/BH_EGY_QM[1] if BH_EGY_QM[1]>0 else 0:.2f}, "
          f"{BH_EGY_RM[2]/BH_EGY_QM[2]:.2f}")
    print("\nSteps 1-3 OK.")

    expected = compute_expected()
    exp_path = HERE / "expected.json"
    with open(exp_path, "w") as fj:
        json.dump(expected, fj, indent=2)
    print(f"\n  wrote {exp_path}")
    for k, v in expected.items():
        print(f"    {k}: {v}")
    print("\nStep 4 OK — expected.json written.")


if __name__ == "__main__":
    main()
