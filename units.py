"""
Unit conversions for TNG-format snapshots.
Used identically for EAGLE and SIMBA TNG-izations.

All functions take code-unit scalars/arrays and return physical-unit values.
The caller is responsible for reading h and a from each snapshot header —
never pass hardcoded values.

Temperature caveat: InternalEnergy for SF gas (StarFormationRate > 0) is on
the effective EoS and is physically meaningless. Apply the SF mask BEFORE
calling gas_temperature; this module does not enforce that.
"""

import numpy as np

# ── physical constants (cgs) ──────────────────────────────────────────────────
_M_P     = 1.67262192e-24   # proton mass          [g]
_K_B     = 1.38064852e-16   # Boltzmann constant   [erg/K]
_C       = 2.99792458e10    # speed of light       [cm/s]
_G       = 6.67430e-8       # gravitational const  [cm³/(g·s²)]
_SIGMA_T = 6.65246e-25      # Thomson cross-sec    [cm²]
_M_SUN   = 1.98892e33       # solar mass           [g]
_KPC     = 3.08568e21       # 1 kpc                [cm]
_YR      = 3.15576e7        # 1 year               [s]

# TNG code-time unit = UnitLength/UnitVelocity = (1 kpc/h)/(1 km/s) = 0.9778/h Gyr.
# The /h cancels with the /h in the mass unit when computing Mdot, so bh_mdot_to_msun_yr
# needs no h argument.  Value below is the h-independent part in Gyr.
_CODE_TIME_GYR = _KPC / 1e5 / (_YR * 1e9)   # ≈ 0.9778


# ── position / velocity / mass / density ─────────────────────────────────────

def comoving_to_physical(x_code, a, h):
    """Coordinates: ckpc/h  →  physical kpc."""
    return x_code * a / h


def code_vel_to_kms(v_code, a):
    """Velocities (stored as v_pec * √a in km/s)  →  peculiar velocity [km/s]."""
    return v_code * np.sqrt(a)


def code_mass_to_msun(m_code, h):
    """Masses / BH_Mass: 1e10 Msun/h  →  Msun."""
    return m_code * 1e10 / h


def code_density_to_msun_kpc3(rho_code, a, h):
    """
    Density: (1e10 Msun/h)/(ckpc/h)³  →  physical Msun/kpc³.

    Derivation:
      mass unit  = 1e10/h Msun
      length unit = (a/h) kpc  (comoving ckpc/h → physical kpc)
      ρ_phys = ρ_code * (1e10/h) / (a/h)³  =  ρ_code * 1e10 * h² / a³
    """
    return rho_code * 1e10 * h**2 / a**3


# ── gas temperature ───────────────────────────────────────────────────────────

def gas_temperature(u_code, xe, gamma, X_H):
    """
    InternalEnergy [(km/s)²] + ElectronAbundance  →  temperature [K].

    xe  : ElectronAbundance  (free electrons per H atom)
    Do NOT pass SF gas (StarFormationRate > 0).
    """
    mu    = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    u_cgs = u_code * 1e10          # (km/s)² → (cm/s)²
    return (gamma - 1.0) * u_cgs * mu * _M_P / _K_B


# ── BH accretion rates ────────────────────────────────────────────────────────

def bh_mdot_to_msun_yr(mdot_code):
    """
    BH_Mdot: TNG code units → Msun/yr.

    Code unit = (1e10 Msun/h) / (0.9778/h Gyr)  =  1e10 Msun / 0.9778 Gyr.
    h cancels → no h argument.

    EAGLE and SIMBA TNG-izations use the same convention (verify per suite
    from the header UnitMass / UnitTime or a published λ_Edd sanity check).
    """
    return mdot_code * 1e10 / (_CODE_TIME_GYR * 1e9)


def eddington_mdot_msun_yr(M_bh_msun, eps_r):
    """
    Eddington accretion rate [Msun/yr] for BH mass M_bh_msun [Msun].

    Mdot_Edd = 4π G M_BH m_p / (ε_r σ_T c)
    """
    M_bh_g   = M_bh_msun * _M_SUN
    mdot_cgs = 4.0 * np.pi * _G * M_bh_g * _M_P / (eps_r * _SIGMA_T * _C)
    return mdot_cgs * _YR / _M_SUN


def lambda_edd(M_bh_msun, mdot_msun_yr, eps_r):
    """Eddington ratio  λ = Mdot / Mdot_Edd  (dimensionless)."""
    return mdot_msun_yr / eddington_mdot_msun_yr(M_bh_msun, eps_r)


# ── cosmological lookback time ────────────────────────────────────────────────

def lookback_time_myr(a_form, a_now, h, Omega_m=0.3089, Omega_L=0.6911, n=512):
    """
    Lookback time [Myr] from scale factor a_now back to a_form.

    Uses trapezoidal integration of the flat ΛCDM Friedmann equation:
      t_lb = (1/H0) * ∫[a_form → a_now] da / (a * sqrt(Ωm/a³ + ΩΛ))

    Supports scalar or 1-D array a_form.  Returns float for scalar input.
    """
    a_form = np.atleast_1d(np.asarray(a_form, dtype=np.float64))
    a_min  = float(np.clip(a_form.min(), 1e-4, a_now))

    a_grid    = np.linspace(a_min, float(a_now), n)
    H_over_H0 = np.sqrt(Omega_m * a_grid**-3 + Omega_L)
    integrand = 1.0 / (a_grid * H_over_H0)

    # cumulative trapezoidal integral from a_min
    cum = np.concatenate([[0.0],
          np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(a_grid))])

    # for each a_form: t_lb = (total − integral_up_to_a_form) / H0
    int_to_a_form = np.interp(a_form, a_grid, cum)
    t_lb_H0inv   = cum[-1] - int_to_a_form        # dimensionless

    H0_inv_myr = 1.0 / (h * 1e2 * 1e3 / 3.0857e22) / (_YR * 1e6)
    t_lb_myr   = t_lb_H0inv * H0_inv_myr

    return t_lb_myr if t_lb_myr.size > 1 else float(t_lb_myr[0])
