"""
selection.py — shared masking layer used identically by every descriptor.

All functions take physical-unit arrays (from loaders.py) and cfg (from
config.PROTOCOL). No descriptor may re-implement these selections.

to_halo_frame() must be called first; it enriches the gas dict with
r / r_hat / v_r which the other functions depend on.
"""

import numpy as np
import units as U


def to_halo_frame(gas, subhalo, box_kpc):
    """
    Recenter gas on SubhaloPos and subtract SubhaloVel.
    Returns a new gas dict (shallow copy) enriched with:
      r     [kpc]          radial distance from subhalo centre
      r_hat [dimensionless] radial unit vector
      v_r   [km/s]         radial peculiar velocity (positive = outflow)

    Minimum-image convention applied for periodic boundaries.
    """
    dpos = gas["pos"] - subhalo["pos"]                        # (N, 3) kpc
    dpos -= box_kpc * np.round(dpos / box_kpc)                # minimum image

    r = np.linalg.norm(dpos, axis=1)                          # (N,) kpc
    # safe unit vector: particles exactly at centre get r_hat = 0
    safe_r = np.where(r > 0, r, 1.0)
    r_hat = dpos / safe_r[:, np.newaxis]                      # (N, 3)

    v_pec = gas["vel"] - subhalo["vel"]                       # (N, 3) km/s
    v_r   = (v_pec * r_hat).sum(axis=1)                       # (N,) km/s

    return {**gas, "r": r, "r_hat": r_hat, "v_r": v_r}


def sf_mask(gas):
    """True where StarFormationRate > 0 (ISM/EoS gas). Exclude before temperature use."""
    return gas["sfr"] > 0.0


def temperature(gas, cfg):
    """
    Gas temperature [K] for all particles.
    Result is physically meaningless for SF gas — apply sf_mask first.
    """
    return U.gas_temperature(gas["u"], gas["xe"], cfg["gamma"], cfg["X_H"])


def hot_mask(gas, cfg):
    """True for hot non-SF gas: T > T_hot AND StarFormationRate == 0."""
    return (temperature(gas, cfg) > cfg["T_hot"]) & ~sf_mask(gas)


def outflow_mask(gas):
    """True where v_r > 0 (outflowing). Requires to_halo_frame."""
    return gas["v_r"] > 0.0


def shell_mask(gas, r200c, cfg):
    """
    True for gas in the η_M shell:
      |r - shell_frac * r200c| < 0.5 * shell_dr_frac * r200c
    """
    r_shell  = cfg["shell_frac"]    * r200c
    half_dr  = 0.5 * cfg["shell_dr_frac"] * r200c
    return np.abs(gas["r"] - r_shell) < half_dr


def aperture_mask(gas, r200c, cfg):
    """True for gas within aperture_frac * r200c."""
    return gas["r"] < cfg["aperture_frac"] * r200c


def inner_mask(gas, r200c, cfg):
    """True for gas within inner_frac * r200c (used by mode_balance P_kin)."""
    return gas["r"] < cfg["inner_frac"] * r200c
