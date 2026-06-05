"""
p★/m★ — outflow momentum per stellar mass formed.

p_out   = sum(m_i * v_r_i)  outflow + non-SF gas in the η_M shell  [Msun·km/s]
m_recent = GFM_InitialMass of genuine stars formed within sf_recent_Myr  [Msun]
p_star  = p_out / m_recent   [km/s]

Caveats encoded in output metadata:
  - Shell momentum contaminated by AGN-driven outflow at the shell
  - Kinetic-wind models (TNG/SIMBA): winds decouple before recoupling
  - SIMBA GFM_InitialMass is reconstructed (BC03/Chabrier), not native

Requires halo['gas'] enriched by selection.to_halo_frame.
"""

import numpy as np
import units as U
import selection as S


def compute(halo, cfg):
    gas   = halo["gas"]
    stars = halo["stars"]
    r200c = halo["subhalo"]["r200c"]
    meta  = halo["meta"]

    # ── outflow momentum in shell (non-SF + outflowing) ──────────────────────
    p_mask = (~S.sf_mask(gas) & S.outflow_mask(gas) & S.shell_mask(gas, r200c, cfg))
    p_out  = float((gas["mass"][p_mask] * gas["v_r"][p_mask]).sum())  # Msun·km/s

    # ── recent stellar mass ───────────────────────────────────────────────────
    h       = meta["h"]
    Omega_m = meta.get("Omega_m", 0.3089)
    Omega_L = meta.get("Omega_L", 0.6911)
    a       = meta["a"]

    t_lb   = U.lookback_time_myr(stars["a_form"], a, h, Omega_m, Omega_L)
    recent = np.atleast_1d(t_lb) < cfg["sf_recent_Myr"]
    m_recent = float(stars["mass_init"][recent].sum())

    flag = ("reconstructed_bc03"
            if meta.get("gfm_initial_mass_reconstructed") else "ok")

    if m_recent <= 0.0:
        return dict(value=np.nan, n_used=int(p_mask.sum()), n_recent_stars=0,
                    p_out_msun_kms=p_out, gfm_initial_mass_flag=flag)

    return dict(
        value                = float(p_out / m_recent),   # km/s
        n_used               = int(p_mask.sum()),
        n_recent_stars       = int(recent.sum()),
        p_out_msun_kms       = p_out,
        m_recent_msun        = m_recent,
        gfm_initial_mass_flag = flag,
    )
