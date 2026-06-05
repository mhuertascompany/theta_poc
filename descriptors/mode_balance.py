"""
mode_balance — emergent kinetic/radiative power proxy.

proxy = P_kin / L_rad

P_kin  = 0.5 * sum(m_i * v_r_i²) / dr   hot+outflow gas inside inner_frac*R200c
L_rad  = eps_r * BH_Mdot * c²            most massive BH in central subhalo

Suite expectations (must NOT be suppressed):
  TNG   — finite proxy, validated against logged RM/QM ratio in validate_modes.py
  EAGLE — proxy ≈ 0 (RM ≡ 0); mode_defined=False flags the zero-kinetic anchor
  SIMBA — emergent-only (no logged ground truth); proxy reported as-is

Requires halo['gas'] enriched by selection.to_halo_frame.
"""

import numpy as np
import units as U
import selection as S

_MSUN_KMS2_KPC_TO_ERGS = U._M_SUN * 1e10 / U._KPC   # Msun*(km/s)²/kpc → erg/s
_MSUN_YR_C2_TO_ERGS    = U._M_SUN / U._YR * U._C**2  # Msun/yr * c² → erg/s


def compute(halo, cfg):
    gas   = halo["gas"]
    bh    = halo["bh"]
    r200c = halo["subhalo"]["r200c"]

    inner = (S.hot_mask(gas, cfg) &
             S.outflow_mask(gas)  &
             S.inner_mask(gas, r200c, cfg))

    dr_kpc    = cfg["inner_frac"] * r200c
    P_kin_nat = 0.5 * (gas["mass"][inner] * gas["v_r"][inner]**2).sum() / dr_kpc
    P_kin_erg = float(P_kin_nat * _MSUN_KMS2_KPC_TO_ERGS)

    if len(bh["mass"]) == 0:
        return dict(value=np.nan, n_used=0, mode_defined=False,
                    P_kin_erg=P_kin_erg, L_rad_erg=0.0)

    idx_central = int(np.argmax(bh["mass"]))
    L_rad_erg   = float(cfg["eps_r"] * bh["mdot"][idx_central] * _MSUN_YR_C2_TO_ERGS)

    if L_rad_erg <= 0.0:
        return dict(value=np.nan, n_used=int(inner.sum()), mode_defined=False,
                    P_kin_erg=P_kin_erg, L_rad_erg=0.0)

    proxy        = P_kin_erg / L_rad_erg
    mode_defined = P_kin_erg > 0.0   # False when no hot outflow in inner region

    return dict(
        value        = float(proxy),
        n_used       = int(inner.sum()),
        mode_defined = bool(mode_defined),
        P_kin_erg    = P_kin_erg,
        L_rad_erg    = L_rad_erg,
    )
