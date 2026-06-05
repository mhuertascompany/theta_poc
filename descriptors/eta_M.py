"""
η_M — hot-phase mass loading factor.

η_M = Mdot_hot_out / SFR_halo

Mdot_hot_out: hot + outflowing + non-SF gas in the η_M shell.
SFR_halo: sum of StarFormationRate within aperture.

Requires halo['gas'] enriched by selection.to_halo_frame.
Cleanest emergent descriptor; expected to separate suites well.
"""

import numpy as np
import selection as S

_KMS_TO_KPC_PER_YR = 1e5 / 3.0857e21 * 3.15576e7   # (km/s) / kpc → yr⁻¹


def compute(halo, cfg):
    gas   = halo["gas"]
    r200c = halo["subhalo"]["r200c"]

    mask  = (S.hot_mask(gas, cfg) &
             S.outflow_mask(gas)  &
             S.shell_mask(gas, r200c, cfg))
    apert = S.aperture_mask(gas, r200c, cfg)

    shell_dr     = cfg["shell_dr_frac"] * r200c
    mdot_hot     = (gas["mass"][mask] * gas["v_r"][mask]).sum()
    mdot_hot     *= _KMS_TO_KPC_PER_YR / shell_dr   # Msun/yr
    sfr_halo     = float(gas["sfr"][apert].sum())

    if sfr_halo <= 0.0:
        return dict(value=np.nan, n_used=int(mask.sum()),
                    mdot_hot_msun_yr=float(mdot_hot), sfr_halo_msun_yr=0.0)

    return dict(
        value            = float(mdot_hot / sfr_halo),
        n_used           = int(mask.sum()),
        mdot_hot_msun_yr = float(mdot_hot),
        sfr_halo_msun_yr = sfr_halo,
    )
