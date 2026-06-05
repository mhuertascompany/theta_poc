"""
f_hot — hot-gas fraction within aperture.

f_hot = M_gas[hot & aperture] / M_gas[aperture]

Cleanly emergent; SIMBA's evacuated low-mass halos expected to sit low.
Requires halo['gas'] enriched by selection.to_halo_frame.
"""

import numpy as np
import selection as S


def compute(halo, cfg):
    gas   = halo["gas"]
    r200c = halo["subhalo"]["r200c"]

    apert = S.aperture_mask(gas, r200c, cfg)
    hot   = S.hot_mask(gas, cfg) & apert

    m_tot = float(gas["mass"][apert].sum())
    m_hot = float(gas["mass"][hot].sum())

    return dict(
        value       = m_hot / m_tot if m_tot > 0.0 else np.nan,
        n_used      = int(apert.sum()),
        n_hot       = int(hot.sum()),
        m_hot_msun  = m_hot,
        m_tot_msun  = m_tot,
    )
