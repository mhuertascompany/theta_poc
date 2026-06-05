"""
ε_ff — SF efficiency per free-fall time.

ε_ff = sum(SFR_i * t_ff,i) / sum(m_gas,i)   over SF gas (SFR > 0)

t_ff,i = sqrt(3π / (32 G ρ_i))    ρ_i in physical Msun/kpc³

HONEST NOTE: for TNG/EAGLE this largely recovers the input KS law
normalisation convolved with the density PDF. For SIMBA the H₂-based law
differs. Report this descriptor as 'recovers_input=True' for TNG/EAGLE;
optionally add an H₂-only recomputation for SIMBA as a bonus diagnostic.
"""

import numpy as np
import units as U
import selection as S

_G_CGS    = U._G
_MSUN_CGS = U._M_SUN
_KPC_CGS  = U._KPC
_YR_S     = U._YR


def compute(halo, cfg):
    gas = halo["gas"]
    sf  = S.sf_mask(gas)

    if sf.sum() == 0:
        return dict(value=np.nan, n_used=0, recovers_input=True)

    rho_cgs = gas["density"][sf].astype(np.float64) * _MSUN_CGS / _KPC_CGS**3
    t_ff_yr = np.sqrt(3.0 * np.pi / (32.0 * _G_CGS * rho_cgs)) / _YR_S

    eps_ff = float((gas["sfr"][sf] * t_ff_yr).sum() / gas["mass"][sf].sum())

    return dict(
        value          = eps_ff,
        n_used         = int(sf.sum()),
        recovers_input = True,   # KS normalisation for TNG/EAGLE; H₂ law for SIMBA
    )
