"""
f_duty — AGN duty cycle from snapshot population.

f_duty = N(λ_Edd > λ_thr) / N_total   within each M_BH bin.

Uses BHs of the central subhalo (already loaded as halo['bh']).
λ_thr and eps_r are declared nuisance parameters in cfg.
"""

import numpy as np
import units as U


def compute(halo, cfg):
    bh      = halo["bh"]
    eps_r   = cfg["eps_r"]
    lam_thr = cfg["lambda_edd_thr"]

    n = len(bh["mass"])
    if n == 0:
        return dict(value=np.nan, n_used=0, n_active=0, lambda_edd=[])

    lam    = U.lambda_edd(bh["mass"], bh["mdot"], eps_r)
    active = lam > lam_thr

    return dict(
        value      = float(active.sum()) / n,
        n_used     = n,
        n_active   = int(active.sum()),
        lambda_edd = lam.tolist(),
    )
