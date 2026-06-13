import numpy as np

# COSMO is intentionally empty here — filled per-suite from the snapshot header.
# Never hardcode h or a; read them from HubbleParam and Time in each header.
COSMO = {}

PROTOCOL = dict(
    z_target       = 0.0,
    T_hot          = 10**5.5,    # K, hot-phase cut
    shell_frac     = 0.25,       # shell centre as fraction of R_200c
    shell_dr_frac  = 0.05,       # shell thickness as fraction of R_200c
    aperture_frac  = 1.0,        # aperture for f_hot and SFR sum
    inner_frac     = 0.10,       # inner region for mode-balance P_kin
    eps_r          = 0.10,       # radiative efficiency (L_Edd, L_rad)
    lambda_edd_thr = 0.01,       # f_duty on/off threshold
    sf_recent_Myr  = 100.0,      # lookback window for p★/m★ stellar mass
    gamma          = 5.0 / 3.0,
    X_H            = 0.76,
)

HALO_SELECT = dict(
    centrals_only = True,   # use GroupFirstSub, NOT SubhaloFlag (absent in SIMBA)
    logM200_min   = 11.5,   # log10 Msun
    logM200_max   = 14.0,
    min_n_gas     = 500,
)

# CAMELS-specific overrides: smaller box (25 Mpc/h) → cap logM200 at 13.
CAMELS_HALO_SELECT = {**HALO_SELECT, "logM200_max": 13.0}

# Pivot mass bin for CAMELS descriptor summaries (P1 figure x = descriptor loci here).
PIVOT = dict(logM200_lo=11.75, logM200_hi=12.25)

HALO_BINS = np.arange(11.5, 14.01, 0.25)   # log10 M200c [Msun]
BH_BINS   = np.arange(6.5,   9.51, 0.25)   # log10 M_BH  [Msun]

NUISANCE = dict(
    T_hot      = [1e5, 3e5, 10**5.5, 1e6],
    shell_frac = [0.25, 0.50],
    lambda_thr = [1e-3, 1e-2, 1e-1],
)
