"""
test_one_halo.py — end-to-end single-galaxy diagnostic on real data.

Loads one central galaxy near logM200 = 12.5 from TNG100-1 at z=0,
runs all 6 descriptors, and prints sanity checks before run_suite.py.

Usage:
    python test_one_halo.py [suite_key]   default: TNG100-1
    python test_one_halo.py TNG100-1
    python test_one_halo.py Eagle100-1
"""

import sys
import numpy as np
import registry as R
import loaders
import selection
import config as C
from descriptors import eta_M, f_hot, f_duty, mode_balance, eps_ff, p_star
import units as U

SUITE_KEY = sys.argv[1] if len(sys.argv) > 1 else "TNG100-1"
TARGET_LOGM = 12.5   # log10 M200c [Msun] — Milky Way-scale


def separator(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def main():
    reg  = R.get_suite(SUITE_KEY)
    path = reg["path"]
    snap = R.snapnum_for_z(SUITE_KEY, 0.0)
    cfg  = C.PROTOCOL

    separator(f"Suite: {SUITE_KEY}  snap={snap}  z=0")

    # ── header ────────────────────────────────────────────────────────────────
    import h5py
    def _il():
        import illustris_python as il
        return il

    snap_file = _il().snapshot.snapPath(path, snap)
    with h5py.File(snap_file, "r") as f:
        hdr = dict(f["Header"].attrs)

    h = float(hdr["HubbleParam"])
    a = float(hdr["Time"])
    box_ckpch = float(hdr["BoxSize"])
    Omega_m = float(hdr.get("Omega0",      0.3089))
    Omega_L = float(hdr.get("OmegaLambda", 0.6911))
    print(f"\n  h={h:.4f}  a={a:.4f}  BoxSize={box_ckpch:.0f} ckpc/h")
    print(f"  Omega_m={Omega_m:.4f}  Omega_L={Omega_L:.4f}")

    # ── select one central near target mass ───────────────────────────────────
    separator("Selecting central galaxy")
    cat = loaders.load_catalog(path, snap, h, a)

    log_m200 = np.log10(cat["m200c"])
    in_range  = (log_m200 >= C.HALO_SELECT["logM200_min"]) & \
                (log_m200 <= C.HALO_SELECT["logM200_max"])
    print(f"  Halos in mass range [{C.HALO_SELECT['logM200_min']}, "
          f"{C.HALO_SELECT['logM200_max']}]: {in_range.sum():,}")

    # pick the central closest to TARGET_LOGM
    diffs = np.abs(log_m200[in_range] - TARGET_LOGM)
    idx_in_range = np.where(in_range)[0]
    halo_id = int(idx_in_range[np.argmin(diffs)])
    subhalo_id = int(cat["first_sub"][halo_id])

    m200c = cat["m200c"][halo_id]
    r200c = cat["r200c"][halo_id]
    print(f"  Chosen halo_id={halo_id}  subhalo_id={subhalo_id}")
    print(f"  logM200c = {np.log10(m200c):.3f}  M200c = {m200c:.3e} Msun")
    print(f"  R200c    = {r200c:.1f} kpc")

    # ── load particles ────────────────────────────────────────────────────────
    separator("Loading particles")
    bh_avail = loaders.snap_bh_fields(path, snap)
    halo = loaders.load_halo(path, snap, halo_id, cat, h, a, box_ckpch,
                             bh_avail=bh_avail)
    # inject cosmology into meta for p_star lookback time
    halo["meta"]["Omega_m"] = Omega_m
    halo["meta"]["Omega_L"] = Omega_L
    halo["meta"]["gfm_initial_mass_reconstructed"] = reg.get(
        "gfm_initial_mass_reconstructed", False)

    n_gas  = len(halo["gas"]["mass"])
    n_star = len(halo["stars"]["mass"])
    n_bh   = len(halo["bh"]["mass"])
    print(f"  gas particles : {n_gas:,}")
    print(f"  real stars    : {n_star:,}  (wind already filtered)")
    print(f"  BHs           : {n_bh}")

    if n_gas < C.HALO_SELECT["min_n_gas"]:
        print(f"  [WARN] n_gas={n_gas} below min_n_gas={C.HALO_SELECT['min_n_gas']}")

    # ── to_halo_frame ─────────────────────────────────────────────────────────
    halo["gas"] = selection.to_halo_frame(
        halo["gas"], halo["subhalo"], halo["meta"]["box_kpc"])

    # velocity dispersion sanity: median |v_r| vs SubhaloVelDisp from catalog
    vdisp_cat = float(_il().groupcat.loadSubhalos(
        path, snap, fields=["SubhaloVelDisp"]
    )["SubhaloVelDisp"][subhalo_id])
    vdisp_gas = float(np.std(halo["gas"]["v_r"]))
    print(f"\n  SubhaloVelDisp (catalog) : {vdisp_cat:.1f} km/s")
    print(f"  gas v_r std (halo frame) : {vdisp_gas:.1f} km/s  (rough check)")

    # ── gas diagnostics ───────────────────────────────────────────────────────
    separator("Gas diagnostics")
    sf   = selection.sf_mask(halo["gas"])
    hot  = selection.hot_mask(halo["gas"], cfg)
    T    = selection.temperature(halo["gas"], cfg)

    print(f"  SF gas fraction          : {sf.mean():.3f}")
    print(f"  Hot non-SF fraction      : {hot.mean():.3f}")
    T_nonSF = T[~sf]
    if len(T_nonSF):
        p10, p50, p90 = np.percentile(T_nonSF, [10, 50, 90])
        print(f"  Non-SF gas T [K] p10/50/90: {p10:.2e} / {p50:.2e} / {p90:.2e}")
        if p50 < 1e4 or p50 > 1e8:
            print("  [WARN] median T outside 10^4–10^8 K — check unit conversion")

    # ── BH diagnostics ────────────────────────────────────────────────────────
    separator("BH diagnostics")
    if n_bh > 0:
        lam = U.lambda_edd(halo["bh"]["mass"], halo["bh"]["mdot"], cfg["eps_r"])
        bh_most_massive = int(np.argmax(halo["bh"]["mass"]))
        print(f"  Most massive BH mass : {halo['bh']['mass'][bh_most_massive]:.3e} Msun")
        print(f"  Most massive BH mdot : {halo['bh']['mdot'][bh_most_massive]:.3e} Msun/yr")
        print(f"  Most massive BH λ_Edd: {lam[bh_most_massive]:.3e}")
        print(f"  λ_Edd distribution   : "
              f"min={lam.min():.2e}  median={np.median(lam):.2e}  max={lam.max():.2e}")
        if lam.max() > 10:
            print("  [WARN] λ_Edd > 10 detected — check BH_Mdot unit conversion")
    else:
        print("  No BHs in central subhalo.")

    # ── run all 6 descriptors ─────────────────────────────────────────────────
    separator("Descriptor values")
    descriptors = [
        ("eta_M",        eta_M),
        ("f_hot",        f_hot),
        ("f_duty",       f_duty),
        ("mode_balance", mode_balance),
        ("eps_ff",       eps_ff),
        ("p_star",       p_star),
    ]

    # expected physical ranges for logM~12.5 TNG galaxy (rough)
    sanity = {
        "eta_M":        (0.01, 100),
        "f_hot":        (0.0,  1.0),
        "f_duty":       (0.0,  1.0),
        "mode_balance": (0.0,  1e6),   # proxy can be large
        "eps_ff":       (1e-4, 1.0),
        "p_star":       (0.1,  1e5),
    }

    for name, mod in descriptors:
        res = mod.compute(halo, cfg)
        v   = res["value"]
        lo, hi = sanity[name]
        if np.isnan(v):
            flag = "[NaN — check n_used or SFR]"
        elif lo <= v <= hi:
            flag = "[OK]"
        else:
            flag = f"[WARN — outside ({lo:.1e}, {hi:.1e})]"
        print(f"  {name:<16} = {v:>12.4g}   n_used={res.get('n_used','-'):>5}  {flag}")

    separator("Done")
    print(f"  Suite {SUITE_KEY}, halo {halo_id} (logM={np.log10(m200c):.2f}) complete.\n")


if __name__ == "__main__":
    main()
