"""
verify_eagle_units.py — cross-check BH_Mdot unit conversion for EAGLE.

Strategy: compute λ_Edd two independent ways and compare.
  (A) Our pipeline: BH_Mdot [code] → bh_mdot_to_msun_yr → lambda_edd(M_BH, Mdot)
  (B) Direct ratio: BH_Mdot / BH_MdotEddington  (both in same code units, no conversion)

If (A) ≈ (B), our unit conversion is correct.
If (A) ≠ (B) by a large factor, BH_Mdot is not in TNG code units for EAGLE.

Also cross-checks TNG and SIMBA using the same BH_MdotEddington field
(present in SIMBA; check if present in TNG).

Usage:
    python verify_eagle_units.py [suite_key]   default: Eagle100-1
"""

import sys
import numpy as np
import h5py
import registry as R
import units as U
import config as C

SUITE_KEY = sys.argv[1] if len(sys.argv) > 1 else "Eagle100-1"
N_SAMPLE  = 200   # load this many BHs (avoids reading the whole snapshot)


def _il():
    import illustris_python as il
    return il


def main():
    reg  = R.get_suite(SUITE_KEY)
    path = reg["path"]
    snap = R.snapnum_for_z(SUITE_KEY, 0.0)

    il = _il()

    # ── check available BH fields ─────────────────────────────────────────────
    snap_file = il.snapshot.snapPath(path, snap)
    with h5py.File(snap_file, "r") as f:
        bh_fields = set(f.get("PartType5", {}).keys())
        hdr = dict(f["Header"].attrs)

    h = float(hdr["HubbleParam"])
    a = float(hdr["Time"])
    print(f"\nSuite : {SUITE_KEY}  (h={h:.4f}  a={a:.4f})")
    print(f"BH fields available: {sorted(bh_fields)}\n")

    has_edd = "BH_MdotEddington" in bh_fields
    if not has_edd:
        print("[WARN] BH_MdotEddington not in snapshot — cannot do direct ratio check.")
        print("       Only the pipeline λ_Edd values will be shown.")

    # ── load a sample of BHs from the group catalog's central subhalos ────────
    cat_g = il.groupcat.loadHalos(path, snap,
                                   fields=["GroupFirstSub", "Group_M_Crit200"])
    first_sub = cat_g["GroupFirstSub"] if isinstance(cat_g, dict) else cat_g
    m200      = (cat_g["Group_M_Crit200"] if isinstance(cat_g, dict) else None)

    # pick centrals in mass range, sample N_SAMPLE
    if m200 is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            logm = np.log10(U.code_mass_to_msun(m200, h))
        in_range = (logm > 11.5) & (logm < 14.0) & (first_sub >= 0)
        idx = np.where(in_range)[0]
    else:
        idx = np.arange(len(first_sub))

    rng = np.random.default_rng(42)
    sample_groups = rng.choice(idx, size=min(N_SAMPLE, len(idx)), replace=False)

    # ── collect BH values ─────────────────────────────────────────────────────
    load_fields = ["BH_Mass", "BH_Mdot"]
    if has_edd:
        load_fields.append("BH_MdotEddington")

    bh_mass_code, bh_mdot_code, bh_mdot_edd_code = [], [], []

    for gid in sample_groups:
        sid = int(first_sub[gid])
        try:
            raw = il.snapshot.loadSubhalo(path, snap, sid, 5,
                                           fields=load_fields)
            if "BH_Mass" not in raw or len(raw["BH_Mass"]) == 0:
                continue
            # most massive BH
            i_max = int(np.argmax(raw["BH_Mass"]))
            bh_mass_code.append(float(raw["BH_Mass"][i_max]))
            bh_mdot_code.append(float(raw["BH_Mdot"][i_max]))
            if has_edd:
                bh_mdot_edd_code.append(float(raw["BH_MdotEddington"][i_max]))
        except Exception:
            continue

    bh_mass_code  = np.array(bh_mass_code,     dtype=np.float64)
    bh_mdot_code  = np.array(bh_mdot_code,     dtype=np.float64)

    # ── Method A: pipeline conversion ─────────────────────────────────────────
    M_bh_msun    = U.code_mass_to_msun(bh_mass_code, h)
    mdot_msun_yr = U.bh_mdot_to_msun_yr(bh_mdot_code)
    lam_A        = U.lambda_edd(M_bh_msun, mdot_msun_yr, C.PROTOCOL["eps_r"])

    print("─── Method A: pipeline conversion (bh_mdot_to_msun_yr + lambda_edd) ───")
    print(f"  BH_Mdot [code]       : min={bh_mdot_code.min():.3e}  "
          f"median={np.median(bh_mdot_code):.3e}  max={bh_mdot_code.max():.3e}")
    print(f"  BH_Mdot [Msun/yr]    : min={mdot_msun_yr.min():.3e}  "
          f"median={np.median(mdot_msun_yr):.3e}  max={mdot_msun_yr.max():.3e}")
    print(f"  λ_Edd (method A)     : min={lam_A.min():.3e}  "
          f"median={np.median(lam_A):.3e}  max={lam_A.max():.3e}")
    print(f"  f_duty (λ>0.01)      : {(lam_A > 0.01).mean():.3f}")

    if has_edd:
        bh_mdot_edd_code = np.array(bh_mdot_edd_code, dtype=np.float64)

        # ── Method B: direct ratio (no unit conversion) ───────────────────────
        lam_B = np.where(bh_mdot_edd_code > 0,
                         bh_mdot_code / bh_mdot_edd_code, np.nan)

        print("\n─── Method B: direct ratio BH_Mdot / BH_MdotEddington ──────────────")
        print(f"  BH_MdotEddington [code]: min={bh_mdot_edd_code.min():.3e}  "
              f"median={np.median(bh_mdot_edd_code):.3e}  "
              f"max={bh_mdot_edd_code.max():.3e}")
        print(f"  λ_Edd (method B)       : min={np.nanmin(lam_B):.3e}  "
              f"median={np.nanmedian(lam_B):.3e}  max={np.nanmax(lam_B):.3e}")
        print(f"  f_duty (λ>0.01)        : {np.nanmean(lam_B > 0.01):.3f}")

        # ── comparison ────────────────────────────────────────────────────────
        ok    = np.isfinite(lam_A) & np.isfinite(lam_B) & (lam_B > 0)
        ratio = lam_A[ok] / lam_B[ok]
        print(f"\n─── A / B ratio (should be ≈1.0 if units correct) ────────────────")
        print(f"  median(λ_A / λ_B) = {np.median(ratio):.4f}")
        print(f"  std(λ_A / λ_B)    = {np.std(ratio):.4f}")
        if 0.5 < np.median(ratio) < 2.0:
            print("  [OK] Unit conversion consistent with BH_MdotEddington ratio.")
        else:
            print(f"  [WARN] Ratio far from 1 — BH_Mdot may not be in TNG code units!")
            print(f"         Expected conversion: 1 code unit = "
                  f"{U.bh_mdot_to_msun_yr(1.0):.3f} Msun/yr")
    else:
        print("\n  BH_MdotEddington absent — skipping direct ratio check.")
        print("  Verify λ_Edd values against published distributions for this suite.")


if __name__ == "__main__":
    main()
