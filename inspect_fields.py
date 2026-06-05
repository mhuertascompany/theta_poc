"""
inspect_fields.py — per-suite field inventory and §A assertion checks.

Usage:
    python inspect_fields.py TNG100-1
    python inspect_fields.py              # all suites in PRIMARY_TRIPLET

Runs on the cluster where /virgotng is accessible.
illustris_python is imported lazily so the file can be imported locally.

Assertions fail loudly if dataset reality contradicts SPEC §A.
"""

import sys
import numpy as np
import registry as R


def _il():
    import illustris_python as il
    return il


def load_header(path, snapnum):
    il = _il()
    with __import__("h5py").File(il.snapPath(path, snapnum), "r") as f:
        return dict(f["Header"].attrs)


def field_list(path, snapnum, part_type_str):
    """Return sorted list of field names for a particle type."""
    il = _il()
    try:
        data = il.snapshot.loadSubset(path, snapnum, part_type_str,
                                      fields=None, mdi=None, sq=False, float32=False)
        return sorted(data.keys())
    except Exception:
        return []


def group_field_list(path, snapnum, catalog_type):
    """Return sorted list of field names from Group or Subhalo catalog."""
    il = _il()
    try:
        data = il.groupcat.loadSingle(path, snapnum)
        return sorted(data.keys())
    except Exception:
        try:
            data = il.groupcat.load(path, snapnum, catalog_type)
            return sorted(data.keys())
        except Exception:
            return []


def check_bh_mode_logs(path, snapnum, suite_key):
    """Load a sample of BH particles; assert §A mode-log expectations."""
    il = _il()
    reg = R.get_suite(suite_key)

    fields = ["BH_Mass", "BH_Mdot", "BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]
    bhs = il.snapshot.loadSubset(path, snapnum, "PartType5", fields=fields, sq=False)

    qm = bhs.get("BH_CumEgyInjection_QM", np.array([]))
    rm = bhs.get("BH_CumEgyInjection_RM", np.array([]))

    family = reg["sim_family"]

    if family == "TNG":
        assert qm.max() > 0, "TNG: BH_CumEgyInjection_QM should be non-zero"
        assert rm.max() > 0, "TNG: BH_CumEgyInjection_RM should be non-zero"
        print(f"  [OK] TNG mode logs: QM max={qm.max():.3e}  RM max={rm.max():.3e}")

    elif family == "EAGLE":
        assert qm.max() > 0, "EAGLE: BH_CumEgyInjection_QM (thermal) should be non-zero"
        assert rm.max() == 0, (
            f"EAGLE: BH_CumEgyInjection_RM should be identically 0 — got max={rm.max():.3e}. "
            "Dataset version may have changed."
        )
        print(f"  [OK] EAGLE zero-kinetic anchor: QM max={qm.max():.3e}  RM≡0 confirmed")

    elif family == "SIMBA":
        assert qm.max() == 0, (
            f"SIMBA: BH_CumEgyInjection_QM should be 0 — got max={qm.max():.3e}"
        )
        assert rm.max() == 0, (
            f"SIMBA: BH_CumEgyInjection_RM should be 0 — got max={rm.max():.3e}"
        )
        print(f"  [OK] SIMBA: both CumEgyInjection fields confirmed zero (no logged ground truth)")

    return dict(qm_max=float(qm.max()), rm_max=float(rm.max()),
                qm_populated=(qm.max() > 0), rm_populated=(rm.max() > 0))


def check_wind_particles(path, snapnum, suite_key):
    """Count wind-phase PT4 particles (GFM_StellarFormationTime < 0)."""
    il = _il()
    reg = R.get_suite(suite_key)
    try:
        gform = il.snapshot.loadSubset(
            path, snapnum, "PartType4",
            fields=["GFM_StellarFormationTime"], sq=False
        )["GFM_StellarFormationTime"]
    except Exception:
        print("  [WARN] Could not load GFM_StellarFormationTime for PT4")
        return None

    n_wind = (gform < 0).sum()
    n_star = (gform > 0).sum()

    if reg["sim_family"] == "TNG":
        assert n_wind > 0, "TNG: expected wind-phase PT4 particles (GFM_StellarFormationTime < 0)"
        print(f"  [OK] TNG wind PT4: {n_wind:,} wind  {n_star:,} real stars")
    else:
        if n_wind > 0:
            print(f"  [WARN] {suite_key}: unexpected wind PT4 particles: {n_wind:,}")
        else:
            print(f"  [OK] {suite_key}: no wind PT4 ({n_star:,} real stars only)")

    return dict(n_wind=int(n_wind), n_real_stars=int(n_star))


def inspect_suite(suite_key):
    reg  = R.get_suite(suite_key)
    path = reg["path"]
    snap = R.snapnum_for_z(suite_key, 0.0)

    print(f"\n{'='*60}")
    print(f"  {suite_key}  (snap {snap}, z=0)")
    print(f"{'='*60}")

    # ── header ────────────────────────────────────────────────────────────────
    hdr = load_header(path, snap)
    print(f"\nHeader:")
    for key in ["HubbleParam", "Time", "Redshift", "BoxSize",
                "UnitMass_in_g", "UnitLength_in_cm", "UnitVelocity_in_cm_per_s"]:
        print(f"  {key}: {hdr.get(key, 'MISSING')}")

    # ── particle fields ───────────────────────────────────────────────────────
    for ptype, label in [("PartType0", "Gas"), ("PartType4", "Stars"), ("PartType5", "BHs")]:
        fields = field_list(path, snap, ptype)
        print(f"\n{label} ({ptype}) fields ({len(fields)}):")
        print("  " + "  ".join(fields))

    # ── required gas fields ───────────────────────────────────────────────────
    print("\nRequired gas field checks:")
    gas_required = ["Coordinates", "Velocities", "Masses", "Density",
                    "InternalEnergy", "ElectronAbundance", "StarFormationRate"]
    gas_fields = field_list(path, snap, "PartType0")
    for f in gas_required:
        status = "[OK]" if f in gas_fields else "[MISSING]"
        print(f"  {status} {f}")
    if "ElectronAbundance" not in gas_fields:
        print("  [WARN] ElectronAbundance absent — temperature needs documented fallback μ")

    # ── required BH fields ────────────────────────────────────────────────────
    print("\nBH mode-log checks (§A assertions):")
    mode_info = check_bh_mode_logs(path, snap, suite_key)

    # ── wind particle check ───────────────────────────────────────────────────
    print("\nWind particle check:")
    wind_info = check_wind_particles(path, snap, suite_key)

    # ── SubhaloFlag ───────────────────────────────────────────────────────────
    il = _il()
    try:
        sub = il.groupcat.loadSubhalos(path, snap, fields=["SubhaloFlag"])
        has_flag = True
        print(f"\n  SubhaloFlag: present ({len(sub):,} subhalos)")
    except Exception:
        has_flag = False
        print("\n  SubhaloFlag: ABSENT — central selection must use GroupFirstSub only")
    if reg["sim_family"] == "SIMBA":
        assert not has_flag, "SIMBA: SubhaloFlag should be absent per §A"

    # ── capability table row ──────────────────────────────────────────────────
    print(f"\nCapability summary for {suite_key}:")
    caps = [
        ("mode_logs_qm",    mode_info["qm_populated"],       reg["mode_logs_qm"]),
        ("mode_logs_rm",    mode_info["rm_populated"],        reg["mode_logs_rm"]),
        ("has_wind_pt4",    wind_info["n_wind"] > 0 if wind_info else False,
                                                              reg["has_wind_pt4"]),
        ("has_subhalo_flag", has_flag,                        reg["has_subhalo_flag"]),
    ]
    all_ok = True
    for name, observed, expected in caps:
        match = observed == expected
        all_ok = all_ok and match
        tag = "[OK]" if match else "[MISMATCH — check dataset version]"
        print(f"  {tag}  {name}: observed={observed}  registry={expected}")

    return all_ok


def main():
    suites = sys.argv[1:] if len(sys.argv) > 1 else list(R.PRIMARY_TRIPLET)
    results = {}
    for key in suites:
        try:
            ok = inspect_suite(key)
            results[key] = "PASS" if ok else "MISMATCH"
        except Exception as e:
            results[key] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
