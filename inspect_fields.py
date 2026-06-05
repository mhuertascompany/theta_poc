"""
inspect_fields.py — per-suite field inventory and §A assertion checks.

Usage:
    python inspect_fields.py TNG100-1
    python inspect_fields.py              # all suites in PRIMARY_TRIPLET

Runs on the cluster where /virgotng is accessible.
illustris_python is imported lazily so the file can be imported locally.
Field lists are read via h5py (no bulk data load) for speed.
"""

import sys
import h5py
import numpy as np
import registry as R


def _il():
    import illustris_python as il
    return il


def _snap_file(path, snapnum):
    return _il().snapshot.snapPath(path, snapnum)


def _gc_file(path, snapnum):
    return _il().groupcat.gcPath(path, snapnum)


# ── field discovery (h5py only — no bulk load) ───────────────────────────────

def snap_field_list(path, snapnum, ptype_int):
    """Return sorted field names for PartType{ptype_int} without loading data."""
    key = f"PartType{ptype_int}"
    try:
        with h5py.File(_snap_file(path, snapnum), "r") as f:
            return sorted(f[key].keys()) if key in f else []
    except Exception as e:
        return [f"ERROR: {e}"]


def gc_field_list(path, snapnum, catalog_type):
    """Return sorted field names for Group or Subhalo catalog without loading data."""
    try:
        with h5py.File(_gc_file(path, snapnum), "r") as f:
            return sorted(f[catalog_type].keys()) if catalog_type in f else []
    except Exception as e:
        return [f"ERROR: {e}"]


def load_header(path, snapnum):
    with h5py.File(_snap_file(path, snapnum), "r") as f:
        return dict(f["Header"].attrs)


# ── §A assertion checks ───────────────────────────────────────────────────────

def check_bh_mode_logs(path, snapnum, suite_key):
    """Load BH cumulative energy fields; assert §A expectations.

    For SIMBA the fields may be entirely absent (not just zeroed) — both
    outcomes mean 'no logged ground truth' and are treated identically.
    """
    il  = _il()
    reg = R.get_suite(suite_key)

    available = set(snap_field_list(path, snapnum, 5))
    load_fields = ["BH_Mass", "BH_Mdot"] + [
        f for f in ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"]
        if f in available
    ]
    bhs = il.snapshot.loadSubset(path, snapnum, 5, fields=load_fields, sq=False)

    qm = bhs.get("BH_CumEgyInjection_QM", np.array([0.0]))
    rm = bhs.get("BH_CumEgyInjection_RM", np.array([0.0]))
    qm_absent = "BH_CumEgyInjection_QM" not in available
    rm_absent = "BH_CumEgyInjection_RM" not in available
    family = reg["sim_family"]

    if family == "TNG":
        assert qm.max() > 0, "TNG: BH_CumEgyInjection_QM should be non-zero"
        assert rm.max() > 0, "TNG: BH_CumEgyInjection_RM should be non-zero"
        print(f"  [OK] TNG mode logs: QM max={qm.max():.3e}  RM max={rm.max():.3e}")

    elif family == "EAGLE":
        assert qm.max() > 0, "EAGLE: BH_CumEgyInjection_QM (thermal) should be non-zero"
        assert rm.max() == 0, (
            f"EAGLE: BH_CumEgyInjection_RM must be identically 0 — got {rm.max():.3e}. "
            "Dataset version may have changed."
        )
        print(f"  [OK] EAGLE zero-kinetic anchor: QM max={qm.max():.3e}  RM≡0 confirmed")

    elif family == "SIMBA":
        # Fields absent OR present-but-zero both mean no logged ground truth — either is OK.
        assert qm.max() == 0, f"SIMBA: QM should be 0 — got {qm.max():.3e}"
        assert rm.max() == 0, f"SIMBA: RM should be 0 — got {rm.max():.3e}"
        qm_status = "absent" if qm_absent else "present but zero"
        rm_status = "absent" if rm_absent else "present but zero"
        print(f"  [OK] SIMBA: QM {qm_status},  RM {rm_status} — no logged ground truth")

    return dict(qm_max=float(qm.max()), rm_max=float(rm.max()),
                qm_populated=(qm.max() > 0), rm_populated=(rm.max() > 0))


def check_wind_particles(path, snapnum, suite_key):
    """Count wind-phase PT4 particles (GFM_StellarFormationTime < 0)."""
    il  = _il()
    reg = R.get_suite(suite_key)
    try:
        data  = il.snapshot.loadSubset(
            path, snapnum, 4,
            fields=["GFM_StellarFormationTime"], sq=False,
        )
        gform = data["GFM_StellarFormationTime"]
    except Exception as e:
        print(f"  [WARN] Could not load GFM_StellarFormationTime: {e}")
        return None

    n_wind = int((gform < 0).sum())
    n_star = int((gform > 0).sum())

    if reg["sim_family"] == "TNG":
        assert n_wind > 0, "TNG: expected wind-phase PT4 (GFM_StellarFormationTime < 0)"
        print(f"  [OK] TNG wind PT4: {n_wind:,} wind  {n_star:,} real stars")
    else:
        tag = "[WARN]" if n_wind > 0 else "[OK]"
        print(f"  {tag} {suite_key}: {n_wind:,} wind PT4  {n_star:,} real stars")

    return dict(n_wind=n_wind, n_real_stars=n_star)


# ── main inspector ────────────────────────────────────────────────────────────

def inspect_suite(suite_key):
    reg  = R.get_suite(suite_key)
    path = reg["path"]
    snap = R.snapnum_for_z(suite_key, 0.0)

    print(f"\n{'='*60}")
    print(f"  {suite_key}  (snap {snap}, z=0)")
    print(f"{'='*60}")

    # header
    hdr = load_header(path, snap)
    print("\nHeader:")
    for key in ["HubbleParam", "Time", "Redshift", "BoxSize",
                "UnitMass_in_g", "UnitLength_in_cm", "UnitVelocity_in_cm_per_s"]:
        print(f"  {key}: {hdr.get(key, 'MISSING')}")

    # particle field lists
    for ptype_int, label in [(0, "Gas"), (4, "Stars"), (5, "BHs")]:
        fields = snap_field_list(path, snap, ptype_int)
        print(f"\n{label} (PartType{ptype_int}) — {len(fields)} fields:")
        print("  " + "  ".join(fields))

    # group catalog field lists
    for cat in ["Group", "Subhalo"]:
        fields = gc_field_list(path, snap, cat)
        print(f"\n{cat} catalog — {len(fields)} fields:")
        print("  " + "  ".join(fields))

    # required gas fields
    print("\nRequired gas field checks:")
    gas_fields = set(snap_field_list(path, snap, 0))
    required = ["Coordinates", "Velocities", "Masses", "Density",
                "InternalEnergy", "ElectronAbundance", "StarFormationRate"]
    for f in required:
        status = "[OK]" if f in gas_fields else "[MISSING]"
        print(f"  {status} {f}")
    if "ElectronAbundance" not in gas_fields:
        print("  [WARN] ElectronAbundance absent — temperature needs documented fallback mu")

    # §A BH mode-log assertions
    print("\nBH mode-log checks (§A assertions):")
    mode_info = check_bh_mode_logs(path, snap, suite_key)

    # wind particle check
    print("\nWind particle check:")
    wind_info = check_wind_particles(path, snap, suite_key)

    # SubhaloFlag
    sub_fields = set(gc_field_list(path, snap, "Subhalo"))
    has_flag   = "SubhaloFlag" in sub_fields
    print(f"\nSubhaloFlag: {'present' if has_flag else 'ABSENT — use GroupFirstSub only'}")
    if reg["sim_family"] == "SIMBA":
        assert not has_flag, "SIMBA: SubhaloFlag should be absent per §A"

    # capability summary
    print(f"\nCapability summary for {suite_key}:")
    n_wind = wind_info["n_wind"] if wind_info else 0
    caps = [
        ("mode_logs_qm",     mode_info["qm_populated"],  reg["mode_logs_qm"]),
        ("mode_logs_rm",     mode_info["rm_populated"],  reg["mode_logs_rm"]),
        ("has_wind_pt4",     n_wind > 0,                 reg["has_wind_pt4"]),
        ("has_subhalo_flag", has_flag,                   reg["has_subhalo_flag"]),
    ]
    all_ok = True
    for name, observed, expected in caps:
        match = observed == expected
        all_ok = all_ok and match
        tag = "[OK]" if match else "[MISMATCH — check dataset version]"
        print(f"  {tag}  {name}: observed={observed}  registry={expected}")

    return all_ok


def main():
    suites  = sys.argv[1:] if len(sys.argv) > 1 else list(R.PRIMARY_TRIPLET)
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
