# θ_feedback PoC — invariants for Claude

## Physics invariants

1. **All physics lives in `selection.py` and `descriptors/`.** `run_suite.py` is pure iteration and I/O — no formulas, no masks, no unit conversions there.

2. **One frozen protocol, zero suite branches inside descriptors.** Per-suite differences (paths, snapshot maps, unit quirks, field availability) belong only in `registry.py` and `units.py`. If you find yourself writing `if suite == "EAGLE"` inside a descriptor or `selection.py`, stop — it is either a registry/units concern or a commensurability violation.

3. **Never hardcode `h`, `a`, or `snapnum`.** Read `HubbleParam` and `Time` (scale factor) from each snapshot header. Read snapshot numbers from the `registry.py` z→snapnum map. TNG h≈0.6774, EAGLE h≈0.6777, SIMBA h=0.68 — all differ; sharing a constant is a silent bug.

4. **Never "fix" a descriptor that returns ~0 or NaN for EAGLE's kinetic mode.** EAGLE has one thermal AGN mode; `BH_CumEgyInjection_RM ≡ 0` by construction. The emergent mode-balance proxy returning ≈0 for EAGLE is a headline result (ground-truth zero anchor), not an error. Return `mode_defined=False` or flag it; do not force a positive number.

## Execution-model invariants (critical)

5. **This laptop has no access to `/virgotng` and cannot import `illustris_python`.** All real runs happen as Slurm jobs on a remote cluster, executed by the user, synced via GitHub. Claude never runs the pipeline end-to-end locally.

6. **`loaders.py` must import `illustris_python` lazily** — inside functions, never at module top. A bare `import illustris_python` at the top of any file breaks every local invocation.

7. **All logic must be testable locally against synthetic TNG-format HDF5 fixtures with known answers.** Every descriptor, unit conversion, and selection function needs a `pytest` test that passes without cluster access. Verify logic locally; the user runs real data on the cluster and pastes back logs/outputs.

## Descriptor signature contract

Every descriptor in `descriptors/` exposes exactly:
```python
def compute(halo: dict, cfg: dict) -> dict: ...
```
`halo` contains physical-unit arrays (post-conversion). Return dict must include `value`, `n_used`, and any diagnostic keys. No other public API.

## Key per-suite facts (encoded in `registry.py`, not rediscovered at runtime)

- **TNG:** wind PT4 particles present (`GFM_StellarFormationTime < 0`); always select stars with `> 0`. Both `BH_CumEgyInjection_QM/RM` populated.
- **EAGLE:** no wind PT4; `RM ≡ 0` (single thermal mode); 29 snapshots; h≈0.6777.
- **SIMBA:** both `CumEgyInjection` fields zero (no logged ground truth); duplicate `ParticleIDs`; no `SubhaloFlag` → central selection via `GroupFirstSub` only; 152 snapshots (snap 126 of L50n512FP corrupt, exclude); `GFM_InitialMass` reconstructed.
- Central selection uses `GroupFirstSub` for **all** suites (not `SubhaloFlag`, which is absent in SIMBA).
- `validate_modes.py` runs TNG (full) + EAGLE (zero anchor) only; must assert-and-skip for SIMBA.
