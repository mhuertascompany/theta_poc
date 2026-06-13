# PILOT_PLAN.md — One-week CAMELS pilot for MANIFOLD (theta_poc extension)

Goal: two preliminary-results deliverables for the ERC AdG 2026 proposal (Panel PE9, Aug 2026):

- **Figure P1 (core, must complete):** identical CAMELS feedback input parameters land at
  different measured descriptor loci in TNG vs SIMBA — input parameters are non-commensurable,
  protocol-measured descriptors are the common language.
- **Figure P2 (stretch):** NPE recovers two descriptors from catalog-level observables, trained
  on CAMELS-TNG LH, with SBC coverage diagnostics; optional cross-code evaluation on SIMBA.

## Execution model (two stages — the key architectural decision)

- **Stage A — heavy extraction, runs ON THE CAMELS BINDER** (binder.flatironinstitute.org,
  CAMELS_PUBLIC; user has an account; the public release is already mounted, so NO downloads).
  Reads multi-GB snapshots, writes TINY per-sim parquet descriptor tables (KB–MB each).
  Download only those parquets out of Binder.
- **Stage B — inference + plotting, runs ANYWHERE (laptop/lab server).** Consumes only the
  parquet tables: SBI training, SBC, and both figures. No simulation data, no GPU needed.

Binder is ephemeral and RAM-limited; the plan is built to be **resumable, incremental, and
memory-frugal** so a session dying mid-run costs nothing. Jean Zay's $DSDIR holds the TNG LH
set locally and is retained as a fallback for the heavy LH extraction only (see §6).

Existing pipeline: `theta_poc` (frozen protocol in `config.py`; suite differences only in
`registry.py`; physics only in `descriptors/` and `selection.py`; an h5py single-file loader
path already exists via `load_halo_from_hdf5`).

---

## 0. Verified CAMELS facts (do NOT deviate; read values from disk where possible)

- **Parameter index mapping:** p1=Omega_m, p2=sigma_8, **p3=A_SN1, p4=A_SN2, p5=A_AGN1,
  p6=A_AGN2**. (Common error: swapping p4/p5. Do not.)
- **Physical meanings differ by suite (this IS the experiment):**
  - TNG: A_SN1 = wind energy per unit SFR; A_SN2 = wind speed; A_AGN1 = kinetic-mode BH energy
    per unit accretion; A_AGN2 = kinetic-mode ejection speed/burstiness.
  - SIMBA: A_SN1 = wind mass-loading factor; A_SN2 = wind speed; A_AGN1 = QSO/jet momentum
    flux; A_AGN2 = jet speed.
- **VERIFIED FROM THE BINDER `CosmoAstroSeed` HEADERS — use the canonical 6-parameter cut
  (p1..p6); IGNORE p7..p28.** The column order in both suites' files is the parameter index:
  p1=Omega0, p2=sigma8, then the four feedback parameters at p3..p6, then fixed nuisance/cosmology
  columns (OmegaBaryon, HubbleParam, n_s, ...). The feedback four, aligned BY ROLE and BY INDEX
  (they coincide here), with the canonical A_SN1/A_AGN1/A_SN2/A_AGN2 ordering:
  - **p3 = A_SN1:** TNG `WindEnergyIn1e51erg` (wind energy per SFR) | SIMBA
    `GALSF_SUBGRID_DAA_LOADFACTOR` (wind mass loading).
  - **p4 = A_AGN1:** TNG `RadioFeedbackFactor` (kinetic-mode BH energy) | SIMBA
    `BH_BAL_KICK_MOMENTUM_FLUX` (QSO/jet momentum flux).
  - **p5 = A_SN2:** TNG `VariableWindVelFactor` (wind speed) | SIMBA `GALSF_SUBGRID_FIREVEL`
    (wind speed).
  - **p6 = A_AGN2:** TNG `RadioFeedbackReiorientationFactor` | SIMBA `BH_QUENCH_JET` (jet speed).
  Same index = same nominal multiplier but DIFFERENT physical quantity across codes — this is the
  non-commensurability thesis, visible in the inputs themselves (note it in the P1 caption).
- **Cosmology is held fixed automatically:** in the p3..p6 rows Omega0=0.3, sigma8=0.8 and all
  nuisance columns are at fiducial — only the one feedback parameter moves. Selecting p3..p6 IS
  the fixed-cosmology cut; do NOT use p1/p2 (those vary cosmology) for P1.
- **EXACT P1 INPUT SET per suite (hard-wired; 16 feedback sims + 1 fiducial = 17):**
  `1P_p3_{n2,n1,1,2}` (A_SN1), `1P_p4_{n2,n1,1,2}` (A_AGN1), `1P_p5_{n2,n1,1,2}` (A_SN2),
  `1P_p6_{n2,n1,1,2}` (A_AGN2), plus `1P_0` (shared fiducial = the `_0` point of every curve).
- **Take exact x-axis values from the file** (the multiplier grid is parameter-specific, e.g.
  TNG p3 WindEnergy = {1.8, 2.7, 3.6, 5.4, 7.2}×1e51 erg ⇒ ×{0.5,0.75,1,1.5,2}); the probe reads
  these into `onep_map.csv`. Directory selection and role mapping above are verified — do not
  re-derive them from indices blindly for p7+.
- **CONFIRMED PATHS ON THIS BINDER** (mount root `/home/jovyan/Data/`):
  - Snapshots: `/home/jovyan/Data/Sims/<suite>/1P/1P_pX_Y/snapshot_090.hdf5`
  - Catalogs (co-located in the same dir; also mirrored under `FOF_Subfind/<suite>/1P/...`):
    `groups_090.hdf5`. Use the `Sims/` copy — snapshot + catalog sit together.
  - z=0 = **`*_090.hdf5`** (confirm Redshift≈0 from header). Many snapshots present → z>0 free.
  - Suites under `Sims/`: IllustrisTNG, SIMBA, Swift-EAGLE, Astrid (+ sets BE, CV, EX, LH, SB28,
    zoom, L25n256/L50n512). **SIMBA + catalogs confirmed.** Swift-EAGLE + Astrid = free bonus (§3).
- Both suites: h=0.6711 (fiducial), 25 Mpc/h box, 256^3, gas mass ~1.27e7 Msun/h.
  Snapshots are Gadget/Arepo HDF5 (`import hdf5plugin` before h5py reads).
- **SIMBA quirks:** snapshots NOT sorted by group membership; duplicate ParticleIDs;
  PartType4 lacks `GFM_InitialMass`. Catalogs are uniform SUBFIND format for both suites.
- Contact on any access anomaly: camel.simulations@gmail.com (F. Villaescusa-Navarro).

## 1. Data inventory and the Day-1 Binder probe

**Available without download:**
- Binder CAMELS_PUBLIC, mount root `/home/jovyan/Data/` (CONFIRMED): `Sims/` and
  `FOF_Subfind/` trees, suites IllustrisTNG / SIMBA / Swift-EAGLE / Astrid, sets 1P / CV / LH,
  snapshots + catalogs. **This is what the pilot uses.** SIMBA + catalogs confirmed present.
- Jean Zay $DSDIR (fallback only): CAMELS-IllustrisTNG LH_0..LH_499, legacy naming
  (snap_000..snap_033), `CosmoAstro_params.txt`, BH details, NO catalogs.

**`probe_binder.py` (FIRST script run on Binder; GATE 0):** report, by inspecting disk —
- confirm mount root `/home/jovyan/Data/` and that `Sims/<suite>/1P/` snapshots exist
  alongside the confirmed `FOF_Subfind/<suite>/1P/` catalogs (open one `snapshot_090.hdf5`);
- **read the 1P parameter values:** the directory→parameter mapping is VERIFIED (p3=A_SN1,
  p4=A_AGN1, p5=A_SN2, p6=A_AGN2 by role; see §0). The probe only needs to parse
  `CosmoAstroSeed_<suite>_L25n256_1P.txt` to extract the exact numeric multiplier grid per
  feedback parameter (relative to the `_0` fiducial) into `provenance/onep_map.csv`, and assert
  that cosmology/nuisance columns are fiducial in all p3..p6 rows (sanity: only one feedback
  value moves per row). Restrict strictly to p3..p6 + `1P_0`; ignore p1, p2, p7..p28;
- confirm the z=0 snapnum from headers (expected `*_090.hdf5`, Redshift≈0) for both a snapshot
  and a catalog;
- which suites have full 1P + catalogs (note Swift-EAGLE / Astrid for the optional multi-code
  P1);
- available RAM (`free` / `/proc/meminfo`) and core count; whether `pyarrow` is importable
  (else Stage A writes CSV/Feather);
- a TIMING TEST: full descriptor extraction on ONE 1P sim end-to-end; report wall-time and
  peak RSS. This calibrates how many sims fit a session.

**GATE 0 outcomes:** SIMBA-1P + catalogs confirmed present and the p3..p6 feedback mapping is
verified (§0), so Figure P1 is unblocked and unambiguous. The probe's remaining job is to read
the exact multiplier grid into `onep_map.csv` and confirm cosmology is fiducial across p3..p6
rows. If peak RSS is near the RAM limit → enforce masked-field loading (§2). If one-sim
wall-time is large → cap Exp 2 LH count (§4) or move Exp 2 to JZ (§6).

## 2. Code changes to theta_poc (suite differences ONLY in registry.py)

- **registry.py:** add entries `camels_tng_1p`, `camels_simba_1p`, `camels_tng_cv0`,
  `camels_simba_cv0`, `camels_tng_lh` (optionally `camels_eagle_1p`, `camels_astrid_1p` for
  the bonus). Confirmed path templates (root `/home/jovyan/Data/`):
  `Sims/<suite>/1P/1P_<i>/snapshot_090.hdf5` and
  `FOF_Subfind/<suite>/1P/1P_<i>/groups_090.hdf5` (verify the `Sims/` leaf in the probe).
  The 1P directory→(parameter,value) mapping comes from `provenance/onep_map.csv` built by the
  probe from `CosmoAstroSeed_<suite>_L25n256_1P.txt` — registry/runner consume that CSV, never
  the `pX` index directly (SB28 indices are not aligned across suites). Confirmed path template
  (root `/home/jovyan/Data/`): `Sims/<suite>/1P/1P_pX_Y/snapshot_090.hdf5` and the co-located
  `groups_090.hdf5` in the same directory. z→snapnum = `090` (confirm from header). Capability flags: SIMBA → `has_GFM_InitialMass=False`,
  `snapshot_sorted_by_group=False`, `duplicate_particle_ids=True`. h read from header.
- **loaders.py:** new `load_halo_camels(snap_path, cat_path, halo_index, cfg)` extending the
  h5py path, **memory-frugal and identical for both suites**:
  1. read Group fields (GroupPos, GroupVel, Group_M_Crit200, Group_R_Crit200, GroupFirstSub)
     and the central SubhaloVel;
  2. read **`PartType0/Coordinates` only** (~200 MB), apply periodic minimum-image to GroupPos,
     build the in-sphere boolean mask within R200c;
  3. read remaining gas fields (Velocities, Masses, Density, InternalEnergy, ElectronAbundance,
     StarFormationRate) **masked**, so peak memory stays < ~1 GB; same for PartType4 where
     p★/m★ needs it.
  Sidesteps SIMBA's unsorted snapshots and duplicate IDs by design (never use group-membership
  offsets or ID matching for CAMELS). Units via existing `units.py` (h, a from header). SIMBA
  p★/m★: fall back to `Masses` for `GFM_InitialMass` behind the flag (logged).
- **config.py additions (and ONLY here):**
  - `PIVOT = dict(logM200_lo=11.75, logM200_hi=12.25)` — descriptor summary bin.
  - `CAMELS_HALO_SELECT = HALO_SELECT | dict(logM200_max=13.0)` — 25 Mpc/h volume cap.
  - AGN-channel descriptors (f_duty, mode_balance) OFF for CAMELS (too few massive BHs; state
    in P1 caption).
- **run_camels.py (RESUMABLE):** iterate sims of a set; if a sim's output parquet exists, SKIP;
  else extract and write immediately. One parquet per sim. Metadata: protocol hash, suite, set,
  sim id, CAMELS release era, input-parameter vector from the CosmoAstro file, n_halos in pivot
  bin, theta_poc git commit. One-line progress log per sim so a dying session leaves a clean
  resume point.
- **inspect_fields.py:** extend with CAMELS expectations (assert fields per suite; assert SIMBA
  lacks GFM_InitialMass rather than silently proceeding). Fail-fast.
- **sbi_pilot.py, fig_p1.py, fig_p2.py:** Stage B (run elsewhere on downloaded parquets).
- Binder env is preinstalled (h5py, hdf5plugin, numpy, pandas). Stage B env: python ≥3.10,
  pandas, pyarrow, matplotlib, torch (CPU), `sbi` (`from sbi.inference import NPE`;
  `from sbi.diagnostics import run_sbc, check_sbc`; `sbc_rank_plot`).

## 3. Experiment 1 — input non-commensurability (core; Stage A on Binder)

- Run frozen protocol at z=0 on the verified P1 input set per suite (§0): `1P_p3_{n2,n1,1,2}`
  (A_SN1), `1P_p4_{n2,n1,1,2}` (A_AGN1), `1P_p5_{n2,n1,1,2}` (A_SN2), `1P_p6_{n2,n1,1,2}`
  (A_AGN2), plus `1P_0` (shared fiducial) = 16 + 1 = 17 sims/suite, 34 total. Cosmology is fixed
  by construction. Descriptors: **η_M, f_hot (headline)**; ε_ff, p★/m★ (secondary). One small
  parquet per sim; resumable runner; download out of Binder.
- **Figure P1 (Stage B):** 2 rows (η_M, f_hot at pivot) × 4 columns (A_SN1, A_AGN1, A_SN2,
  A_AGN2); x = input multiplier (log) from `onep_map.csv`, 5 points/curve (n2,n1,fiducial,1,2);
  TNG and SIMBA overlaid; shaded 16–84% bootstrap halo-to-halo bands; dotted line at fiducial.
  Caption (draft): "Emergent feedback descriptors measured with a single frozen protocol on the
  CAMELS one-parameter sets of IllustrisTNG (AREPO) and SIMBA (GIZMO) at z=0, fixed cosmology,
  pivot halo mass log M200c = 12.0 ± 0.25. At each index the two codes vary a DIFFERENT physical
  quantity (e.g. p3 = wind energy in TNG vs wind mass loading in SIMBA), so identical nominal
  multipliers (dotted line: shared fiducial) land at systematically different, differently
  sloped descriptor loci: input parameters are non-commensurable across codes, whereas
  protocol-measured descriptors provide the common physical language. Bands: 16–84% halo-to-halo
  scatter. AGN-channel descriptors descoped at this box size."
- **Optional multi-code bonus (free, Swift-EAGLE + Astrid also mounted):** add EAGLE and Astrid
  1P feedback sims to P1 as 2 more overlaid curves per panel. Turns a 2-code into a 4-code
  non-commensurability statement — substantially stronger for the proposal — for the cost of
  two more registry entries and extra extraction time. Treat as stretch within Exp 1: do TNG +
  SIMBA first (the must-have), add EAGLE/Astrid if Day-2 timing allows. NOTE EAGLE/Astrid may
  have their own field quirks (e.g. SWIFT-EAGLE field names) — run inspect_fields on one sim of
  each before trusting them; if a suite needs nontrivial loader work, defer it rather than risk
  the core figure.
- **Bonus (Stage B, 1 hr):** CAMELS-TNG fiducial vs existing TNG100-1 measurement, same
  protocol = first point of the proposal's resolution-drift map. Keep separate from P1 (never
  pool resolutions silently).

## 4. Experiment 2 — easy SBI (stretch)

- **Stage A (Binder), RESUMABLE:** extract descriptors for LH-TNG. Default target **N≈200 LH**
  (statistically ample for a 2D NPE pilot and friendlier to Binder sessions); upside to the
  full 500 if sessions/timing permit, or run the full set on JZ (§6). Also compute, from
  CATALOGS ONLY, the cheap observable summaries x per sim (SMF counts ~5 bins, median f_gas vs
  M* ~3 bins, median sSFR; ~10–12 dim) — catalog-only x keeps inference fast and snapshot-free.
  Outputs: one small parquet per LH sim (θ + x + provenance); download them all.
- **Stage B (anywhere):** θ = (η_M, f_hot) at pivot; train NPE on the LH set; hold out ~20% for
  inject-and-recover; SBC rank histograms over the held-out set (uniform = calibrated). Prior:
  uniform over the empirical descriptor range of the training set. CPU-only, minutes. Stretch:
  evaluate the trained posterior on SIMBA CV_0 summaries — expected miscalibration is itself a
  non-commensurability illustration; report honestly either way.
- **Figure P2:** corner plot for one held-out sim (truth crosshairs) + SBC rank inset.
- Descopes: ≥150 valid LH θ suffices; SBC strongly non-uniform → report 1D marginals per
  descriptor; if Binder LH extraction too slow → run it on JZ $DSDIR (§6); if SIMBA cross-code
  eval blocked → drop it (optional stretch only).

## 5. Day-by-day (one person, Mon–Fri + weekend buffer)

- **D1:** open Binder; run `probe_binder.py` → **GATE 0** (confirm root `/home/jovyan/Data/`
  and `Sims/` snapshot path; resolve the 1P scheme + build `onep_map.csv` from
  `CosmoAstroSeed.txt`; confirm snapnum 090 / Redshift≈0; RAM; pyarrow; one-sim timing). Code
  loader + registry against the confirmed paths; deploy theta_poc into the Binder session (git
  clone or upload). **GATE A:** inspect_fields passes on 1 TNG + 1 SIMBA file.
- **D2:** run Exp 1 on both 1P sets + fiducials (resumable; ~36 sims). Download parquets as they
  appear. Sanity-check fiducial η_M, f_hot. **GATE C:** ≥~20–30 centrals in pivot bin per sim
  (else pivot → 11.75±0.25 or widen bin).
- **D3 (Stage B, local):** assemble Exp 1 tables; bootstrap bands; build Figure P1; the TNG100-1
  resolution-drift bonus point.
- **D4 (Stage A, Binder):** launch resumable LH-TNG extraction (target N≈200; let it run across
  sessions if needed); build catalog-only x. **GATE D:** ≥~150 LH sims yield valid pivot θ.
- **D5 (Stage B, local):** train NPE; held-out recovery; SBC; optional SIMBA cross-code; draft
  Figure P2.
- **Weekend:** finalize figures; archive parquets + protocol hash + provenance; write B2
  paragraph.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Binder session times out mid-run | per-sim resumable runner (skip existing parquets); write + download outputs incrementally; never rely on Binder persistence |
| Binder RAM too small for full snapshot | masked-field loader (read Coordinates → sphere mask → read other fields masked); peak < ~1 GB |
| SIMBA-1P or catalogs not mounted on Binder (GATE 0 fail) | email CAMELS team; relay SIMBA-1P only (download externally + rsync); Exp 1 is the sole dependency |
| Binder too slow for full LH extraction | reduce to N≈150–200; OR run LH extraction on JZ ($DSDIR TNG LH is local — needs only the ~8 GB catalogs pushed in once via rsync, then a 500-task job array) |
| Pivot-bin statistics thin | lower pivot to 11.75; widen bin to ±0.3; report medians of ≥3 mass bins |
| NPE miscalibrated | more held-out SBC; catalog-only summaries; 1D marginals fallback |
| SIMBA field surprises | inspect_fields fail-fast; capability flags; spatial loader avoids ID/offset issues by design |

## 7. B2-ready preliminary-results paragraph (draft)

"As proof of concept, we applied the frozen six-descriptor protocol to the public CAMELS
one-parameter sets of two independent codes — IllustrisTNG (AREPO) and SIMBA (GIZMO) — at z=0.
At fixed pivot halo mass (log M200c = 12.0±0.25), simulations sharing identical nominal feedback
input parameters occupy systematically different, differently sloped loci in descriptor space
(Fig. P1): input parameters are not commensurable across codes, whereas the emergent descriptors
furnish a common, physically interpretable coordinate system — the premise of O1 demonstrated on
designed parameter sweeps. We further trained a neural posterior estimator on [N] CAMELS-
IllustrisTNG latin-hypercube simulations to recover these descriptors from inexpensive
catalogue-level observables with calibrated coverage (simulation-based calibration ranks
consistent with uniform; Fig. P2), establishing the descriptor-space inference engine of O3 at
pilot scale. The closest existing analyses (e.g. Medlock et al. 2025; Pandya et al. 2021)
quantify feedback energetics or mass loading within or across suites, but none measures a
unified, protocol-frozen descriptor vector across code families nor performs inference in that
space."

(Adjust [N]; cite Villaescusa-Navarro et al. 2021 for CAMELS; keep the differentiation
sentence — reviewers will check.)

## 8. Provenance requirements (non-negotiable for proposal-grade results)

Every parquet: protocol hash, suite, set, sim id, CAMELS release era, input-parameter vector
read from disk, n_halos in pivot bin, theta_poc git commit. Archive the probe report, gate
outcomes, and the exact commit used for every figure.
