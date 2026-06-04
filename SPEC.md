# θ_feedback proof-of-concept — implementation brief

**Goal.** Compute the six `θ_feedback` descriptors with **one frozen protocol applied identically** to three TNG-ized suites (IllustrisTNG, EAGLE, SIMBA), all rewritten into identical Illustris/TNG format on `/virgotng/universe/`, and produce two figures: (1) descriptor loci vs halo/BH mass for the three suites, (2) validation of the emergent AGN mode-balance proxy against the **logged** per-mode injected energy — full kinetic/thermal ground truth in TNG, a ground-truth *zero-kinetic anchor* in EAGLE (which logs only a thermal mode), and emergent-only for SIMBA (whose logs are unavailable). See §A for the per-suite facts that drive these choices.

**Guiding principle — read this first.** All three suites share *format* (HDF5 layout, field names, `illustris_python` loadability), **not** *physics*. The entire scientific value of this PoC is keeping those two things separate. A uniform pipeline will run cleanly on all three and emit six numbers per suite; for some descriptors those numbers are genuinely commensurable (η_M, f_hot, f_duty), for others they collapse toward each suite's own input (ε_ff, p★/m★) or are structurally absent (the kinetic AGN mode in single-mode EAGLE). The code must surface that, not hide it. **Do not "fix" a descriptor that returns ~0 or NaN for EAGLE's kinetic mode — that is the correct, informative answer, and EAGLE's own logs confirm it.**

> **The READMEs are authoritative — read §A before coding.** The three suites were reprocessed by the TNG team (Subfind re-run from scratch, so group-catalog indices do *not* correspond to the original public releases — irrelevant for us, we work per-snapshot). But each has documented field removals, zeroed fields, and conventions that directly determine descriptor computability. These are captured in §A and must be encoded in `registry.py`, not rediscovered at runtime.

---

## A. Per-suite facts (from the dataset READMEs) — encode these in `registry.py`

Paths (both the alias names and the `L…` names appear on disk; verify which directory holds the `output/` with snapshots + group catalogs, and use that as the `illustris_python` basePath):
```
/virgotng/universe/IllustrisTNG/   TNG50-1 (=L35n2160TNG)  TNG100-1 (=L75n1820TNG)  TNG300-1 (=L205n2500TNG) ...
/virgotng/universe/Eagle/          Eagle100-1 (=L68n1504FP, 100 cMpc, h≈0.6777, 29 snaps)   RecalL0025N0752 (also TNG-ized)
/virgotng/universe/Simba/          Simba100-1 (=L100n1024FP)  Simba50-1 (=L50n512FP)  Simba25-1 (=L25n512FP)  (152 snaps; h=0.68)
```

**Resolution matching — you CAN match, so do it (this strengthens the result).** Gas-element masses (M⊙):

| tier | TNG | EAGLE | SIMBA |
|---|---|---|---|
| ~10⁵ | TNG50-1  (8.5e4) | Recal25  (2.3e5) | — |
| **~10⁶ (primary)** | **TNG100-1  (1.4e6)** | **Eagle100-1  (1.8e6)** | **Simba25-1  (2.3e6)** |
| ~10⁷ | TNG300-1 / TNG100-2  (1.1e7) | — | Simba100-1 / Simba50-1  (1.8e7) |

- **Primary science comparison = the ~10⁶ matched triplet:** TNG100-1 + Eagle100-1 + Simba25-1, all within ×1.6 in mass resolution. Differences in θ here are physics, not numerics — this is the defensible cross-suite statement and should be Figure 1.
- **Volume tradeoff:** Simba25-1 is a small box (~37 cMpc, ~20× less volume than Eagle100-1) and runs out of halos above ~10¹³ M⊙. Extend the high-mass / AGN end with the flagship **Simba100-1** (largest volume), shown alongside, *licensed by* the within-SIMBA convergence below.
- **Within-suite convergence (robustness panel):** show each descriptor is resolution-stable along each suite's own ladder — TNG {50-1, 100-1, 300-1}, SIMBA {m25, m100}, EAGLE {Recal25, Eagle100-1}. If each suite converges *and* the matched triplet separates, the suite ordering is bulletproof.
- `RecalL0025N0752` is **also TNG-ized** — the same `illustris_python` loader works on it. Use it for the EAGLE high-resolution convergence point (2.3×10⁵ M⊙ gas, ×8 finer than Eagle100-1); the flagship `Eagle100-1` carries the main analysis. (Note its different h/box are read from the header like any other run.)

`registry.py` must hold an explicit `z_target → snapnum` map per suite (TNG100: 100 snaps, z=0→99; EAGLE: 29 snaps; SIMBA: 152 snaps) — never hardcode 99. SIMBA `L50n512FP` snapshot 126 is corrupt — exclude it. Read h and a from each header (TNG 0.6774, EAGLE 0.6777, SIMBA 0.68 — all different).

**SIMBA (`L100n1024FP`):**
- `BH_CumEgyInjection_{QM,RM}` and `BH_CumMassGrowth_{QM,RM}` are **all zero (n/a)** → **no logged mode ground truth.** SIMBA mode balance is emergent-proxy-only, licensed by the TNG/EAGLE validation.
- `BH_Mdot` = sum of torque-limited + Bondi (Davé+2019 Eq. 6); `BH_MdotBondi` not separately meaningful. λ_Edd and f_duty are still computable from total `BH_Mdot`; flag the accretion-model difference in interpretation.
- `BH_BPressure, BH_Density, BH_HostHaloMass, BH_Pressure, BH_U` all zero — do not use.
- **Duplicate `ParticleID` values exist** (multiple stars from one gas particle; enrichment). No `SubLink_gal` trees, **no `SubhaloFlag` field.** → (a) never use ID-based matching for SIMBA without the modulus-to-progenitor trick; (b) central selection must use `GroupFirstSub` from the group catalog, **not** `SubhaloFlag`.
- `GFM_InitialMass` was **reconstructed** (BC03/Chabrier), not native — fine for "mass formed" in p★/m★, but flag it as reconstructed.
- Extra fields: `Simba_FractionH2` (on-the-fly H₂ mass fraction; SIMBA's SF law is H₂-based) and `Simba_DustMass`/`Simba_DustMetallicity` (metals moved from gas to dust). **Keep the common SF-gas mask as `StarFormationRate > 0` for all suites**; optionally *also* report ε_ff on H₂ gas for SIMBA as a bonus diagnostic, but never let it replace the common protocol.
- Removed: `GFM_CoolingRate, GFM_AGNRadiation, GFM_WindDMVelDisp, GFM_WindHostHaloMass, MagneticField*, Machnumber, EnergyDissipation, CenterOfMass, GFM_MetalsTagged`, PT4 `BirthPos/BirthVel`. None are needed by the six descriptors — confirm your code never references them.

**EAGLE (`L68n1504FP`):**
- `BH_CumEgyInjection_QM` = total energy released by cumulative accreted mass (the single thermal mode); `BH_CumEgyInjection_RM` and `BH_CumMassGrowth_RM` are **zero by construction** (EAGLE has one mode). → **EAGLE is the ground-truth zero-kinetic anchor** for the validation figure: its logged kinetic fraction is exactly 0, and the emergent proxy must recover ≈0.
- **No wind-phase `PartType4` particles — all PT4 are genuine stars.** (Contrast TNG below.)
- `BH_HostHaloMass, BH_U, BH_BPressure`, all `Potential`, and `GFM_AGNRadiation, GFM_CoolingRate, GFM_WindDMVelDisp, GFM_WindHostHaloMass, MagneticField*, Machnumber, EnergyDissipation` are zero; `CenterOfMass` set equal to `Coordinates`. `GFM_Metals` are the smoothed values. None affect the six descriptors — confirm no references.
- h ≈ 0.6777 and box = 100 cMpc (≠ TNG's h, ≠ SIMBA's h=0.68). **Read h and a from each header; never share a hardcoded h across suites.**

**IllustrisTNG (`TNG100-1`):**
- Full two-mode AGN with `BH_CumEgyInjection_QM` (thermal/quasar) **and** `BH_CumEgyInjection_RM` (kinetic/radio) both populated → full mode-ratio ground truth (§6).
- **`PartType4` contains wind-phase particles**, flagged by `GFM_StellarFormationTime < 0`. **Select genuine stars with `GFM_StellarFormationTime > 0`** for all stellar-mass and recent-SF quantities. (EAGLE has none; SIMBA — apply the same `>0` guard, harmless if all positive.)

**Cross-suite consequence for the descriptor table:** AGN mode balance has logged ground truth in two suites (TNG full, EAGLE zero) and emergent-only in SIMBA; f_duty is computable in all three but rests on different accretion models; η_M / f_hot are cleanly emergent everywhere; ε_ff / p★/m★ are computable everywhere but partly recover inputs. This is exactly the informative spread — record it in the capability table.

---

Write a small `inspect_fields.py` that, for each suite at the chosen snapshot, prints:

- Available particle types and the **exact field list** for `PartType0` (gas), `PartType4` (stars), `PartType5` (BHs).
- Group/Subhalo catalog field list.
- Header values: `HubbleParam` (h), `Time` (scale factor a), `Redshift`, `BoxSize`, `UnitMass_in_g`, `UnitLength_in_cm`, `UnitVelocity_in_cm_per_s` if present.
- For gas: confirm presence of `Density`, `InternalEnergy`, `ElectronAbundance`, `StarFormationRate`, `Masses`, `Velocities`, `Coordinates`.
- For BHs: confirm `BH_Mass`, `BH_Mdot`, and the per-mode logs `BH_CumEgyInjection_QM` / `BH_CumEgyInjection_RM`. Per §A, expect: TNG both populated; EAGLE `QM` populated + `RM` zero; SIMBA both zero. **The script should assert these against §A and fail loudly if reality differs** — a mismatch means the dataset version changed.

**Output a per-suite capability table.** This table decides which descriptors are computable per suite and is itself a deliverable (the honest "per-suite accessibility" content for B2). It should encode the §A facts as verified-at-runtime booleans (mode logs present? wind PT4 present? `ElectronAbundance` present? `SubhaloFlag` present?), not just expected ones.

Do **not** assume field availability is identical across suites just because the format is. Verify `ElectronAbundance` exists for all three (needed for temperature) — it should, but the SIMBA/EAGLE removals list (§A) is long, so check rather than trust. If a suite lacks it, flag it; temperature there needs a documented fallback μ and that becomes a named systematic.

---

## 1. Repository layout

```
theta_poc/
  config.py            # frozen protocol constants + nuisance grids
  registry.py          # per-suite paths, snapshot↔redshift map, unit quirks
  units.py             # a-factor / h handling, temperature, Eddington ratio
  loaders.py           # thin wrappers over illustris_python (suite-agnostic)
  selection.py         # SHARED: shell, outflow rest-frame, hot-phase, SF-gas mask
  descriptors/
    eps_ff.py
    p_star.py
    eta_M.py
    f_hot.py
    f_duty.py
    mode_balance.py
  run_suite.py         # iterate halos → per-halo descriptor table (parquet)
  validate_modes.py    # mode-proxy vs logged ratio: TNG (full) + EAGLE (zero anchor); SIMBA skipped
  compare.py           # load 3 tables → the two money figures
  inspect_fields.py    # section 0
```

Each descriptor module exposes one pure function with an **identical signature** so the protocol cannot drift between descriptors:

```python
def compute(halo, cfg) -> dict:
    """halo: standardized dict of physical-unit arrays (see loaders).
       returns {value, plus any diagnostics, plus n_used for resolution checks}."""
```

`run_suite.py` must never contain physics — only iteration and I/O. All physics lives in `descriptors/` and `selection.py`. This is what guarantees the protocol is identical across suites.

---

## 2. Frozen protocol (`config.py`)

These are the declared, fixed choices. Everything physical is computed with these; sensitivity scans vary only the nuisance grid.

```python
COSMO = dict(  # filled per-suite from header, do not hardcode h
)
PROTOCOL = dict(
    z_target        = 0.0,        # PoC at z=0; add z=2 later, not now
    T_hot           = 10**5.5,    # K, hot-phase cut (nuisance-scanned)
    shell_frac      = 0.25,       # of R_200c for η_M flux shell (nuisance-scanned)
    shell_dr_frac   = 0.05,       # shell thickness as fraction of R_200c
    aperture_frac   = 1.0,        # of R_200c for f_hot (also test 0.25)
    inner_frac      = 0.10,       # of R_200c for AGN kinetic-power region
    eps_r           = 0.10,       # radiative efficiency for L_Edd and L_rad
    lambda_edd_thr  = 0.01,       # f_duty accretion-on threshold (nuisance-scanned)
    sf_recent_Myr   = 100.0,      # window for "recent" stellar mass (p★/m★)
    gamma           = 5.0/3.0,
    X_H             = 0.76,
)
HALO_SELECT = dict(
    centrals_only   = True,       # use GroupFirstSub (NOT SubhaloFlag — absent in SIMBA, §A)
    logM200_min     = 11.5,       # Msun
    logM200_max     = 14.0,
    min_n_gas       = 500,        # resolution floor; record n_gas always
)
HALO_BINS = np.arange(11.5, 14.01, 0.25)   # log10 M200c [Msun]
BH_BINS   = np.arange(6.5, 9.51, 0.25)     # log10 M_BH  [Msun]

NUISANCE = dict(  # run the grid, show loci are robust / report where they aren't
    T_hot       = [1e5, 3e5, 10**5.5, 1e6],
    shell_frac  = [0.25, 0.50],
    lambda_thr  = [1e-3, 1e-2, 1e-1],
)
```

**Rule:** any constant that affects a descriptor value lives here. No magic numbers inside descriptor modules.

---

## 3. Units, a-factors, temperature (`units.py`) — the highest-risk code

Get this wrong and every descriptor is silently off. TNG conventions; verify per suite from the header in section 0.

- **Length:** `Coordinates` are comoving ckpc/h → physical kpc: `x_phys = x_code * a / h`.
- **Velocity:** snapshot `Velocities` carry a √a factor → peculiar velocity `v_pec[km/s] = Velocities * sqrt(a)`. Confirm this holds for EAGLE/SIMBA TNG-izations (it should if faithfully reprocessed, but check one halo's velocity dispersion against `SubhaloVelDisp`).
- **Mass:** `Masses`, `BH_Mass` in 1e10 Msun/h → `M[Msun] = Masses * 1e10 / h`.
- **Density:** (1e10 Msun/h)/(ckpc/h)³ → physical: multiply by `1e10 * h^2 / a^3` (Msun/kpc³). Derive carefully and unit-test against a known TNG halo gas mass (sum of cell masses ≈ ∫ρdV).

**Temperature** (exclude SF gas before trusting it):
```
mu  = 4 / (1 + 3*X_H + 4*X_H*ElectronAbundance)
u_cgs = InternalEnergy * 1e10            # (km/s)^2 -> (cm/s)^2
T   = (gamma-1) * u_cgs * mu * m_p / k_B  # Kelvin
```
For gas with `StarFormationRate > 0`, T is on the effective EoS and is **fictitious** — never apply the hot-phase cut to it; exclude it first (see selection).

**Eddington ratio:**
```
Mdot_Edd = 4*pi*G*M_BH*m_p / (eps_r * sigma_T * c)   # in g/s, convert to Msun/yr
lambda_Edd = BH_Mdot / Mdot_Edd
```
**BH_Mdot units differ and are a classic error** — TNG `BH_Mdot` is in code units (1e10 Msun/h per code time 0.978 Gyr/h). Convert to Msun/yr and verify against a published TNG λ_Edd distribution before trusting f_duty. The READMEs state EAGLE and SIMBA are rewritten with units *identical to TNG*, so the same conversion should apply — **but verify per suite**, because (a) SIMBA `BH_Mdot` is a torque-limited + Bondi sum (Davé+2019 Eq. 6), a different accretion physics whose λ_Edd means something different, and (b) EAGLE's native Mdot convention was converted, not native. Sanity-check each suite's λ_Edd distribution independently.

For the **per-mode logs** used in §6, treat per suite (see §A): TNG has both `QM` and `RM`; EAGLE has `QM` (thermal) with `RM ≡ 0`; SIMBA has both zero (skip).

Provide unit tests: (a) total gas mass in one TNG halo matches `SubhaloMassType`; (b) median T of non-SF gas in a 1e13 halo is ~1e6 K; (c) λ_Edd distribution sane.

---

## 4. Shared selection layer (`selection.py`) — identical across all suites

This module is where commensurability is won or lost. One implementation, used by every descriptor.

- `to_halo_frame(gas, subhalo)`: subtract central `SubhaloVel` (in km/s, a-corrected) from gas peculiar velocity; recenter coordinates on `GroupPos`/`SubhaloPos` (minimum-image with `BoxSize`). Compute `r`, radial unit vector, `v_r = v·r̂`.
- `sf_mask(gas)`: `StarFormationRate > 0` → True means **ISM/SF gas, exclude from all hot-phase logic**. Use this everywhere temperature is used.
- `hot_mask(gas, cfg)`: `(T > cfg.T_hot) & ~sf_mask`.
- `outflow_mask(gas)`: `v_r > 0` in halo frame.
- `shell_mask(gas, R200, cfg)`: `|r - shell_frac*R200| < 0.5*shell_dr_frac*R200`.
- `aperture_mask(gas, R200, cfg)`: `r < aperture_frac*R200`.

Every selection takes physical-unit arrays and `cfg`. No descriptor re-implements selection.

---

## 5. The six descriptors (`descriptors/`)

### 5.1 η_M(M_halo) — hot-phase mass loading  *(cleanly emergent; expect clean 3-suite separation)*
Through the shell, hot + outflowing + non-SF gas:
```
Mdot_hot_out = sum( m_i * v_r,i ) / (shell_dr_frac * R200)   # mass flux
eta_M = Mdot_hot_out / SFR_halo
```
- `SFR_halo`: sum gas SFR within the aperture (define once; reuse for ε_ff normalization context).
- Convert v_r [km/s] and R200 [kpc] consistently to get Mdot in Msun/yr (1 km/s = 1.0227e-9 kpc/yr).
- Bin by `Group_M_Crit200`. **This is your cleanest descriptor — make it the lead panel.**

### 5.2 f_hot — hot-gas fraction  *(cleanly emergent)*
```
f_hot = M_gas[hot_mask & aperture_mask] / M_gas[aperture_mask]
```
Bin by M200c. Trivial, robust, and physically interpretable; SIMBA's evacuated low-mass halos should sit low.

### 5.3 f_duty(M_BH) — AGN duty cycle  *(emergent, threshold-dependent)*
Population duty cycle from a single snapshot:
```
f_duty(bin) = N(lambda_Edd > lambda_edd_thr) / N_total   within each M_BH bin
```
Use central BHs (most massive BH per central subhalo). Bin by `BH_Mass`. Carry `lambda_edd_thr` as a declared nuisance. (Optional later: true time-fraction across snapshots — not needed for PoC.)

### 5.4 mode_balance — P_kin / L_rad  *(the hard one; validated against TNG logs)*
Emergent proxy:
```
L_rad  = eps_r * BH_Mdot * c^2                                   # bolometric
P_kin  = 0.5 * sum( m_i * v_r,i^2 ) / dr   over hot+outflow gas in r < inner_frac*R200
proxy  = P_kin / L_rad
```
Expectations the code must NOT suppress (per §A):
- **TNG:** finite; validatable against *both* logged modes (§6).
- **SIMBA:** expected high at low f_Edd (bipolar jets) — two-mode but both kinetic. **No logs to validate against (both `CumEgyInjection` fields are zero)** — the proxy here is emergent-only, licensed by the TNG fit and the EAGLE zero-anchor.
- **EAGLE:** single thermal mode; logged `RM ≡ 0`, so the *true* kinetic fraction is exactly zero. The emergent proxy must recover ≈0. **Return a sentinel (`mode_defined=False` or `proxy≈0` with a flag) rather than a forced positive number.** This — emergent proxy ≈ 0 *and* logs confirm 0 — is a headline result, not an error.

### 5.5 ε_ff — SF efficiency per free-fall time  *(measurable, but recovers TNG/EAGLE input)*
Over SF gas (`StarFormationRate > 0`):
```
t_ff,i  = sqrt(3*pi / (32*G*rho_i))          # rho_i physical mass density
eps_ff  = sum(SFR_i * t_ff,i) / sum(m_gas,i)  # mass-weighted dimensionless efficiency
```
**Report honestly:** for TNG/EAGLE this largely returns the input KS normalization convolved with the density PDF; for SIMBA the H₂-based law differs. Annotate the figure/text accordingly — this is the framework correctly reporting "design choice, not emergent physics" on this axis. *Optional SIMBA-only diagnostic:* recompute using `Simba_FractionH2` to define the dense/molecular phase, reported alongside (not replacing) the common-protocol value.

### 5.6 p★/m★ — momentum per stellar mass formed  *(noisiest; flag heavily)*
```
p_out   = sum( m_i * v_r,i )  over outflow+non-SF gas in the shell   # radial momentum flux
m_recent = stellar mass formed in last sf_recent_Myr
p_star  = p_out / m_recent
```
- `m_recent`: select genuine stars with **`GFM_StellarFormationTime > 0`** (excludes TNG wind-phase PT4; harmless for EAGLE/SIMBA which have none) and formation time within `sf_recent_Myr`; sum `GFM_InitialMass` (mass at birth). **SIMBA's `GFM_InitialMass` is reconstructed (BC03/Chabrier), not native** — record that as a per-suite metadata flag on this descriptor.
- Caveats to encode in output metadata: contaminated by AGN-driven outflow at the shell; kinetic-wind models (TNG/SIMBA) decouple winds hydrodynamically then recouple — measure outside the decoupling region. Treat as supporting, not headline.

---

## 6. Mode-balance validation (`validate_modes.py`) — the centerpiece figure

The single most persuasive panel: show the emergent mode-balance proxy reproduces the *logged* injected ratio where logs exist, with EAGLE anchoring the zero. Three points, not one:

**TNG — full ground truth.** For BHs matched across two adjacent snapshots (TNG IDs are unique):
```
dE_RM = BH_CumEgyInjection_RM[t2] - BH_CumEgyInjection_RM[t1]   # kinetic/radio
dE_QM = BH_CumEgyInjection_QM[t2] - BH_CumEgyInjection_QM[t1]   # thermal/quasar
true_ratio = dE_RM / dE_QM
```
Compare `true_ratio` to the emergent `proxy = P_kin/L_rad` (§5.4), per BH and per mass bin; plot proxy vs logged ratio; quantify correlation and bias.

**EAGLE — ground-truth zero anchor.** `RM ≡ 0`, `QM` = total thermal energy. So `true_ratio = 0` exactly. Compute the emergent proxy with the *same* code and show it returns ≈0. This point pins the bottom of the validation relation and proves the proxy doesn't manufacture a kinetic signal where there is none. (Differencing `QM` across EAGLE's snapshots also gives a thermal-injection-rate cross-check on `L_rad`, though EAGLE's 29-snapshot cadence makes Δt coarse — use a wider baseline.)

**SIMBA — no logs, emergent-only.** Both `CumEgyInjection` fields are zero. `validate_modes.py` must **skip SIMBA's validation** (assert-and-skip), and the proxy value for SIMBA is reported as emergent, its credibility resting on the TNG fit + EAGLE zero.

**Argument this licenses:** the proxy tracks the logged ratio across TNG's dynamic range *and* correctly returns zero at EAGLE's logged zero → it is justified to apply the same proxy to SIMBA, where the kinetic/thermal split cannot be read from logs. State this explicitly in the figure caption. This two-suite-anchored validation is materially stronger than a single-suite fit and is the panel that answers the "is θ even well-posed?" objection head-on.

**ID-matching caution:** never reuse this matching code on SIMBA (duplicate ParticleIDs, §A). It is needed only for TNG (and optionally EAGLE), both of which have unique BH IDs.

---

## 7. Runner and outputs

`run_suite.py` per suite:
1. Load group/subhalo catalog; select centrals in the mass range; apply `min_n_gas`.
2. For each halo: load its gas/star/BH particles (use `illustris_python` halo-scoped loads to avoid reading whole snapshots), convert to physical units once, build the halo dict.
3. Call all six descriptor `compute()` functions.
4. Append a row: `suite, halo_id, M200c, R200c, M_BH, SFR, n_gas, <6 descriptors>, <diagnostics>, <protocol hash>`.
5. Write `tables/{suite}_z{z}_protocol{hash}.parquet`.

Store the **protocol hash** (hash of the frozen `PROTOCOL` + `HALO_SELECT`) in every row so figures can never mix runs with different conventions.

`compare.py` produces:
- **Figure 1 (separation):** small multiples — η_M(M200c), f_hot(M200c), f_duty(M_BH), and mode-balance per suite, three suites overlaid. Expect clean separation on η_M/f_hot, structural EAGLE≈0 on mode balance, near-input convergence flagged on ε_ff.
- **Figure 2 (validation):** the §6 proxy-vs-logged-ratio panel — TNG across its dynamic range plus the EAGLE zero-anchor point; SIMBA shown as emergent-only (its proxy plotted but with no log to compare).

---

## 8. Build order (do not skip the single-halo step)

1. `inspect_fields.py` → per-suite capability table. **Stop and read it** before coding descriptors.
2. `units.py` + unit tests (gas-mass closure, temperature sanity, λ_Edd sanity) on **one TNG halo**.
3. `selection.py` + a visual sanity check (radial profile of one halo: hot fraction rising outward, outflow gas at positive v_r).
4. η_M and f_hot end-to-end on one halo, then the full TNG mass range — confirm sensible η_M(M_halo).
5. f_duty and the mode-balance proxy on TNG; then `validate_modes.py` on TNG. **Land the TNG validation before touching the other suites.**
6. ε_ff and p★/m★ (with their honesty flags: `GFM_StellarFormationTime > 0` star selection; SIMBA reconstructed-mass flag).
7. Bring in **EAGLE next** (not SIMBA) — it is the cheaper, cleaner second suite and gives you the validation zero-anchor immediately: run the full pipeline + add EAGLE's `RM≡0` point to the validation figure. Then SIMBA. **Change only `registry.py` paths/snapshot maps, nothing in `descriptors/` or `selection.py`.** If you find yourself special-casing a suite inside a descriptor, stop: either it's a genuine convention difference (→ `registry.py`/`units.py`) or you're about to break commensurability.
8. Resolution: the **primary Figure 1 uses the matched ~10⁶ triplet** (TNG100-1, Eagle100-1, Simba25-1). Add the within-suite convergence panel (TNG {50-1,100-1,300-1}, SIMBA {m25,m100}, EAGLE {Recal25, Eagle100-1}) — all through the same loader — and only then extend the high-mass end with Simba100-1.
9. `compare.py` → the two figures.

---

## 9. Known systematics to carry, not bury

- **Resolution (now a controlled check, not an unavoidable systematic):** matched-resolution runs exist (§A) — the primary comparison TNG100-1 + Eagle100-1 + Simba25-1 sits within ×1.6 in gas-element mass, so the suite separation there is physics. Still record `n_gas` per halo, demonstrate within-suite convergence along each ladder, and only trust Simba100-1 at the high-mass end after the m25↔m100 convergence check passes.
- **Effective-EoS temperature:** SF gas T is fictitious in all three — the `StarFormationRate > 0` exclusion must be airtight and applied identically before any hot-phase cut.
- **Wind/star particle convention:** TNG stores wind-phase cells as PartType4 with `GFM_StellarFormationTime < 0`; always select stars with `>0`. EAGLE has no wind PT4; SIMBA winds are decoupled gas, not PT4 — the `>0` guard is still correct everywhere.
- **Aperture/threshold dependence:** η_M, f_hot, f_duty all move with the protocol constants — that is why the nuisance grid exists. Show the *ordering of the three suites* is robust even where absolute values shift.
- **Accretion-model difference (f_duty):** TNG/EAGLE (modified Bondi) vs SIMBA (torque-limited + Bondi sum) — λ_Edd is a clean emergent measurement but its interpretation differs; freeze the threshold and declare it.
- **SIMBA dust:** some metals are sequestered into dust (`Simba_DustMass`), so gas-phase metallicity is lower than total; irrelevant to the six descriptors but note it if metallicity ever enters a derived quantity. Confirm the gas `Masses` field is total cell mass (it is) so mass fluxes are unaffected.
- **SIMBA duplicate ParticleIDs / no SubhaloFlag:** no ID-based cross-snapshot matching without the modulus-to-progenitor trick; central selection via `GroupFirstSub` only.
- **Format ≠ physics:** re-state in the README. The §0 capability table (verified at runtime against §A) is the standing record of where the three suites genuinely differ.

---

### What "success" looks like for the proposal
Four descriptors (η_M, f_hot, f_duty, and the validated mode proxy) cleanly separating the three structurally-different feedback models in a common measured space; the mode proxy reproducing TNG's logged kinetic/thermal ratio across its range **and** returning ≈0 at EAGLE's logged zero; SIMBA's emergent mode value placed credibly on that validated relation; EAGLE correctly located in the no-kinetic-mode corner; ε_ff/p★/m★ flagged as recovering input on the suites where they do. That figure converts "the central object may be ill-posed" into "here it is, populated, validated against two independent logged anchors, and resolving real model differences."
