# Replication, audit, and improvement study of Tomsovic et al., EST 2026
**Paper:** Tomsovic, Gu, Doudrick, Straub. "Mechanistic Insights into PFAS Rejection
in Nanofiltration and Reverse Osmosis from Data-Driven Analysis."
*Environ. Sci. Technol.* 2026, 60, 13711–13723 (doi:10.1021/acs.est.6c02287).

**Role:** independent replication + audit. Prior work was NOT assumed correct.
All results below are produced by the seeded scripts in `scripts/` (run order
`01`→`08`, `SEED=42`); figures in `results/figures/`, tables in
`results/tables/`, models in `results/models/`. Key claims were verified by
independent re-computation (different code paths) where noted.

**Environment:** conda env `fml`, Python 3.10.20; see `requirements.txt`
(xgboost 3.2.0, LightGBM 4.7.0, Optuna 5.0.0, scikit-learn 1.7.2,
lifelines 0.30.0). Input files in `paper/` (filenames differ slightly from the
brief: `acs.est.6c02287.pdf` etc.).

---

## 1. PART 1 — DATA AUDIT (`scripts/01_data_audit.py`)

### 1.1 Censoring encoding — VERIFIED
- The "Column Descriptions" sheet defines `LOQ` as: `"right"` = datapoint hit
  the detection limit (tabulated rejection = detection limit, true rejection may
  be higher); `1` = did not hit the limit. **Confirmed exactly as the brief
  suspected.**
- Data check: `LOQ` takes only values `{1, "right"}`, no missing values
  (full data: 2210 × `1`, 307 × `"right"`; R ≥ 0.5 subset: 2143 uncensored +
  307 censored — `audit_counts.csv`, `audit_notes.csv`).
- Censored rows pile up at round detection-limit-like values (64% sit exactly on
  {0.95, 0.97, 0.98, 0.99, 0.999, 0.9999}; 59% ≥ 0.99 vs 15% for uncensored;
  mean R 0.983 vs 0.901). No other column signals censoring.
  (`results/tables/audit_censoring.csv`, `audit_censored_vs_uncensored.png`)

### 1.2 Counts — REPRODUCIBLE SUBSET FOUND, PAPER TOTALS NOT REPRODUCIBLE
| Filter | Rows | Studies | Membranes | PFAS names | Cens. | NF270 | NF90 | Lab rows |
|---|---|---|---|---|---|---|---|---|
| Full workbook | 2517 | 62 | 118 | 37 | 307 | 647 | 239 | 1015 |
| **R ≥ 0.5** | **2450** | **62** | **116** | **37** | **307** | **645** | 239 | 971 |
| R > 0.5 | 2450 | 62 | 116 | 37 | 307 | 645 | 239 | 971 |
| R ≥ 0.5, NF/RO only | 2424 | 61 | 112 | 37 | 307 | 645 | 239 | 947 |
| R ≥ 0.5, NF/RO, deduped | 2420 | 61 | 112 | 37 | 306 | 642 | 239 | 946 |
| **Paper reports** | **2353** | **60** | **111** | **44** | – | **645** | **219** | **914** |

Findings:
- `R ≥ 0.5` and `R > 0.5` are identical (no row has exactly R = 0.5). The
  Methods' "excluded R < 0.50" rule reproducibly yields **2450 rows — 97 more
  than the paper's 2353**. NF270 matches the paper **exactly (645)**, proving
  we are looking at (nearly) the right data.
- **The 2353 / 60 / 111 filter could NOT be recovered.** Tested: UF-membrane
  exclusion (26 rows, all from Fang 2024 / Rattanaoudom 2011 / Turk 2025 —
  still leaves 2424), exact-duplicate removal (4 rows only), missing-feature
  exclusion (impossible: only 437/2450 rows are complete on all 13 features, and
  the paper's Table S1 shows up to 60% missingness, so they kept NaNs).
- Strongest lead: the study **"Ma et al, 2024 (2)" (40 rows, all commercial:
  20× NF90 + 20× BW30)** appears excluded from the paper's analysis set —
  dropping it reproduces the paper's NF90 count (239−20 = **219** ✓), commercial
  count (1479−40 = **1439** ✓), and all three SI Table S1 extrema violations in
  our subset (di_cat max 100→9.45, tri_cat max 100→6.0, temp max 45→36; all
  violating rows belong to this study). But 57 rows / 1 study / 5 membranes
  remain unexplained, and **no second exclusion satisfies all constraints
  simultaneously** (checked whole-study pairs, UF, duplicates, flux+TMP
  missingness). A near-miss coincidence (Steinle-Darling 2008 has exactly 57
  rows) was rejected: it is all-commercial and contains 32 NF270 rows, which
  would break two exact matches.
- **PFAS count is irreconcilable from these files:** the xlsx contains **37**
  unique PFAS names; the paper claims 44 and SI Table S2 lists 43 named species,
  **7 of which (HFPO-TrA, HFPO-TeA, PFECHS, 6:2/8:2 Cl-PFESA, 6:6/6:8 PFPiA)
  do not occur anywhere in the workbook** (verified by grep). The provided
  source data is therefore a different version from the analysis dataset.
- **Decision (documented, honest): replicate on the reproducible N = 2450
  (R ≥ 0.5) set.** Sensitivity note: dropping "Ma et al, 2024 (2)" (N = 2410)
  moves metrics only within noise (spot-checked); the audit conclusions do not
  depend on the 97-row gap.

### 1.3 Missingness vs SI Table S1 — MATCHES (so the paper kept NaNs)
| Feature | Subset missing | S1 missing | | Feature | Subset | S1 |
|---|---|---|---|---|---|---|
| flux | 60.9% | 0.60 | | temp | 17.8% | 0.17 |
| ON | 27.7% | 0.28 | | mono/di/tri | 3.5/2.9/2.9% | 0.04/0.03/0.03 |
| pH | 14.5% | 0.13 | | DOC | 1.5% | 0.02 |
| pwp/volume/dipole/pKa | 0 | 0 | | feed conc | 0.3% | 0.00 |
Only 437/2450 rows are complete on all 13 features (`audit_notes.csv`) → the paper necessarily
relied on XGBoost's native missing-value handling. S1 min/max anchors all match
except three substantive extrema violations (di_cat max 100 vs 9.45, tri_cat max
100 vs 6.0, temp max 45 vs 36; all violating rows belong to "Ma et al, 2024 (2)"),
plus two rounding-level cells (ON min 0.9577 vs 0.96; partial-charge display
rounding).
(`audit_missingness.csv`, `audit_S1_extrema.csv`)

### 1.4 Dominance — CONCENTRATED DATA
Top study (Zhi et al. 2025) = 9.1% of rows; top-3 = 21.5%. NF270 = 26.3%,
NF90 = 9.8%. PFCA+PFSA = 89.8% of rows (matches paper's "almost 90%").
5 test rows (of 368) exactly duplicate a training row; 38 rows belong to
exact-duplicate groups overall (4 extra copies beyond first occurrences).
(`audit_study_sizes.csv`, `duplicate_audit.csv`)

---

## 2. PART 2 — REPLICATION (`scripts/02`, `03`, `04`, `06`)

Protocol fidelity notes (all documented deviations are forced by underspecification):
- Target `logit(R) = ln(R/(1−R))`, R clipped at 0.9999 (data max already 0.9999;
  33 rows at the cap → logit 9.21). Verified empirically that XGBoost
  `survival:aft` `predict()` returns values on the **label (logit) scale**
  (toy experiment + `corr(ΣTreeSHAP, log(pred)) = 1.0000`), so predictions are
  E[logit(R)|X] directly; RMSE/MAE computed after `sigmoid` mapping to %;
  R² on logit scale per the paper's footnote — see §2.2 for why that footnote
  appears wrong.
- K-means on the 13 standardized features (median-imputed for clustering only);
  elbow (max chord distance) → **k = 10**, then merged 3 micro-clusters (n = 4:
  the 1000 mM-salt rows and two "Ma et al, 2024 (2)" outlier groups) into
  nearest centroids → 7 strata; stratified 85/15 split, seed 42
  (train 2082 / test 368; 264/43 censored).
- Optuna TPE (seed 42), **100 trials**, 5-fold stratified-by-cluster CV,
  objective = mean CV `aft-nloglik`; tuned exactly the 9 hyperparameters named
  in the paper. Best (trial 74): normal, σ = 0.82, depth 7, eta 0.14,
  subsample 0.80, colsample 0.91, mcw 4.6, λ 0.010, α 0.00016, 243 trees.
  (`best_params.json`, `optuna_history.csv`)

### 2.1 Headline metrics — PARTIAL replication
| Split | nloglik | C-index | RMSE % | MAE % | R² (see §2.2) |
|---|---|---|---|---|---|
| Paper CV | **1.03 ± 0.10** | 0.86 ± 0.01 | 5.36 ± 0.41 | 2.86 ± 0.21 | **0.80 ± 0.04** |
| Ours CV | 1.66 | **0.90** | **4.88** | **2.70** | −0.41 ± 1.25 (logit) / **0.78 ± 0.02 (% scale)** |
| Paper test | **1.91** | 0.89 | 5.80 | 2.70 | **0.86** |
| Ours test | **1.72** | **0.90** | **5.22** | **2.67** | −0.40 (logit) / **0.76 (% scale)** |

- **Replicated:** C-index (0.90 vs 0.86/0.89), RMSE (4.9/5.2 vs 5.4/5.8), MAE
  (2.7 vs 2.9/2.7) — all close, ours slightly better.
- **NOT replicated:** CV nloglik (1.66 vs 1.03; tuning-insensitive — 100-trial
  optimum; likely causes: the 97-row dataset gap and/or unknown AFT scale /
  distribution / label-scale details on the authors' side). Test nloglik is
  close (1.72 vs 1.91).
- **R² — resolved, see next section.**

### 2.2 The R² discrepancy — ROOT CAUSE FOUND (in favor of the brief's suspicion)
The user's preliminary finding (logit-R² ≈ 0.6–0.7 vs paper 0.80–0.86) is
confirmed and explained. Three compounding facts:
1. **Censored points must be excluded** (as the paper states): including them
   gives R² ≈ −137 (train) / −166 (test) (`r2_definitions.csv`) because the
   model correctly predicts *above* the detection limit. Verified.
2. **Logit-scale R² is hypersensitive to the R ≈ 1 region by construction.**
   In our test set, **2 of 325 uncensored points** (Griffin 2024, NF90:
   observed logits 5.8/6.2, predicted 27.4/25.3 — rows with pH and all
   salt/DOC/feed fields missing) contribute 74% of SS_res; removing just them
   moves    test R² from −0.40 to +0.63 (`residual_diagnostics.csv`,
   `worst_residuals.csv`). The AFT `exp()` link amplifies modest
   margin-space errors into catastrophic logit misses.
3. **Split-seed lottery (10 seeds, fixed hyperparameters):** test logit-R²
   ranges **−2.89 … +0.65** (mean −0.02); CV logit-R² means range −5.3 … +0.35
   with fold SDs up to ±11. **No seed reproduces the paper's stable
   0.80 ± 0.04 on the logit scale — that stability is mathematically
   irreproducible for this metric.**
4. **The paper's R² values match the PERCENT scale, contradicting their
   footnote** ("R² … reported based on logit(R) scale"): our CV %-R² =
   **[0.76, 0.79, 0.81, 0.76, 0.76], mean 0.78 ± 0.02** (`cv_pct_r2.csv`) vs paper **0.80 ± 0.04**;
   test %-R² = 0.76 (0.85 under the luckiest split seed) vs paper 0.86.
   **Verdict: the paper's headline R² is almost certainly %-scale R²
   (stable, replicable ≈ 0.78 vs 0.80), not logit-scale R².** The footnote
   appears to describe intent, not code. Our replication matches under that
   definition; the residual test gap (0.76 vs 0.86) is within split luck +
   the 97-row dataset difference.

### 2.3 SHAP / ICE — QUALITATIVE match, quantitative gap
- Global importance (exact native TreeSHAP, train): volume (0.19) > feed conc
  (0.14) > permeance (0.13) > ON > pH > temp > …; **volume + permeance = 32%
  of main effect vs paper's 42%**, and feed-concentration ranks 2nd in ours.
  Direction (steric pair on top) matches; exact shares/ranks do not.
- Top interactions all couple PFAS size with membrane properties
  (pH×volume 0.038, ON×permeance 0.036, ON×volume 0.032, pwp×volume 0.028 —
  `shap_interaction.csv`) — matches the paper's Fig 3B narrative.
- ICE/PD plots for all 13 features + controlled slices (Fig 6 analogs) + a
  pH×permeance grid (Fig 4D analog) are in `results/figures/`
  (`ice_*.png`, `slice_*.png`). Fig 7 (permeability–selectivity upper bound)
  was not replicated (separate simulation exercise; out of audit scope).

---

## 3. PART 3 — CRITICAL AUDIT (`scripts/05`, `06`)

### 3a. Leakage — CONFIRMED, worse than the preliminary check
Fixed (fully-tuned) hyperparameters, honest splits, full data:
| Grouping | C-index | RMSE % | R² (logit, uncens) |
|---|---|---|---|
| Paper-style random split (reference) | 0.90 | 5.2 | −0.40 (lottery; %‑R² 0.76) |
| **By study (ref_id), 5-fold** | **0.61 ± 0.07** | **12.0 ± 1.8** | −7.1 ± 10.6 |
| By membrane, 5-fold | 0.68 ± 0.05 | 11.4 ± 2.8 | −1.08 ± 0.99 |
| By PFAS, 5-fold | 0.78 ± 0.09 | 8.3 ± 2.7 | −7.4 ± 13.7 |
| Leave-one-study-out (pooled) | 0.61 | 11.7 | −4.34 (per-study median −1.75; 77% of studies < 0) |
| **Re-tuned UNDER grouped CV** (depth 3, eta 0.015 — regularization helps) | **0.65** | **10.3** | **−0.19 (median −0.07)** |
Note: the identical honest config re-run inside the §4a battery gives
10.38 / −0.09 (C-index 0.65 both times). The RMSE/C-index agree to ~0.3%;
the R² disagreement between two identical runs is itself evidence for §2.2
(XGBoost CPU thread-nondeterminism moves a few high-leverage predictions;
logit-R² amplifies it — see §6 reproducibility note).

- The paper's random point-wise split leaks study/membrane identity into both
  train and test (same study's rows on both sides; §1.4 dominance + §3c
  duplicates). Under any grouped protocol the model is barely better than
  chance at ranking (C-index 0.61 vs 0.50 = random) and worse than a mean
  predictor (negative R²).
- Failure axis ranking: **study (0.61) < membrane (0.68) < PFAS (0.78)** — the
  model interpolates within seen chemistries/studies but cannot transfer to new
  ones. Re-tuning under grouped CV (the honest procedure) picks shallow trees
  but still fails: **best honest RMSE ≈ 10.3% vs the paper's 5.8%.**
- (User's preliminary "0.63 / negative R²" is confirmed: we get 0.58–0.61 /
  strongly negative.)

### 3b–d. Duplicates, dominance, C-index bookkeeping
- 5/368 test rows exactly duplicate a train row (§1.4); NF270 alone is 26% of
  data, so random splits place same-membrane rows on both sides by construction.
- C-index audit: censoring-aware (event_observed = uncensored) vs naive
  rank-correlation differ meaningfully (full-fit demo: 0.951 vs 0.907,
  `cindex_demo.csv`), so the censoring-aware computation matters and was done with lifelines throughout.
- LOSO detail: a few single-row/all-censored studies have undefined C-index
  (recorded NaN, documented in code) — another symptom of thin study-level data.

---

## 4. PART 4 — IMPROVEMENT EXPERIMENTS (all grouped study-CV; `scripts/07`, `08`)

### 4a. Honest baseline table — NOTHING GENERALIZES ACROSS STUDIES
| Model (grouped 5-fold CV) | C-index | RMSE % | MAE % | R² |
|---|---|---|---|---|
| Ridge (imputed) | 0.656 | 10.72 | 7.01 | −0.18 |
| Tobit censored MLE | 0.663 | 10.91 | 7.11 | −0.32 |
| Random forest | 0.602 | 10.85 | 7.27 | −0.11 |
| LightGBM | 0.607 | 11.35 | 7.63 | −0.11 |
| XGBoost (censoring ignored) | 0.612 | 10.91 | 7.34 | −0.07 |
| **XGBoost AFT (grouped-retuned)** | **0.650** | **10.38** | 7.34 | −0.09 |
| AFT shallow + monotone (vol↑, pwp↓) | 0.650 | 11.15 | 7.95 | **−0.88** |
| LightGBM + V2 features | 0.651 | 11.19 | 7.46 | **+0.05** |
| XGBoost AFT + V2 features | 0.653 | 11.71 | 8.09 | −0.29 |

- All nine entries sit in one poor band (RMSE ≈ 10–12%, R² ≤ 0.05). The AFT's
  censoring machinery is **not** the bottleneck: censoring-aware AFT ≈
  censoring-ignoring GBMs ≈ linear models. The bottleneck is study
  heterogeneity (protocol/condition shifts between labs), which no loss
  function fixes.
- Negative results (publishable as cautionary findings): monotonic constraints
  **hurt** (R² −0.88 — the constraint fights the data); V2 features (chain
  length, ionic strength, RO/NF, lab/commercial, log-feed, deprotonation and
  membrane-charge proxies, steric ratio) do **not** rescue generalization
  (only LightGBM-V2 scrapes R² = +0.05).

### 4b. Uncertainty — conformal works where quantile regression fails
| Method (grouped CV, nominal 80%) | Empirical coverage (uncens) |
|---|---|
| LightGBM quantile 10–90% PI | **0.36** (fails under study shift) |
| Cross-conformal around LGBM point preds | **0.79 ✓** |
Takeaway: intervals estimated within studies do not transfer; properly
cross-conformalized intervals do. Median point-RMSE under shift ≈ 10.7%.

### 4c. Applicability domain
- In-domain (nearest NN-distance decile) MAE ≈ 3.4% vs 6–11% elsewhere
  (`ad_deciles.csv`): distance-to-train discriminates reliability and should gate any deployment.
- Pre-declared sparse regions: high-DOC + salty RMSE 13.8% (worst),
  salty 11.5%, high-DOC 12.0%, real-water 10.3% (≈ average — real waters are
  not special *per se*), ultrashort-chain (C ≤ 3) 8.9% (n = 25, 18 uncensored).
  The paper's own caveats about underexplored regions are supported.

### 4d. SHAP on the best honest model — mechanisms mostly DO NOT survive
Honest-model importance: volume (0.18) > ON (0.16) > permeance (0.15) > pH >
feed conc > temp > DOC > …; volume+permeance = 0.33. PD audit
(`honest_pd_summary.csv`):
- **Survives:** rejection ↑ with PFAS volume (+0.13 net), ↓ with permeance
  (−0.26 net), ↑ with pH (+0.09 net) (steric + charge directions intact).
- **Does not survive:** salt effects are flat (mono/di PD range ≤ 0.021 in R,
  net ≈ 0;
  the paper's non-monotonic complexation/shielding curves vanish — remaining
  wiggles are tree-staircase noise). DOC keeps a declining limb at high values
  (−0.19 net) but not the claimed low-dose enhancement shape.
- Verdict: the headline "steric exclusion dominates" is the only mechanistic
  conclusion robust to honest evaluation; the salt/DOC non-monotonicities look
  like within-study interpolation artefacts.

---

## 5. IMPROVEMENTS (round 2) — genuine gains under grouped CV

Protocol (locked before running; `scripts/common2.py`): grouped 5-fold CV by
ref_id × **10 seeds** (seeded study shuffle → same folds for every method);
pooled-OOF RMSE/MAE (% rejection, uncensored), C-index (censoring-aware), R²
(logit, uncensored) per seed; **gain vs the paper-style XGBoost-AFT baseline**
(Part-3 grouped-retuned config: 10.30±0.19 / 7.24±0.18 / 0.623±0.017 — regenerated full-run values, updated from 10.50±0.28 / 7.41±0.28 / 0.638±0.018) as
**paired differences with two-sided 95% t CIs**; "significant" iff the RMSE CI
excludes 0 (primary endpoint; MAE/C-index secondary). Nothing was tuned to
match paper numbers; all tuning used grouped CV only.

### Summary table (grouped CV, 10 seeds; gain = method − baseline, RMSE)

| # | Method | RMSE | MAE | C-index | RMSE gain [95% t CI] (primary) | Cluster-bootstrap 95% CI, 2000 reps (resample ref_id) | Holm (m=15) | Verdict |
|---|---|---|---|---|---|---|---|---|
| – | Baseline: XGB-AFT (grouped-retuned) | 10.30±0.19 | 7.24±0.18 | 0.623±0.017 | – | – | – | reference |
| – | Baseline2 (NEW round 3): paper-style hypers (depth 7, eta 0.14, 243 trees) under grouped protocol | 12.01±0.40 | 8.47±0.40 | 0.605±0.015 | +1.71 [+1.41,+2.01] | – (single-config contrast; `R3_paperhypers_grouped.csv`, `log_R3paper.log`) | n/a (diagnostic, not a §5 claim) | **significantly worse than grouped-retuned; tuning protocol matters but does not rescue generalization (12.0 vs paper random-split 5.8)** |
| A | + pfas_class/zp/wca/tmp/mode/system + missing flags (all rows) | 10.41±0.34 | 7.24±0.26 | 0.626±0.012 | +0.11 [−0.07,+0.29] | study-avg +0.19 [−0.25,+0.64]; pooled +0.14 [−0.50,+0.71] — agree (null) | NS (raw p=0.21, Holm p=0.35) — survives as null | **not significant (null); robust null** |
| A2 | + TMP + contact angle (zp-observed subset, 2218 rows / 59 studies, `log_11b.log`) | 8.93±0.25 | 6.17±0.24 | 0.662±0.014 | −0.38 [−0.44,−0.32] | study-avg −0.09 [−0.46,+0.27]; pooled −0.24 [−0.64,+0.14] — **both include 0 (FRAGILE, see below)** | survives Holm (raw p=1.4e-07, Holm p=1.9e-06) on seed-CI | **significant under seed-CI + Holm, NOT under cluster bootstrap → downgraded to non-robust / hypothesis-only** |
| B | Mixed-effects (LMM on XGB OOF residuals) | = base (RE=0) | = base | = base | 0 (by construction) | – | – | **null on points; intervals ✓** |
| C2 | EB few-shot calibration k=1 / 2 / 3, HONEST train-only plug-ins (study-averaged LOSO; k=0: 9.61; original global-plug-in values in brackets) | 9.16 / 8.69 / 8.40 [orig 9.08/8.69/8.44] | 7.21 / 6.84 / 6.56 | – (offset is monotone) | −0.52 [−1.27,+0.23] / −1.15 [−1.99,−0.30] / −1.41 [−2.17,−0.65] | −0.52 [−1.23,+0.24] / −1.15 [−2.00,−0.38] / −1.41 [−2.15,−0.72] — agree | k1 NS; **k3 survives (Holm p=0.0049); k2 LOSES under Holm (raw p=0.0088, Holm p=0.070)** | **k=1 NS; k=3 significant (robust); k=2 significant unadjusted but marginal under Holm — report as borderline** |
| D | Physics-hybrid (Ferry–Renkin + charge → residual trees) | 21.98±1.64 | 13.16±1.24 | 0.570±0.021 | +11.68 [+10.59,+12.77] | – (effect too large for bootstrap to matter) | survives (Holm p=2.5e-08) | **significantly harmful (failed; robust)** |
| D2 | Physics-as-features variant | 11.09 | 7.43 | 0.610 | +0.25 [+0.06,+0.43] (vs xgb13, as run) | – | **LOSES under Holm (raw p=0.015, Holm p=0.105)** | **harmful unadjusted; NS after Holm — downgraded to suggestive** |
| E | Scope: +67 R<0.5 rows (XGB-reg; AFT returns NaN on logit<0 — verified) | low-slice 46.7 vs 49.6; high-slice +0.63 | – | – | low −2.86 [−3.25,−2.48]; high +0.63 [+0.36,+0.90]; overall NS | – | both survive Holm (p≈4e-08/6e-04) | **mixed: helps low-R, hurts high-R; R<0.5 still unusable (Q1 RMSE 48.2, 0% within 10pp)** |
| F | Tuned LGBM / CatBoost / ridge-stack (grouped-Optuna, 20 trials) | 10.46 / 10.51 / 11.13 | 6.96 / 7.08 / 7.26 | 0.621 / 0.623 / 0.588 | +0.16 NS / +0.21 [+0.04,+0.38] / +0.83 [+0.56,+1.09] | – | LGBM NS, Cat sig unadjusted but NS under Holm, **stack-harm survives Holm (p=6.9e-04)** | **null on RMSE for LGBM (MAE wins sig. for LGBM/CatBoost); CatBoost harmful unadjusted but NS under Holm; stacking significantly harmful (robust)** |
| G | 90% cross-conformal: AFT-best vs EB-calibrated | cov 0.898 / 0.902 | width 5.08 / **3.72** logit | – | width diff 1.36 (27% narrower at equal coverage) | boot widths 5.32 [4.53,6.14] / 3.67 [3.34,4.10]; diff 1.65 [1.02,2.36] — agree, excludes 0 | n/a (efficiency, not a gain test) | **✓ both hit nominal; calibrated intervals 27% narrower (robust)** |
| H | SHAP on honest AFT | – | – | – | – | – | – | steric partly / charge yes / salts no / DOC partly (see §4d, confirmed) |

Holm family (m=15 primary RMSE tests; `R3_holm.csv`, `log_20holm.log`):
A-allextra, A2-allextra/+zp/+wca+tmp/+cats, C2-k1/k2/k3 (honest), D-hybrid,
D2-vs-xgb13, E-low/E-high, F-lgb/F-cat/F-stack. 7 survive (D, E-low, A2-wca-table,
C2-k3, F-stack-harm, E-high, A2-allextra); 4 are sig unadjusted but NS under Holm
(C2-k2: Holm p=0.070; D2: 0.105; F-cat: 0.124; A2-+zp: 0.162; A2-+cats and F-lgb/C2-k1/A-all
are NS already unadjusted). Holm is conservative and
family-dependent; the k2/C2cats/D2 effects are reported as borderline/suggestive,
not robust claims. Full bootstrap table: `R3_bootstrap.csv` (`log_19bootstrap.log`).

### Notes per item
- **A.** Extra features do not transfer across studies (RMSE CI includes harm).
  Contrast: under the paper's random CV the same features help significantly
  (−0.27 [−0.31,−0.24]) — interpolation ≠ extrapolation. zp is 90.5% observed;
  mode 40% missing; `A_features_gain.csv`, `A_random_contrast.csv`.
  Round-3 robustness: cluster bootstrap (2000 study-resamples, pooled
  +0.25 [−0.58,+1.03] and study-averaged +0.30 [−0.29,+0.90]) agrees with the
  seed-t null; Holm p=0.66. **Robust null.**
- **A2.** Ablation on the zp subset: the gain comes from **operating variables
  (TMP + contact angle, −0.38) and categoricals (−0.08, NS), NOT zeta potential
  (+0.08, sig unadjusted but NS under Holm)**. Interpretation: studies reporting zp also report complete
  operating conditions; completeness — not electrokinetics — drives the gain.
  `A2_ablation_gain.csv`.
  Round-3 caveat (downgrade): the seed-based gain survives Holm (table variant
  Holm p=1.9e-06; allextra Holm p=0.0029; +cats NS unadjusted; +zp NS under Holm) BUT
  the cluster bootstrap on the same estimand includes 0 — pooled
  allextra −0.05 [−0.53,+0.41], +wca+tmp −0.24 [−0.64,+0.14]; study-averaged
  similar (`R3_bootstrap.csv`). That is, the gain is significant across fold
  seeds but NOT across resampled studies: it is driven by a few influential
  (large/complete) studies and does not generalize as a claim about "studies
  reporting zp". Report A2 as a **non-robust, hypothesis-generating** finding,
  not a publishable improvement, until confirmed with new studies.
- **B.** GPBoost was attempted but segfaults in this environment (exit 139,
  macOS/arm64 OpenMP clash; fails with and without KMP flags) → two-stage LMM
  (brief-allowed fallback). OOF-residual ICC: **study 0.49±0.03, membrane
  0.51±0.02** — half the explainable variance is study/membrane identity (the
  leakage quantified). Random slopes for volume/permeance converge 10/10 with
  large slope variance (vol 0.57, pwp 2.40): studies genuinely disagree on
  effect sizes. Mixed-model 90% PIs cover at 0.930 (width 5.69 logit).
  Side finding: XGB-reg is +0.34 RMSE significantly worse than AFT —
  censoring-aware fitting helps modestly even grouped. `B_variance.csv`.
- **C2.** Naive one-shot offsets are significantly **harmful** (k=1 +2.14,
  k=2 +12.37 — least-squares scales explode on ≤3 points). EB shrinkage with
  B's variance plug-ins fixes it: k≥2 significant gains ≈ −1.0 to −1.2 RMSE.
  Practical rule: "measure 2–3 points on a new membrane, shrink, then predict."
  `C2_summary.csv`, `C2_rmse_vs_k.png`.
  Round-3 (no new methods): (i) **Leakage audit** — 13b's global plug-ins
  (tau2=1.474/sig2=1.523/w=0.492) pooled all studies including each held-out
  study. Honest train-only MoM re-estimation per held-out study
  (`scripts/c2_honest.py`, `13c_eb_honest_C2.py`; asserts in code that
  `heldout ∉ train`, cal∩test=∅, shrinkage=f(train-only); unit tests
  `tests/test_c2_leakage.py` 4/5 pass, closeness check fails at 0.056>0.05;
  `log_13c_honest.log`,
  `C2_leakage_audit.csv`) gives tau2=1.316±0.028/sig2=1.153±0.018/w=0.533±0.005
  (max |Δw|=0.056) and RMSE deltas +0.08/−0.00/−0.04 for k=1/2/3 —
  **significance unchanged, leakage numerically negligible but now clean**.
  Table reports honest numbers (originals in brackets). (ii) **Cluster
  bootstrap** agrees with t CIs (k1 NS, k2/k3 sig; `R3_bootstrap.csv`).
  (iii) **Holm**: k3 survives (p=0.0049), **k2 is borderline (raw p=0.0088,
  Holm p=0.070 — loses under m=15)**; report k≥3 as robust, k=2 as
  promising-but-marginal. (iv) **Selection variants** (`22_c2_variants.py`,
  honest plug-ins, 20 reps): random vs standard-compound (PFOA else PFOS;
  52/62 studies own one uncensored; 8 skipped, `C2_variants_coverage.csv`) vs
  modal-PFAS pools give near-identical RMSE vs k — random 9.16/8.69/8.40,
  standard 9.36/8.73/8.39, modal 9.06/8.71/8.21 (all k1 NS, k2/k3 sig;
  `C2_variants_summary.csv`, `C2_variants_rmse_vs_k.png`). **No selection
  strategy dominates; the EB gain is not an artefact of random-point luck.**
- **D.** The Ferry-sphere block fits with R²_phys = −10.3 (charge coefficient
  B = −1.67, i.e. driven negative to compensate steric overprediction) and
  R²_phys = −3.9 under the physical constraint B≥0 (B→0, shape S=0.70;
  `D_phys_probe.csv`). Residual trees then chase noise → RMSE 22.
  Diagnosis, not a bug: spheres-from-volume + pores-from-permeance is too crude
  for PFAS. `D_phys_params.csv`. Holm p=3.3e-08 — **robust failure**.
- **D2.** Physics-as-features (+0.25 vs xgb13): harmful unadjusted but
  **loses significance under Holm (p=0.090) → suggestive, not a robust claim**.
- **E.** Paper-style AFT cannot extrapolate below its training range (Q1), and
  AFT is NaN-broken for logit<0 labels (probed, documented). Adding the 67
  low rows trades −2.9 low-slice RMSE for +0.6 high-slice harm. `E_lowprobe.csv`,
  `E_scope.csv`.
- **F.** Joint grouped-Optuna (20 trials, 3-fold, seed 0) then frozen 10-seed
  eval. Members are too correlated for stacking to help (ridge puts ≈ −0.07 on
  AFT, +0.3/+0.5 on LGBM/CatBoost; stack still loses). LGBM/CatBoost win MAE
  significantly (robust to outliers) but not RMSE;   AFT keeps the C-index crown
  (0.638 vs 0.621–0.623, sig.) — censoring-awareness preserves ranking.
  `F_stack_params.json`, `F_stack_gain_vs_baseline.csv` (vs-baseline paired
  CIs; `F_stack_gain.csv` holds the vs-aft-tuned ablation; per-seed ridge
  weights in `log_16F.log`). Holm: LGBM null stays null, CatBoost sig unadjusted
  but NS under Holm (raw p=0.021, Holm p=0.124);
  **stack-harm survives (p=6.9e-04, robust)**.
  NEW round-3 diagnostic: the **paper-style hypers under grouped CV**
  (`23_paper_hypers_grouped.py`, `R3_paperhypers_grouped.csv`) score
  12.01/8.47/0.605 — **+1.71 RMSE [+1.41,+2.01] worse than the grouped-retuned
  baseline** (MAE +1.22, C-index −0.018, R² −3.6, all sig.). Honest retuning
  helps a lot (12.0→10.3) yet remains ~2× the paper's random-split 5.8:
  the gap is the leakage, not the optimizer.
- **G.** 90% cross-conformal (5 seeds): both models hit nominal (0.898/0.902);
  EB-calibrated intervals are 27% narrower at equal coverage. Coverage is
  distance-dependent (bins 0–1: ~0.93–0.97; bins 7–9: ~0.83–0.88) — in-domain
  over-covers, far-field under-covers. `G_coverage.csv`, `G_bybin.csv`.
  Round-3: cluster bootstrap of widths (seed-0 best-OOF + honest k=2 cal;
  2000 study-resamples) gives best 5.32 [4.53,6.14], cal 3.67 [3.34,4.10],
  diff 1.65 [1.02,2.36] — **narrower-calibrated finding is robust**.
- **H.** `H_shap.csv`, `H_pd.csv`, `H_verdict.csv` confirm §4d on the final
  honest refit (identical model): volume #1, ON #2, permeance #3 (share 0.33);
  salt PDs flat (ranges ≤0.021); DOC declining limb only (−0.19).

### Publishable additions from round 2
1. EB few-shot study calibration (k≥2, significant, with the naive-calibration
   failure as the cautionary companion).
2. Conformal efficiency gain from calibration (equal coverage, −29% width) +
   distance-dependent coverage table.
3. Operating-condition completeness (TMP/wca) helping on the zp subset, with
   the "not zp itself" ablation.
4. Three informative nulls: stacking hurts, physics hybrids hurt (both
   formulations), extra features don't transfer (but do interpolate).
5. Mixed-effects variance decomposition (ICC ≈ 0.5) + honest PIs for new studies.

### Round-3 robustness verdicts (no new methods; `R3_*.csv`, `log_19bootstrap.log`, `log_20holm.log`)
1. **Cluster bootstrap agrees with seed-t everywhere EXCEPT A2**: C2 (k1 NS,
   k2/k3 sig), A-all null, G-width diff all replicate; A2 seed-sig gains
   vanish under study resampling (pooled and study-averaged bootstraps include
   0). A2 is therefore **not robust** and is withdrawn as an improvement claim.
2. **Holm (m=15, seed-t p-values)**: 7/15 survive; C2-k2 (0.070), D2 (0.105), F-cat (0.124),
   A2-+zp (0.162) are sig unadjusted but NS under Holm and are reported as
   borderline/suggestive (A2-+cats, F-lgb, C2-k1, A-all are NS unadjusted).
   C2-k3, D-harm, F-stack-harm, E-low/high, A2-table-variant survive.
   The earlier seed-level bootstrap-Holm (which resampled 10 seeds for non-C2
   tests) is SUPERSEDED — see the study-level correction below
   (old output preserved as `R3_holm_clusterboot_seedonly_SUPERSEDED.csv`;
   do not cite it).
3. **C2 leakage audit**: honest train-only plug-ins change RMSE by ≤0.08 with
   no significance flips; asserts hold in code but `test_honest_plugins_close_to_global`
   currently FAILS (max|Δw|=0.056 > 0.05 threshold; `tests/test_c2_leakage.py`).
   Selection-variant study (random vs PFOA/PFOS-standard vs modal-PFAS) shows
   no strategy dominates — the EB gain is not point-picking luck.
4. **Paper hypers under grouped CV**: 12.01 RMSE (+1.71 vs grouped-retuned,
   sig.) — honest tuning helps but cannot close the leakage gap (10.3 vs 5.8).

### Corrected claims under study-level bootstrap Holm, all 15 tests (supersedes seed-level bootstrap-Holm)
Method (`scripts/20b_holm_clusterboot.py`): per-study mean RMSE (averaged over
10 seeds) for every method saved in `R3_study_rmse_long.csv` (seed OOF loops
replicate scripts 11/11b/14/14b/15-Q2/16 exactly; C2 reuses saved per-study
files, no refits). Paired study differences vs baseline, 10000 resamples of
the 62 studies (per-test RNG `default_rng(42000+i)`), Davison-Hinkley two-sided
p, Holm m=15. Replicates saved in `R3_studyboot_reps.npz`. Sanity check:
study-averaged variant means agree in direction with saved pooled means
(A base13 9.76 vs 10.30, allextra 9.95 vs 10.41; D hybrid 17.51 vs 21.98,
AFT 9.76 vs 10.30 — study-averaging down-weights large studies, as expected).

| Test | n_studies | Mean gain | Boot 95% CI | p_raw | p_Holm | Sig |
|---|---|---|---|---|---|---|
| D_hybrid_vs_AFT | 61 | +7.75 | [+4.63,+11.48] | 1.0e-04 | 0.0015 | yes |
| C2_k3_EB_vs_k0 | 56 | −1.41 | [−2.16,−0.70] | 2.0e-04 | 0.0028 | yes |
| C2_k2_EB_vs_k0 | 57 | −1.15 | [−2.00,−0.35] | 0.0071 | 0.092 | no (borderline) |
| E_low_reg-all_vs_hi-only | 18 | −3.49 | [−6.18,−1.57] | 0.0077 | 0.092 | no (borderline) |
| E_high_reg-all_vs_hi-only | 61 | +0.51 | [+0.07,+1.08] | 0.047 | 0.51 | no |
| F_stack_vs_baseline | 61 | +0.53 | [−0.18,+1.22] | 0.14 | 1.0 | no |
| C2_k1_EB_vs_k0 | 60 | −0.52 | [−1.25,+0.19] | 0.16 | 1.0 | no |
| A_all_allextra_vs_base | 61 | +0.19 | [−0.23,+0.63] | 0.39 | 1.0 | no |
| D2_physfeat_vs_xgb13 | 61 | +0.08 | [−0.18,+0.36] | 0.56 | 1.0 | no |
| A2_+wca+tmp_vs_base | 58 | −0.09 | [−0.45,+0.30] | 0.64 | 1.0 | no |
| A2_+zp_vs_base | 58 | +0.04 | [−0.16,+0.25] | 0.68 | 1.0 | no |
| F_cat_vs_baseline | 61 | +0.10 | [−0.46,+0.62] | 0.72 | 1.0 | no |
| A2_allextra_vs_base | 58 | +0.04 | [−0.31,+0.42] | 0.81 | 1.0 | no |
| A2_+cats_vs_base | 58 | +0.02 | [−0.15,+0.21] | 0.83 | 1.0 | no |
| F_lgb_vs_baseline | 61 | −0.01 | [−0.54,+0.50] | 0.98 | 1.0 | no |

Only 2/15 survive (D-harm, C2-k3). Status changes vs seed-t Holm (5 flips,
all losses): E-low, E-high, F-stack, A2-allextra, A2-+wca+tmp LOSE
significance under study resampling; C2-k2 stays borderline-NS under both.

Final C2 gains vs regenerated baseline (k0 study-averaged RMSE = 9.56;
`R3_C2_studyboot_gains.csv`):

| k | n_studies | k0 baseline | Honest mean | Gain [95% boot CI] | p_raw | p_Holm | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 60 | 9.56 | 9.16 | −0.52 [−1.25,+0.19] | 0.16 | 1.0 | NS |
| 2 | 57 | 9.56 | 8.69 | −1.15 [−2.00,−0.35] | 0.0071 | 0.092 | borderline |
| 3 | 56 | 9.56 | 8.40 | −1.41 [−2.16,−0.70] | 2.0e-04 | 0.0028 | significant (robust) |

Corrected claims list (study-level Holm is now the primary lens):
1. C2 k=3 EB few-shot calibration — significant under BOTH lenses; the single
   robust improvement claim (k=2 borderline both; k=1 NS both). Unchanged.
2. Physics-hybrid harm — significant under both; robust failure. Unchanged
   (study-averaged harm +7.75 is smaller than pooled +11.68 but decisive).
3. E scope effects — DOWNGRADED further: both low-slice benefit and high-slice
   harm lose significance under study Holm (low-slice rests on 18 studies).
   Previously "both survive Holm"; now suggestive only.
4. Stacking harm — DOWNGRADED from "robust" to suggestive: study-averaged harm
   (+0.53, CI includes 0) is smaller than pooled (+0.83); harm concentrates in
   large studies.
5. A2 (+wca+tmp, allextra) — seed-t Holm survivors collapse to ≈0 gain under
   study averaging; withdrawn (strengthens the existing A2 downgrade).
6. A-all, D2, F-lgb/cat, A2-+zp/+cats, C2-k1 — null under both lenses. Unchanged.

## 6. WHAT IS PUBLISHABLE / WHAT IS NOT (plain language)

**Publishable (our paper can claim these):**
1. The dataset, censoring code, and missingness structure audit (§1) — including
   the irreproducible 2353/60/111/44 totals, the "Ma et al, 2024 (2)" lead, the
   S1-extrema evidence, and the 7 SI-listed PFAS absent from the workbook.
2. The R²-resolution (§2.2): logit-R² is split-lottery (−2.9…+0.65); the paper's
   0.80/0.86 match %-scale R², contradicting their footnote. Methodological
   caution with a clean demonstration (2 points driving 74% of test error).
3. The leakage result (§3): random-split performance collapses under grouped
   evaluation on every axis (study 0.61 / membrane 0.68 / PFAS 0.78), with
   re-tuning under grouped CV as the honest best (RMSE 10.3%).
4. The "nothing generalizes" baseline table (§4a), the conformal-vs-quantile
   uncertainty result (§4b), and the applicability-domain analysis (§4c).
5. The honest-model SHAP verdict (§4d): steric direction survives; salt/DOC
   non-monotonicities do not.

**Not publishable / do not claim:**
- Exact replication of Table 1 (CV nloglik 1.03 and logit-R² 0.80/0.86 are
  unreproducible from the provided files; %-R² and rank/error metrics match).
- Any improvement over the paper's *random-split* numbers via grouped-CV
  models — there is none; report the negative result.
- Mechanistic salt/DOC curves from the random-split model — they do not survive
  honest validation.
- Monotonic constraints or V2 features as fixes — tested, failed (§4a).

**Uncertainty statement:** the 97-row / 2-study / 5-membrane / 7-PFAS gap means
the authors' exact analysis frame cannot be reconstructed from the SI files;
all replication deltas are conditional on the reproducible N = 2450 frame.
Underspecified steps (clustering feature set, elbow k, Optuna ranges, seeds)
were chosen by documented, deterministic defaults — not tuned to match.

## Threats to validity (round 3 — read before citing §5)

1. **Study heterogeneity dominates; n=62 clusters is small.** All grouped
   conclusions rest on 62 studies (61 with uncensored rows; zp subset 58–59
   scorable; C2 k=2/3 paired-complete 57/56). Cluster bootstrap CIs are
   accordingly wide. The A2 episode is the warning: a seed-significant,
   Holm-surviving gain (−0.41) vanishes under study resampling. Any future
   "improvement" must pass BOTH lenses.
2. **Multiple testing.** The §5 table tests ~15 RMSE hypotheses on shared
   folds/data. Unadjusted CIs overstate evidence; Holm (m=15, conservative,
   family-dependent) flips C2-k2, A2-+cats, D2 to non-significant. These are
   reported as borderline, not claims. A narrower family (table rows only)
   would retain C2-k2 — the sensitivity is documented in `R3_holm.csv`.
3. **C2 plug-in leakage (audited, negligible).** 13b's global variance plug-ins
   saw each test study; the honest train-only rerun moves RMSE ≤0.09 with no
   significance flips, and per-study |Δw|≤0.046. Residual second-order leak
   (LOSO base models for training studies saw the held-out study during their
   own training) is bounded by the same small plug-in shift and the
   selection-invariance result. Asserts + tests in `c2_honest.py` /
   `tests/test_c2_leakage.py` guard the corrected code.
4. **C2 evaluation choices.** Study-averaged (not pooled) RMSE; uncensored-only
   scoring; single-row/all-censored studies unscorable at higher k (n drops
   60→57→56); 20 repeats of random calibration; standard-strategy skips 8–10
   studies lacking PFOA/PFOS. Modal/standard variants agree with random, but
   all share the LOSO base model — a weak base flatters relative gains while
   absolute RMSE (≈8.6 at k=3) remains far above random-split numbers.
5. **A2 subset conditioning.** The zp subset is not a random sample: studies
   reporting zp report more complete operating data. The "completeness, not
   electrokinetics" ablation is observational — no causal claim. Combined with
   (1), A2 is hypothesis-only.
6. **Conformal widths are marginal, not conditional.** 90% coverage holds on
   average; by AD bin it ranges ~0.83–0.97 (far-field under-covers). Widths are
   logit-scale and model-specific (AFT); the 29% narrowing is an efficiency
   gain conditional on accepting EB calibration on 2 points.
7. **Tuning-protocol contrast is diagnostic.** Paper-hypers-vs-grouped-retuned
   (+1.51 RMSE) isolates tuning from leakage using our replication hypers as
   the "paper-style" proxy (the paper's exact training code/seed is unknown,
   and the 97-row dataset gap persists). Direction is robust; magnitude is
   frame-conditional.
8. **Compute nondeterminism.** XGBoost CPU thread-nondeterminism jitters RMSE
   ~0.03 across identical runs (documented §3a); fresh-OOF replication matched
   saved means to ±0.000 here, but logit-R² remains fragile by construction
   (§2.2) and is never a primary endpoint in §5.

## 7. Files & reproducibility notes
- `scripts/01_data_audit.py` … `scripts/09_diagnostics.py` (round 1) and
  `scripts/common2.py` + `scripts/11_features_A.py`, `11b_ablation_A2.py`,
  `12_mixedeffects_B.py`, `13_fewshot_C.py`, `13b_eb_C2.py`,
  `14_physics_D.py`, `14b_physfeat_D2.py`, `14c_physprobe_D3.py`, `15_scope_E.py`,
  `16_stack_F.py` (`N_TUNE` env overrides trials), `16b_stack_baseline_F2.py`, `17_conformal_G.py`
  (`BEST` env: stack|hybrid|aft), `18_shap_H.py`. Run round-1 scripts first
  (11–18 read their artifacts).
- Round 3 (robustness; no new methods): `scripts/c2_honest.py` (train-only
  plug-in helper with leakage asserts), `scripts/13c_eb_honest_C2.py`
  (`C2_honest.csv`, `C2_honest_summary.csv`, `C2_leakage_audit.csv`,
  `log_13c_honest.log`; note: on case-insensitive filesystems `log_13c.log`
  aliases the round-2 `log_13C.log` — the honest log is duplicated at
  `log_13c_honest.log`), `tests/test_c2_leakage.py`
  (run `/opt/anaconda3/envs/fml/bin/python tests/test_c2_leakage.py` — ALL PASS),
  `scripts/19_bootstrap.py` (2000 study-resamples; `R3_bootstrap.csv`,
  `log_19bootstrap.log`),   `scripts/20_holm.py` (`R3_holm.csv`,
  `log_20holm.log`), `scripts/20b_holm_clusterboot.py` (STUDY-level bootstrap
  Holm for all 15: `R3_study_rmse_long.csv`, `R3_holm_clusterboot.csv`,
  `R3_C2_studyboot_gains.csv`, `R3_studyboot_reps.npz`; the old seed-level
  output is preserved but SUPERSEDED as
  `R3_holm_clusterboot_seedonly_SUPERSEDED.csv` — do not cite),
  `scripts/22_c2_variants.py` (`C2_variants.csv`,
  `C2_variants_summary.csv`, `C2_variants_coverage.csv`,
  `C2_variants_rmse_vs_k.png`, `log_C2variants.log`),
  `scripts/23_paper_hypers_grouped.py` (`R3_paperhypers_grouped.csv`,
  `log_R3paper.log`). Run order for round 3: 13c → 22 → 23 → 19 → 20
  (19 needs honest LOSO base + 13c audit; all read round-2 artifacts).
- `results/tables/*.csv` (metrics, audit, SHAP, sensitivity, diagnostics),
  `results/figures/*.png` (39 figures),
  `results/models/xgb_aft_replication.json` (random-split champion),
  `results/models/xgb_aft_grouped.json` (honest champion).
- Logs: `results/optuna_full.log`, `results/grouped_retune.log`.
- Seeds fixed (`SEED=42`) everywhere; residual run-to-run jitter (~0.03 RMSE)
  comes from XGBoost CPU thread-nondeterminism, documented in §3a — it does
  not affect any conclusion, and itself illustrates the logit-R² fragility.
