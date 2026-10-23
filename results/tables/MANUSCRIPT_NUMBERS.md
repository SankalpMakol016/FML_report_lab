# Manuscript numbers (paste-ready)

All numbers are regenerated from the tracked `results/tables/*.csv` outputs; no method changes were made.
Source: `scripts/`, `results/tables/`, `REPORT.md` §2–§8.

Conventions: RMSE/MAE in % rejection (uncensored rows only); C-index censoring-aware (lifelines);
bootstrap CIs are study-level cluster bootstrap (resample ref_id); Holm m=15, alpha=0.05.

---

## 1. AFT hyperparameters (grouped-retuned honest champion)

Source: `results/tables/grouped_best_params.json` (written by `scripts/05_grouped_cv.py`,
fixed `N_TUNE`, `SEED=42`).

| Parameter | Value |
|---|---|
| objective | survival:aft (normal) |
| max_depth | 3 |
| eta (learning_rate) | 0.0081 |
| subsample | 0.64 |
| colsample_bytree | 0.79 |
| min_child_weight | 20.1 |
| lambda | 0.083 |
| alpha | 0.0012 |
| aft_loss_distribution_scale | 0.69 |
| n_estimators (trees) | 711 |
| mean grouped CV nloglik | 1.946 |

Used as the fixed AFT config in `scripts/24_ensemble_random_vs_grouped.py` and as the
"re-tuned UNDER grouped CV" baseline in REPORT §3a / §4a (RMSE 10.3, C-index 0.65).

---

## 2. S24 ensemble: random split vs study-grouped CV

Source: `results/tables/S24_ensemble_random_vs_grouped.csv` (written by
`scripts/24_ensemble_random_vs_grouped.py`). 8 extra features added to the 13 base:
carbon count, ionic strength, charge term, volume/permeance, log permeance, RO flag,
lab flag, pfas_class one-hot. No tuning (fixed defaults).

| Protocol | Variant | RMSE (±SD) | MAE (±SD) | C-index (±SD) | RMSE gain vs AFT [study-boot 95% CI] | gain sig? |
|---|---|---:|---:|---:|---:|---|
| Random 85/15 holdout | AFT | 7.22 ± 0.50 | 4.52 ± 0.24 | 0.800 ± 0.009 | reference | — |
| Random holdout | LightGBM | 5.06 ± 0.75 | 2.64 ± 0.29 | 0.890 ± 0.009 | −1.51 [−2.02, −0.98] | yes |
| Random holdout | CatBoost | 6.16 ± 0.70 | 3.31 ± 0.33 | 0.866 ± 0.010 | −1.06 [−1.45, −0.62] | yes |
| Random holdout | Ensemble (mean) | 5.82 ± 0.66 | 3.18 ± 0.28 | 0.871 ± 0.008 | −1.16 [−1.45, −0.86] | yes |
| Random 5-fold CV | AFT | 7.22 ± 0.06 | 4.57 ± 0.03 | 0.793 ± 0.001 | reference | — |
| Random 5-fold CV | LightGBM | 5.29 ± 0.11 | 2.72 ± 0.04 | 0.885 ± 0.002 | −1.45 [−2.01, −0.87] | yes |
| Random 5-fold CV | CatBoost | 6.23 ± 0.09 | 3.35 ± 0.04 | 0.864 ± 0.002 | −0.86 [−1.34, −0.40] | yes |
| Random 5-fold CV | Ensemble (mean) | 5.90 ± 0.07 | 3.26 ± 0.03 | 0.867 ± 0.001 | −1.05 [−1.39, −0.69] | yes |
| Study-grouped 10-seed CV | AFT | 10.50 ± 0.36 | 7.37 ± 0.23 | 0.626 ± 0.015 | reference | — |
| Study-grouped CV | LightGBM | 11.13 ± 0.68 | 7.41 ± 0.48 | 0.648 ± 0.016 | +0.31 [−0.25, +0.84] | no |
| Study-grouped CV | CatBoost | 10.76 ± 0.23 | 7.14 ± 0.15 | 0.632 ± 0.010 | +0.03 [−0.52, +0.55] | no |
| Study-grouped CV | Ensemble (mean) | 10.41 ± 0.37 | 6.99 ± 0.26 | 0.643 ± 0.010 | −0.23 [−0.67, +0.16] | no |

Interpretation: random-split gains (RMSE ~5–6 vs AFT ~7.2) are a leakage artefact;
under honest study-grouped CV all methods collapse to one poor band (~10.4–11.1) and
**no variant shows a significant gain vs AFT** (all gain CIs include 0).

---

## 3. Replication Table 1 (CV and test)

Source: `results/tables/replication_metrics.csv` (written by `scripts/03_train_evaluate.py`,
`N_TRIALS=100`, `SEED=42`, reproducible N=2450 frame).

| Split | nloglik | C-index | RMSE % | MAE % | R² |
|---|---:|---:|---:|---:|---|
| Paper CV (reported) | 1.03 ± 0.10 | 0.86 ± 0.01 | 5.36 ± 0.41 | 2.86 ± 0.21 | 0.80 ± 0.04 |
| Ours CV | 1.66 | 0.90 | 4.88 | 2.70 | −0.41 logit / 0.78 %-scale |
| Paper test (reported) | 1.91 | 0.89 | 5.80 | 2.70 | 0.86 |
| Ours test | 1.72 | 0.90 | 5.22 | 2.67 | −0.40 logit / 0.76 %-scale |

Replicated: C-index, RMSE, MAE (close; ours slightly better). Not replicated: CV nloglik
(1.66 vs 1.03) and logit-scale R² (negative; paper's 0.80/0.86 match %-scale R², see
REPORT §2.2).

---

## 4. Source file + row for each cited number

| Number | Value | Source file | Row / column |
|---|---|---|---|
| Study-grouped C-index (by study) | 0.61 | `results/tables/grouped_study.csv` | `cindex` mean over 5 folds = 0.606 |
| Study-grouped C-index (by membrane) | 0.68 | `results/tables/grouped_membrane.csv` | `cindex` mean over 5 folds = 0.677 |
| Study-grouped C-index (by PFAS) | 0.78 | `results/tables/grouped_pfas.csv` | `cindex` mean over 5 folds = 0.785 |
| LOSO C-index | 0.61 | `REPORT.md` §3a leakage table | row "Leave-one-study-out (pooled)" |
| OOF-residual ICC (study) | 0.49 | `results/tables/B_variance.csv` | `study_icc` mean over 10 seeds = 0.491 |
| OOF-residual ICC (membrane) | 0.51 | `results/tables/B_variance.csv` | `mem_icc` mean over 10 seeds = 0.506 |
| C2 k=1 RMSE | 9.16 | `results/tables/R3_C2_studyboot_gains.csv` | k=1, `honest_mean_rmse` |
| C2 k=1 gain [CI] | −0.52 [−1.25, +0.19] | `results/tables/R3_C2_studyboot_gains.csv` | k=1, `gain` / `boot_lo` / `boot_hi` |
| C2 k=1 Holm p | 1.0 (NS) | `results/tables/R3_C2_studyboot_gains.csv` | k=1, `p_holm_boot` |
| C2 k=2 RMSE | 8.69 | `results/tables/R3_C2_studyboot_gains.csv` | k=2, `honest_mean_rmse` |
| C2 k=2 gain [CI] | −1.15 [−2.00, −0.35] | `results/tables/R3_C2_studyboot_gains.csv` | k=2 |
| C2 k=2 Holm p | 0.092 (borderline) | `results/tables/R3_C2_studyboot_gains.csv` | k=2, `p_holm_boot` |
| C2 k=3 RMSE | 8.40 | `results/tables/R3_C2_studyboot_gains.csv` | k=3, `honest_mean_rmse` |
| C2 k=3 gain [CI] | −1.41 [−2.16, −0.70] | `results/tables/R3_C2_studyboot_gains.csv` | k=3 |
| C2 k=3 Holm p | 0.0028 (significant) | `results/tables/R3_C2_studyboot_gains.csv` | k=3, `p_holm_boot` |
| C2 k0 baseline RMSE | 9.56 | `results/tables/R3_C2_studyboot_gains.csv` | `k0_baseline_rmse` = 9.562 |
| Conformal width AFT-best | 5.08 logit | `results/tables/G_coverage.csv` | `width` mean (aft-best) ≈ 5.08 |
| Conformal width aft-calibrated | 3.72 logit | `results/tables/G_coverage.csv` | `width` mean (aft-calibrated) ≈ 3.72 |
| Conformal coverage AFT-best | 0.898 | `results/tables/G_coverage.csv` | `coverage` mean (aft-best) ≈ 0.898 |
| Conformal coverage aft-calibrated | 0.902 | `results/tables/G_coverage.csv` | `coverage` mean (aft-calibrated) ≈ 0.902 |
| Honest SHAP share volume | 0.17 | `results/tables/H_shap.csv` | `feature=volume_Å3`, `share` = 0.173 |
| Honest SHAP share ON | 0.16 | `results/tables/H_shap.csv` | `feature=ON`, `share` = 0.164 |
| Honest SHAP share permeance | 0.16 | `results/tables/H_shap.csv` | `feature=pwp_LMH/bar`, `share` = 0.156 |
| Honest SHAP volume+permeance | 0.33 | `results/tables/H_shap.csv` | 0.173 + 0.156 = 0.329 |

Scripts provenance:
- grouped_study/membrane/pfas.csv → `scripts/05_grouped_cv.py` (§3a leakage table)
- B_variance.csv → `scripts/12_mixedeffects_B.py` (§5-B)
- R3_C2_studyboot_gains.csv → `scripts/20b_holm_clusterboot.py` (§5 C2 row)
- G_coverage.csv → `scripts/17_conformal_G.py` (§5 G row)
- H_shap.csv → `scripts/18_shap_H.py` (§4d / §5 H)

---

## 5. Holm survivor summary (REPORT §5, primary lens = study-bootstrap Holm)

Source: `results/tables/R3_holm_clusterboot.csv` (`scripts/20b_holm_clusterboot.py`,
m=15, B=10000) superseding `results/tables/R3_holm.csv` (`scripts/20_holm.py`, seed-t).

| Protocol / lens | Tests surviving Holm (m=15) | Survivors |
|---|---:|---|
| Seed-t Holm (`20_holm.py`, SUPERSEDED) | 7 / 15 | D, E-low, A2-wca-table/+wca+tmp, C2-k3, F-stack-harm, E-high, A2-allextra |
| Study-bootstrap Holm (`20b`, PRIMARY) | 2 / 15 | D (physics-hybrid harm), C2-k3 |

Only 2/15 survive under the primary study-bootstrap lens. C2-k2 stays borderline-NS under both
(p_holm 0.070 seed-t / 0.092 study-boot); C2-k3 p-values: seed-t Holm 0.0049 vs study-boot
p_raw 2.0e-04, p_holm 0.0028 (k0 baseline RMSE 9.56).

---

## 6. Figures inventory (`results/figures/`)

Total: 42 PNG files.

| File | What it shows |
|---|---|
| `audit_censored_vs_uncensored.png` | Rejection histogram by censoring status (censored vs uncensored) **(a)** |
| `audit_rejection_hist.png` | Rejection distribution, R>=0.5 subset |
| `audit_study_dominance.png` | Rows per study ranked (62 studies) |
| `audit_membrane_dominance.png` | Rows per membrane ranked (116 membranes) |
| `cluster_elbow.png` | K-means elbow for cluster stratification |
| `C_rmse_vs_k.png` | C2 few-shot RMSE vs calibration points k (round 2) |
| `C2_rmse_vs_k.png` | C2 honest RMSE vs k **(c)** |
| `C2_variants_rmse_vs_k.png` | C2 selection-variant RMSE vs k |
| `D_rmse.png` | Physics-hybrid RMSE |
| `E_rmseslice.png` | E scope low/high slice RMSE |
| `F_rmse.png` | Stacked ensemble vs tuned bases boxplot |
| `G_covwidth.png` | Conformal coverage vs width (fold-level) |
| `G_coverage_by_bin.png` | Conformal coverage by distance bin, aft-best vs aft-calibrated **(d)** |
| `H_importance.png` | Honest-model SHAP importance |
| `honest_pd_*.png` (13) | Partial dependence per feature: volume, permeance (pwp), pH, salts (mon/di/tri_cat), DOC, etc. **(e)** |
| `ice_*.png` (13) | ICE curves per feature |
| `improve_rmse.png` | Round-2 improvement RMSE |
| `loso_hist.png` | LOSO histogram |
| `rep_pred_vs_obs_logit_train/test.png` | Replication pred vs obs (logit) train/test |
| `seed_sensitivity.png` | Seed sensitivity |
| `shap_importance.png` | SHAP importance (random-split) |
| `shap_interaction_heatmap.png` | SHAP interaction heatmap |
| `slice_*.png` (6) | Controlled slice / PD grid plots |

**Best suited for:**
- (a) censored vs uncensored audit → `audit_censored_vs_uncensored.png`
- (b) random vs grouped validation bars → `S24_random_vs_grouped.png` (new; RMSE bar chart with ±SD error bars across random-holdout / random-5fold / grouped-10seed protocols)
- (c) RMSE vs number of measured points k → `C2_rmse_vs_k.png`
- (d) conformal coverage by distance bin → `G_coverage_by_bin.png` (new; coverage vs bin 0–9 with nominal 90% line)
- (e) partial dependence of volume, permeance, pH, salts, DOC → `honest_pd_volume_Å3.png`, `honest_pd_pwp_LMH_bar.png`, `honest_pd_pH.png`, `honest_pd_mon_cat_mM.png` / `honest_pd_di_cat_mM.png` / `honest_pd_tri_cat_mM.png`, `honest_pd_doc_mg_L.png`

All figures at 300 dpi, labelled axes, error bars where applicable.
