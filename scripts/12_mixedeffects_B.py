"""Round-2 B — Mixed-effects variance decomposition + honest PIs.

GPBoost was attempted first (joint tree + random-effects fitting) but
segfaults in this environment (exit 139, macOS/arm64 OpenMP clash; fails
identically with KMP_DUPLICATE_LIB_OK=TRUE). Fallback, as explicitly
allowed in the brief: two-stage linear mixed model (statsmodels MixedLM)
on boosted-tree OOF residuals.

Per seed (10 seeded grouped folds, same folds as every round-2 method):
  stage 1: XGBoost regression OOF point predictions (fixed effects);
  stage 2: MixedLM(resid ~ 1) with (a) random intercepts per study,
    (b) random intercepts + random slopes for PFAS-volume and permeance
    per study, (c) random intercepts per membrane.
For UNSEEN studies the BLUP is 0 by construction, so the mixed model
cannot beat the base learner on point metrics (hypothesis: null result);
its value is (i) quantifying between-study vs within-study variance
(ICC = the leakage diagnostic) and (ii) honest prediction intervals
pred +/- z*sqrt(var_between + var_within) with empirical coverage.

Outputs: results/tables/B_mixed_oof.csv (per-seed point metrics),
  B_variance.csv (variance components, ICC, PI coverage),
  results/figures/B_variance.png, B_picoverage.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import statsmodels.api as sm
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, fit_aft, SEEDS, FEATURES_13)

Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def main():
    frame, raw = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    mem = frame["meta_mem"].to_numpy()
    vol = frame["volume_Å3"].to_numpy(float)
    pwp = frame["pwp_LMH/bar"].to_numpy(float)
    vol_s = (vol - np.nanmean(vol)) / np.nanstd(vol)
    pwp_s = (pwp - np.nanmean(pwp)) / np.nanstd(pwp)

    oof_rows, var_rows = [], []
    for s in SEEDS:
        px = np.empty(len(y))  # xgb-reg OOF
        pa = np.empty(len(y))  # baseline AFT OOF (paired comparator)
        for tri, vai in grouped_folds(groups, s):
            mx = xgb.train({"objective": "reg:squarederror", "eta": 0.05,
                            "max_depth": 6, "subsample": 0.8,
                            "colsample_bytree": 0.8, "seed": 42},
                           xgb.DMatrix(X[tri], label=y[tri]),
                           num_boost_round=500)
            px[vai] = mx.predict(xgb.DMatrix(X[vai]))
            ba = fit_aft(X[tri], y[tri], cens[tri])
            pa[vai] = ba.predict(xgb.DMatrix(X[vai]))
        mx = oof_metrics(y, cens, R, px)
        mx.update({"variant": "xgb-reg", "seed": s})
        ma = oof_metrics(y, cens, R, pa)
        ma.update({"variant": "AFT-baseline", "seed": s})
        oof_rows += [mx, ma]

        r = y - px  # OOF residuals (honest: never in-sample)
        d = pd.DataFrame({"r": r, "study": groups, "mem": mem,
                          "vol": vol_s, "pwp": pwp_s})
        # (a) study intercepts
        fa = sm.MixedLM.from_formula("r ~ 1", groups="study",
                                     data=d).fit(reml=True, disp=0)
        vb, vw = float(fa.cov_re.iloc[0, 0]), float(fa.scale)
        # (b) study intercepts + slopes for volume, permeance
        try:
            fb = sm.MixedLM.from_formula("r ~ vol + pwp", groups="study",
                                         re_formula="~vol + pwp",
                                         data=d).fit(reml=True, disp=0,
                                                     maxiter=200)
            slope_ok = True
            sl = np.diag(fb.cov_re).tolist()
        except Exception as e:  # noqa - non-convergence is a finding
            print(f"seed {s}: slope model failed ({type(e).__name__})")
            slope_ok, sl = False, [np.nan] * 3
        # (c) membrane intercepts
        fm = sm.MixedLM.from_formula("r ~ 1", groups="mem",
                                     data=d).fit(reml=True, disp=0)
        vbm, vwm = float(fm.cov_re.iloc[0, 0]), float(fm.scale)
        row = {"seed": s, "study_between": vb, "study_within": vw,
               "study_icc": vb / (vb + vw),
               "mem_between": vbm, "mem_within": vwm,
               "mem_icc": vbm / (vbm + vwm),
               "slope_ok": slope_ok, "slope_vol_var": sl[1] if slope_ok
               else np.nan, "slope_pwp_var": sl[2] if slope_ok else np.nan,
               "fe_vol": float(fb.params["vol"]) if slope_ok else np.nan,
               "fe_pwp": float(fb.params["pwp"]) if slope_ok else np.nan}
        # PIs from (a): RE=0 point = xgb pred + global intercept
        c0 = float(fa.params["Intercept"])
        for level, z in Z.items():
            w = z * np.sqrt(vb + vw)
            cov = ((np.abs(y[~cens] - (px[~cens] + c0)) <= w)).mean()
            row[f"pi{int(level*100)}_cov"] = float(cov)
            row[f"pi{int(level*100)}_width"] = float(2 * w)
        var_rows.append(row)
        print(f"seed {s}: study ICC={row['study_icc']:.3f} "
              f"mem ICC={row['mem_icc']:.3f} "
              f"PI90cov={row['pi90_cov']:.3f}")
    pd.DataFrame(oof_rows).to_csv("results/tables/B_mixed_oof.csv",
                                  index=False)
    var = pd.DataFrame(var_rows)
    var.to_csv("results/tables/B_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(var))
    ax.bar(x, var["study_between"], label="between-study")
    ax.bar(x, var["study_within"], bottom=var["study_between"],
           label="within-study")
    ax.set_xlabel("seed"); ax.set_ylabel("OOF residual variance (logit²)")
    ax.set_title("Variance decomposition: study vs within-study")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/B_variance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, nom in [("pi80_cov", 0.80), ("pi90_cov", 0.90),
                     ("pi95_cov", 0.95)]:
        ax.scatter(var["seed"], var[col], label=f"{col} (nom {nom})")
        ax.axhline(nom, ls="--", c="gray")
    ax.set_xlabel("seed"); ax.set_ylabel("Empirical PI coverage")
    ax.set_title("Mixed-model PI coverage on unseen studies")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/B_picoverage.png", dpi=150)

    bx = pd.DataFrame(oof_rows)
    bx = bx[bx["variant"] == "xgb-reg"].set_index("seed")
    ba = pd.DataFrame(oof_rows)
    ba = ba[ba["variant"] == "AFT-baseline"].set_index("seed")
    print("paired xgb-reg minus AFT RMSE CI:",
          paired_ci(bx["rmse"] - ba["rmse"]))
    print("study ICC mean±std: %.3f±%.3f"
          % (var["study_icc"].mean(), var["study_icc"].std()))
    print("mem ICC mean±std: %.3f±%.3f"
          % (var["mem_icc"].mean(), var["mem_icc"].std()))
    print("Done.")


if __name__ == "__main__":
    main()
