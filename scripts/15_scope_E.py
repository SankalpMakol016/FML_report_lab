"""Round-2 E — Data scope: include Rejection < 0.5 rows.

Key constraint found by probing: XGBoost AFT *accepts* non-positive labels
but returns NaN predictions (silently broken), so every E model is
XGBoost-regression on logit(R) (censoring flag kept for bookkeeping; the 67
low rows are all uncensored). Documented, not hidden.

  Q1 extrapolation: baseline AFT trained on R>=0.5 rows from OTHER studies
      (leave-study-out over the 18 studies owning low rows) predicts the 67
      low rows -> honest low-retention error of the paper-style model.
  Q2 scope expansion: XGB-reg trained on ALL 2517 rows vs on R>=0.5 only,
      same 10-seed grouped folds (folds built on all rows) -> paired
      comparison overall, on the low slice (R<0.5) and the high slice.
Reports whether low-retention coverage improves and at what cost to the
high-retention regime.

Outputs: results/tables/E_scope.csv, E_lowprobe.csv,
  results/figures/E_rmseslice.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (grouped_folds, oof_metrics, paired_ci, fit_aft,
                     SEEDS, FEATURES_13, INF)

XGBR = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42}


def main():
    raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                        sheet_name="PFAS_literature_dataset_2b").reset_index(
                            drop=True)
    Xa = raw[FEATURES_13].to_numpy()
    Ra = raw["Rejection"].to_numpy(float)
    ya = np.log(np.clip(Ra, 1e-12, 0.9999)
                / (1 - np.clip(Ra, 1e-12, 0.9999)))
    ca = (raw["LOQ"] == "right").to_numpy()
    ga = raw["ref_id"].to_numpy()
    hi = Ra >= 0.5
    print(f"all={len(raw)} hi={hi.sum()} low={(~hi).sum()} "
          f"censored-low={ca[~hi].sum()} studies-low={len(np.unique(ga[~hi]))}")

    # ---- Q1: paper-style AFT (R>=0.5, other studies) -> low rows -----------
    low_idx = np.where(~hi)[0]
    plow = np.empty(len(low_idx))
    for j, i in enumerate(low_idx):
        tr = hi & (ga != ga[i])
        bst = fit_aft(Xa[tr], ya[tr], ca[tr])
        plow[j] = bst.predict(xgb.DMatrix(Xa[i].reshape(1, -1)))[0]
    yl, Rl = ya[low_idx], Ra[low_idx]
    err = (1.0 / (1.0 + np.exp(-plow)) - Rl) * 100.0
    q1 = {"n_low": len(low_idx),
          "rmse_low": float(np.sqrt(np.mean(err ** 2))),
          "mae_low": float(np.mean(np.abs(err))),
          "frac_within_10pp": float((np.abs(err) < 10).mean())}
    pd.DataFrame([q1]).to_csv("results/tables/E_lowprobe.csv", index=False)
    print("Q1 AFT(R>=0.5, other studies) on R<0.5 rows:", q1)

    # ---- Q2: reg on all vs reg on hi-only, same folds -----------------------
    rows = []
    for s in SEEDS:
        p_all = np.empty(len(raw))
        p_hi = np.empty(len(raw))
        for tri, vai in grouped_folds(ga, s):
            m_all = xgb.train(XGBR, xgb.DMatrix(Xa[tri], label=ya[tri]),
                              num_boost_round=500)
            p_all[vai] = m_all.predict(xgb.DMatrix(Xa[vai]))
            htr = tri[hi[tri]]
            m_hi = xgb.train(XGBR, xgb.DMatrix(Xa[htr], label=ya[htr]),
                             num_boost_round=500)
            p_hi[vai] = m_hi.predict(xgb.DMatrix(Xa[vai]))
        for tag, p in [("reg-all", p_all), ("reg-hi-only", p_hi)]:
            mall = oof_metrics(ya, ca, Ra, p)
            mlo = oof_metrics(ya[~hi], ca[~hi], Ra[~hi], p[~hi])
            mhi = oof_metrics(ya[hi], ca[hi], Ra[hi], p[hi])
            rows.append({"variant": tag, "seed": s,
                         "rmse_all": mall["rmse"], "mae_all": mall["mae"],
                         "cindex_all": mall["cindex"],
                         "rmse_low": mlo["rmse"], "mae_low": mlo["mae"],
                         "rmse_hi": mhi["rmse"], "mae_hi": mhi["mae"],
                         "r2_hi": mhi["r2"]})
    q2 = pd.DataFrame(rows)
    q2.to_csv("results/tables/E_scope.csv", index=False)
    a = q2[q2["variant"] == "reg-all"].set_index("seed")
    h = q2[q2["variant"] == "reg-hi-only"].set_index("seed")
    print("\nreg-all means:\n",
          a[["rmse_all", "rmse_low", "rmse_hi"]].mean().to_string())
    print("\nreg-hi-only means:\n",
          h[["rmse_all", "rmse_low", "rmse_hi"]].mean().to_string())
    for col in ["rmse_all", "rmse_low", "rmse_hi", "cindex_all"]:
        print(col, "paired (all - hi-only):",
              paired_ci(a[col] - h[col]))

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    x = np.arange(3)
    ax.bar(x - w / 2, a[["rmse_low", "rmse_hi",
                         "rmse_all"]].mean(), w, yerr=a[[
                             "rmse_low", "rmse_hi",
                             "rmse_all"]].std(), capsize=4,
           label="reg-all")
    ax.bar(x + w / 2, h[["rmse_low", "rmse_hi",
                         "rmse_all"]].mean(), w, yerr=h[[
                             "rmse_low", "rmse_hi",
                             "rmse_all"]].std(), capsize=4,
           label="reg-hi-only")
    ax.set_xticks(x, ["low (R<0.5)", "high (R>=0.5)", "all"])
    ax.set_ylabel("Grouped-CV RMSE (% rejection)")
    ax.set_title("Scope expansion: train on all rows vs R>=0.5 only")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/E_rmseslice.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
