"""Part 2d — Metric-definition forensics (saved artifacts for REPORT §2.2).

  * cv_pct_r2.csv: retrain the 5 stratified folds with the tuned config and
    record BOTH logit-scale and percent-scale R2 per fold (the %-scale folds
    are cited in the report but were previously only in stdout).
  * residual_diagnostics.csv: test-split worst-12 uncensored residuals,
    SS_tot/SS_res, R2 with and without the top-2 leverage points.
  * cindex_demo.csv: full-fit C-index censoring-aware vs censoring-ignored
    vs Spearman rank correlation (bookkeeping audit, REPORT §3d).

Deterministic (SEED=42). Runtime ~3 min.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from lifelines.utils import concordance_index
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs as _ensure_dirs
    _ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        _os.makedirs(_d, exist_ok=True)

SEED = 42
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]
INF = np.inf


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def dm(X, y, cens):
    return xgb.DMatrix(X, label_lower_bound=y.astype(float),
                       label_upper_bound=np.where(cens, INF, y.astype(float)))


def main():
    frame = pd.read_csv("results/modeling_frame.csv")
    X = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    clu = frame["cluster"].to_numpy()
    R = frame["meta_Rejection"].to_numpy()
    tr = (frame["split"] == "train").to_numpy()
    with open("results/tables/best_params.json") as f:
        bp = json.load(f)
    params = {"objective": "survival:aft", "seed": SEED,
              "eta": bp["eta"],
              "aft_loss_distribution": bp["aft_loss_distribution"],
              "aft_loss_distribution_scale":
                  bp["aft_loss_distribution_scale"],
              "max_depth": bp["max_depth"], "subsample": bp["subsample"],
              "colsample_bytree": bp["colsample_bytree"],
              "min_child_weight": bp["min_child_weight"],
              "lambda": bp["lambda"], "alpha": bp["alpha"]}
    n_est = bp["n_estimators"]

    # ---- per-fold logit-R2 and %-R2 -----------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for i, (tri, vai) in enumerate(skf.split(X[tr], clu[tr])):
        b = xgb.train(params, dm(X[tr][tri], y[tr][tri], cens[tr][tri]),
                      num_boost_round=n_est)
        pv = b.predict(dm(X[tr][vai], y[tr][vai], cens[tr][vai]))
        mv = ~cens[tr][vai]
        rows.append({"fold": i,
                     "r2_logit_uncens": float(r2_score(y[tr][vai][mv],
                                                      pv[mv])),
                     "r2_pct_uncens": float(r2_score(
                         R[tr][vai][mv] * 100, sigmoid(pv[mv]) * 100))})
    pct = pd.DataFrame(rows)
    pct.to_csv("results/tables/cv_pct_r2.csv", index=False)
    print(pct.to_string(index=False))
    print("mean logit-R2=%.2f sd=%.2f | mean %%-R2=%.2f sd=%.2f"
          % (pct["r2_logit_uncens"].mean(), pct["r2_logit_uncens"].std(),
             pct["r2_pct_uncens"].mean(), pct["r2_pct_uncens"].std()))

    # ---- test residual forensics --------------------------------------------
    model = xgb.Booster()
    model.load_model("results/models/xgb_aft_replication.json")
    te = frame[~tr].reset_index(drop=True)
    p = model.predict(dm(X[~tr], y[~tr], cens[~tr]))
    te["pred_logit"] = p
    te["resid"] = te["pred_logit"] - te["y_logit"]
    te["sq_resid"] = te["resid"] ** 2
    u = te[te["censored"] == 0]
    ss_tot = float(((u["y_logit"] - u["y_logit"].mean()) ** 2).sum())
    ss_res = float(u["sq_resid"].sum())
    worst = u.nlargest(12, "sq_resid")
    top2 = worst["sq_resid"].iloc[:2].sum()
    diag = pd.DataFrame([{
        "n_test_uncens": int(len(u)), "ss_tot": ss_tot, "ss_res": ss_res,
        "r2_logit": float(1 - ss_res / ss_tot),
        "top2_ss_share": float(top2 / ss_res),
        "r2_logit_excl_top2": float(
            1 - (ss_res - top2) / ((u["y_logit"] - u["y_logit"].mean()
                                    ) ** 2).sum()),
    }])
    diag.to_csv("results/tables/residual_diagnostics.csv", index=False)
    worst[["meta_ref_id", "meta_mem", "meta_PFAS", "meta_Rejection",
           "y_logit", "pred_logit", "resid", "sq_resid",
           "pH", "mon_cat_mM", "doc_mg/L",
           "pfas_feed_conc_µg/L"]].to_csv(
        "results/tables/worst_residuals.csv", index=False)
    print(diag.to_string(index=False))

    # ---- C-index bookkeeping demo --------------------------------------------
    dall = xgb.DMatrix(X)
    pfull = model.predict(dall)
    demo = pd.DataFrame([{
        "cindex_censoring_aware": float(concordance_index(y, pfull, ~cens)),
        "cindex_ignoring_censoring": float(concordance_index(
            y, pfull, np.ones(len(y), bool))),
        "spearman_y_pred": float(spearmanr(y, pfull)[0]),
    }])
    demo.to_csv("results/tables/cindex_demo.csv", index=False)
    print(demo.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
