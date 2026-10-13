"""Part 2b — XGBoost AFT replication (paper Methods, "Model Development").

Pipeline:
  * DMatrix with right-censoring: censored -> lower=logit(R), upper=+inf;
    uncensored -> lower=upper=logit(R).  (XGBoost AFT `predict` returns
    values on the label/original scale -- verified empirically in
    notebooks/probe_aft_scale.py -- so predictions are E[logit(R)|X].)
  * Optuna TPE (seeded) over the 9 hyperparameters named in the paper,
    objective = mean 5-fold stratified (by cluster) aft-nloglik on train.
  * Refit best config on full train; report CV (mean+/-sd, same folds) and
    test metrics:
      - aft-nloglik  (xgboost built-in, whole split incl. censored)
      - C-index      (lifelines, event_observed = uncensored)
      - RMSE/MAE (%) on uncensored only, predictions mapped to R scale
      - R2 on logit(R) scale, uncensored only  (+ variants for §R2 probe)
  * R2-definition probe: same predictions scored under 6 definitions to
    explain the CV/test R2 gap vs the paper (censoring, scale, outliers).

Outputs:
  results/tables/best_params.json, optuna_history.csv,
    replication_metrics.csv, r2_definitions.csv, cv_fold_metrics.csv
  results/figures/rep_pred_vs_obs_logit.png, rep_pred_vs_obs_pct.png
  results/models/xgb_aft_replication.json

Usage: N_TRIALS=100 python scripts/03_train_evaluate.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lifelines.utils import concordance_index
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs as _ensure_dirs
    _ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        os.makedirs(_d, exist_ok=True)

SEED = 42
N_TRIALS = int(os.environ.get("N_TRIALS", "100"))
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]
INF = np.inf


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def make_dmatrix(X, y, cens):
    lower = y.astype(float)
    upper = np.where(cens, INF, y.astype(float))
    dm = xgb.DMatrix(X, label_lower_bound=lower, label_upper_bound=upper)
    return dm


def aft_nloglik(model, dm):
    """Mean negative log-lik via xgboost's own aft-nloglik metric."""
    res = model.eval(dm)
    # res like '[0]\tVALID-aft-nloglik:1.234'
    return float(res.split("aft-nloglik:")[1])


def suggest_params(trial):
    return {
        "objective": "survival:aft",
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "aft_loss_distribution": trial.suggest_categorical(
            "aft_loss_distribution", ["normal", "logistic", "extreme"]),
        "aft_loss_distribution_scale": trial.suggest_float(
            "aft_loss_distribution_scale", 0.1, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 30.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 30.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "seed": SEED,
    }, trial.suggest_int("n_estimators", 100, 1500)


def full_metrics(y_true_logit, R_true, pred_logit, cens, dm_for_nll=None,
                 model=None):
    """All metrics for one data split."""
    out = {}
    if model is not None and dm_for_nll is not None:
        out["nloglik"] = aft_nloglik(model, dm_for_nll)
    out["cindex"] = concordance_index(y_true_logit, pred_logit,
                                      ~np.asarray(cens, bool))
    m = ~np.asarray(cens, bool)  # uncensored only, per paper
    pred_R = sigmoid(pred_logit[m])
    err_pct = (pred_R - R_true[m]) * 100.0
    out["rmse_pct"] = float(np.sqrt(np.mean(err_pct ** 2)))
    out["mae_pct"] = float(np.mean(np.abs(err_pct)))
    out["r2_logit_uncens"] = float(r2_score(y_true_logit[m], pred_logit[m]))
    return out


def r2_probe_table(y_logit, R_obs, pred_logit, cens):
    """Score identical predictions under competing R2 definitions."""
    m = ~np.asarray(cens, bool)
    rows = []
    rows.append(("logit scale, uncens only [paper stated]",
                 r2_score(y_logit[m], pred_logit[m])))
    rows.append(("logit scale, ALL points (incl. censored)",
                 r2_score(y_logit, pred_logit)))
    pred_R, R = sigmoid(pred_logit), np.asarray(R_obs)
    rows.append(("% rejection scale, uncens only",
                 r2_score(R[m] * 100, pred_R[m] * 100)))
    rows.append(("% rejection scale, ALL points",
                 r2_score(R * 100, pred_R * 100)))
    hi = y_logit < 9.0  # drop R=0.9999 pile-up (logit 9.21)
    rows.append(("logit, uncens, excl. logit>=9.0 outliers",
                 r2_score(y_logit[m & hi], pred_logit[m & hi])))
    rows.append(("logit, uncens, CORR^2 (no bias penalty)",
                 float(np.corrcoef(y_logit[m], pred_logit[m])[0, 1] ** 2)))
    return pd.DataFrame(rows, columns=["definition", "R2"])


def main():
    frame = pd.read_csv("results/modeling_frame.csv")
    X = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    clu = frame["cluster"].to_numpy()
    is_tr = (frame["split"] == "train").to_numpy()
    R = frame["meta_Rejection"].to_numpy()
    Xtr, ytr, ctr, clutr, Rtr = X[is_tr], y[is_tr], cens[is_tr], clu[is_tr], R[is_tr]
    Xte, yte, cte, Rte = X[~is_tr], y[~is_tr], cens[~is_tr], R[~is_tr]
    print(f"train={len(Xtr)} test={len(Xte)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xtr, clutr))

    def objective(trial):
        params, n_est = suggest_params(trial)
        nlls = []
        for tri, vai in folds:
            dm_tr = make_dmatrix(Xtr[tri], ytr[tri], ctr[tri])
            dm_va = make_dmatrix(Xtr[vai], ytr[vai], ctr[vai])
            bst = xgb.train(params, dm_tr, num_boost_round=n_est,
                            evals=[(dm_va, "VALID")], verbose_eval=False)
            nlls.append(aft_nloglik(bst, dm_va))
        trial.set_user_attr("mean_nll", float(np.mean(nlls)))
        return float(np.mean(nlls))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print("Best trial:", study.best_trial.number,
          "mean CV nll=%.4f" % study.best_value)
    print("Best params:", study.best_params)
    with open("results/tables/best_params.json", "w") as f:
        json.dump({"trial": study.best_trial.number,
                   "mean_cv_nll": study.best_value,
                   **study.best_params}, f, indent=2)
    hist = pd.DataFrame(
        [{"trial": t.number, "mean_cv_nll": t.value,
          **t.params} for t in study.trials if t.value is not None])
    hist.to_csv("results/tables/optuna_history.csv", index=False)

    # ---- refit best on full train -----------------------------------------
    bp = study.best_params
    params = {"objective": "survival:aft", "seed": SEED,
              "eta": bp["eta"],
              "aft_loss_distribution": bp["aft_loss_distribution"],
              "aft_loss_distribution_scale": bp["aft_loss_distribution_scale"],
              "max_depth": bp["max_depth"], "subsample": bp["subsample"],
              "colsample_bytree": bp["colsample_bytree"],
              "min_child_weight": bp["min_child_weight"],
              "lambda": bp["lambda"], "alpha": bp["alpha"]}
    n_est = bp["n_estimators"]
    dm_tr_full = make_dmatrix(Xtr, ytr, ctr)
    dm_te = make_dmatrix(Xte, yte, cte)
    final = xgb.train(params, dm_tr_full, num_boost_round=n_est)
    final.save_model("results/models/xgb_aft_replication.json")

    # ---- CV metrics with final config (same folds) + test -----------------
    fold_rows = []
    for i, (tri, vai) in enumerate(folds):
        dm_tri = make_dmatrix(Xtr[tri], ytr[tri], ctr[tri])
        dm_vai = make_dmatrix(Xtr[vai], ytr[vai], ctr[vai])
        bst = xgb.train(params, dm_tri, num_boost_round=n_est)
        pv = bst.predict(dm_vai)
        m = full_metrics(ytr[vai], Rtr[vai], pv, ctr[vai],
                         dm_for_nll=dm_vai, model=bst)
        m["fold"] = i
        fold_rows.append(m)
    cv = pd.DataFrame(fold_rows)
    cv.to_csv("results/tables/cv_fold_metrics.csv", index=False)

    p_tr = final.predict(dm_tr_full)
    p_te = final.predict(dm_te)
    m_tr = full_metrics(ytr, Rtr, p_tr, ctr, dm_tr_full, final)
    m_te = full_metrics(yte, Rte, p_te, cte, dm_te, final)
    rep = pd.DataFrame([
        {"split": "CV (mean±sd)",
         "nloglik": f"{cv['nloglik'].mean():.2f} ± {cv['nloglik'].std():.2f}",
         "cindex": f"{cv['cindex'].mean():.2f} ± {cv['cindex'].std():.2f}",
         "rmse_pct": f"{cv['rmse_pct'].mean():.2f} ± {cv['rmse_pct'].std():.2f}",
         "mae_pct": f"{cv['mae_pct'].mean():.2f} ± {cv['mae_pct'].std():.2f}",
         "r2_logit": f"{cv['r2_logit_uncens'].mean():.2f} ± {cv['r2_logit_uncens'].std():.2f}",
         **{k: cv[k].mean() for k in
            ["nloglik", "cindex", "rmse_pct", "mae_pct", "r2_logit_uncens"]}},
        {"split": "train (refit)",
         "nloglik": f"{m_tr['nloglik']:.2f}", "cindex": f"{m_tr['cindex']:.2f}",
         "rmse_pct": f"{m_tr['rmse_pct']:.2f}", "mae_pct": f"{m_tr['mae_pct']:.2f}",
         "r2_logit": f"{m_tr['r2_logit_uncens']:.2f}", **m_tr},
        {"split": "test",
         "nloglik": f"{m_te['nloglik']:.2f}", "cindex": f"{m_te['cindex']:.2f}",
         "rmse_pct": f"{m_te['rmse_pct']:.2f}", "mae_pct": f"{m_te['mae_pct']:.2f}",
         "r2_logit": f"{m_te['r2_logit_uncens']:.2f}", **m_te},
        {"split": "PAPER CV", "nloglik": "1.03 ± 0.10", "cindex": "0.86 ± 0.01",
         "rmse_pct": "5.36 ± 0.41", "mae_pct": "2.86 ± 0.21",
         "r2_logit": "0.80 ± 0.04"},
        {"split": "PAPER test", "nloglik": "1.91", "cindex": "0.89",
         "rmse_pct": "5.80", "mae_pct": "2.70", "r2_logit": "0.86"},
    ])
    rep.to_csv("results/tables/replication_metrics.csv", index=False)
    print(rep[["split", "nloglik", "cindex", "rmse_pct", "mae_pct",
               "r2_logit"]].to_string(index=False))

    # ---- R2 definition probe (test split) ----------------------------------
    probe = r2_probe_table(yte, Rte, p_te, cte)
    probe["split"] = "test"
    probe_tr = r2_probe_table(ytr, Rtr, p_tr, ctr)
    probe_tr["split"] = "train"
    probe_all = pd.concat([probe_tr, probe], ignore_index=True)
    probe_all.to_csv("results/tables/r2_definitions.csv", index=False)
    print("\n== R2 under competing definitions ==")
    print(probe_all.to_string(index=False))

    # ---- Fig 3A analogs ----------------------------------------------------
    for name, yo, po, co in [("train", ytr, p_tr, ctr),
                             ("test", yte, p_te, cte)]:
        m = ~co
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(yo[m], po[m], s=8, alpha=0.4)
        lo = min(yo[m].min(), po[m].min())
        hi = max(yo[m].max(), po[m].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Observed logit(R) (uncensored)")
        ax.set_ylabel("Predicted logit(R)")
        ax.set_title(f"{name}: R²={r2_score(yo[m], po[m]):.2f} "
                     f"(n={m.sum()} uncens)")
        fig.tight_layout()
        fig.savefig(f"results/figures/rep_pred_vs_obs_logit_{name}.png",
                    dpi=150)
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
