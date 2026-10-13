"""Part 3 — Critical audit: leakage via grouped evaluation.

  * (a) Fixed-hyperparameter evaluation under GroupKFold by ref_id
    (study), leave-one-study-out (LOSO, mean over studies), GroupKFold by
    membrane, and GroupKFold by PFAS species. Compares against the
    paper's random (point-wise) split protocol.
  * (a2) Re-tune hyperparameters UNDER grouped (study) 5-fold CV with
    Optuna TPE and report the best honest performance. Fixed protocol:
    N_TRIALS=30, SEED=42 (override with N_TRIALS env only for --fast
    smoke tests, e.g. N_TRIALS=2). Always writes
    results/tables/grouped_best_params.json.
  * (c) Duplicate / near-duplicate analysis across the paper-style random
    train/test split: exact duplicates, and near-duplicates (same
    study+membrane+PFAS with (near-)identical features and target).
  * (d) C-index definition audit: censoring-aware (event_observed=uncensored) vs a
    naive rank correlation ignoring censoring.

Fixed hyperparameters = best config from scripts/03 (best_params.json);
the point is to isolate the effect of the SPLIT, not the tuning.

Outputs: results/tables/grouped_metrics.csv, loso_metrics.csv,
  grouped_best_params.json (always), duplicate_audit.csv,
  results/figures/loso_hist.png
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from scipy.stats import spearmanr

SEED = 42
# Fixed grouped-retune protocol: 30 Optuna trials, seed 42.
# Override with N_TRIALS env only for --fast smoke tests.
N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]
INF = np.inf


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def make_dmatrix(X, y, cens):
    return xgb.DMatrix(X, label_lower_bound=y.astype(float),
                       label_upper_bound=np.where(cens, INF, y.astype(float)))


def aft_nloglik(model, dm):
    return float(model.eval(dm).split("aft-nloglik:")[1])


def fit_predict(params, n_est, Xtr, ytr, ctr, Xva):
    bst = xgb.train(params, make_dmatrix(Xtr, ytr, ctr),
                    num_boost_round=n_est)
    return bst, bst.predict(xgb.DMatrix(Xva))


def fold_metrics(y_va, R_va, c_va, p_va, bst, dm_va):
    m = ~c_va
    out = {"nloglik": aft_nloglik(bst, dm_va)}
    try:
        out["cindex"] = concordance_index(y_va, p_va, m)
    except ZeroDivisionError:  # single-uncensored holdouts: undefined
        out["cindex"] = np.nan
    err = (sigmoid(p_va[m]) - R_va[m]) * 100.0
    out["rmse_pct"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae_pct"] = float(np.mean(np.abs(err)))
    out["r2_logit_uncens"] = float(r2_score(y_va[m], p_va[m])
                                    if m.sum() > 2 else np.nan)
    out["n_va"] = int(len(y_va))
    out["n_va_uncens"] = int(m.sum())
    return out


def grouped_eval(X, y, cens, R, groups, gname, params, n_est, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for i, (tri, vai) in enumerate(gkf.split(X, y, groups)):
        bst, p = fit_predict(params, n_est, X[tri], y[tri], cens[tri],
                             X[vai])
        m = fold_metrics(y[vai], R[vai], cens[vai], p, bst,
                         make_dmatrix(X[vai], y[vai], cens[vai]))
        m.update({"fold": i, "grouping": gname})
        rows.append(m)
    return pd.DataFrame(rows)


def loso(X, y, cens, R, groups, params, n_est):
    """Leave-one-study-out: one fit per study, metrics pooled + per-study."""
    studies = np.unique(groups)
    per, preds = [], np.empty(len(y))
    for s in studies:
        va = groups == s
        bst, p = fit_predict(params, n_est, X[~va], y[~va], cens[~va],
                             X[va])
        preds[va] = p
        m = fold_metrics(y[va], R[va], cens[va], p, bst,
                         make_dmatrix(X[va], y[va], cens[va]))
        m.update({"study": s, "n": int(va.sum())})
        per.append(m)
    per = pd.DataFrame(per)
    pooled = fold_metrics(y, R, cens, preds,
                          xgb.train(params, make_dmatrix(X, y, cens),
                                    num_boost_round=n_est),
                          make_dmatrix(X, y, cens))
    return per, pooled


def main():
    import sys as _sys
    import os as _os
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    try:
        from setup_paths import ensure_dirs
        ensure_dirs()
    except ImportError:
        for _d in ("results/tables", "results/figures", "results/models"):
            _os.makedirs(_d, exist_ok=True)
    frame = pd.read_csv("results/modeling_frame.csv")
    X = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    R = frame["meta_Rejection"].to_numpy()
    g_study = frame["meta_ref_id"].to_numpy()
    g_mem = frame["meta_mem"].to_numpy()
    g_pfas = frame["meta_PFAS"].to_numpy()
    print(f"studies={len(np.unique(g_study))} mems={len(np.unique(g_mem))} "
          f"pfas={len(np.unique(g_pfas))}")

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
    print("Fixed params from:", bp.get("trial"), bp["aft_loss_distribution"],
          "n_est=", n_est)

    # ---- (a/b) grouped 5-fold ---------------------------------------------
    res = []
    for g, name in [(g_study, "study(ref_id)"), (g_mem, "membrane"),
                    (g_pfas, "pfas")]:
        df = grouped_eval(X, y, cens, R, g, name, params, n_est)
        df.to_csv(f"results/tables/grouped_{name.split('(')[0]}.csv",
                  index=False)
        agg = {"grouping": name,
               "nloglik": f"{df['nloglik'].mean():.2f}±{df['nloglik'].std():.2f}",
               "cindex": f"{df['cindex'].mean():.2f}±{df['cindex'].std():.2f}",
               "rmse_pct": f"{df['rmse_pct'].mean():.2f}±{df['rmse_pct'].std():.2f}",
               "r2_logit": f"{df['r2_logit_uncens'].mean():.2f}±{df['r2_logit_uncens'].std():.2f}"}
        res.append(agg)
        print(name, agg)

    # ---- (a) LOSO ----------------------------------------------------------
    per, pooled = loso(X, y, cens, R, g_study, params, n_est)
    per.to_csv("results/tables/loso_per_study.csv", index=False)
    print("\nLOSO pooled:", {k: round(v, 3) for k, v in pooled.items()
                             if k not in ("n_va", "n_va_uncens")})
    print("LOSO per-study R2: median=%.2f frac_neg=%.2f"
          % (per["r2_logit_uncens"].median(),
             (per["r2_logit_uncens"] < 0).mean()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per["r2_logit_uncens"].clip(-1, 1), bins=25)
    ax.set_xlabel("Per-study test R² (logit, uncens; clipped to [-1,1])")
    ax.set_ylabel("# studies")
    ax.set_title("Leave-one-study-out R² distribution")
    fig.tight_layout()
    fig.savefig("results/figures/loso_hist.png", dpi=150)
    plt.close(fig)

    res.append({"grouping": "LOSO-pooled",
                "nloglik": f"{pooled['nloglik']:.2f}",
                "cindex": f"{pooled['cindex']:.2f}",
                "rmse_pct": f"{pooled['rmse_pct']:.2f}",
                "r2_logit": f"{pooled['r2_logit_uncens']:.2f}"})
    pd.DataFrame(res).to_csv("results/tables/grouped_metrics.csv",
                             index=False)

    # ---- (c) duplicates across paper-style random split --------------------
    tr_idx = frame.index[frame["split"] == "train"].to_numpy()
    te_idx = frame.index[frame["split"] == "test"].to_numpy()
    key_all = frame[FEATURES_13 + ["y_logit"]].copy()
    # exact duplicates anywhere
    dup_any = frame.duplicated(
        subset=FEATURES_13 + ["y_logit", "censored"], keep=False)
    # test rows with an exact match in train
    train_keys = set(map(tuple, key_all.iloc[tr_idx].astype(str).itertuples(
        index=False, name=None)))
    te_match = [tuple(r) in train_keys for r in key_all.iloc[te_idx].astype(
        str).itertuples(index=False, name=None)]
    # near-dupes: same study+mem+PFAS, features within 1% and |Δlogit|<0.05
    frame["_smp"] = g_study + "|" + g_mem + "|" + g_pfas
    near = 0
    for _, grp in frame.groupby("_smp"):
        ii = grp.index.to_numpy()
        te_ii = ii[np.isin(ii, te_idx)]
        tr_ii = ii[np.isin(ii, tr_idx)]
        if len(te_ii) == 0 or len(tr_ii) == 0:
            continue
        Xa, Xb = X[te_ii][:, None, :], X[tr_ii][None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(Xa - Xb) / (np.abs(Xb) + 1e-9)
        rel = np.where(np.isnan(Xa) | np.isnan(Xb), np.inf, rel)
        close = (rel.max(axis=2) < 0.01) & (
            np.abs(y[te_ii][:, None] - y[tr_ii][None, :]) < 0.05)
        near += int(close.any(axis=1).sum())
    dup_tbl = pd.DataFrame([
        {"check": "exact-dupe rows in full frame (13 feats+y+cens)",
         "n": int(dup_any.sum())},
        {"check": "test rows (n=%d) exactly matching a train row"
         % len(te_idx), "n": int(np.sum(te_match))},
        {"check": "test rows with near-dupe in train (same study/mem/PFAS,"
         " feats within 1%, |dlogit|<0.05)", "n": near},
    ])
    dup_tbl.to_csv("results/tables/duplicate_audit.csv", index=False)
    print("\n", dup_tbl.to_string(index=False))

    # ---- (d) C-index definition audit --------------------------------------
    bst_full = xgb.train(params, make_dmatrix(X, y, cens),
                         num_boost_round=n_est)
    p_full = bst_full.predict(xgb.DMatrix(X))
    ci_cens = concordance_index(y, p_full, ~cens)
    ci_naive = concordance_index(y, p_full,
                                 np.ones(len(y), bool))  # ignore censoring
    rho, _ = spearmanr(y, p_full)
    print(f"\nC-index censoring-aware: {ci_cens:.4f} | ignoring censoring: "
          f"{ci_naive:.4f} | Spearman(y, pred): {rho:.4f}")

    # ---- (a2) re-tune UNDER grouped CV (always; fixed N_TRIALS/SEED) -----
    # N_TRIALS defaults to 30 (see header); --fast smoke uses N_TRIALS=2.
    n_trials_eff = max(int(N_TRIALS), 1)
    if True:
        gkf = GroupKFold(n_splits=5)
        gfolds = list(gkf.split(X, y, g_study))

        def objective(trial):
            pr = {"objective": "survival:aft", "seed": SEED,
                  "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                  "aft_loss_distribution": trial.suggest_categorical(
                      "aft_loss_distribution",
                      ["normal", "logistic", "extreme"]),
                  "aft_loss_distribution_scale": trial.suggest_float(
                      "aft_loss_distribution_scale", 0.1, 5.0, log=True),
                  "max_depth": trial.suggest_int("max_depth", 2, 6),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float(
                      "colsample_bytree", 0.5, 1.0),
                  "min_child_weight": trial.suggest_float(
                      "min_child_weight", 1.0, 30.0, log=True),
                  "lambda": trial.suggest_float("lambda", 1e-3, 30.0,
                                                log=True),
                  "alpha": trial.suggest_float("alpha", 1e-4, 10.0,
                                               log=True)}
            n_e = trial.suggest_int("n_estimators", 100, 800)
            nlls = []
            for tri, vai in gfolds:
                b, _ = fit_predict(pr, n_e, X[tri], y[tri], cens[tri],
                                   X[vai])
                nlls.append(aft_nloglik(
                    b, make_dmatrix(X[vai], y[vai], cens[vai])))
            return float(np.mean(nlls))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials_eff)
        print("Grouped-tuning best:", study.best_trial.number,
              "nll=%.4f" % study.best_value, study.best_params)
        with open("results/tables/grouped_best_params.json", "w") as f:
            json.dump({"mean_grouped_cv_nll": study.best_value,
                       **study.best_params}, f, indent=2)
        gb = study.best_params
        gparams = {"objective": "survival:aft", "seed": SEED,
                   "eta": gb["eta"],
                   "aft_loss_distribution": gb["aft_loss_distribution"],
                   "aft_loss_distribution_scale":
                       gb["aft_loss_distribution_scale"],
                   "max_depth": gb["max_depth"],
                   "subsample": gb["subsample"],
                   "colsample_bytree": gb["colsample_bytree"],
                   "min_child_weight": gb["min_child_weight"],
                   "lambda": gb["lambda"], "alpha": gb["alpha"]}
        df = grouped_eval(X, y, cens, R, g_study, "study(ref_id)-retuned",
                          gparams, gb["n_estimators"])
        print(df[["nloglik", "cindex", "rmse_pct", "mae_pct",
                  "r2_logit_uncens"]].mean().to_string())
        df.to_csv("results/tables/grouped_study_retuned.csv", index=False)
        bst = xgb.train(gparams, make_dmatrix(X, y, cens),
                        num_boost_round=gb["n_estimators"])
        bst.save_model("results/models/xgb_aft_grouped.json")
    print("Done.")


if __name__ == "__main__":
    main()
