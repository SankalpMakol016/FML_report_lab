"""Round-2 F — Stacked ensemble tuned under grouped CV.

Level-0 (grouped-OOF per seed): XGBoost-AFT (paper-style, censored),
LightGBM-regression, CatBoost-regression (both censoring-ignored).
Level-1: ridge meta-learner. Honest nesting: for each outer grouped fold,
the meta-learner is fit on the OTHER folds' OOF predictions and applied
to the held-out fold (no leakage); final weights refit on all OOF for
inspection only.
Hyperparameters (all three bases + ridge alpha) are tuned JOINTLY with
Optuna TPE under grouped 3-fold CV on seed 0 (20 trials, RMSE objective),
then FROZEN for the 10-seed grouped 5-fold evaluation. Tuned bases are
also scored alone (stacking ablation).

Outputs: results/tables/F_stack_tune.csv (trials), F_stack_params.json,
  F_stack_oof.csv (per-seed metrics: stack + 3 tuned bases + AFT baseline),
  F_stack_gain.csv (paired CIs), results/figures/F_rmse.png
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.linear_model import Ridge
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, dm_aft, SEEDS, FEATURES_13, SEED)

N_TUNE = int(os.environ.get("N_TUNE", "20"))


def fit_xgb_aft(Xtr, ytr, ctr, p, n):
    pr = {"objective": "survival:aft", "seed": SEED,
          "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": p["scale"],
          "eta": p["eta"], "max_depth": p["depth"],
          "subsample": p["sub"], "colsample_bytree": p["col"],
          "min_child_weight": p["mcw"], "lambda": p["lam"],
          "alpha": p["alf"]}
    return xgb.train(pr, dm_aft(Xtr, ytr, ctr), num_boost_round=n)


def fit_lgb(Xtr, ytr, p, n):
    return lgb.train({"objective": "regression",
                      "learning_rate": p["eta"], "num_leaves": p["leaves"],
                      "min_data_in_leaf": 20, "feature_fraction": p["feat"],
                      "seed": SEED, "verbosity": -1},
                     lgb.Dataset(Xtr, label=ytr), num_boost_round=n)


def fit_cat(Xtr, ytr, p, n):
    m = cb.CatBoostRegressor(iterations=n, depth=p["depth"],
                             learning_rate=p["eta"],
                             l2_leaf_reg=p["l2"], random_seed=SEED,
                             verbose=False, allow_writing_files=False)
    return m.fit(Xtr, ytr)


def predict_all(models, Xva):
    ma, ml, mc = models
    import xgboost as _x
    return (ma.predict(_x.DMatrix(Xva)), ml.predict(Xva),
            mc.predict(Xva))


def oof_block(X, y, cens, groups, seed, P, n_splits=5):
    """OOF level-0 preds + nested ridge-stack preds for one seed."""
    pa = np.empty(len(y))
    pl = np.empty(len(y))
    pc = np.empty(len(y))
    folds = grouped_folds(groups, seed, n_splits=n_splits)
    for tri, vai in folds:
        ma = fit_xgb_aft(X[tri], y[tri], cens[tri], P["aft"], P["n_aft"])
        ml = fit_lgb(X[tri], y[tri], P["lgb"], P["n_lgb"])
        mc = fit_cat(X[tri], y[tri], P["cat"], P["n_cat"])
        a, l, c = predict_all((ma, ml, mc), X[vai])
        pa[vai], pl[vai], pc[vai] = a, l, c
    Z = np.column_stack([pa, pl, pc])
    ps = np.empty(len(y))
    rgs = []
    for tri, vai in folds:
        rg = Ridge(alpha=P["ridge"]).fit(Z[tri], y[tri])
        ps[vai] = rg.predict(Z[vai])
        rgs.append(rg.coef_)
    return {"aft": pa, "lgb": pl, "cat": pc, "stack": ps,
            "coef": np.mean(rgs, axis=0)}


def rmse_of(y, cens, R, pred):
    m = ~cens
    e = (1.0 / (1.0 + np.exp(-pred[m])) - R[m]) * 100.0
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    frame, _ = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)

    def suggest(trial):
        return {
            "aft": {"eta": trial.suggest_float("a_eta", 0.005, 0.15,
                                               log=True),
                    "depth": trial.suggest_int("a_depth", 2, 5),
                    "scale": trial.suggest_float("a_scale", 0.3, 1.5,
                                                 log=True),
                    "sub": trial.suggest_float("a_sub", 0.6, 1.0),
                    "col": trial.suggest_float("a_col", 0.6, 1.0),
                    "mcw": trial.suggest_float("a_mcw", 3.0, 20.0,
                                               log=True),
                    "lam": trial.suggest_float("a_lam", 0.01, 5.0,
                                               log=True),
                    "alf": trial.suggest_float("a_alf", 1e-4, 1.0,
                                               log=True)},
            "lgb": {"eta": trial.suggest_float("l_eta", 0.01, 0.15,
                                               log=True),
                    "leaves": trial.suggest_int("l_leaves", 15, 63),
                    "feat": trial.suggest_float("l_feat", 0.6, 1.0)},
            "cat": {"eta": trial.suggest_float("c_eta", 0.01, 0.15,
                                               log=True),
                    "depth": trial.suggest_int("c_depth", 3, 7),
                    "l2": trial.suggest_float("c_l2", 1.0, 20.0,
                                              log=True)},
            "ridge": trial.suggest_float("ridge", 0.1, 100.0, log=True),
            "n_aft": trial.suggest_int("n_aft", 200, 700),
            "n_lgb": trial.suggest_int("n_lgb", 200, 600),
            "n_cat": trial.suggest_int("n_cat", 200, 600),
        }

    def objective(trial):
        P = suggest(trial)
        return rmse_of(y, cens, R,
                       oof_block(X, y, cens, groups, 0, P,
                                 n_splits=3)["stack"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TUNE)
    P = suggest(study.best_trial)
    # rebind best values
    bp = study.best_params
    P = {"aft": {"eta": bp["a_eta"], "depth": bp["a_depth"],
                 "scale": bp["a_scale"], "sub": bp["a_sub"],
                 "col": bp["a_col"], "mcw": bp["a_mcw"], "lam": bp["a_lam"],
                 "alf": bp["a_alf"]},
         "lgb": {"eta": bp["l_eta"], "leaves": bp["l_leaves"],
                 "feat": bp["l_feat"]},
         "cat": {"eta": bp["c_eta"], "depth": bp["c_depth"],
                 "l2": bp["c_l2"]},
         "ridge": bp["ridge"], "n_aft": bp["n_aft"], "n_lgb": bp["n_lgb"],
         "n_cat": bp["n_cat"]}
    with open("results/tables/F_stack_params.json", "w") as f:
        json.dump({"tune_rmse_seed0_3fold": study.best_value, **bp},
                  f, indent=2)
    pd.DataFrame(
        [{"trial": t.number, "rmse": t.value, **t.params}
         for t in study.trials if t.value is not None]).to_csv(
        "results/tables/F_stack_tune.csv", index=False)
    print("tune best RMSE(seed0,3fold)=%.3f" % study.best_value)

    rows = []
    for s in SEEDS:
        O = oof_block(X, y, cens, groups, s, P, n_splits=5)
        print(f"seed {s} stack coefs: "
              + str(np.round(O['coef'], 3)))
        for v in ["aft", "lgb", "cat", "stack"]:
            m = oof_metrics(y, cens, R, O[v])
            m.update({"variant": ("stack" if v == "stack"
                                  else v + "-tuned"), "seed": s})
            rows.append(m)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/F_stack_oof.csv", index=False)
    b = oof[oof["variant"] == "aft-tuned"].set_index("seed")
    gains = []
    for v in ["lgb-tuned", "cat-tuned", "stack"]:
        d = oof[oof["variant"] == v].set_index("seed")
        for metric in ["rmse", "mae", "cindex", "r2"]:
            ci = paired_ci(d[metric] - b[metric])
            gains.append({"variant": v, "metric": metric,
                          "var_mean": float(d[metric].mean()), **ci})
    gain = pd.DataFrame(gains)
    gain.to_csv("results/tables/F_stack_gain.csv", index=False)
    print(gain[gain["metric"] == "rmse"].to_string(index=False))
    print(oof.groupby("variant")[["rmse", "mae", "cindex", "r2"]].agg(
        ["mean", "std"]).round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.5, 4))
    piv = oof.pivot(index="seed", columns="variant", values="rmse")
    ax.boxplot([piv[c] for c in piv.columns], labels=piv.columns)
    ax.set_ylabel("Grouped-CV RMSE (% rejection)")
    ax.set_title("Stacked ensemble vs tuned bases (10 seeds)")
    fig.tight_layout()
    fig.savefig("results/figures/F_rmse.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
