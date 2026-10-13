"""R2/nloglik stability probe: split-seed sensitivity with FIXED hyperparameters.

Question: how much of the CV/test metric gap vs the paper is just the
luck of the random 85/15 split? For SEEDS 0..9: redo the stratified
(cluster) split, train the fixed best config, and record CV + test
metrics. If logit-R2 swings wildly while %-metrics stay put, the paper's
headline R2 is split-dependent and not a robust replication target.

Outputs: results/tables/seed_sensitivity.csv,
         results/figures/seed_sensitivity.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
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
SEEDS = list(range(10))
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
    df_raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                           sheet_name="PFAS_literature_dataset_2b")
    d = df_raw[df_raw["Rejection"] >= 0.5].reset_index(drop=True)
    frame = pd.read_csv("results/modeling_frame.csv")
    X = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    clu = frame["cluster"].to_numpy()
    R = frame["meta_Rejection"].to_numpy()
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

    rows = []
    for s in SEEDS:
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=0.15, random_state=s,
                                  stratify=clu)
        bst = xgb.train(params, dm(X[tr], y[tr], cens[tr]),
                        num_boost_round=n_est)
        p = bst.predict(dm(X[te], y[te], cens[te]))
        m = ~cens[te]
        err = (sigmoid(p[m]) - R[te][m]) * 100.0
        # 5-fold CV on this seed's train (same protocol as paper)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cvr2 = []
        for tri, vai in skf.split(X[tr], clu[tr]):
            b = xgb.train(params, dm(X[tr][tri], y[tr][tri], cens[tr][tri]),
                          num_boost_round=n_est)
            pv = b.predict(dm(X[tr][vai], y[tr][vai], cens[tr][vai]))
            mv = ~cens[tr][vai]
            cvr2.append(r2_score(y[tr][vai][mv], pv[mv]))
        rows.append({
            "seed": s,
            "test_nll": float(bst.eval(dm(X[te], y[te], cens[te])).split(
                "aft-nloglik:")[1]),
            "test_cindex": concordance_index(y[te], p, m),
            "test_rmse": float(np.sqrt(np.mean(err ** 2))),
            "test_mae": float(np.mean(np.abs(err))),
            "test_r2_logit": float(r2_score(y[te][m], p[m])),
            "test_r2_pct": float(r2_score(R[te][m] * 100,
                                          sigmoid(p[m]) * 100)),
            "cv_r2_mean": float(np.mean(cvr2)),
            "cv_r2_sd": float(np.std(cvr2)),
        })
        print(f"seed {s}: test R2_logit={rows[-1]['test_r2_logit']:.2f} "
              f"R2_pct={rows[-1]['test_r2_pct']:.2f} "
              f"CV R2={rows[-1]['cv_r2_mean']:.2f}±{rows[-1]['cv_r2_sd']:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/seed_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(out["seed"], out["test_r2_logit"], "o-", label="logit R²")
    axes[0].plot(out["seed"], out["test_r2_pct"], "s-", label="% R²")
    axes[0].axhline(0.86, c="r", ls="--", label="paper test R²=0.86")
    axes[0].set_xlabel("split seed"); axes[0].set_ylabel("Test R²")
    axes[0].legend(); axes[0].set_title("Test R² vs split seed")
    axes[1].errorbar(out["seed"], out["cv_r2_mean"], yerr=out["cv_r2_sd"],
                     fmt="o-", capsize=3)
    axes[1].axhline(0.80, c="r", ls="--", label="paper CV R²=0.80")
    axes[1].set_xlabel("split seed"); axes[1].set_ylabel("CV R² (logit)")
    axes[1].legend(); axes[1].set_title("CV R² vs split seed")
    fig.tight_layout()
    fig.savefig("results/figures/seed_sensitivity.png", dpi=150)
    print("Wrote results/tables/seed_sensitivity.csv")


if __name__ == "__main__":
    main()
