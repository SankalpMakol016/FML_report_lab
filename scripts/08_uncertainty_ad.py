"""Part 4b — Uncertainty, applicability domain, and honest-model SHAP.

  * Quantile LightGBM (alpha=.1/.5/.9) under grouped study CV:
    80% prediction-interval coverage (uncensored points) + median RMSE.
  * Split-conformal intervals around grouped-CV point predictions
    (absolute-residual, 80% level): empirical coverage per fold.
  * Applicability domain: 5-NN distance (standardized, median-imputed)
    to grouped-fold train; error/coverage by distance decile; error in
    pre-defined sparse regions: ultrashort PFAS (carbons<=3), real water
    feeds, high DOC+salts.
  * SHAP (native TreeSHAP, margin space) on the best honest model
    (grouped-retuned AFT refit on all data) + PD curves for the 8
    high-interest features to test whether steric dominance and
    non-monotonic salt/DOC effects survive honest fitting.

Outputs: results/tables/uncertainty_coverage.csv, ad_regions.csv,
  honest_shap_importance.csv, results/figures/ad_error.png,
  honest_pd_*.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
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


def main():
    frame = pd.read_csv("results/modeling_frame.csv")
    raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                        sheet_name="PFAS_literature_dataset_2b")
    raw = raw[raw["Rejection"] >= 0.5].reset_index(drop=True)
    X = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    R = frame["meta_Rejection"].to_numpy()
    groups = frame["meta_ref_id"].to_numpy()
    carbons = raw["carbons"].to_numpy()
    feed_type = raw["feed_type"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))

    # ---- quantile LightGBM, grouped CV ------------------------------------
    cov_rows = []
    qpred = {}
    for a in (0.1, 0.5, 0.9):
        p = np.empty(len(y))
        for tri, vai in folds:
            m = lgb.train({"objective": "quantile", "alpha": a,
                           "learning_rate": 0.05, "num_leaves": 31,
                           "min_data_in_leaf": 20, "seed": SEED,
                           "verbosity": -1},
                          lgb.Dataset(X[tri], label=y[tri]),
                          num_boost_round=500)
            p[vai] = m.predict(X[vai])
        qpred[a] = p
    m = ~cens
    lo, hi = qpred[0.1][m], qpred[0.9][m]
    cov = ((y[m] >= lo) & (y[m] <= hi)).mean()
    med_err = (sigmoid(qpred[0.5][m]) - R[m]) * 100.0
    cov_rows.append({"method": "lgb-quantile-80%",
                     "coverage_uncens": float(cov),
                     "median_rmse_pct": float(np.sqrt(np.mean(med_err**2))),
                     "mean_width_logit": float(np.mean(hi - lo))})
    print("quantile PI coverage (target 0.80): %.3f, width %.2f logit"
          % (cov, np.mean(hi - lo)))

    # ---- cross-conformal on grouped folds (honest) ---------------------------
    # OOF point predictions (model never saw the point's study); for each
    # test fold, calibrate q on OOF residuals of all OTHER folds, then
    # predict the test fold with a model fit on all non-test studies.
    def fit_lgb(Xa, ya):
        return lgb.train({"objective": "regression", "learning_rate": 0.05,
                          "num_leaves": 31, "min_data_in_leaf": 20,
                          "seed": SEED, "verbosity": -1},
                         lgb.Dataset(Xa, label=ya), num_boost_round=500)

    oof = np.empty(len(y))
    for tri, vai in folds:
        oof[vai] = fit_lgb(X[tri], y[tri]).predict(X[vai])
    cover = []
    for tri, vai in folds:
        cal = ~np.isin(np.arange(len(y)), vai) & ~cens
        q = np.quantile(np.abs(oof[cal] - y[cal]), 0.8)
        pv = fit_lgb(X[tri], y[tri]).predict(X[vai])
        mv = ~cens[vai]
        cover.append(float((np.abs(pv[mv] - y[vai][mv]) <= q).mean()))
    cov_rows.append({"method": "cross-conformal-80% (lgb point)",
                     "coverage_uncens": float(np.mean(cover)),
                     "median_rmse_pct": np.nan,
                     "mean_width_logit": np.nan})
    pd.DataFrame(cov_rows).to_csv(
        "results/tables/uncertainty_coverage.csv", index=False)
    print("cross-conformal coverage (target 0.80): %.3f" % np.mean(cover))

    # ---- applicability domain ----------------------------------------------
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xs = sc.fit_transform(imp.fit_transform(X))
    nn_dist = np.empty(len(y))
    fold_pred = np.empty(len(y))
    with open("results/tables/grouped_best_params.json") as fh:
        gb = json.load(fh)
    ap = {"objective": "survival:aft", "seed": SEED,
          "eta": gb["eta"],
          "aft_loss_distribution": gb["aft_loss_distribution"],
          "aft_loss_distribution_scale": gb["aft_loss_distribution_scale"],
          "max_depth": gb["max_depth"], "subsample": gb["subsample"],
          "colsample_bytree": gb["colsample_bytree"],
          "min_child_weight": gb["min_child_weight"],
          "lambda": gb["lambda"], "alpha": gb["alpha"]}
    for tri, vai in folds:
        nn = NearestNeighbors(n_neighbors=5).fit(Xs[tri])
        nn_dist[vai] = nn.kneighbors(Xs[vai])[0].mean(axis=1)
        bst = xgb.train(ap, xgb.DMatrix(
            X[tri], label_lower_bound=y[tri],
            label_upper_bound=np.where(cens[tri], INF, y[tri])),
            num_boost_round=gb["n_estimators"])
        fold_pred[vai] = bst.predict(xgb.DMatrix(X[vai]))
    dec = pd.qcut(nn_dist, 10, labels=False, duplicates="drop")
    ad = pd.DataFrame({"decile": dec, "abs_err_pct": np.abs(
        sigmoid(fold_pred) - R) * 100.0, "cens": cens})
    ad = ad[~cens].groupby("decile")["abs_err_pct"].agg(["mean", "count"])
    ad.to_csv("results/tables/ad_deciles.csv")
    print("\nMAE%% by NN-distance decile (uncens):\n", ad.to_string())

    ios = raw["ios_mM"].to_numpy(dtype=float)
    doc = frame["doc_mg/L"].to_numpy(dtype=float)
    regions = {
        "ultrashort (C<=3)": carbons <= 3,
        "real water feed": feed_type == "Raw",
        "high DOC (>5 mg/L)": np.nan_to_num(doc) > 5,
        "salty (ios>20 mM)": np.nan_to_num(ios) > 20,
        "high DOC + salty": (np.nan_to_num(doc) > 5)
        & (np.nan_to_num(ios) > 20),
    }
    reg_rows = []
    for name, mask in regions.items():
        mu = mask & ~cens
        reg_rows.append({
            "region": name, "n": int(mask.sum()),
            "n_uncens": int(mu.sum()),
            "rmse_pct": float(np.sqrt(np.mean(
                ((sigmoid(fold_pred[mu]) - R[mu]) * 100.0) ** 2)))
            if mu.sum() > 2 else np.nan,
            "overall_rmse_pct": float(np.sqrt(np.mean(
                ((sigmoid(fold_pred[~cens]) - R[~cens]) * 100.0) ** 2)))})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv("results/tables/ad_regions.csv", index=False)
    print("\n", reg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ad.index, ad["mean"], "o-")
    ax.set_xlabel("5-NN distance decile (0=in-domain)")
    ax.set_ylabel("MAE (% rejection, uncens)")
    ax.set_title("Error vs applicability-domain distance")
    fig.tight_layout()
    fig.savefig("results/figures/ad_error.png", dpi=150)

    # ---- honest-model SHAP + PD --------------------------------------------
    bst_full = xgb.train(
        ap, xgb.DMatrix(X, label_lower_bound=y,
                        label_upper_bound=np.where(cens, INF, y)),
        num_boost_round=gb["n_estimators"])
    bst_full.save_model("results/models/xgb_aft_grouped.json")
    d_all = xgb.DMatrix(X)
    sv = bst_full.predict(d_all, pred_contribs=True)[:, :len(FEATURES_13)]
    himp = pd.DataFrame({"feature": FEATURES_13,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
    himp = himp.sort_values("mean_abs_shap", ascending=False)
    himp["share"] = himp["mean_abs_shap"] / himp["mean_abs_shap"].sum()
    himp.to_csv("results/tables/honest_shap_importance.csv", index=False)
    print("\nHonest-model SHAP:\n", himp.to_string(index=False))
    print("honest volume+pwp share: %.3f" % himp[
        himp["feature"].isin(["volume_Å3", "pwp_LMH/bar"])]["share"].sum())

    rng = np.random.default_rng(SEED)
    base = X[rng.choice(len(X), size=70, replace=False)]
    pd_rows = []
    for feat in ["pwp_LMH/bar", "ON", "pH", "volume_Å3", "mon_cat_mM",
                 "di_cat_mM", "doc_mg/L", "pfas_feed_conc_µg/L"]:
        j = FEATURES_13.index(feat)
        v = X[:, j]
        v = v[~np.isnan(v)]
        g = np.unique(np.quantile(v, np.linspace(0, 1, 26)))
        cur = np.empty((70, len(g)))
        Xi = base.copy()
        for k, val in enumerate(g):
            Xi[:, j] = val
            cur[:, k] = sigmoid(bst_full.predict(xgb.DMatrix(Xi)))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i in range(70):
            ax.plot(g, cur[i], c="gray", lw=0.6, alpha=0.5)
        ax.plot(g, cur.mean(axis=0), c="red", lw=2)
        ax.set_xlabel(feat)
        ax.set_ylabel("Predicted rejection")
        ax.set_title(f"Honest model PD: {feat}")
        if feat in ("mon_cat_mM", "pfas_feed_conc_µg/L"):
            ax.set_xscale("symlog", linthresh=1.0)
        fig.tight_layout()
        fig.savefig(f"results/figures/honest_pd_{feat.replace('/', '_')}.png",
                    dpi=150)
        plt.close(fig)
        pdm = cur.mean(axis=0)
        dd = np.diff(pdm)
        pd_rows.append({"feature": feat, "pd_min": float(pdm.min()),
                        "pd_max": float(pdm.max()),
                        "pd_range": float(pdm.max() - pdm.min()),
                        "pd_net": float(pdm[-1] - pdm[0]),
                        "sign_changes": int(
                            ((dd[:-1] > 0) != (dd[1:] > 0)).sum())})
    pd.DataFrame(pd_rows).to_csv(
        "results/tables/honest_pd_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
