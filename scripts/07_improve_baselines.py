"""Part 4a — Honest baselines, ALL under grouped (study) 5-fold CV.

Compares, on identical GroupKFold(ref_id) splits:
  * ridge (median-imputed + standardized; censoring ignored)
  * Tobit censored MLE (normal errors, right-censoring; imputed+scaled)
  * random forest (imputed; censoring ignored)
  * LightGBM (native NaN; censoring ignored)
  * XGBoost reg:squarederror (native NaN; censoring ignored)
  * XGBoost AFT with grouped-retuned hyperparameters (Part 3 champion)
  * XGBoost AFT shallow (depth<=3) + monotonic constraints
      (R non-decreasing in PFAS volume, non-increasing in permeance)
  * LightGBM + XGBoost-AFT on V2 engineered features:
      carbons, ionic strength, membrane class (RO/NF), lab-vs-commercial,
      log feed conc, PFAS deprotonated fraction f(pKa,pH),
      membrane-charge proxy O:N x f(pH), steric ratio volume/pwp

Metrics per fold (pooled predictions per model for R2/C-index):
nloglik where a likelihood exists; C-index (censoring-aware);
RMSE/MAE (%) + R2 (logit), uncensored only.

Outputs: results/tables/improvement_baselines.csv,
         results/figures/improve_rmse.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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


def dm_aft(X, y, cens):
    return xgb.DMatrix(X, label_lower_bound=y.astype(float),
                       label_upper_bound=np.where(cens, INF, y.astype(float)))


def tobit_nll(theta, X, y, cens):
    beta, b0, ls = theta[:-2], theta[-2], theta[-1]
    s = np.exp(ls)
    mu = X @ beta + b0
    z = (y - mu) / s
    nll_u = -norm.logpdf(z[~cens]) + ls
    nll_c = -np.log(norm.sf(z[cens]) + 1e-300)
    return float(nll_u.sum() + nll_c.sum())


def fit_tobit(Xtr, ytr, ctr):
    th0 = np.zeros(Xtr.shape[1] + 2)
    th0[-1] = 0.0
    res = minimize(tobit_nll, th0, args=(Xtr, ytr, ctr),
                   method="L-BFGS-B",
                   options={"maxiter": 2000})
    return res.x[:-2], res.x[-2], np.exp(res.x[-1])


def evaluate(name, preds,model_nlls, y, cens, R):
    m = ~cens
    err = (sigmoid(preds[m]) - R[m]) * 100.0
    return {"model": name,
            "nloglik": float(np.mean(model_nlls))
            if model_nlls is not None else np.nan,
            "cindex": float(concordance_index(y, preds, m)),
            "rmse_pct": float(np.sqrt(np.mean(err ** 2))),
            "mae_pct": float(np.mean(np.abs(err))),
            "r2_logit_uncens": float(r2_score(y[m], preds[m]))}


def main():
    frame = pd.read_csv("results/modeling_frame.csv")
    raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                        sheet_name="PFAS_literature_dataset_2b")
    raw = raw[raw["Rejection"] >= 0.5].reset_index(drop=True)
    X13 = frame[FEATURES_13].to_numpy()
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    R = frame["meta_Rejection"].to_numpy()
    groups = frame["meta_ref_id"].to_numpy()

    # ---- V2 engineered features -------------------------------------------
    f = frame.copy()
    f["carbons"] = raw["carbons"].to_numpy()
    f["ios_mM"] = raw["ios_mM"].to_numpy()
    f["is_RO"] = (raw["mem_class"] == "RO").astype(int).to_numpy()
    f["is_lab"] = (raw["mem_source"] == "Lab").astype(int).to_numpy()
    f["log_feed"] = np.log10(raw["pfas_feed_conc_µg/L"].clip(
        lower=1e-4).to_numpy())
    pka, ph = raw["pKa"].to_numpy(dtype=float), f["pH"].to_numpy(dtype=float)
    f["deprot"] = 1.0 / (1.0 + 10.0 ** (pka - ph))  # anionic fraction
    on = f["ON"].to_numpy(dtype=float)
    f["mem_charge"] = on * (1.0 / (1.0 + 10.0 ** (4.0 - ph)))  # (-) at high pH
    f["steric"] = f["volume_Å3"].to_numpy(dtype=float) / f[
        "pwp_LMH/bar"].to_numpy(dtype=float)
    V2 = FEATURES_13 + ["carbons", "ios_mM", "is_RO", "is_lab", "log_feed",
                        "deprot", "mem_charge", "steric"]
    Xv2 = f[V2].to_numpy()
    print("V2 missing frac:", pd.DataFrame(Xv2).isna().mean().mean())

    with open("results/tables/grouped_best_params.json") as fh:
        gb = json.load(fh)
    aft_params = {"objective": "survival:aft", "seed": SEED,
                  "eta": gb["eta"],
                  "aft_loss_distribution": gb["aft_loss_distribution"],
                  "aft_loss_distribution_scale":
                      gb["aft_loss_distribution_scale"],
                  "max_depth": gb["max_depth"],
                  "subsample": gb["subsample"],
                  "colsample_bytree": gb["colsample_bytree"],
                  "min_child_weight": gb["min_child_weight"],
                  "lambda": gb["lambda"], "alpha": gb["alpha"]}
    aft_n = gb["n_estimators"]

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X13, y, groups))
    results, resid = {}, {}

    def run(name, fit_predict_fn, needs_impute=False, Xuse=None):
        Xu = X13 if Xuse is None else Xuse
        preds = np.empty(len(y))
        nlls = []
        for tri, vai in folds:
            if needs_impute:
                imp = SimpleImputer(strategy="median")
                sc = StandardScaler()
                Ztr = sc.fit_transform(imp.fit_transform(Xu[tri]))
                Zva = sc.transform(imp.transform(Xu[vai]))
                p, nll = fit_predict_fn(Ztr, y[tri], cens[tri], Zva,
                                        y[vai], cens[vai])
            else:
                p, nll = fit_predict_fn(Xu[tri], y[tri], cens[tri],
                                        Xu[vai], y[vai], cens[vai])
            preds[vai] = p
            if nll is not None:
                nlls.append(nll)
        results[name] = evaluate(
            name, preds, nlls if nlls else None, y, cens, R)
        print(name, {k: round(v, 3) for k, v in results[name].items()
                     if k != "model"})
        return preds

    # ridge
    run("ridge", lambda Ztr, ytr, ctr, Zva, yva, cva: (
        Ridge(alpha=1.0, random_state=SEED).fit(Ztr, ytr).predict(Zva),
        None), needs_impute=True)
    # Tobit
    def tobit_fp(Ztr, ytr, ctr, Zva, yva, cva):
        beta, b0, s = fit_tobit(Ztr, ytr, ctr)
        z = (yva - (Zva @ beta + b0)) / s
        nll = float((-norm.logpdf(z[~cva]) + np.log(s)).mean()
                    if (~cva).sum() else np.nan)
        return Zva @ beta + b0, nll
    run("tobit", tobit_fp, needs_impute=True)
    # random forest
    run("rf", lambda Xtr, ytr, ctr, Xva, yva, cva: (
        RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                              n_jobs=-1, random_state=SEED
                              ).fit(np.nan_to_num(Xtr, nan=-9999.0),
                                    ytr).predict(
                                        np.nan_to_num(Xva, nan=-9999.0)),
        None))
    # lightGBM
    def lgb_fp(Xtr, ytr, ctr, Xva, yva, cva):
        m = lgb.train({"objective": "regression", "learning_rate": 0.05,
                       "num_leaves": 31, "min_data_in_leaf": 20,
                       "seed": SEED, "verbosity": -1},
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=500)
        return m.predict(Xva), None
    run("lightgbm", lgb_fp)
    # xgboost regression (ignore censoring)
    def xgbr_fp(Xtr, ytr, ctr, Xva, yva, cva):
        m = xgb.train({"objective": "reg:squarederror", "eta": 0.05,
                       "max_depth": 6, "subsample": 0.8,
                       "colsample_bytree": 0.8, "seed": SEED},
                      xgb.DMatrix(Xtr, label=ytr), num_boost_round=500)
        return m.predict(xgb.DMatrix(Xva)), None
    run("xgboost-reg", xgbr_fp)
    # xgboost AFT grouped-tuned (honest champion)
    def aft_fp(Xtr, ytr, ctr, Xva, yva, cva):
        m = xgb.train(aft_params, dm_aft(Xtr, ytr, ctr),
                      num_boost_round=aft_n)
        dva = dm_aft(Xva, yva, cva)
        return m.predict(dva), float(
            m.eval(dva).split("aft-nloglik:")[1])
    run("xgboost-AFT(grouped-tuned)", aft_fp)
    # AFT shallow + monotonic (volume>=0 idx3, pwp<=0 idx0)
    mono = ["0"] * 13
    mono[0], mono[3] = "-1", "1"
    mono = "(" + ",".join(mono) + ")"
    def aftm_fp(Xtr, ytr, ctr, Xva, yva, cva):
        pr = dict(aft_params)
        pr.update({"max_depth": 3, "monotone_constraints": mono,
                   "eta": 0.05})
        m = xgb.train(pr, dm_aft(Xtr, ytr, ctr), num_boost_round=600)
        dva = dm_aft(Xva, yva, cva)
        return m.predict(dva), float(
            m.eval(dva).split("aft-nloglik:")[1])
    run("xgboost-AFT-shallow-monotone", aftm_fp)
    # V2 features
    run("lightgbm-V2", lgb_fp, Xuse=Xv2)
    run("xgboost-AFT-V2", aft_fp, Xuse=Xv2)

    out = pd.DataFrame(list(results.values()))
    out.to_csv("results/tables/improvement_baselines.csv", index=False)
    print("\n", out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(out["model"], out["rmse_pct"])
    ax.set_xlabel("Grouped-CV RMSE (% rejection, uncensored)")
    ax.set_title("Honest baselines: 5-fold grouped CV by study")
    fig.tight_layout()
    fig.savefig("results/figures/improve_rmse.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
