"""Round-2 G — 90% cross-conformal intervals for the best and calibrated
models, by distance-to-training bin.

Models (BEST env: 'stack' default | 'hybrid' | 'aft'):
  best: 5-seed grouped-OOF predictions. 'stack' reloads the tuned level-0
        configs from F_stack_params.json and refits per fold; 'hybrid'
        refits physics+residual per fold (D params refit); 'aft' refits the
        grouped-retuned AFT per fold.
  calibrated: EB few-shot (k=2, mean over 20 fresh repeats with SEED+1)
        applied to LOSO base predictions (C_loso_base.npy); no refits.
Cross-conformal (per seed, per test fold): q90 = 90th percentile of
|OOF residual| over uncensored points OUTSIDE the fold; interval =
test-fold prediction +/- q90 (logit scale). Coverage/width on uncensored
test-fold points. Distance bins: per-point mean 5-NN distance to points
from OTHER studies (seed-independent AD proxy), deciles.

Outputs: results/tables/G_coverage.csv, G_bybin.csv,
  results/figures/G_covwidth.png
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
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, fit_aft,
                     dm_aft, SEED, FEATURES_13)

BEST = os.environ.get("BEST", "stack")
SEEDS5 = list(range(5))
XGBR = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED}


def phys_R(P, pwp, V, pka, ph, ios, on, on_med):
    A, B, C = P
    rs = 0.1 * (3.0 * np.asarray(V, float) / (4.0 * np.pi)) ** (1.0 / 3.0)
    rp = A * np.sqrt(np.asarray(pwp, float))
    lam = np.clip(rs / np.maximum(rp, 1e-9), 0, 0.999)
    Rs = np.clip(1.0 - (2.0 * (1 - lam) ** 2 - (1 - lam) ** 4),
                 1e-6, 1 - 1e-6)
    ph = np.asarray(ph, float)
    sig = np.where(np.isnan(ph), 0.5, 1.0 / (1.0 + 10.0 ** (4.0 - ph)))
    dep = np.where(np.isnan(ph), 1.0,
                   1.0 / (1.0 + 10.0 ** (np.asarray(pka, float) - ph)))
    on = np.where(np.isnan(np.asarray(on, float)), on_med,
                  np.asarray(on, float))
    io = np.nan_to_num(np.asarray(ios, float), nan=0.0)
    scr = np.exp(-C * np.sqrt(np.maximum(io, 0.0)))
    lo = np.log(Rs / (1.0 - Rs)) + B * dep * on * sig * scr
    return 1.0 / (1.0 + np.exp(-lo))


def main():
    frame, raw = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    print("G best-model =", BEST)
    P = None
    if BEST == "stack":
        with open("results/tables/F_stack_params.json") as f:
            P = json.load(f)

    PWP = frame["pwp_LMH/bar"].to_numpy(float)
    VOL = frame["volume_Å3"].to_numpy(float)
    PKA = frame["pKa"].to_numpy(float)
    PH = frame["pH"].to_numpy(float)
    IOS = raw["ios_mM"].to_numpy(float)
    ON = frame["ON"].to_numpy(float)

    def fit_best(Xtr, ytr, ctr):
        if BEST == "aft":
            return ("aft", fit_aft(Xtr, ytr, ctr))
        pa = P  # stack level-0 configs from F
        ma = xgb.train({"objective": "survival:aft", "seed": SEED,
                        "aft_loss_distribution": "normal",
                        "aft_loss_distribution_scale": pa["a_scale"],
                        "eta": pa["a_eta"], "max_depth": pa["a_depth"],
                        "subsample": pa["a_sub"],
                        "colsample_bytree": pa["a_col"],
                        "min_child_weight": pa["a_mcw"],
                        "lambda": pa["a_lam"], "alpha": pa["a_alf"]},
                       dm_aft(Xtr, ytr, ctr), num_boost_round=pa["n_aft"])
        ml = lgb.train({"objective": "regression",
                        "learning_rate": pa["l_eta"],
                        "num_leaves": pa["l_leaves"],
                        "min_data_in_leaf": 20,
                        "feature_fraction": pa["l_feat"], "seed": SEED,
                        "verbosity": -1},
                       lgb.Dataset(Xtr, label=ytr),
                       num_boost_round=pa["n_lgb"])
        mc = cb.CatBoostRegressor(
            iterations=pa["n_cat"], depth=pa["c_depth"],
            learning_rate=pa["c_eta"], l2_leaf_reg=pa["c_l2"],
            random_seed=SEED, verbose=False,
            allow_writing_files=False).fit(Xtr, ytr)
        rg_holder = {}
        return (("stack", ma, ml, mc, rg_holder))

    def predict_best(fit, Xva):
        if fit[0] == "aft":
            return fit[1].predict(xgb.DMatrix(Xva))
        ma, ml, mc = fit[1], fit[2], fit[3]
        return (ma.predict(xgb.DMatrix(Xva)), ml.predict(Xva),
                mc.predict(Xva))

    # EB-calibrated OOF (k=2, mean over 20 repeats), computed once
    base = np.load("results/tables/C_loso_base.npy")
    bv = pd.read_csv("results/tables/B_variance.csv")
    tau2, sig2 = float(bv["study_between"].mean()), float(
        bv["study_within"].mean())
    lam_a, lam_b = sig2 / tau2, sig2 / 0.09
    rng = np.random.default_rng(SEED + 1)
    studies = np.unique(groups)
    cal_oof = np.zeros((len(y), 20))
    for rep in range(20):
        pc = np.empty(len(y))
        for s in studies:
            idx = np.where(groups == s)[0]
            unc = idx[~cens[idx]]
            pool = unc if len(unc) >= 2 else idx
            cal = rng.choice(pool, size=min(2, len(pool)), replace=False)
            yc, pp = y[cal], base[cal]
            Xc2 = np.column_stack([np.ones(len(cal)), pp])
            A = Xc2.T @ Xc2 + np.diag([lam_a, lam_b])
            sol = np.linalg.solve(A, Xc2.T @ (yc - pp))
            a, b = float(sol[0]), 1.0 + float(sol[1])
            tst = idx[~np.isin(idx, cal)]
            pc[tst] = a + b * base[tst]
            pc[cal] = base[cal]  # cal points: keep base (unscored anyway)
        cal_oof[:, rep] = pc
    cal_mean = cal_oof.mean(axis=1)

    # distance bins (other-study 5-NN)
    Xs = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X))
    dist = np.empty(len(y))
    for i in range(len(y)):
        d = np.sqrt(((Xs - Xs[i]) ** 2).sum(axis=1))
        d[groups == groups[i]] = np.inf
        dist[i] = np.sort(d)[:5].mean()
    dbin = pd.qcut(dist, 10, labels=False, duplicates="drop").astype(int)

    cov_rows, bin_rows = [], []
    for s in SEEDS5:
        folds = grouped_folds(groups, s)
        # OOF best-model preds
        pb = np.empty(len(y))
        if BEST == "hybrid":
            for tri, vai in folds:
                on_med = np.nanmedian(ON[tri])
                m = ~cens[tri]

                def resid(Pp):
                    Rp = phys_R(Pp, PWP[tri][m], VOL[tri][m],
                                PKA[tri][m], PH[tri][m], IOS[tri][m],
                                ON[tri][m], on_med)
                    return np.log(Rp / (1 - Rp)) - y[tri][m]

                sol = least_squares(
                    resid, [0.12, 1.0, 0.3],
                    bounds=([0.02, -5, 0.0], [1.0, 5, 5.0]), max_nfev=200)
                Pp = sol.x
                Rp_va = phys_R(Pp, PWP[vai], VOL[vai], PKA[vai], PH[vai],
                               IOS[vai], ON[vai], on_med)
                rtr = y[tri] - np.log(phys_R(
                    Pp, PWP[tri], VOL[tri], PKA[tri], PH[tri], IOS[tri],
                    ON[tri], on_med) / (1 - phys_R(
                        Pp, PWP[tri], VOL[tri], PKA[tri], PH[tri],
                        IOS[tri], ON[tri], on_med)))
                mr = xgb.train(XGBR, xgb.DMatrix(X[tri], label=rtr),
                               num_boost_round=500)
                pb[vai] = (np.log(Rp_va / (1 - Rp_va))
                           + mr.predict(xgb.DMatrix(X[vai])))
        else:
            # per-fold refit + nested ridge for stack
            Z = {}
            if BEST == "stack":
                Z = {v: np.empty(len(y)) for v in ["a", "l", "c"]}
            for tri, vai in folds:
                fit = fit_best(X[tri], y[tri], cens[tri])
                if BEST == "aft":
                    pb[vai] = predict_best(fit, X[vai])
                else:
                    a, l, c = predict_best(fit, X[vai])
                    Z["a"][vai], Z["l"][vai], Z["c"][vai] = a, l, c
            if BEST == "stack":
                Zmat = np.column_stack([Z["a"], Z["l"], Z["c"]])
                with open("results/tables/F_stack_params.json") as f:
                    ra = json.load(f)["ridge"]
                for tri, vai in folds:
                    rg = Ridge(alpha=ra).fit(Zmat[tri], y[tri])
                    pb[vai] = rg.predict(Zmat[vai])
        for tag, po in [("best", pb), ("calibrated", cal_mean)]:
            for tri, vai in folds:
                cal_mask = ~np.isin(np.arange(len(y)), vai) & ~cens
                q = np.quantile(np.abs(po[cal_mask] - y[cal_mask]), 0.90)
                mv = ~cens[vai]
                cov = ((np.abs(po[vai][mv] - y[vai][mv]) <= q)).mean()
                cov_rows.append({"model": f"{BEST}-{tag}", "seed": s,
                                "fold": int(np.where(
                                    [np.array_equal(v, vai)
                                     for _, v in folds])[0][0]),
                                "coverage": float(cov),
                                "width": float(2 * q)})
                for b in range(int(dbin.max()) + 1):
                    bb = vai[(dbin[vai] == b) & mv]
                    if len(bb) == 0:
                        continue
                    bin_rows.append(
                        {"model": f"{BEST}-{tag}", "seed": s, "bin": b,
                         "n": len(bb),
                         "coverage": float(
                             (np.abs(po[bb] - y[bb]) <= q).mean()),
                         "width": float(2 * q)})
    pd.DataFrame(cov_rows).to_csv("results/tables/G_coverage.csv",
                                  index=False)
    bint = pd.DataFrame(bin_rows)
    bint.to_csv("results/tables/G_bybin.csv", index=False)
    agg = bint.groupby(["model", "bin"])[["coverage", "width",
                                          "n"]].mean().reset_index()
    print(agg.to_string(index=False))
    print(pd.DataFrame(cov_rows).groupby("model")[
        ["coverage", "width"]].agg(["mean", "std"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, sub in agg.groupby("model"):
        axes[0].plot(sub["bin"], sub["coverage"], "o-", label=m)
    axes[0].axhline(0.90, c="r", ls="--", label="nominal 90%")
    axes[0].set_xlabel("distance-to-training decile")
    axes[0].set_ylabel("coverage")
    axes[0].legend(fontsize=8)
    axes[0].set_title("90% cross-conformal coverage by AD bin")
    for m, sub in agg.groupby("model"):
        axes[1].plot(sub["bin"], sub["width"], "o-", label=m)
    axes[1].set_xlabel("distance-to-training decile")
    axes[1].set_ylabel("mean width (logit)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("interval width by AD bin")
    fig.tight_layout()
    fig.savefig("results/figures/G_covwidth.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
