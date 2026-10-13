"""Round-2 D — Physics-hybrid: steric Ferry-Renkin + Donnan term, then
boosted trees on the residual.

Physics block (3 fitted global params A, B, C; NaN fallbacks documented):
  rs = 0.1*(3V/4pi)^(1/3)            PFAS Stokes radius [nm] from vdW vol
  rp = A*sqrt(pwp)                   pore-radius proxy [nm] from permeance
  lam = min(rs/rp, 0.999)
  R_s = 1 - (2(1-lam)^2 - (1-lam)^4)  Ferry-Renkin sieving rejection
  q = deprot(pKa,pH) * ON * sig(pH)   charge product, sig=1/(1+10^(4-pH))
                                      (negative limb only; pH NaN->0.5,
                                      deprot NaN->1, ios NaN->0,
                                      ON NaN->train median)
  scr = exp(-C*sqrt(ios))             Debye screening (ios = ios_mM)
  logit(R_phys) = logit(clip(R_s)) + B*q*scr ;  R_phys = sigmoid(.)
Params fit per train fold by least squares on uncensored logit(R).
Then XGBoost-regression (fixed config) on train residuals; final test
prediction = physics(test) + residual-pred(test).
Comparators on IDENTICAL folds: pure XGB-reg (same config) and the AFT
baseline. 10-seed grouped protocol, paired CIs.

Outputs: results/tables/D_hybrid_oof.csv, D_hybrid_gain.csv,
  D_phys_params.csv, results/figures/D_rmse.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.optimize import least_squares
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, fit_aft, SEEDS, FEATURES_13)

XGBR = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42}


def phys_R(P, pwp, V, pka, ph, ios, on, on_med):
    A, B, C = P
    rs = 0.1 * (3.0 * np.asarray(V, float) / (4.0 * np.pi)) ** (1.0 / 3.0)
    rp = A * np.sqrt(np.asarray(pwp, float))
    lam = np.clip(rs / np.maximum(rp, 1e-9), 0, 0.999)
    Rs = 1.0 - (2.0 * (1 - lam) ** 2 - (1 - lam) ** 4)
    Rs = np.clip(Rs, 1e-6, 1 - 1e-6)
    ph = np.asarray(ph, float)
    sig = np.where(np.isnan(ph), 0.5, 1.0 / (1.0 + 10.0 ** (4.0 - ph)))
    dep = np.where(np.isnan(ph), 1.0,
                   1.0 / (1.0 + 10.0 ** (np.asarray(pka, float) - ph)))
    on = np.where(np.isnan(np.asarray(on, float)), on_med,
                  np.asarray(on, float))
    io = np.nan_to_num(np.asarray(ios, float), nan=0.0)
    q = dep * on * sig
    scr = np.exp(-C * np.sqrt(np.maximum(io, 0.0)))
    lo = np.log(Rs / (1.0 - Rs)) + B * q * scr
    return 1.0 / (1.0 + np.exp(-lo))


def main():
    frame, raw = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    PWP = frame["pwp_LMH/bar"].to_numpy(float)
    VOL = frame["volume_Å3"].to_numpy(float)
    PKA = frame["pKa"].to_numpy(float)
    PH = frame["pH"].to_numpy(float)
    IOS = raw["ios_mM"].to_numpy(float)
    ON = frame["ON"].to_numpy(float)

    variants = ["hybrid", "xgb-reg", "AFT-baseline"]
    rows, parows = [], []
    for s in SEEDS:
        pred = {v: np.empty(len(y)) for v in variants}
        for tri, vai in grouped_folds(groups, s):
            on_med = np.nanmedian(ON[tri])
            m = ~cens[tri]

            def resid(P):
                Rp = phys_R(P, PWP[tri][m], VOL[tri][m], PKA[tri][m],
                            PH[tri][m], IOS[tri][m], ON[tri][m], on_med)
                return np.log(Rp / (1 - Rp)) - y[tri][m]

            sol = least_squares(resid, [0.12, 1.0, 0.3],
                                bounds=([0.02, -5, 0.0], [1.0, 5, 5.0]),
                                max_nfev=200)
            P = sol.x
            parows.append({"seed": s, "A": P[0], "B": P[1], "C": P[2],
                           "cost": float(sol.cost)})
            Rp_tr = phys_R(P, PWP[tri], VOL[tri], PKA[tri], PH[tri],
                           IOS[tri], ON[tri], on_med)
            Rp_va = phys_R(P, PWP[vai], VOL[vai], PKA[vai], PH[vai],
                           IOS[vai], ON[vai], on_med)
            rtr = y[tri] - np.log(Rp_tr / (1 - Rp_tr))
            mr = xgb.train(XGBR, xgb.DMatrix(X[tri], label=rtr),
                           num_boost_round=500)
            pred["hybrid"][vai] = (np.log(Rp_va / (1 - Rp_va))
                                   + mr.predict(xgb.DMatrix(X[vai])))
            mx = xgb.train(XGBR, xgb.DMatrix(X[tri], label=y[tri]),
                           num_boost_round=500)
            pred["xgb-reg"][vai] = mx.predict(xgb.DMatrix(X[vai]))
            ba = fit_aft(X[tri], y[tri], cens[tri])
            pred["AFT-baseline"][vai] = ba.predict(xgb.DMatrix(X[vai]))
        for v in variants:
            mm = oof_metrics(y, cens, R, pred[v])
            mm.update({"variant": v, "seed": s})
            rows.append(mm)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/D_hybrid_oof.csv", index=False)
    pd.DataFrame(parows).to_csv("results/tables/D_phys_params.csv",
                                index=False)
    print("physics params mean±std:")
    print(pd.DataFrame(parows)[["A", "B", "C"]].agg(["mean", "std"]
                                                    ).to_string())
    b = oof[oof["variant"] == "AFT-baseline"].set_index("seed")
    gains = []
    for v in ["hybrid", "xgb-reg"]:
        d = oof[oof["variant"] == v].set_index("seed")
        for metric in ["rmse", "mae", "cindex", "r2"]:
            ci = paired_ci(d[metric] - b[metric])
            gains.append({"variant": v, "metric": metric,
                          "var_mean": float(d[metric].mean()), **ci})
    gain = pd.DataFrame(gains)
    gain.to_csv("results/tables/D_hybrid_gain.csv", index=False)
    print(gain[gain["metric"] == "rmse"].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    piv = oof.pivot(index="seed", columns="variant", values="rmse")
    ax.boxplot([piv[c] for c in variants], labels=variants)
    ax.set_ylabel("Grouped-CV RMSE (% rejection)")
    ax.set_title("Physics-hybrid vs pure ML (10 seeds)")
    fig.tight_layout()
    fig.savefig("results/figures/D_rmse.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
