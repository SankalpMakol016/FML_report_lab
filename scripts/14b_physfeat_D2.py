"""Round-2 D2 — Physics-as-features variant (follow-up to the failed
residual hybrid in script 14).

Same Ferry-Renkin + charge block, but used as 4 extra FEATURES
(R_phys, lambda, q_charge, scr_screen, with FIXED params
A=0.13/S=0.70/B=0/C=0.06 from the constrained fit in script-14 probing)
appended to the 13 ML features, XGBoost-regression, identical 10-seed
grouped folds. Distinguishes "physics useless" from "residual
formulation useless".

Outputs: results/tables/D2_physfeat_oof.csv, D2_gain.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, SEEDS, FEATURES_13, SEED)

A, S, B, C = 0.13, 0.70, 0.0, 0.06
XGBR = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED}


def phys_feats(pwp, V, pka, ph, ios, on, on_med):
    rs = S * 0.1 * (3.0 * np.asarray(V, float) / (4.0 * np.pi)) ** (1 / 3)
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
    q = dep * on * sig
    scr = np.exp(-C * np.sqrt(np.maximum(io, 0.0)))
    return np.column_stack([Rs, lam, q, scr])


def main():
    frame, raw = load_frame()
    X13 = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    PWP = frame["pwp_LMH/bar"].to_numpy(float)
    VOL = frame["volume_Å3"].to_numpy(float)
    PKA = frame["pKa"].to_numpy(float)
    PH = frame["pH"].to_numpy(float)
    IOS = raw["ios_mM"].to_numpy(float)
    ON = frame["ON"].to_numpy(float)
    on_med = float(np.nanmedian(ON))
    PF = phys_feats(PWP, VOL, PKA, PH, IOS, ON, on_med)
    X17 = np.column_stack([X13, PF])

    rows = []
    for s in SEEDS:
        p13 = np.empty(len(y))
        p17 = np.empty(len(y))
        for tri, vai in grouped_folds(groups, s):
            m13 = xgb.train(XGBR, xgb.DMatrix(X13[tri], label=y[tri]),
                            num_boost_round=500)
            p13[vai] = m13.predict(xgb.DMatrix(X13[vai]))
            m17 = xgb.train(XGBR, xgb.DMatrix(X17[tri], label=y[tri]),
                            num_boost_round=500)
            p17[vai] = m17.predict(xgb.DMatrix(X17[vai]))
        for tag, p in [("xgb13", p13), ("xgb13+physfeat", p17)]:
            mm = oof_metrics(y, cens, R, p)
            mm.update({"variant": tag, "seed": s})
            rows.append(mm)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/D2_physfeat_oof.csv", index=False)
    b = oof[oof["variant"] == "xgb13"].set_index("seed")
    d = oof[oof["variant"] == "xgb13+physfeat"].set_index("seed")
    gains = []
    for metric in ["rmse", "mae", "cindex", "r2"]:
        ci = paired_ci(d[metric] - b[metric])
        gains.append({"variant": "xgb13+physfeat", "metric": metric,
                      "var_mean": float(d[metric].mean()), **ci})
    gain = pd.DataFrame(gains)
    gain.to_csv("results/tables/D2_gain.csv", index=False)
    print(gain.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
