"""Round-2 C — One/few-shot study calibration (LOSO + offset).

For each study s (62 total): train the baseline XGBoost-AFT on all OTHER
studies, predict s. Then for k in {0,1,2,3}, 20 repeats: sample k
calibration points from s (uncensored only — a real lab would recalibrate
on a quantifiable measurement; fallback to any points if fewer than k
uncensored exist), estimate a study offset in logit space
  k=1: delta = y_cal - p_cal (scale fixed to 1);
  k>=2: least-squares y_cal ~ a + b*p_cal (offset a, scale b),
apply to the remaining points of s, and score ONLY non-calibration points
(RMSE/MAE %, uncensored). Simulates "measure one PFAS on a new membrane".

Outputs: results/tables/C_fewshot.csv (per study/k/repeat),
  C_fewshot_summary.csv (mean over studies+repeats, paired study-level
  CI of (k - 0) RMSE/MAE), results/figures/C_rmse_vs_k.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, paired_ci, fit_aft,
                     SEED, FEATURES_13)

KS = [0, 1, 2, 3]
REPEATS = 20


def main():
    frame, _ = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    studies = np.unique(groups)
    rng = np.random.default_rng(SEED)

    # LOSO base predictions (train on all-other-studies, honest config)
    base = np.empty(len(y))
    for s in studies:
        va = groups == s
        bst = fit_aft(X[~va], y[~va], cens[~va])
        base[va] = bst.predict(xgb.DMatrix(X[va]))
    np.save("results/tables/C_loso_base.npy", base)

    rows = []
    for s in studies:
        idx = np.where(groups == s)[0]
        unc = idx[~cens[idx]]
        pool = unc if len(unc) >= 1 else idx
        for k in KS:
            for rep in range(REPEATS if k else 1):
                if k == 0:
                    cal = np.array([], dtype=int)
                    a, b = 0.0, 1.0
                else:
                    cal = rng.choice(pool,
                                     size=min(k, len(pool)),
                                     replace=False)
                    yc, pc = y[cal], base[cal]
                    if k == 1 or len(cal) < 2 or np.std(pc) < 1e-9:
                        a, b = float(yc.mean() - pc.mean()), 1.0
                    else:
                        b, a = np.polyfit(pc, yc, 1)
                        b = float(np.clip(b, 0.2, 5.0))
                tst = idx[~np.isin(idx, cal)]
                tst = tst[~cens[tst]]  # score uncensored only
                if len(tst) == 0:
                    continue
                p = a + b * base[tst]
                err = (1.0 / (1.0 + np.exp(-p)) - R[tst]) * 100.0
                rows.append({"study": s, "k": k, "rep": rep,
                             "n_test": len(tst),
                             "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mae": float(np.mean(np.abs(err)))})
    det = pd.DataFrame(rows)
    det.to_csv("results/tables/C_fewshot.csv", index=False)

    # study-level means per k, paired (k - 0) over studies
    agg = det.groupby(["study", "k"])[["rmse", "mae"]].mean().reset_index()
    piv = agg.pivot(index="study", columns="k", values=["rmse", "mae"])
    summ = []
    for k in [1, 2, 3]:
        for metric in ["rmse", "mae"]:
            ci = paired_ci(piv[(metric, k)] - piv[(metric, 0)])
            summ.append({"k": k, "metric": metric,
                         "mean_k0": float(piv[(metric, 0)].mean()),
                         "mean_k": float(piv[(metric, k)].mean()),
                         "gain_mean": ci["mean"], "ci_lo": ci["lo"],
                         "ci_hi": ci["hi"],
                         "significant": ci["significant"]})
    summ = pd.DataFrame(summ)
    summ.to_csv("results/tables/C_fewshot_summary.csv", index=False)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    g = agg.groupby("k")[["rmse", "mae"]].agg(["mean", "std"])
    ax.errorbar(g.index, g[("rmse", "mean")], yerr=g[("rmse", "std")],
                fmt="o-", capsize=4, label="RMSE")
    ax.errorbar(g.index, g[("mae", "mean")], yerr=g[("mae", "std")],
                fmt="s-", capsize=4, label="MAE")
    ax.set_xlabel("# revealed calibration points (k)")
    ax.set_ylabel("% rejection (study-averaged)")
    ax.set_title("Few-shot study calibration: error vs k")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/C_rmse_vs_k.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
