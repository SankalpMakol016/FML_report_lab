"""Round-3 task 5 — Paper-style (random-tuned) hypers under grouped protocol.

Question: how much of the honest-performance gap is the TUNING protocol vs
the model? The §5 baseline is the grouped-retuned AFT champion
(grouped_best_params.json: depth 3, eta 0.015, 688 trees). Here we evaluate
the paper-style champion (best_params.json: depth 7, eta 0.14, 243 trees,
tuned under random stratified-cluster CV) under the IDENTICAL grouped
protocol (grouped 5-fold by ref_id x 10 seeds, same folds as every round-2
method via common2.grouped_folds), as an extra baseline row.

Outputs: results/tables/R3_paperhypers_grouped.csv (per-seed metrics for
both configs + paired CIs), console log.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, dm_aft, SEEDS, FEATURES_13, SEED)


def params_from(d):
    return {"objective": "survival:aft", "seed": SEED,
            "eta": d["eta"],
            "aft_loss_distribution": d["aft_loss_distribution"],
            "aft_loss_distribution_scale": d["aft_loss_distribution_scale"],
            "max_depth": d["max_depth"], "subsample": d["subsample"],
            "colsample_bytree": d["colsample_bytree"],
            "min_child_weight": d["min_child_weight"],
            "lambda": d["lambda"], "alpha": d["alpha"]}


def main():
    frame, _ = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    with open("results/tables/best_params.json") as f:
        bp = json.load(f)
    with open("results/tables/grouped_best_params.json") as f:
        gb = json.load(f)
    p_paper = params_from(bp)
    n_paper = bp["n_estimators"]
    p_group = params_from(gb)
    n_group = gb["n_estimators"]
    print(f"paper-style hypers: depth={bp['max_depth']} eta={bp['eta']:.4f} "
          f"scale={bp['aft_loss_distribution_scale']:.3f} n={n_paper}")
    print(f"grouped-retuned hypers: depth={gb['max_depth']} "
          f"eta={gb['eta']:.4f} scale={gb['aft_loss_distribution_scale']:.3f} "
          f"n={n_group}")

    rows = []
    for s in SEEDS:
        for tag, pr, nn in [("paper-hypers", p_paper, n_paper),
                            ("grouped-retuned", p_group, n_group)]:
            pred = np.empty(len(y))
            for tri, vai in grouped_folds(groups, s):
                bst = xgb.train(pr, dm_aft(X[tri], y[tri], cens[tri]),
                                num_boost_round=nn)
                pred[vai] = bst.predict(xgb.DMatrix(X[vai]))
            m = oof_metrics(y, cens, R, pred)
            m.update({"variant": tag, "seed": s})
            rows.append(m)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/R3_paperhypers_grouped.csv", index=False)
    a = oof[oof["variant"] == "grouped-retuned"].set_index("seed")
    b = oof[oof["variant"] == "paper-hypers"].set_index("seed")
    print("\nmeans:")
    print(oof.groupby("variant")[["rmse", "mae", "cindex", "r2"]].agg(
        ["mean", "std"]).round(3).to_string())
    for metric in ["rmse", "mae", "cindex", "r2"]:
        # positive = paper worse (for rmse/mae); report paper - grouped
        ci = paired_ci(b[metric] - a[metric])
        print(f"{metric}: paper-grouped gain {ci['mean']:+.3f} "
              f"[{ci['lo']:.3f},{ci['hi']:.3f}] sig={ci['significant']}")
    print("Done.")


if __name__ == "__main__":
    main()
