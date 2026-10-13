"""Round-2 F2 — Stack-vs-baseline paired gains (saved artifact).

F_stack_gain.csv holds gains vs the jointly-tuned aft-tuned member; the
REPORT §5-F verdicts are stated vs the Part-3 grouped-retuned AFT baseline
(A_features_oof.csv, variant base13 — identical folds, paired by seed).
This script computes those paired 95% CIs (and the per-seed stack RMSE /
LGBM / CatBoost means already visible in F_stack_oof.csv) into
results/tables/F_stack_gain_vs_baseline.csv. No model fits. Seconds.
"""
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from common2 import paired_ci


def main():
    a = pd.read_csv("results/tables/A_features_oof.csv")
    a = a[a["variant"] == "base13"].set_index("seed")
    f = pd.read_csv("results/tables/F_stack_oof.csv")
    rows = []
    for v in ["aft-tuned", "lgb-tuned", "cat-tuned", "stack"]:
        d = f[f["variant"] == v].set_index("seed")
        for metric in ["rmse", "mae", "cindex", "r2"]:
            ci = paired_ci(d[metric] - a[metric])
            rows.append({"variant": v, "metric": metric,
                         "baseline_mean": float(a[metric].mean()),
                         "var_mean": float(d[metric].mean()), **ci})
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/F_stack_gain_vs_baseline.csv", index=False)
    print(out[out["metric"] == "rmse"].to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
