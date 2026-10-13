"""Round-3 task 2 — Holm correction across the main §5 table (RMSE, primary).

Family (15 primary RMSE gain tests, each mapped to a §5 row):
  A_all_allextra      A_features_oof allextra-base (n=10 seeds)
  A2_allextra         A2_ablation_oof allextra-base on zp subset (n=10)
  A2_+zp / +wca+tmp / +cats  same file (n=10); table row = +wca+tmp
  C2_k1/k2/k3         honest EB vs k0, paired study-level (n=60/57/56)
  D_hybrid            D_hybrid_oof hybrid vs AFT-baseline (n=10)
  D2_physfeat         D2_physfeat_oof xgb13+physfeat vs xgb13 (n=10, as run)
  E_low / E_high      E_scope reg-all vs reg-hi-only, rmse_low/high (n=10)
  F_lgb / F_cat / F_stack  F_stack_oof vs Part-3 baseline A-base13 (n=10;
                      baseline from A_features_oof, same folds)

Raw two-sided paired t p-values via scipy.stats.ttest_rel; Holm step-down
adjusted p = max over k<=i of (m-k+1)*p_(k) (ordered), alpha=0.05.
"Survives" iff Holm-adjusted p < 0.05. Also reports unadjusted significance
(CI excludes 0) for comparison.

Outputs: results/tables/R3_holm.csv, console table, results/log_20holm.log.
No model fits (seconds).
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import sys
sys.path.insert(0, "scripts")
from common2 import paired_ci


def raw_p(diff):
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return np.nan
    return float(ttest_rel(d, np.zeros_like(d)).pvalue
                 if np.std(d, ddof=1) > 0 else 1.0)


def holm(pvals):
    """Holm adjusted p-values. Returns array aligned to input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj_val = (m - rank) * pvals[idx]
        running = max(running, min(adj_val, 1.0))
        adj[idx] = running
    return adj


def main():
    tests = []  # (name, diff array, n, mean, lo, hi, unadj_sig)

    def add(name, diff):
        d = np.asarray(diff, float)
        d = d[~np.isnan(d)]
        ci = paired_ci(d)
        tests.append({"test": name, "n": len(d), "mean": ci["mean"],
                      "lo": ci["lo"], "hi": ci["hi"],
                      "unadj_sig": ci["significant"],
                      "p_raw": raw_p(d)})

    # ---- A ----
    a = pd.read_csv("results/tables/A_features_oof.csv")
    ab = a[a["variant"] == "base13"].set_index("seed")["rmse"]
    ae = a[a["variant"] == "allextra"].set_index("seed")["rmse"]
    add("A_all_allextra_vs_base", ae - ab)

    # ---- A2 ablation (zp subset) ----
    a2 = pd.read_csv("results/tables/A2_ablation_oof.csv")
    b2 = a2[a2["variant"] == "base13"].set_index("seed")["rmse"]
    for v, nm in [("allextra", "A2_allextra_vs_base"),
                  ("+zp", "A2_+zp_vs_base"),
                  ("+wca+tmp", "A2_+wca+tmp_vs_base[TABLE]"),
                  ("+cats", "A2_+cats_vs_base")]:
        d = a2[a2["variant"] == v].set_index("seed")["rmse"]
        add(nm, d - b2)

    # ---- C2 honest (study-level) ----
    hon = pd.read_csv("results/tables/C2_honest.csv")
    few = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = few[few["k"] == 0].groupby("study")["rmse"].mean()
    aggh = hon.groupby(["study", "k"])["rmse"].mean().reset_index()
    for k in [1, 2, 3]:
        sub = aggh[aggh["k"] == k].set_index("study")["rmse"]
        common = sub.index.intersection(agg0.index)
        add(f"C2_k{k}_EB_vs_k0", sub.loc[common] - agg0.loc[common])

    # ---- D ----
    dh = pd.read_csv("results/tables/D_hybrid_oof.csv")
    db = dh[dh["variant"] == "AFT-baseline"].set_index("seed")["rmse"]
    dhy = dh[dh["variant"] == "hybrid"].set_index("seed")["rmse"]
    add("D_hybrid_vs_AFT", dhy - db)

    # ---- D2 (as run: vs xgb13) ----
    d2 = pd.read_csv("results/tables/D2_physfeat_oof.csv")
    b13 = d2[d2["variant"] == "xgb13"].set_index("seed")["rmse"]
    p17 = d2[d2["variant"] == "xgb13+physfeat"].set_index("seed")["rmse"]
    add("D2_physfeat_vs_xgb13", p17 - b13)

    # ---- E (scope: reg-all vs hi-only) ----
    e = pd.read_csv("results/tables/E_scope.csv")
    ea = e[e["variant"] == "reg-all"].set_index("seed")
    eh = e[e["variant"] == "reg-hi-only"].set_index("seed")
    add("E_low_reg-all_vs_hi-only", ea["rmse_low"] - eh["rmse_low"])
    add("E_high_reg-all_vs_hi-only", ea["rmse_hi"] - eh["rmse_hi"])

    # ---- F (vs Part-3 baseline, same folds) ----
    f = pd.read_csv("results/tables/F_stack_oof.csv")
    for v, nm in [("lgb-tuned", "F_lgb_vs_baseline"),
                  ("cat-tuned", "F_cat_vs_baseline"),
                  ("stack", "F_stack_vs_baseline")]:
        d = f[f["variant"] == v].set_index("seed")["rmse"]
        # align seeds with baseline ab (same SEEDS 0..9)
        add(nm, d - ab)

    out = pd.DataFrame(tests)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    out["holm_sig"] = out["p_holm"] < 0.05
    out["verdict_change"] = np.where(
        out["unadj_sig"] == out["holm_sig"], "same",
        np.where(out["holm_sig"], "becomes sig (impossible here)",
                 "LOSES significance under Holm"))
    out = out.sort_values("p_raw").reset_index(drop=True)
    out.to_csv("results/tables/R3_holm.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["test", "n", "mean", "lo", "hi", "unadj_sig",
               "p_raw", "p_holm", "holm_sig", "verdict_change"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nFamily m={len(out)}; {out['holm_sig'].sum()} survive Holm "
          f"at 0.05; {(out['unadj_sig'] != out['holm_sig']).sum()} change "
          f"verdict vs unadjusted.")
    print("Done.")


if __name__ == "__main__":
    main()
