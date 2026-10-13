"""Verification: REPORT §5 numbers vs regenerated tables (no tuning).

Reads final output CSVs, recomputes the §5 headline figures with the same
paired-CI/Holm logic, and writes results/tables/VERIFY_report_vs_regenerated.csv
with columns: item, metric, report_value, regen_value, file_row,
abs_diff, flag_gt_0_05_RMSE_or_verdict_flip.
REPORT values are hardcoded constants transcribed from REPORT.md §5
(they are the claim under test, never edited to match).
"""
import re
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from common2 import paired_ci


def mean_ci_str(mean, lo, hi):
    return f"{mean:.2f} [{lo:.2f},{hi:.2f}]"


def num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


rows = []


def add(item, metric, report, regen, file_row, is_rmse=False,
        verdict_report=None, verdict_regen=None):
    try:
        diff = abs(float(regen) - float(report))
    except Exception:
        diff = np.nan
    flag = ""
    if is_rmse and np.isfinite(diff) and diff > 0.05:
        flag += ">0.05 RMSE; "
    if verdict_report is not None and verdict_regen is not None \
            and verdict_report != verdict_regen:
        flag += "VERDICT FLIP; "
    rows.append({"item": item, "metric": metric, "report_value": report,
                 "regen_value": regen, "file_row": file_row,
                 "abs_diff": diff, "flag": flag.strip()})


def main():
    # ---- Baseline (A_features_oof base13) ----
    a = pd.read_csv("results/tables/A_features_oof.csv")
    ab = a[a["variant"] == "base13"]
    add("baseline", "RMSE", 10.50, round(ab["rmse"].mean(), 2),
        "A_features_oof.csv:variant=base13 mean rmse", True)
    add("baseline", "MAE", 7.41, round(ab["mae"].mean(), 2),
        "A_features_oof.csv:variant=base13 mean mae", False)
    add("baseline", "C-index", 0.638, round(ab["cindex"].mean(), 3),
        "A_features_oof.csv:variant=base13 mean cindex", False)

    # ---- Baseline2 ----
    p = pd.read_csv("results/tables/R3_paperhypers_grouped.csv")
    pp = p[p["variant"] == "paper-hypers"].set_index("seed")
    pg = p[p["variant"] == "grouped-retuned"].set_index("seed")
    add("Baseline2", "RMSE", 12.01, round(pp["rmse"].mean(), 2),
        "R3_paperhypers_grouped.csv:variant=paper-hypers mean rmse", True)
    ci = paired_ci(pp["rmse"] - pg["rmse"])
    add("Baseline2", "RMSE gain", 1.51, round(ci["mean"], 2),
        "R3_paperhypers_grouped.csv:paired paper-grouped", True,
        "sig", "sig" if ci["significant"] else "NS")

    # ---- A ----
    g = pd.read_csv("results/tables/A_features_gain.csv")
    ga = g[(g["variant"] == "allextra") & (g["metric"] == "rmse")].iloc[0]
    add("A", "RMSE", 10.73, round(float(ga["var_mean"]), 2),
        "A_features_gain.csv:variant=allextra,metric=rmse var_mean", True)
    add("A", "RMSE gain", 0.22, round(float(ga["gain_mean"]), 2),
        "A_features_gain.csv:gain_mean", True,
        "NS", "NS" if not bool(ga["significant"]) else "sig")
    h = pd.read_csv("results/tables/R3_holm.csv")
    ha = h[h["test"] == "A_all_allextra_vs_base"].iloc[0]
    add("A", "Holm raw p", 0.13, float(ha["p_raw"]),
        "R3_holm.csv:test=A_all_allextra_vs_base p_raw", False)
    add("A", "Holm adj p", 0.66, float(ha["p_holm"]),
        "R3_holm.csv:p_holm", False,
        "NS", "NS" if not bool(ha["holm_sig"]) else "sig")
    b = pd.read_csv("results/tables/R3_bootstrap.csv")
    bsa = b[b["estimand"] == "A_all_gain_allextra-base"].iloc[0]
    bpa = b[b["estimand"] == "A_all_gain_allextra-base_POOLEDboot"].iloc[0]
    add("A", "boot study-avg", 0.30, round(float(bsa["boot_mean"]), 2),
        "R3_bootstrap.csv:estimand=A_all_gain_allextra-base boot_mean",
        True)
    add("A", "boot pooled", 0.25, round(float(bpa["boot_mean"]), 2),
        "R3_bootstrap.csv:POOLEDboot boot_mean", True)

    # ---- A2 (+wca+tmp table variant) ----
    g2 = pd.read_csv("results/tables/A2_ablation_gain.csv")
    gw = g2[(g2["variant"] == "+wca+tmp") & (g2["metric"] == "rmse")].iloc[0]
    add("A2", "RMSE", 9.17, round(float(gw["var_mean"]), 2),
        "A2_ablation_gain.csv:variant=+wca+tmp,metric=rmse var_mean", True)
    add("A2", "RMSE gain", -0.41, round(float(gw["mean"]), 2),
        "A2_ablation_gain.csv:mean", True,
        "sig", "sig" if bool(gw["significant"]) else "NS")
    hw = h[h["test"] == "A2_+wca+tmp_vs_base[TABLE]"].iloc[0]
    add("A2", "Holm raw p", 2.4e-05, float(hw["p_raw"]),
        "R3_holm.csv:test=A2_+wca+tmp p_raw", False)
    add("A2", "Holm adj p", 3.2e-04, float(hw["p_holm"]),
        "R3_holm.csv:p_holm", False,
        "sig", "sig" if bool(hw["holm_sig"]) else "NS")

    # ---- B ----
    bv = pd.read_csv("results/tables/B_variance.csv")
    add("B", "study ICC", 0.49, round(bv["study_icc"].mean(), 2),
        "B_variance.csv:study_icc mean", False)
    add("B", "mem ICC", 0.51, round(bv["mem_icc"].mean(), 2),
        "B_variance.csv:mem_icc mean", False)
    add("B", "PI90 cov", 0.930, round(bv["pi90_cov"].mean(), 3),
        "B_variance.csv:pi90_cov mean", False)
    add("B", "PI90 width", 5.69, round(bv["pi90_width"].mean(), 2),
        "B_variance.csv:pi90_width mean", False)

    # ---- C2 honest k=1/2/3 ----
    ch = pd.read_csv("results/tables/C2_honest_summary.csv")
    co = pd.read_csv("results/tables/C2_summary.csv")
    for k, rep, orig_rep in [(1, 9.36, 9.27), (2, 8.88, 8.88),
                             (3, 8.58, 8.61)]:
        rh = ch[(ch["k"] == k) & (ch["metric"] == "rmse")].iloc[0]
        add(f"C2 k={k}", "RMSE", rep, round(float(rh["honest_mean"]), 2),
            f"C2_honest_summary.csv:k={k},metric=rmse honest_mean", True)
        ro = co[(co["k"] == k) & (co["metric"] == "rmse")].iloc[0]
        add(f"C2 k={k}", "orig RMSE", orig_rep, round(float(ro["eb_mean"]), 2),
            f"C2_summary.csv:k={k} eb_mean", True)
        hk = h[h["test"] == f"C2_k{k}_EB_vs_k0"].iloc[0]
        rep_raw = {1: 0.05, 2: 0.0073, 3: 0.001}.get(k)
        rep_holm = {1: 0.2, 2: 0.059, 3: 0.004}.get(k)
        add(f"C2 k={k}", "Holm raw p", rep_raw, float(hk["p_raw"]),
            f"R3_holm.csv:test=C2_k{k} p_raw", False)
        add(f"C2 k={k}", "Holm adj p", rep_holm, float(hk["p_holm"]),
            f"R3_holm.csv:p_holm", False)

    # ---- D ----
    dh = pd.read_csv("results/tables/D_hybrid_gain.csv")
    dhr = dh[(dh["variant"] == "hybrid") & (dh["metric"] == "rmse")].iloc[0]
    add("D", "RMSE", 21.98, round(float(dhr["var_mean"]), 2),
        "D_hybrid_gain.csv:variant=hybrid,metric=rmse var_mean", True)
    add("D", "RMSE gain", 11.48, round(float(dhr["mean"]), 2),
        "D_hybrid_gain.csv:mean", True)
    hd = h[h["test"] == "D_hybrid_vs_AFT"].iloc[0]
    add("D", "Holm adj p", 3.3e-08, float(hd["p_holm"]),
        "R3_holm.csv:test=D_hybrid p_holm", False,
        "sig", "sig" if bool(hd["holm_sig"]) else "NS")

    # ---- D2 ----
    d2 = pd.read_csv("results/tables/D2_gain.csv")
    d2r = d2[d2["metric"] == "rmse"].iloc[0] if "variant" not in d2.columns \
        else d2[(d2["variant"] == "xgb13+physfeat") & (d2["metric"] == "rmse")].iloc[0] \
        if (d2["variant"] == "xgb13+physfeat").any() else d2[d2["metric"] == "rmse"].iloc[0]
    # D2_gain.csv columns: variant,metric,var_mean,mean,lo,hi,significant
    try:
        d2row = pd.read_csv("results/tables/D2_gain.csv")
        # find physfeat row
        cand = d2row[d2row["variant"].str.contains("physfeat", na=False)]
        r2 = cand[cand["metric"] == "rmse"].iloc[0] if len(cand) else d2row[d2row["metric"] == "rmse"].iloc[0]
        add("D2", "RMSE gain", 0.25, round(float(r2["mean"]), 2),
            "D2_gain.csv:physfeat rmse mean", True)
    except Exception as e:
        add("D2", "RMSE gain", 0.25, np.nan, f"D2_gain.csv:read fail {e}", True)
    hd2 = h[h["test"] == "D2_physfeat_vs_xgb13"].iloc[0]
    add("D2", "Holm raw p", 0.015, float(hd2["p_raw"]),
        "R3_holm.csv:test=D2 p_raw", False)
    add("D2", "Holm adj p", 0.090, float(hd2["p_holm"]),
        "R3_holm.csv:p_holm", False,
        "NS", "NS" if not bool(hd2["holm_sig"]) else "sig")

    # ---- E ----
    e = pd.read_csv("results/tables/E_scope.csv")
    ea = e[e["variant"] == "reg-all"].set_index("seed")
    eh = e[e["variant"] == "reg-hi-only"].set_index("seed")
    from common2 import paired_ci as _ci
    cl = _ci(ea["rmse_low"] - eh["rmse_low"])
    chh = _ci(ea["rmse_hi"] - eh["rmse_hi"])
    add("E", "low gain", -2.86, round(cl["mean"], 2),
        "E_scope.csv:paired reg-all-reg-hi rmse_low", True)
    add("E", "high gain", 0.63, round(chh["mean"], 2),
        "E_scope.csv:paired rmse_hi", True)
    add("E", "low-slice reg-all", 46.7, round(ea["rmse_low"].mean(), 1),
        "E_scope.csv:reg-all rmse_low mean", True)
    add("E", "low-slice hi-only", 49.6, round(eh["rmse_low"].mean(), 1),
        "E_scope.csv:reg-hi-only rmse_low mean", True)
    el = pd.read_csv("results/tables/E_lowprobe.csv").iloc[0]
    add("E", "Q1 RMSE", 45.7, round(float(el["rmse_low"]), 1),
        "E_lowprobe.csv:rmse_low", True)

    # ---- F ----
    fo = pd.read_csv("results/tables/F_stack_oof.csv")
    for v, rep in [("lgb-tuned", 10.46), ("cat-tuned", 10.51),
                   ("stack", 11.13)]:
        add("F", f"{v} RMSE", rep,
            round(float(fo[fo["variant"] == v]["rmse"].mean()), 2),
            f"F_stack_oof.csv:variant={v} mean rmse", True)
    fb = pd.read_csv("results/tables/F_stack_gain_vs_baseline.csv")
    for v in ["lgb-tuned", "cat-tuned", "stack"]:
        r = fb[(fb["variant"] == v) & (fb["metric"] == "rmse")].iloc[0]
        rep = {"lgb-tuned": -0.05, "cat-tuned": 0.01, "stack": 0.62}[v]
        add("F", f"{v} gain", rep, round(float(r["mean"]), 2),
            f"F_stack_gain_vs_baseline.csv:{v} rmse mean", True)

    # ---- G ----
    g = pd.read_csv("results/tables/G_coverage.csv")
    gb = g[g["model"] == "aft-best"]["width"].mean()
    gc = g[g["model"] == "aft-calibrated"]["width"].mean()
    add("G", "width best", 5.26, round(float(gb), 2),
        "G_coverage.csv:model=aft-best width mean", False)
    add("G", "width cal", 3.74, round(float(gc), 2),
        "G_coverage.csv:model=aft-calibrated width mean", False)
    add("G", "width diff", 1.52, round(float(gb - gc), 2),
        "G_coverage.csv:best-cal width diff", False)
    add("G", "cov best", 0.899,
        round(float(g[g["model"] == "aft-best"]["coverage"].mean()), 3),
        "G_coverage.csv:aft-best coverage mean", False)
    add("G", "cov cal", 0.900,
        round(float(g[g["model"] == "aft-calibrated"]["coverage"].mean()), 3),
        "G_coverage.csv:aft-calibrated coverage mean", False)
    bb = pd.read_csv("results/tables/R3_bootstrap.csv")
    for est, rep in [("G_width_best", 5.41), ("G_width_cal", 3.71),
                     ("G_width_diff_best-cal", 1.71)]:
        r = bb[bb["estimand"] == est].iloc[0]
        add("G-boot", est, rep, round(float(r["boot_mean"]), 2),
            f"R3_bootstrap.csv:estimand={est} boot_mean", False)

    # ---- C2 leakage audit ----
    au = pd.read_csv("results/tables/C2_leakage_audit.csv")
    add("C2-audit", "max|dw|", 0.046,
        round(float((au["w_train"] - au["w_global"]).abs().max()), 3),
        "C2_leakage_audit.csv:max|w_train-w_global|", False)

    out = pd.DataFrame(rows)
    out.to_csv("results/tables/VERIFY_report_vs_regenerated.csv", index=False)
    nflag = (out["flag"].str.len() > 0).sum()
    print(out.to_string(index=False))
    print(f"\n{len(out)} comparisons; {nflag} flagged.")
    print("Done.")


if __name__ == "__main__":
    main()
