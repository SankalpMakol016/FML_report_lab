"""Round-2 A — Extra features under grouped CV (10 seeds, paired CIs).

Variants (XGBoost AFT, grouped-retuned hyperparameters, identical folds):
  base13      13 paper features (sanity: reproduces Part-3 baseline)
  base13+ind  + missing-indicator flags
  allextra    + zp_mV, wca, tmp_bar (native NaN) + one-hot
                pfas_class/mode/system + missing indicators
Protocol: grouped 5-fold by ref_id x 10 seeds (common2); pooled OOF
metrics per seed; paired t 95% CI of (variant - base13) RMSE.
Second analysis: zp-observed subset only (paired folds rebuilt on subset).
Third: paper-style random stratified-cluster CV x 10 seeds for
{base13, allextra} (contrast: leakage inflates both).

Outputs: results/tables/A_features_oof.csv (per-seed metrics),
  A_features_gain.csv (paired CIs), A_zp_subset.csv,
  A_random_contrast.csv, results/figures/A_rmse.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, fit_aft, build_matrix, encode_apply,
                     SEEDS, FEATURES_13)
from sklearn.model_selection import StratifiedKFold


def run_variant(Xdf, cat_levels, y, cens, R, groups, seeds, tag, out):
    for s in seeds:
        pred = np.empty(len(y))
        for tri, vai in grouped_folds(groups, s):
            Xtr, Xva = encode_apply(Xdf.iloc[tri], Xdf.iloc[vai],
                                    cat_levels, indicators=("ind" in tag
                                                             or "extra" in tag))
            bst = fit_aft(Xtr, y[tri], cens[tri])
            pred[vai] = bst.predict(xgb.DMatrix(Xva))
        m = oof_metrics(y, cens, R, pred)
        m.update({"variant": tag, "seed": s})
        out.append(m)
    return out


def main():
    frame, raw = load_frame()
    y, cens, R, groups = base_arrays(frame)
    print("extra-col missing (all rows):")
    for c in ["zp_mV", "wca_º", "tmp_bar", "pfas_class", "mode", "system"]:
        print(f"  {c}: {raw[c].isna().mean():.3f}")
    print("mode values:", raw["mode"].dropna().unique().tolist())
    print("system values:", raw["system"].dropna().unique().tolist())
    print("pfas_class values:", raw["pfas_class"].dropna().unique().tolist())

    df_all, levs_all = build_matrix(raw, frame, extra=True)
    df_base, _ = build_matrix(raw, frame, extra=False)

    rows = []
    rows = run_variant(df_base, {}, y, cens, R, groups, SEEDS, "base13",
                       rows)
    rows = run_variant(df_base, {}, y, cens, R, groups, SEEDS,
                       "base13+ind", rows)
    rows = run_variant(df_all, levs_all, y, cens, R, groups, SEEDS,
                       "allextra", rows)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/A_features_oof.csv", index=False)

    gains = []
    b = oof[oof["variant"] == "base13"].set_index("seed")
    for v in ["base13+ind", "allextra"]:
        d = oof[oof["variant"] == v].set_index("seed")
        for metric in ["rmse", "mae", "cindex", "r2"]:
            ci = paired_ci(d[metric] - b[metric])
            gains.append({"variant": v, "metric": metric,
                          "base_mean": float(b[metric].mean()),
                          "var_mean": float(d[metric].mean()),
                          "gain_mean": ci["mean"], "ci_lo": ci["lo"],
                          "ci_hi": ci["hi"],
                          "significant": ci["significant"]})
    gain = pd.DataFrame(gains)
    gain.to_csv("results/tables/A_features_gain.csv", index=False)
    print(gain[gain["metric"] == "rmse"].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    piv = oof.pivot(index="seed", columns="variant", values="rmse")
    piv = piv[["base13", "base13+ind", "allextra"]]
    ax.boxplot([piv[c] for c in piv.columns], labels=piv.columns)
    ax.set_ylabel("Grouped-CV RMSE (% rejection)")
    ax.set_title("Extra features under grouped CV (10 seeds)")
    fig.tight_layout()
    fig.savefig("results/figures/A_rmse.png", dpi=150)

    # ---- zp-observed subset -------------------------------------------------
    zp = raw["zp_mV"].notna().to_numpy()
    print(f"\nzp-observed subset: {zp.sum()} rows, "
          f"{len(np.unique(groups[zp]))} studies")
    fs, ys, cs, Rs, gs = (df_all[zp].reset_index(drop=True), y[zp],
                          cens[zp], R[zp], groups[zp])
    fb = df_base[zp].reset_index(drop=True)
    sub = []
    for tag, Xd, lv in [("base13", fb, {}), ("allextra", fs, levs_all)]:
        for s in SEEDS:
            pred = np.empty(len(ys))
            for tri, vai in grouped_folds(gs, s):
                Xtr, Xva = encode_apply(Xd.iloc[tri], Xd.iloc[vai], lv,
                                        indicators=(tag == "allextra"))
                pred[vai] = fit_aft(Xtr, ys[tri], cs[tri]).predict(
                    xgb.DMatrix(Xva))
            m = oof_metrics(ys, cs, Rs, pred)
            m.update({"variant": tag, "seed": s})
            sub.append(m)
    sub = pd.DataFrame(sub)
    sub.to_csv("results/tables/A_zp_subset.csv", index=False)
    bs = sub[sub["variant"] == "base13"].set_index("seed")
    ds = sub[sub["variant"] == "allextra"].set_index("seed")
    print("zp-subset paired RMSE gain (allextra-base13):",
          paired_ci(ds["rmse"] - bs["rmse"]))

    # ---- random-CV contrast ---------------------------------------------------
    clu = frame["cluster"].to_numpy()
    rnd = []
    for tag, Xd, lv in [("base13", df_base, {}),
                        ("allextra", df_all, levs_all)]:
        for s in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=s)
            pred = np.empty(len(y))
            for tri, vai in skf.split(Xd, clu):
                Xtr, Xva = encode_apply(Xd.iloc[tri], Xd.iloc[vai], lv,
                                        indicators=(tag == "allextra"))
                pred[vai] = fit_aft(Xtr, y[tri], cens[tri]).predict(
                    xgb.DMatrix(Xva))
            m = oof_metrics(y, cens, R, pred)
            m.update({"variant": tag, "seed": s})
            rnd.append(m)
    rnd = pd.DataFrame(rnd)
    rnd.to_csv("results/tables/A_random_contrast.csv", index=False)
    print("\nrandom-CV RMSE means:")
    print(rnd.groupby("variant")["rmse"].agg(["mean", "std"]).to_string())
    print("Done.")


if __name__ == "__main__":
    main()
