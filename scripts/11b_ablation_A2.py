"""Round-2 A2 — Ablation of the zp-subset gain (which extra feature helps?).

Same 10-seed grouped protocol, zp-observed subset only, XGBoost AFT:
  base13 | +zp | +wca+tmp | +cats (mode/system/pfas_class one-hot) | allextra
(missing indicators on wherever they apply, as in script 11).

Outputs: results/tables/A2_ablation_oof.csv, A2_ablation_gain.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, fit_aft, build_matrix, encode_apply, SEEDS)


def run(Xdf, levs, ind, ys, cs, Rs, gs, tag, out):
    for s in SEEDS:
        pred = np.empty(len(ys))
        for tri, vai in grouped_folds(gs, s):
            Xtr, Xva = encode_apply(Xdf.iloc[tri], Xdf.iloc[vai], levs,
                                    indicators=ind)
            pred[vai] = fit_aft(Xtr, ys[tri], cs[tri]).predict(
                xgb.DMatrix(Xva))
        m = oof_metrics(ys, cs, Rs, pred)
        m.update({"variant": tag, "seed": s})
        out.append(m)
    return out


def sub(df, cols_extra=None, cats=None):
    use = list(df.columns)
    if cols_extra is not None:
        drop_num = {"zp_mV", "wca_º", "tmp_bar"} - set(cols_extra)
        use = [c for c in use if c not in drop_num]
    if cats is not None:
        use = [c for c in use if (not c.startswith("_cat_"))
               or c in cats]
    else:
        use = [c for c in use if not c.startswith("_cat_")]
    return df[use]


def main():
    frame, raw = load_frame()
    y, cens, R, groups = base_arrays(frame)
    zp = raw["zp_mV"].notna().to_numpy()
    print(f"zp subset: {zp.sum()} rows, {len(np.unique(groups[zp]))} studies")
    df_all, levs_all = build_matrix(raw, frame, extra=True)
    df_base, _ = build_matrix(raw, frame, extra=False)
    fb = df_base[zp].reset_index(drop=True)
    dall = df_all[zp].reset_index(drop=True)
    ys, cs, Rs, gs = y[zp], cens[zp], R[zp], groups[zp]
    variants = {
        "base13": (fb, {}, False),
        "+zp": (sub(dall, ["zp_mV"]), {}, True),
        "+wca+tmp": (sub(dall, ["wca_º", "tmp_bar"]), {}, True),
        "+cats": (sub(dall, [], ["_cat_system", "_cat_mode",
                                 "_cat_pfas"]),
                  {k: levs_all[k] for k in
                   ["_cat_system", "_cat_mode", "_cat_pfas"]}, False),
        "allextra": (dall, levs_all, True),
    }
    rows = []
    for tag, (Xd, lv, ind) in variants.items():
        rows = run(Xd, lv, ind, ys, cs, Rs, gs, tag, rows)
    oof = pd.DataFrame(rows)
    oof.to_csv("results/tables/A2_ablation_oof.csv", index=False)
    b = oof[oof["variant"] == "base13"].set_index("seed")
    gains = []
    for v in ["+zp", "+wca+tmp", "+cats", "allextra"]:
        d = oof[oof["variant"] == v].set_index("seed")
        for metric in ["rmse", "mae", "cindex", "r2"]:
            ci = paired_ci(d[metric] - b[metric])
            gains.append({"variant": v, "metric": metric, **ci,
                          "var_mean": float(d[metric].mean())})
    gain = pd.DataFrame(gains)
    gain.to_csv("results/tables/A2_ablation_gain.csv", index=False)
    print(gain[gain["metric"].isin(["rmse", "cindex"])].to_string(
        index=False))
    print("Done.")


if __name__ == "__main__":
    main()
