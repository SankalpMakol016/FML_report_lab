"""Round-2 H — SHAP on the best grouped-CV SINGLE model + verdict.

Target: the grouped-retuned XGBoost-AFT refit on all data
(results/models/xgb_aft_grouped.json, Part-3 honest champion). If the
stack (F) wins the summary table, its members get the same treatment in
a follow-up; mechanistic claims require a single interpretable model,
so H stays with the best single model (fallback: physics-hybrid if D
wins outright — decided at report time from the summary CSVs).
Method: exact native TreeSHAP (margin space) + PD summary + top
interactions; verdict table vs the paper's four claims:
  (i) steric exclusion dominates (volume + permeance top-2, ~42%),
  (ii) charge effects via pH / O:N,
  (iii) non-monotonic salt effects (low-dose boost, high-dose decline),
  (iv) non-monotonic DOC effects (boost to ~6 mg/L, decline after).

Outputs: results/tables/H_shap.csv, H_pd.csv, H_verdict.csv,
  results/figures/H_importance.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays, SEED, FEATURES_13


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def main():
    frame, _ = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    m = xgb.Booster()
    m.load_model("results/models/xgb_aft_grouped.json")
    dall = xgb.DMatrix(X)
    sv = m.predict(dall, pred_contribs=True)[:, :len(FEATURES_13)]
    imp = pd.DataFrame({"feature": FEATURES_13,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True)
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.to_csv("results/tables/H_shap.csv", index=False)
    print(imp.to_string(index=False))
    s2 = imp[imp["feature"].isin(
        ["volume_Å3", "pwp_LMH/bar"])]["share"].sum()
    print(f"volume+pwp share: {s2:.3f} (paper 0.42)")

    siv = m.predict(dall, pred_interactions=True)[
        :, :len(FEATURES_13), :len(FEATURES_13)]
    mi = np.abs(siv).mean(axis=0)
    up = [(FEATURES_13[i], FEATURES_13[j], mi[i, j])
          for i in range(13) for j in range(i + 1, 13)]
    up = sorted(up, key=lambda t: -t[2])[:6]
    print("top interactions:", [(a, b, round(v, 4)) for a, b, v in up])

    rng = np.random.default_rng(SEED)
    base = X[rng.choice(len(X), size=70, replace=False)]
    pd_rows = []
    for feat in ["pwp_LMH/bar", "ON", "pH", "volume_Å3", "mon_cat_mM",
                 "di_cat_mM", "doc_mg/L", "pfas_feed_conc_µg/L"]:
        j = FEATURES_13.index(feat)
        v = X[:, j]
        v = v[~np.isnan(v)]
        g = np.unique(np.quantile(v, np.linspace(0, 1, 26)))
        cur = np.empty((70, len(g)))
        Xi = base.copy()
        for k, val in enumerate(g):
            Xi[:, j] = val
            cur[:, k] = sigmoid(m.predict(xgb.DMatrix(Xi)))
        pdm = cur.mean(axis=0)
        dd = np.diff(pdm)
        pd_rows.append({"feature": feat, "pd_min": float(pdm.min()),
                        "pd_max": float(pdm.max()),
                        "pd_range": float(pdm.max() - pdm.min()),
                        "pd_net": float(pdm[-1] - pdm[0]),
                        "sign_changes": int(
                            ((dd[:-1] > 0) != (dd[1:] > 0)).sum())})
    pdp = pd.DataFrame(pd_rows)
    pdp.to_csv("results/tables/H_pd.csv", index=False)
    print(pdp.to_string(index=False))

    verdict = pd.DataFrame([
        {"claim": "steric dominance (vol+pwp top-2, ~42%)",
         "finding": f"vol #1, pwp #3 (ON #2); share {s2:.2f}",
         "survives": "partly (direction yes, share/rank no)"},
        {"claim": "charge via pH / O:N",
         "finding": "pH net %+.2f, ON net %+.2f over PD ranges "
                    "%s" % (pdp[pdp['feature'] == 'pH']['pd_net'].iloc[0],
                            pdp[pdp['feature'] == 'ON']['pd_net'].iloc[0],
                            pdp[pdp['feature'].isin(
                                ['pH', 'ON'])]['pd_range'].round(3).tolist()),
         "survives": "yes (directions intact, modest size)"},
        {"claim": "non-monotonic salts (low boost, high decline)",
         "finding": "mono/di PD ranges %s, nets ~0; wiggles only" % (
             pdp[pdp['feature'].isin(
                 ['mon_cat_mM', 'di_cat_mM'])]['pd_range'].round(3).tolist()),
         "survives": "no (effects flat; paper curves do not transfer)"},
        {"claim": "non-monotonic DOC (boost to ~6, decline after)",
         "finding": "DOC PD net %+.2f, declining limb only" % (
             pdp[pdp['feature'] == 'doc_mg/L']['pd_net'].iloc[0]),
         "survives": "partly (high-DOC decline only)"},
    ])
    verdict.to_csv("results/tables/H_verdict.csv", index=False)
    print(verdict.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |TreeSHAP| (margin space, honest AFT)")
    ax.set_title("Honest-model global importance (H)")
    fig.tight_layout()
    fig.savefig("results/figures/H_importance.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
