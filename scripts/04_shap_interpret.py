"""Part 2c — SHAP + ICE interpretation (paper Figs 3B, 4-6, S4-S7 analogs).

Loads the tuned XGBoost AFT model and:
  * global SHAP importance (mean |value|); share of PFAS-volume + permeance
    (paper: 42% of total main effect)
  * SHAP interaction heatmap (mean |interaction|, Fig 3B analog)
  * ICE + partial-dependence plots on the rejection scale for the 13
    features (Figs 4/5/S4-S6 analogs); centered ICE for salts/DOC/feed
    (change vs zero-additive baseline, as in paper Fig 5)
  * controlled-subset slices (Fig 6 analogs) and pure-water pH x permeance
    grids at two PFAS volumes (Fig 4D analog)

ICE protocol (paper): vary one feature over its observed range, hold the
other 12 at values from 70 randomly sampled training rows; average = PD.
Predictions mapped logit -> R via sigmoid for display.

Outputs: results/tables/shap_importance.csv, shap_interaction.csv,
  results/figures/shap_importance.png, shap_interaction_heatmap.png,
  ice_*.png, slice_*.png   (seeded; model file must exist)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs as _ensure_dirs
    _ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        _os.makedirs(_d, exist_ok=True)

SEED = 42
N_ICE = 70
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]
PRETTY = {
    "pwp_LMH/bar": "Water permeance (LMH/bar)",
    "ON": "O:N ratio", "pH": "Feed pH",
    "volume_Å3": "PFAS volume (Å³)", "dipole_debye": "Dipole (debye)",
    "pKa": "PFAS pKa", "temp_ºC": "Temperature (°C)",
    "flux_LMH": "Permeate flux (LMH)",
    "pfas_feed_conc_µg/L": "PFAS feed conc. (µg/L)",
    "mon_cat_mM": "Monovalent cations (mM)",
    "di_cat_mM": "Divalent cations (mM)",
    "tri_cat_mM": "Trivalent cations (mM)",
    "doc_mg/L": "DOC (mg/L)",
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def main():
    rng = np.random.default_rng(SEED)
    frame = pd.read_csv("results/modeling_frame.csv")
    X = frame[FEATURES_13].to_numpy()
    is_tr = (frame["split"] == "train").to_numpy()
    Xtr = X[is_tr]
    model = xgb.Booster()
    model.load_model("results/models/xgb_aft_replication.json")

    # ---- exact TreeSHAP via XGBoost natively --------------------------------
    # (shap.TreeExplainer cannot parse xgboost>=3 model files, and
    # Booster.predict lacks the old predcontrib kwarg; the current
    # snake_case API below is exact TreeSHAP in the model's margin space.
    # Verified: contributions sum to log(predict) to machine precision,
    # i.e. predict() = exp(margin); the AFT "location" is E[log(label)].)
    dtr = xgb.DMatrix(Xtr)
    sv = model.predict(dtr, pred_contribs=True)[:, :len(FEATURES_13)]
    assert np.allclose(np.exp(
        model.predict(dtr, pred_contribs=True).sum(axis=1)),
        model.predict(dtr)), "TreeSHAP additivity check failed"

    # ---- SHAP values + interactions (train) --------------------------------
    # (sv already computed above via native TreeSHAP)
    imp = pd.DataFrame({"feature": FEATURES_13,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True)
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.to_csv("results/tables/shap_importance.csv", index=False)
    share2 = imp[imp["feature"].isin(
        ["volume_Å3", "pwp_LMH/bar"])]["share"].sum()
    print(f"volume+pwp share of main effect: {share2:.3f} (paper: 0.42)")
    print(imp.to_string(index=False))

    siv = model.predict(dtr, pred_interactions=True)[
        :, :len(FEATURES_13), :len(FEATURES_13)]  # (n,13,13), margin space
    mi = np.abs(siv).mean(axis=0)
    mi_tbl = pd.DataFrame(mi, index=FEATURES_13, columns=FEATURES_13)
    mi_tbl.to_csv("results/tables/shap_interaction.csv")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(mi, cmap="hot", interpolation="nearest")
    ax.set_xticks(range(13), FEATURES_13, rotation=90, fontsize=7)
    ax.set_yticks(range(13), FEATURES_13, fontsize=7)
    ax.set_title("Mean |SHAP interaction| (train)")
    fig.tight_layout()
    fig.savefig("results/figures/shap_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mi, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(13), FEATURES_13, rotation=90, fontsize=8)
    ax.set_yticks(range(13), FEATURES_13, fontsize=8)
    ax.set_title("Mean |SHAP interaction value| heatmap (Fig 3B analog)")
    fig.colorbar(im, ax=ax, label="mean |interaction|")
    fig.tight_layout()
    fig.savefig("results/figures/shap_interaction_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- ICE / PD -----------------------------------------------------------
    base = Xtr[rng.choice(len(Xtr), size=N_ICE, replace=False)]
    grids = {}
    for j, f in enumerate(FEATURES_13):
        v = Xtr[:, j]
        v = v[~np.isnan(v)]
        if f in ("mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
                 "pfas_feed_conc_µg/L"):
            g = np.unique(np.quantile(v, np.linspace(0, 1, 26)))
            g[0] = 0.0 if f != "pfas_feed_conc_µg/L" else g[0]
        else:
            g = np.linspace(v.min(), v.max(), 26)
        grids[f] = np.unique(g)

    def ice(feature, centered=False):
        j = FEATURES_13.index(feature)
        g = grids[feature]
        curves = np.empty((N_ICE, len(g)))
        Xi = base.copy()
        for k, val in enumerate(g):
            Xi[:, j] = val
            curves[:, k] = sigmoid(model.predict(xgb.DMatrix(Xi)))
        if centered:
            curves = curves - curves[:, [0]]
        return g, curves

    CENTER = {"mon_cat_mM", "di_cat_mM", "doc_mg/L", "pfas_feed_conc_µg/L",
              "tri_cat_mM"}
    for f in FEATURES_13:
        g, curves = ice(f, centered=(f in CENTER))
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for i in range(N_ICE):
            ax.plot(g, curves[i], c="gray", lw=0.6, alpha=0.5)
        ax.plot(g, curves.mean(axis=0), c="red", lw=2, label="mean (PD)")
        v = Xtr[:, FEATURES_13.index(f)]
        ax.plot(v[~np.isnan(v)],
                np.full((~np.isnan(v)).sum(),
                        curves.min() - 0.02 * (curves.max() - curves.min() + 1e-9)),
                "|", c="blue", ms=6, alpha=0.5)
        ax.set_xlabel(PRETTY[f])
        ax.set_ylabel("Δ predicted rejection" if f in CENTER
                      else "Predicted rejection")
        ax.set_title(f"ICE + PD: {PRETTY[f]}")
        if f in ("mon_cat_mM", "pfas_feed_conc_µg/L"):
            ax.set_xscale("symlog", linthresh=1.0)
        fig.tight_layout()
        fig.savefig(f"results/figures/ice_{f.replace('/', '_')}.png",
                    dpi=150)
        plt.close(fig)

    # ---- controlled slices (Fig 6 analogs) ----------------------------------
    means = np.nanmean(Xtr.astype(float), axis=0)  # nan-aware column means

    def predict_row(**kw):
        row = means.copy()
        for f, val in kw.items():
            row[FEATURES_13.index(f)] = val
        return float(sigmoid(model.predict(
            xgb.DMatrix(row.reshape(1, -1))))[0])

    # 6A: mono cation sweep at two pH x two volumes, pwp=12
    g_mono = np.concatenate([[0], np.logspace(-2, 3, 40)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vol in zip(axes, [150, 300]):
        for ph in [3.0, 10.0]:
            ys = [predict_row(**{"pwp_LMH/bar": 12.0, "pH": ph,
                                 "volume_Å3": float(vol),
                                 "mon_cat_mM": float(m),
                                 "di_cat_mM": 0.0, "tri_cat_mM": 0.0,
                                 "doc_mg/L": 0.0}) for m in g_mono]
            base0 = ys[0]
            ax.plot(g_mono, np.array(ys) - base0, label=f"pH {ph}")
        ax.set_xscale("log")
        ax.set_xlabel("Monovalent cations (mM)")
        ax.set_title(f"PFAS vol {vol} Å³, pwp=12")
        ax.legend()
    axes[0].set_ylabel("Δ predicted rejection vs 0 mM")
    fig.suptitle("Mono-cation effect by PFAS size and pH (Fig 6A analog)")
    fig.tight_layout()
    fig.savefig("results/figures/slice_mono_ph_vol.png", dpi=150)
    plt.close(fig)

    # 6B: di cation sweep at two pwps x two volumes
    g_di = np.concatenate([[0], np.logspace(-2, 1.2, 40)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vol in zip(axes, [150, 300]):
        for pwp in [5.0, 12.0, 25.0]:
            ys = [predict_row(**{"pwp_LMH/bar": pwp, "pH": 7.0,
                                 "volume_Å3": float(vol),
                                 "di_cat_mM": float(m),
                                 "mon_cat_mM": 0.0, "tri_cat_mM": 0.0,
                                 "doc_mg/L": 0.0}) for m in g_di]
            ax.plot(g_di, np.array(ys) - ys[0], label=f"pwp {pwp}")
        ax.set_xscale("log")
        ax.set_xlabel("Divalent cations (mM)")
        ax.set_title(f"PFAS vol {vol} Å³, pH 7")
        ax.legend()
    axes[0].set_ylabel("Δ predicted rejection vs 0 mM")
    fig.suptitle("Di-cation effect by permeance and PFAS size "
                 "(Fig 6B analog)")
    fig.tight_layout()
    fig.savefig("results/figures/slice_di_pwp_vol.png", dpi=150)
    plt.close(fig)

    # 6C: DOC sweep, pure water vs 4 mM di, at two volumes, pwp=12
    g_doc = np.concatenate([[0], np.logspace(-2, 1.5, 40)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vol in zip(axes, [150, 300]):
        for di in [0.0, 4.0]:
            ys = [predict_row(**{"pwp_LMH/bar": 12.0, "pH": 7.0,
                                 "volume_Å3": float(vol),
                                 "doc_mg/L": float(o),
                                 "di_cat_mM": di, "mon_cat_mM": 0.0,
                                 "tri_cat_mM": 0.0}) for o in g_doc]
            ax.plot(g_doc, np.array(ys) - ys[0],
                    label=f"di {di} mM")
        ax.set_xscale("log")
        ax.set_xlabel("DOC (mg/L)")
        ax.set_title(f"PFAS vol {vol} Å³, pwp=12")
        ax.legend()
    axes[0].set_ylabel("Δ predicted rejection vs 0 DOC")
    fig.suptitle("DOC effect ± divalent cations (Fig 6C analog)")
    fig.tight_layout()
    fig.savefig("results/figures/slice_doc_di_vol.png", dpi=150)
    plt.close(fig)

    # 6D: feed-conc sweep in pure water, vol=250, pwp 5/12/25, pH 3/7
    g_fc = np.logspace(-3, 4, 40)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ph in zip(axes, [3.0, 7.0]):
        for pwp in [5.0, 12.0, 25.0]:
            ys = [predict_row(**{"pwp_LMH/bar": pwp, "pH": ph,
                                 "volume_Å3": 250.0,
                                 "pfas_feed_conc_µg/L": float(f),
                                 "mon_cat_mM": 0.0, "di_cat_mM": 0.0,
                                 "tri_cat_mM": 0.0,
                                 "doc_mg/L": 0.0}) for f in g_fc]
            ax.plot(g_fc, np.array(ys) - ys[0], label=f"pwp {pwp}")
        ax.set_xscale("log")
        ax.set_xlabel("PFAS feed conc. (µg/L)")
        ax.set_title(f"pH {ph}, vol 250 Å³")
        ax.legend()
    axes[0].set_ylabel("Δ predicted rejection vs lowest conc.")
    fig.suptitle("Feed-concentration effect (Fig 6D analog)")
    fig.tight_layout()
    fig.savefig("results/figures/slice_feedconc.png", dpi=150)
    plt.close(fig)

    # Fig 4D analog: pure-water R over a pH x permeance grid at two volumes
    g_ph = np.linspace(2.5, 11, 30)
    g_pwp = np.linspace(1.5, 40, 30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, vol in zip(axes, [150, 300]):
        Z = np.empty((len(g_ph), len(g_pwp)))
        for i, ph in enumerate(g_ph):
            for k, pw in enumerate(g_pwp):
                Z[i, k] = predict_row(**{"pwp_LMH/bar": float(pw),
                                         "pH": float(ph),
                                         "volume_Å3": float(vol),
                                         "mon_cat_mM": 0.0,
                                         "di_cat_mM": 0.0,
                                         "tri_cat_mM": 0.0,
                                         "doc_mg/L": 0.0})
        cs = ax.contourf(g_pwp, g_ph, Z, levels=20, cmap="viridis",
                         vmin=0.5, vmax=1.0)
        ax.set_xlabel("Water permeance (LMH/bar)")
        ax.set_title(f"PFAS vol {vol} Å³, pure water")
        # pH sensitivity at loose vs tight permeance
        ax.plot([], [], " ", label=" " .join(
            [f"dR/dpH[{g_pwp[2]:.0f}->{g_pwp[-3]:.0f}] pwp: "
             f"{Z[:, 2].max()-Z[:, 2].min():.2f} vs "
             f"{Z[:, -3].max()-Z[:, -3].min():.2f}"]))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Feed pH")
    fig.colorbar(cs, ax=axes, label="Predicted rejection")
    fig.suptitle("pH x permeance grids (Fig 4D analog): charge effects "
                 "should dominate for small PFAS / loose membranes")
    fig.tight_layout()
    fig.savefig("results/figures/slice_ph_pwp_grid.png", dpi=150)
    plt.close(fig)
    print("Wrote SHAP tables + ICE/slice figures.")


if __name__ == "__main__":
    main()
