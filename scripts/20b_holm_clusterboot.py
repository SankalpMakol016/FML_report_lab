"""Round-3 — Holm with STUDY-level cluster-bootstrap p-values for ALL 15 tests.

Supersedes the previous seed-level version of this script (whose output is
preserved as results/tables/R3_holm_clusterboot_seedonly_SUPERSEDED.csv):
resampling 10 seeds is not a cluster bootstrap and is strictly weaker.
Here every test resamples the 62 studies (10000 reps).

Method (identical for all 15 primary RMSE gain tests):
  * Per-study mean RMSE (averaged over seeds) for every method is saved in
    results/tables/R3_study_rmse_long.csv (long format:
    study, block, variant, seed, rmse, n_rows). Seed-level OOF loops replicate
    scripts 11 (A), 11b (A2), 14 (D), 14b (D2), 15-Q2 (E), 16 (F, frozen
    F_stack_params.json) exactly — same folds (common2.grouped_folds),
    same configs, same censoring/uncensored-only RMSE. C2 reuses the saved
    per-study files (C2_honest.csv, C_fewshot.csv; already study-level,
    averaged over reps) with no refits.
  * For each test, paired differences per study (variant − baseline,
    paired-complete), resample studies with replacement (B=10000, per-test
    RNG default_rng(42000+test_index)), mean gain per replicate.
  * Two-sided bootstrap p (Davison-Hinkley):
      p = (1 + #{|g* − mean(g*)| >= |obs|}) / (B+1);
    percentile 95% CI; Holm step-down over m=15, alpha=0.05.

Outputs: results/tables/R3_study_rmse_long.csv,
  results/tables/R3_studyboot_reps.npz (replicate gains per test),
  results/tables/R3_holm_clusterboot.csv (study-boot Holm + seed-t
  reference columns from R3_holm.csv for side-by-side),
  results/tables/R3_C2_studyboot_gains.csv (final C2 k=1/2/3 table).
Checkpointed per (block, seed): safe to re-run/resume; slow blocks are
the refits (A/A2/D/D2/E/F ≈ 1 hr total), bootstrap itself is seconds.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from scipy.stats import ttest_rel
import os
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, paired_ci,
                     fit_aft, build_matrix, encode_apply, dm_aft,
                     SEEDS, SEED, FEATURES_13)

B = 10000
LONG_CSV = "results/tables/R3_study_rmse_long.csv"

# The 15-test family (same as scripts/20_holm.py), as (test, block, variant,
# baseline_block, baseline_variant). E slices are handled by using
# slice-specific rmse values stored under variant names "reg-all:low" etc.
# F tests use the A-block base13 per-study RMSE as baseline (identical folds,
# same as 16b/F_stack_gain_vs_baseline.csv which pairs by seed).
FAMILY = [
    ("A_all_allextra_vs_base", "A", "allextra", "A", "base13"),
    ("A2_allextra_vs_base", "A2", "allextra", "A2", "base13"),
    ("A2_+zp_vs_base", "A2", "+zp", "A2", "base13"),
    ("A2_+wca+tmp_vs_base[TABLE]", "A2", "+wca+tmp", "A2", "base13"),
    ("A2_+cats_vs_base", "A2", "+cats", "A2", "base13"),
    ("C2_k1_EB_vs_k0", "C2", "k1", "C2", "k0"),
    ("C2_k2_EB_vs_k0", "C2", "k2", "C2", "k0"),
    ("C2_k3_EB_vs_k0", "C2", "k3", "C2", "k0"),
    ("D_hybrid_vs_AFT", "D", "hybrid", "D", "AFT-baseline"),
    ("D2_physfeat_vs_xgb13", "D2", "xgb13+physfeat", "D2", "xgb13"),
    ("E_low_reg-all_vs_hi-only", "E", "reg-all:low", "E", "reg-hi-only:low"),
    ("E_high_reg-all_vs_hi-only", "E", "reg-all:hi", "E", "reg-hi-only:hi"),
    ("F_lgb_vs_baseline", "F", "lgb-tuned", "A", "base13"),
    ("F_cat_vs_baseline", "F", "cat-tuned", "A", "base13"),
    ("F_stack_vs_baseline", "F", "stack", "A", "base13"),
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def study_rmse(pred, y, cens, R, groups, studies):
    """Per-study RMSE (% rejection, uncensored rows only)."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    cens = np.asarray(cens, bool)
    R = np.asarray(R, float)
    groups = np.asarray(groups)
    out = {}
    for s in studies:
        m = (groups == s) & (~cens)
        if int(m.sum()) == 0:
            out[s] = (np.nan, 0)
        else:
            err = (sigmoid(pred[m]) - R[m]) * 100.0
            out[s] = (float(np.sqrt(np.mean(err ** 2))), int(m.sum()))
    return out


def load_long():
    if os.path.exists(LONG_CSV):
        return pd.read_csv(LONG_CSV)
    return pd.DataFrame(
        columns=["study", "block", "variant", "seed", "rmse", "n_rows"])


def done_seeds(df, block):
    """Seeds already complete for a block (all its variants present).

    NOTE: the seed column reads back as str (mixed int/str seeds across
    blocks, e.g. C2's "mean_over_reps"), so comparisons use str(s).
    """
    if len(df) == 0:
        return set()
    d = df[df["block"] == block]
    if len(d) == 0:
        return set()
    d = d.copy()
    d["seed"] = d["seed"].astype(str)
    variants = sorted(d["variant"].unique().tolist())
    seeds = sorted(d["seed"].unique().tolist())
    good = set()
    for s in seeds:
        vs = set(d[d["seed"] == s]["variant"].unique().tolist())
        if vs == set(variants) and len(variants) > 0:
            good.add(s)
    return good


def block_complete(df, block, seeds=SEEDS):
    """True iff every seed in `seeds` is complete for `block`."""
    return all(str(s) in done_seeds(df, block) for s in seeds)


def append_rows(df, rows):
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(LONG_CSV, index=False)
    return df


# ---------------------------------------------------------------- A block
def block_A(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "A")]
    if not need:
        print("A: complete, skipping refits")
        return df
    frame, raw = load_frame()
    y, cens, R, groups = base_arrays(frame)
    studies = sorted(np.unique(groups).tolist())
    df_all, levs_all = build_matrix(raw, frame, extra=True)
    df_base, _ = build_matrix(raw, frame, extra=False)
    for s in need:
        preds = {}
        for tag, Xdf, lv, ind in [
                ("base13", df_base, {}, False),
                ("allextra", df_all, levs_all, True)]:
            pred = np.empty(len(y))
            for tri, vai in grouped_folds(groups, s):
                Xtr, Xva = encode_apply(Xdf.iloc[tri], Xdf.iloc[vai], lv,
                                        indicators=ind)
                bst = fit_aft(Xtr, y[tri], cens[tri])
                pred[vai] = bst.predict(xgb.DMatrix(Xva))
            preds[tag] = pred
        rows = []
        for tag, pred in preds.items():
            for st, (rm, n) in study_rmse(
                    pred, y, cens, R, groups, studies).items():
                rows.append({"study": st, "block": "A", "variant": tag,
                             "seed": s, "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"A: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- A2 block
def a2_sub(df_all, cols_extra=None, cats=None):
    use = list(df_all.columns)
    if cols_extra is not None:
        drop_num = {"zp_mV", "wca_º", "tmp_bar"} - set(cols_extra)
        use = [c for c in use if c not in drop_num]
    if cats is not None:
        use = [c for c in use if (not c.startswith("_cat_"))
               or c in cats]
    else:
        use = [c for c in use if not c.startswith("_cat_")]
    return df_all[use]


def block_A2(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "A2")]
    if not need:
        print("A2: complete, skipping refits")
        return df
    frame, raw = load_frame()
    y, cens, R, groups = base_arrays(frame)
    zp = raw["zp_mV"].notna().to_numpy()
    df_all, levs_all = build_matrix(raw, frame, extra=True)
    df_base, _ = build_matrix(raw, frame, extra=False)
    fb = df_base[zp].reset_index(drop=True)
    dall = df_all[zp].reset_index(drop=True)
    ys, cs, Rs, gs = y[zp], cens[zp], R[zp], groups[zp]
    studies = sorted(np.unique(gs).tolist())
    variants = {
        "base13": (fb, {}, False),
        "+zp": (a2_sub(dall, ["zp_mV"]), {}, True),
        "+wca+tmp": (a2_sub(dall, ["wca_º", "tmp_bar"]), {}, True),
        "+cats": (a2_sub(dall, [], ["_cat_system", "_cat_mode",
                                    "_cat_pfas"]),
                  {k: levs_all[k] for k in
                   ["_cat_system", "_cat_mode", "_cat_pfas"]}, False),
        "allextra": (dall, levs_all, True),
    }
    for s in need:
        rows = []
        for tag, (Xd, lv, ind) in variants.items():
            pred = np.empty(len(ys))
            for tri, vai in grouped_folds(gs, s):
                Xtr, Xva = encode_apply(Xd.iloc[tri], Xd.iloc[vai], lv,
                                        indicators=ind)
                pred[vai] = fit_aft(Xtr, ys[tri], cs[tri]).predict(
                    xgb.DMatrix(Xva))
            for st, (rm, n) in study_rmse(
                    pred, ys, cs, Rs, gs, studies).items():
                rows.append({"study": st, "block": "A2", "variant": tag,
                             "seed": s, "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"A2: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- D block
XGBR = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42}


def phys_R(P, pwp, V, pka, ph, ios, on, on_med):
    A, B, C = P
    rs = 0.1 * (3.0 * np.asarray(V, float) / (4.0 * np.pi)) ** (1.0 / 3.0)
    rp = A * np.sqrt(np.asarray(pwp, float))
    lam = np.clip(rs / np.maximum(rp, 1e-9), 0, 0.999)
    Rs = 1.0 - (2.0 * (1 - lam) ** 2 - (1 - lam) ** 4)
    Rs = np.clip(Rs, 1e-6, 1 - 1e-6)
    ph = np.asarray(ph, float)
    sig = np.where(np.isnan(ph), 0.5, 1.0 / (1.0 + 10.0 ** (4.0 - ph)))
    dep = np.where(np.isnan(ph), 1.0,
                   1.0 / (1.0 + 10.0 ** (np.asarray(pka, float) - ph)))
    on = np.where(np.isnan(np.asarray(on, float)), on_med,
                  np.asarray(on, float))
    io = np.nan_to_num(np.asarray(ios, float), nan=0.0)
    q = dep * on * sig
    scr = np.exp(-C * np.sqrt(np.maximum(io, 0.0)))
    lo = np.log(Rs / (1.0 - Rs)) + B * q * scr
    return 1.0 / (1.0 + np.exp(-lo))


def block_D(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "D")]
    if not need:
        print("D: complete, skipping refits")
        return df
    frame, raw = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    studies = sorted(np.unique(groups).tolist())
    PWP = frame["pwp_LMH/bar"].to_numpy(float)
    VOL = frame["volume_Å3"].to_numpy(float)
    PKA = frame["pKa"].to_numpy(float)
    PH = frame["pH"].to_numpy(float)
    IOS = raw["ios_mM"].to_numpy(float)
    ON = frame["ON"].to_numpy(float)
    for s in need:
        pred = {"hybrid": np.empty(len(y)),
                "AFT-baseline": np.empty(len(y))}
        for tri, vai in grouped_folds(groups, s):
            on_med = np.nanmedian(ON[tri])
            m = ~cens[tri]

            def resid(P):
                Rp = phys_R(P, PWP[tri][m], VOL[tri][m], PKA[tri][m],
                            PH[tri][m], IOS[tri][m], ON[tri][m], on_med)
                return np.log(Rp / (1 - Rp)) - y[tri][m]

            sol = least_squares(resid, [0.12, 1.0, 0.3],
                                bounds=([0.02, -5, 0.0], [1.0, 5, 5.0]),
                                max_nfev=200)
            P = sol.x
            Rp_tr = phys_R(P, PWP[tri], VOL[tri], PKA[tri], PH[tri],
                           IOS[tri], ON[tri], on_med)
            Rp_va = phys_R(P, PWP[vai], VOL[vai], PKA[vai], PH[vai],
                           IOS[vai], ON[vai], on_med)
            rtr = y[tri] - np.log(Rp_tr / (1 - Rp_tr))
            mr = xgb.train(XGBR, xgb.DMatrix(X[tri], label=rtr),
                           num_boost_round=500)
            pred["hybrid"][vai] = (np.log(Rp_va / (1 - Rp_va))
                                   + mr.predict(xgb.DMatrix(X[vai])))
            ba = fit_aft(X[tri], y[tri], cens[tri])
            pred["AFT-baseline"][vai] = ba.predict(xgb.DMatrix(X[vai]))
        rows = []
        for tag, p in pred.items():
            for st, (rm, n) in study_rmse(
                    p, y, cens, R, groups, studies).items():
                rows.append({"study": st, "block": "D", "variant": tag,
                             "seed": s, "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"D: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- D2 block
A_PHYS, S_PHYS, B_PHYS, C_PHYS = 0.13, 0.70, 0.0, 0.06


def phys_feats(pwp, V, pka, ph, ios, on, on_med):
    rs = S_PHYS * 0.1 * (3.0 * np.asarray(V, float)
                         / (4.0 * np.pi)) ** (1 / 3)
    rp = A_PHYS * np.sqrt(np.asarray(pwp, float))
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
    scr = np.exp(-C_PHYS * np.sqrt(np.maximum(io, 0.0)))
    return np.column_stack([Rs, lam, q, scr])


def block_D2(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "D2")]
    if not need:
        print("D2: complete, skipping refits")
        return df
    frame, raw = load_frame()
    X13 = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    studies = sorted(np.unique(groups).tolist())
    PWP = frame["pwp_LMH/bar"].to_numpy(float)
    VOL = frame["volume_Å3"].to_numpy(float)
    PKA = frame["pKa"].to_numpy(float)
    PH = frame["pH"].to_numpy(float)
    IOS = raw["ios_mM"].to_numpy(float)
    ON = frame["ON"].to_numpy(float)
    on_med = float(np.nanmedian(ON))
    PF = phys_feats(PWP, VOL, PKA, PH, IOS, ON, on_med)
    X17 = np.column_stack([X13, PF])
    for s in need:
        p13 = np.empty(len(y))
        p17 = np.empty(len(y))
        for tri, vai in grouped_folds(groups, s):
            m13 = xgb.train(XGBR, xgb.DMatrix(X13[tri], label=y[tri]),
                            num_boost_round=500)
            p13[vai] = m13.predict(xgb.DMatrix(X13[vai]))
            m17 = xgb.train(XGBR, xgb.DMatrix(X17[tri], label=y[tri]),
                            num_boost_round=500)
            p17[vai] = m17.predict(xgb.DMatrix(X17[vai]))
        rows = []
        for tag, p in [("xgb13", p13), ("xgb13+physfeat", p17)]:
            for st, (rm, n) in study_rmse(
                    p, y, cens, R, groups, studies).items():
                rows.append({"study": st, "block": "D2", "variant": tag,
                             "seed": s, "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"D2: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- E block
def block_E(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "E")]
    if not need:
        print("E: complete, skipping refits")
        return df
    raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                        sheet_name="PFAS_literature_dataset_2b").reset_index(
                            drop=True)
    Xa = raw[FEATURES_13].to_numpy()
    Ra = raw["Rejection"].to_numpy(float)
    ya = np.log(np.clip(Ra, 1e-12, 0.9999)
                / (1 - np.clip(Ra, 1e-12, 0.9999)))
    ca = (raw["LOQ"] == "right").to_numpy()
    ga = raw["ref_id"].to_numpy()
    hi = Ra >= 0.5
    studies = sorted(np.unique(ga).tolist())
    for s in need:
        p_all = np.empty(len(raw))
        p_hi = np.empty(len(raw))
        for tri, vai in grouped_folds(ga, s):
            m_all = xgb.train(XGBR, xgb.DMatrix(Xa[tri], label=ya[tri]),
                              num_boost_round=500)
            p_all[vai] = m_all.predict(xgb.DMatrix(Xa[vai]))
            htr = tri[hi[tri]]
            m_hi = xgb.train(XGBR, xgb.DMatrix(Xa[htr], label=ya[htr]),
                             num_boost_round=500)
            p_hi[vai] = m_hi.predict(xgb.DMatrix(Xa[vai]))
        rows = []
        for tag, p in [("reg-all", p_all), ("reg-hi-only", p_hi)]:
            for sl, sm in [("low", ~hi), ("hi", hi)]:
                pp, yy, cc, RR, gg = (p[sm], ya[sm], ca[sm], Ra[sm], ga[sm])
                for st, (rm, n) in study_rmse(
                        pp, yy, cc, RR, gg, studies).items():
                    rows.append({"study": st, "block": "E",
                                 "variant": f"{tag}:{sl}", "seed": s,
                                 "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"E: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- F block
def block_F(df):
    need = [s for s in SEEDS if str(s) not in done_seeds(df, "F")]
    if not need:
        print("F: complete, skipping refits")
        return df
    with open("results/tables/F_stack_params.json") as f:
        bp = json.load(f)
    P = {"aft": {"eta": bp["a_eta"], "depth": bp["a_depth"],
                 "scale": bp["a_scale"], "sub": bp["a_sub"],
                 "col": bp["a_col"], "mcw": bp["a_mcw"],
                 "lam": bp["a_lam"], "alf": bp["a_alf"]},
         "lgb": {"eta": bp["l_eta"], "leaves": bp["l_leaves"],
                 "feat": bp["l_feat"]},
         "cat": {"eta": bp["c_eta"], "depth": bp["c_depth"],
                 "l2": bp["c_l2"]},
         "ridge": bp["ridge"], "n_aft": bp["n_aft"],
         "n_lgb": bp["n_lgb"], "n_cat": bp["n_cat"]}
    frame, _ = load_frame()
    X = frame[FEATURES_13].to_numpy()
    y, cens, R, groups = base_arrays(frame)
    studies = sorted(np.unique(groups).tolist())

    def fit_xgb_aft(Xtr, ytr, ctr, p, n):
        pr = {"objective": "survival:aft", "seed": SEED,
              "aft_loss_distribution": "normal",
              "aft_loss_distribution_scale": p["scale"],
              "eta": p["eta"], "max_depth": p["depth"],
              "subsample": p["sub"], "colsample_bytree": p["col"],
              "min_child_weight": p["mcw"], "lambda": p["lam"],
              "alpha": p["alf"]}
        return xgb.train(pr, dm_aft(Xtr, ytr, ctr), num_boost_round=n)

    def fit_lgb(Xtr, ytr, p, n):
        return lgb.train({"objective": "regression",
                          "learning_rate": p["eta"],
                          "num_leaves": p["leaves"],
                          "min_data_in_leaf": 20,
                          "feature_fraction": p["feat"],
                          "seed": SEED, "verbosity": -1},
                         lgb.Dataset(Xtr, label=ytr),
                         num_boost_round=n)

    def fit_cat(Xtr, ytr, p, n):
        m = cb.CatBoostRegressor(iterations=n, depth=p["depth"],
                                 learning_rate=p["eta"],
                                 l2_leaf_reg=p["l2"],
                                 random_seed=SEED,
                                 verbose=False,
                                 allow_writing_files=False)
        return m.fit(Xtr, ytr)

    for s in need:
        pa = np.empty(len(y))
        pl = np.empty(len(y))
        pc = np.empty(len(y))
        folds = grouped_folds(groups, s, n_splits=5)
        for tri, vai in folds:
            ma = fit_xgb_aft(X[tri], y[tri], cens[tri], P["aft"],
                             P["n_aft"])
            ml = fit_lgb(X[tri], y[tri], P["lgb"], P["n_lgb"])
            mc = fit_cat(X[tri], y[tri], P["cat"], P["n_cat"])
            pa[vai] = ma.predict(xgb.DMatrix(X[vai]))
            pl[vai] = ml.predict(X[vai])
            pc[vai] = mc.predict(X[vai])
        Z = np.column_stack([pa, pl, pc])
        ps = np.empty(len(y))
        for tri, vai in folds:
            rg = Ridge(alpha=P["ridge"]).fit(Z[tri], y[tri])
            ps[vai] = rg.predict(Z[vai])
        rows = []
        for tag, p in [("lgb-tuned", pl), ("cat-tuned", pc),
                       ("stack", ps)]:
            for st, (rm, n) in study_rmse(
                    p, y, cens, R, groups, studies).items():
                rows.append({"study": st, "block": "F", "variant": tag,
                             "seed": s, "rmse": rm, "n_rows": n})
        df = append_rows(df, rows)
        print(f"F: seed {s} done", flush=True)
    return df


# ---------------------------------------------------------------- C2 block
def block_C2(df):
    if "C2" in (df["block"].unique().tolist() if len(df) else []):
        print("C2: complete, reusing saved files")
        return df
    hon = pd.read_csv("results/tables/C2_honest.csv")
    few = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = few[few["k"] == 0].groupby("study")["rmse"].mean()
    aggh = hon.groupby(["study", "k"])["rmse"].mean()
    rows = []
    for st, v in agg0.items():
        rows.append({"study": st, "block": "C2", "variant": "k0",
                     "seed": "mean_over_reps", "rmse": float(v),
                     "n_rows": -1})
    for (st, k), v in aggh.items():
        rows.append({"study": st, "block": "C2",
                     "variant": f"k{k}", "seed": "mean_over_reps",
                     "rmse": float(v), "n_rows": -1})
    df = append_rows(df, rows)
    print("C2: done (no refits)", flush=True)
    return df


# ---------------------------------------------------------------- bootstrap
def holm(pvals):
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
    import time
    t0 = time.time()
    df = load_long()
    for blk, fn in [("A", block_A), ("A2", block_A2), ("D", block_D),
                    ("D2", block_D2), ("E", block_E), ("F", block_F),
                    ("C2", block_C2)]:
        print(f"--- block {blk} ---", flush=True)
        df = fn(df)
    print(f"refits done in {(time.time()-t0)/60:.1f} min", flush=True)

    # study-averaged means per (study, block, variant)
    avg = df.groupby(["study", "block", "variant"])["rmse"].mean(
        ).reset_index()

    # validation: study-averaged variant means vs saved pooled-seed means
    # (weighting differs so exact match is NOT expected; direction and
    # rough magnitude must agree — a large flip signals a mapping bug)
    print("validation (study-avg mean vs saved pooled-seed mean):",
          flush=True)
    for blk, var, path, sel in [
            ("A", "base13", "results/tables/A_features_oof.csv", "base13"),
            ("A", "allextra", "results/tables/A_features_oof.csv",
             "allextra"),
            ("D", "hybrid", "results/tables/D_hybrid_oof.csv", "hybrid"),
            ("D", "AFT-baseline", "results/tables/D_hybrid_oof.csv",
             "AFT-baseline")]:
        sa = float(avg[(avg["block"] == blk)
                       & (avg["variant"] == var)]["rmse"].mean())
        sv = pd.read_csv(path)
        pooled = float(sv[sv["variant"] == sel]["rmse"].mean())
        print(f"  {blk}/{var}: study-avg {sa:.2f} vs pooled {pooled:.2f}",
              flush=True)

    def get(block, variant):
        d = avg[(avg["block"] == block) & (avg["variant"] == variant)]
        return d.set_index("study")["rmse"]

    tests = []
    for name, blk, var, base_blk, base_var in FAMILY:
        v = get(blk, var)
        b = get(base_blk, base_var)
        common = v.index.intersection(b.index)
        d = (v.loc[common] - b.loc[common]).dropna()
        tests.append({"test": name, "studies": common.tolist(),
                      "diff": d.to_numpy(float)})

    seedt = pd.read_csv("results/tables/R3_holm.csv").set_index("test")
    reps_out = {}
    rows = []
    for i, t in enumerate(tests):
        rng = np.random.default_rng(42000 + i)
        d = t["diff"]
        n = len(d)
        if n < 3:
            raise RuntimeError(
                f"{t['test']}: only {n} paired studies; "
                f"check R3_study_rmse_long.csv coverage")
        obs = float(np.mean(d))
        reps = np.empty(B)
        for b in range(B):
            j = rng.integers(0, n, n)
            reps[b] = d[j].mean()
        lo, hi = float(np.percentile(reps, 2.5)), float(
            np.percentile(reps, 97.5))
        stat = np.abs(reps - reps.mean())
        p = float((1 + np.sum(stat >= abs(obs))) / (B + 1))
        reps_out[t["test"]] = reps
        st = seedt.loc[t["test"]]
        rows.append({"test": t["test"], "n_studies": n,
                     "mean_gain": obs, "boot_lo": lo, "boot_hi": hi,
                     "p_raw_boot": p,
                     "seed_t_mean": float(st["mean"]),
                     "seed_t_p_raw": float(st["p_raw"]),
                     "seed_t_p_holm": float(st["p_holm"]),
                     "seed_t_holm_sig": bool(st["holm_sig"])})
    out = pd.DataFrame(rows)
    out["p_holm_boot"] = holm(out["p_raw_boot"].to_numpy())
    out["holm_sig_boot"] = out["p_holm_boot"] < 0.05
    out["status_change_vs_seedt_holm"] = np.where(
        out["holm_sig_boot"] == out["seed_t_holm_sig"], "same",
        np.where(out["holm_sig_boot"], "becomes sig under study-boot Holm",
                 "LOSES sig under study-boot Holm"))
    out = out.sort_values("p_raw_boot").reset_index(drop=True)
    out.to_csv("results/tables/R3_holm_clusterboot.csv", index=False)
    np.savez_compressed("results/tables/R3_studyboot_reps.npz",
                        **{f"i{i}_{t['test'][:40]}": reps_out[t["test"]]
                           for i, t in enumerate(tests)})
    pd.set_option("display.width", 250)
    print(out[["test", "n_studies", "mean_gain", "boot_lo", "boot_hi",
               "p_raw_boot", "p_holm_boot", "holm_sig_boot",
               "seed_t_p_holm", "seed_t_holm_sig",
               "status_change_vs_seedt_holm"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"))
    nch = (out["holm_sig_boot"] != out["seed_t_holm_sig"]).sum()
    print(f"\n{nch}/{len(out)} change Holm status vs seed-t Holm "
          f"(study-level bootstrap, B={B}, all 15 tests).")

    # final C2 gains table vs regenerated baseline
    c0 = avg[(avg["block"] == "C2") & (avg["variant"] == "k0")]
    k0 = float(c0["rmse"].mean())
    c2rows = []
    for k in [1, 2, 3]:
        r = out[out["test"] == f"C2_k{k}_EB_vs_k0"].iloc[0]
        hk = avg[(avg["block"] == "C2") & (avg["variant"] == f"k{k}")]
        c2rows.append({"k": k, "n_studies": int(r["n_studies"]),
                       "k0_baseline_rmse": k0,
                       "honest_mean_rmse": float(hk["rmse"].mean()),
                       "gain": float(r["mean_gain"]),
                       "boot_lo": float(r["boot_lo"]),
                       "boot_hi": float(r["boot_hi"]),
                       "p_raw_boot": float(r["p_raw_boot"]),
                       "p_holm_boot": float(r["p_holm_boot"]),
                       "holm_sig_boot": bool(r["holm_sig_boot"])})
    c2 = pd.DataFrame(c2rows)
    c2.to_csv("results/tables/R3_C2_studyboot_gains.csv", index=False)
    print("\nFinal C2 gains vs regenerated baseline "
          f"(k0={k0:.2f}):")
    print(c2.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Done.")


if __name__ == "__main__":
    main()
