"""Round-3 task 4 — AFT / LightGBM / CatBoost + mean ensemble with extra
features, evaluated under (a) paper-style random split & random 5-fold CV and
(b) study-grouped 10-seed CV.

Extra features (added to the 13 base features): carbon count, ionic strength,
charge term (total_partial_charge), volume/permeance, log permeance, RO flag,
lab flag, and pfas_class (one-hot, train-fit levels only).

No tuning: fixed hyperparameters (grouped-best AFT from
results/tables/grouped_best_params.json; fixed defaults for LGBM/CatBoost) so
nothing is tuned to show a gain.

Metrics: RMSE / MAE (% rejection, uncensored rows) and censoring-aware
C-index (pure-NumPy, event_observed=1 for uncensored rows), computed per
(mode, variant, seed), plus study-level cluster bootstrap CIs (B=2000,
resample ref_id) for the per-study-averaged metric and for the gain vs AFT.

Self-contained: needs only numpy/pandas/sklearn/xgboost/lightgbm/catboost
(runnable with the default `python3`).

Outputs: results/tables/S24_ensemble_random_vs_grouped.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs as _ensure_dirs
    _ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        os.makedirs(_d, exist_ok=True)

NBOOT = 2000
BRNG = np.random.default_rng(42)
OUT_CSV = "results/tables/S24_ensemble_random_vs_grouped.csv"
SEED = 42
SEEDS = list(range(10))
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]

# Fixed hyperparameters (no tuning; grouped-best AFT from round 2).
GROUPED_PARAMS_PATH = "results/tables/grouped_best_params.json"
_GB = None


def _load_grouped_best():
    global _GB
    if _GB is None:
        with open(GROUPED_PARAMS_PATH) as f:
            _GB = json.load(f)
    return _GB


def get_baseline_aft():
    g = _load_grouped_best()
    return {"objective": "survival:aft", "seed": SEED,
            "aft_loss_distribution": g["aft_loss_distribution"],
            "aft_loss_distribution_scale": g["aft_loss_distribution_scale"],
            "eta": g["eta"], "max_depth": g["max_depth"],
            "subsample": g["subsample"],
            "colsample_bytree": g["colsample_bytree"],
            "min_child_weight": g["min_child_weight"],
            "lambda": g["lambda"], "alpha": g["alpha"]}


def get_baseline_n_est():
    return _load_grouped_best()["n_estimators"]


LGB_PARAMS = {"objective": "regression", "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 20,
              "feature_fraction": 0.9, "seed": SEED, "verbosity": -1}
LGB_N = 500
CAT_PARAMS = {"iterations": 500, "depth": 6, "learning_rate": 0.05,
              "l2_leaf_reg": 3.0, "random_seed": SEED, "verbose": False,
              "allow_writing_files": False}
CAT_N = 500


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def load_frame():
    frame = pd.read_csv("results/modeling_frame.csv")
    raw = pd.read_excel("paper/es6c02287_si_002.xlsx",
                        sheet_name="PFAS_literature_dataset_2b")
    raw = raw[raw["Rejection"] >= 0.5].reset_index(drop=True)
    assert len(raw) == len(frame), "frame/raw misaligned"
    return frame, raw


def base_arrays(frame):
    y = frame["y_logit"].to_numpy()
    cens = frame["censored"].to_numpy().astype(bool)
    R = frame["meta_Rejection"].to_numpy()
    groups = frame["meta_ref_id"].to_numpy()
    return y, cens, R, groups


def grouped_folds(groups, seed, n_splits=5):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_splits)
    idx = np.arange(len(groups))
    return [(idx[~np.isin(groups, c)], idx[np.isin(groups, c)])
            for c in chunks]


def _cindex(ytrue, pred):
    """Concordance index (all rows treated as observed events)."""
    ytrue = np.asarray(ytrue, float)
    pred = np.asarray(pred, float)
    n = len(ytrue)
    if n < 3 or len(np.unique(pred)) < 2:
        return np.nan
    concordant = 0.0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ytrue[i] == ytrue[j]:
                continue
            total += 1
            d = (pred[i] - pred[j]) * (ytrue[i] - ytrue[j])
            if d > 0:
                concordant += 1.0
            elif d == 0:
                concordant += 0.5
    if total == 0:
        return np.nan
    return concordant / total


def build_extras(frame, raw):
    """Extra-feature DataFrame aligned to frame rows."""
    df = pd.DataFrame(index=frame.index)
    df["carbons"] = pd.to_numeric(raw["carbons"], errors="coerce").to_numpy()
    df["ios_mM"] = pd.to_numeric(raw["ios_mM"], errors="coerce").to_numpy()
    df["charge_term"] = pd.to_numeric(
        raw["total_partial_charge"], errors="coerce").to_numpy()
    pwp = frame["pwp_LMH/bar"].to_numpy(float)
    df["vol_over_pwp"] = frame["volume_Å3"].to_numpy() / pwp
    df["log_pwp"] = np.log1p(pwp)
    df["ro_flag"] = (raw["mem_class"].astype(str).str.strip() == "RO").astype(float)
    df["lab_flag"] = (raw["mem_source"].astype(str).str.strip() == "Lab").astype(float)
    df["_cat_pfas"] = raw["pfas_class"].to_numpy()
    return df


def encode_extras(Xd_tr, Xd_va, levs=None):
    """Numeric + one-hot pfas_class (train levels; NaN -> all-zero row)."""
    num_cols = [c for c in Xd_tr.columns if c != "_cat_pfas"]
    Xtr_num = Xd_tr[num_cols].to_numpy(float)
    Xva_num = Xd_va[num_cols].to_numpy(float)
    if levs is None:
        levs = sorted(pd.Series(Xd_tr["_cat_pfas"]).dropna().unique().tolist())

    def oh(v):
        v = np.asarray(v)
        m = np.zeros((len(v), len(levs)))
        for j, lv in enumerate(levs):
            m[:, j] = (v == lv).astype(float)
        m[pd.isna(v), :] = 0.0
        return m

    Xtr = np.column_stack([Xtr_num, oh(Xd_tr["_cat_pfas"].to_numpy())])
    Xva = np.column_stack([Xva_num, oh(Xd_va["_cat_pfas"].to_numpy())])
    return Xtr, Xva, levs


def fit_models(Xtr, ytr, ctr):
    """Fit AFT, LGBM, CatBoost on (logit) target. Returns predict callable."""
    INF = np.inf
    aft = xgb.train(get_baseline_aft(),
                    xgb.DMatrix(Xtr, label_lower_bound=np.asarray(ytr, float),
                                label_upper_bound=np.where(
                                    np.asarray(ctr, bool), INF,
                                    np.asarray(ytr, float))),
                    num_boost_round=get_baseline_n_est(),
                    verbose_eval=False)
    lgbm = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, label=ytr),
                   num_boost_round=LGB_N)
    cat = cb.CatBoostRegressor(**CAT_PARAMS).fit(Xtr, ytr)

    def predict(Xva):
        a = aft.predict(xgb.DMatrix(Xva))
        l = lgbm.predict(Xva)
        c = cat.predict(Xva)
        return a, l, c, (a + l + c) / 3.0

    return predict


def per_study_metrics(pred, y, cens, R, groups):
    """Per-study (rmse, mae, cindex, n_uncens); NaN where not scorable."""
    out = {}
    y = np.asarray(y, float)
    cens = np.asarray(cens, bool)
    R = np.asarray(R, float)
    groups = np.asarray(groups)
    pred = np.asarray(pred, float)
    for st in np.unique(groups):
        m = (groups == st) & (~cens)
        n = int(m.sum())
        if n < 1:
            out[st] = (np.nan, np.nan, np.nan, 0)
            continue
        err = (sigmoid(pred[m]) - R[m]) * 100.0
        out[st] = (float(np.sqrt(np.mean(err ** 2))),
                   float(np.mean(np.abs(err))),
                   _cindex(y[m], pred[m]), n)
    return out


def pooled_metrics(pred, y, cens, R):
    m = ~np.asarray(cens, bool)
    err = (sigmoid(np.asarray(pred)[m]) - np.asarray(R)[m]) * 100.0
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "cindex": _cindex(np.asarray(y)[m], np.asarray(pred)[m])}


def average_per_study(seed_dicts):
    studies = set()
    for d in seed_dicts:
        studies.update(d.keys())
    avg = {}
    for st in studies:
        cols = [[d[st][i] for d in seed_dicts if st in d]
                for i in range(3)]
        # require a valid RMSE in at least one seed to keep the study
        rmse_vals = [v for v in cols[0] if v is not None and not np.isnan(v)]
        if not rmse_vals:
            continue
        avg[st] = tuple(float(np.mean([v for v in c if v is not None
                                       and not np.isnan(v)])) if any(
                v is not None and not np.isnan(v) for v in c) else np.nan
                        for c in cols)
    return avg


def bootstrap_studies(ids, v0, vk, nboot=NBOOT, rng=BRNG):
    ids = np.asarray(ids)
    v0 = np.asarray(v0, float)
    vk = np.asarray(vk, float)
    mask = ~(np.isnan(v0) | np.isnan(vk))
    ids, v0, vk = ids[mask], v0[mask], vk[mask]
    n = len(ids)
    res = {"n": int(n)}
    if n < 3:
        for k in ("m0", "mk", "gain", "m0_lo", "m0_hi", "mk_lo", "mk_hi",
                  "g_lo", "g_hi"):
            res[k] = np.nan
        res["gain_sig"] = False
        return res
    m0 = np.empty(nboot)
    mk = np.empty(nboot)
    mg = np.empty(nboot)
    for b in range(nboot):
        j = rng.integers(0, n, n)
        m0[b] = v0[j].mean()
        mk[b] = vk[j].mean()
        mg[b] = (vk[j] - v0[j]).mean()
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    res["m0"] = float(v0.mean())
    res["mk"] = float(vk.mean())
    res["gain"] = float((vk - v0).mean())
    res["m0_lo"], res["m0_hi"] = q(m0)
    res["mk_lo"], res["mk_hi"] = q(mk)
    res["g_lo"], res["g_hi"] = q(mg)
    res["gain_sig"] = bool((res["g_lo"] > 0) or (res["g_hi"] < 0))
    return res


def main():
    frame, raw = load_frame()
    Xb = frame[FEATURES_13].to_numpy(float)
    y, cens, R, groups = base_arrays(frame)
    extras = build_extras(frame, raw)
    variants = ["AFT", "LGBM", "CatBoost", "Ensemble"]

    def eval_split(Xtr, ytr, ctr, Xva):
        predict = fit_models(Xtr, ytr, ctr)
        a, l, c, ens = predict(Xva)
        return {"AFT": a, "LGBM": l, "CatBoost": c, "Ensemble": ens}

    # ---- mode (b): study-grouped 10-seed CV ------------------------------
    print("== mode (b): study-grouped 10-seed CV ==")
    grouped = {v: [] for v in variants}
    grouped_pooled = {v: [] for v in variants}
    for s in SEEDS:
        preds = {v: np.empty(len(y)) for v in variants}
        for tri, vai in grouped_folds(groups, s):
            Xtr, Xva, _ = encode_extras(
                pd.DataFrame(extras.iloc[tri]), pd.DataFrame(extras.iloc[vai]))
            Xtr = np.column_stack([Xb[tri], Xtr])
            Xva = np.column_stack([Xb[vai], Xva])
            part = eval_split(Xtr, y[tri], cens[tri], Xva)
            for v in variants:
                preds[v][vai] = part[v]
        for v in variants:
            grouped[v].append(per_study_metrics(preds[v], y, cens, R, groups))
            grouped_pooled[v].append(pooled_metrics(preds[v], y, cens, R))
        print(f"  grouped seed {s} done", flush=True)

    # ---- mode (a): paper-style random 10-seed holdout --------------------
    print("== mode (a): paper-style random 10-seed holdout ==")
    random_hold = {v: [] for v in variants}
    random_hold_pooled = {v: [] for v in variants}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(y))
        n_tr = int(0.85 * len(y))
        tr, te = perm[:n_tr], perm[n_tr:]
        Xtr, Xva, _ = encode_extras(
            pd.DataFrame(extras.iloc[tr]), pd.DataFrame(extras.iloc[te]))
        Xtr = np.column_stack([Xb[tr], Xtr])
        Xva = np.column_stack([Xb[te], Xva])
        part = eval_split(Xtr, y[tr], cens[tr], Xva)
        for v in variants:
            random_hold[v].append(
                per_study_metrics(part[v], y[te], cens[te], R[te],
                                  groups[te]))
            m = ~cens[te]
            err = (sigmoid(part[v][m]) - R[te][m]) * 100.0
            random_hold_pooled[v].append({
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "cindex": _cindex(y[te][m], part[v][m])})
        print(f"  random holdout seed {s} done", flush=True)

    # ---- mode (a2): random 5-fold CV -------------------------------------
    print("== mode (a2): random 5-fold CV (10 seeds) ==")
    random_cv = {v: [] for v in variants}
    random_cv_pooled = {v: [] for v in variants}
    for s in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        preds = {v: np.empty(len(y)) for v in variants}
        for tri, vai in kf.split(Xb):
            Xtr, Xva, _ = encode_extras(
                pd.DataFrame(extras.iloc[tri]), pd.DataFrame(extras.iloc[vai]))
            Xtr = np.column_stack([Xb[tri], Xtr])
            Xva = np.column_stack([Xb[vai], Xva])
            part = eval_split(Xtr, y[tri], cens[tri], Xva)
            for v in variants:
                preds[v][vai] = part[v]
        for v in variants:
            random_cv[v].append(per_study_metrics(preds[v], y, cens, R, groups))
            random_cv_pooled[v].append(pooled_metrics(preds[v], y, cens, R))
        print(f"  random CV seed {s} done", flush=True)

    # ---- aggregate + bootstrap -------------------------------------------
    modes = {
        "grouped_10seed_cv": (grouped, None),
        "random_10seed_holdout": (random_hold, None),
        "random_10seed_5fold_cv": (random_cv, None),
    }
    pooled_sources = {
        "grouped_10seed_cv": grouped_pooled,
        "random_10seed_holdout": random_hold_pooled,
        "random_10seed_5fold_cv": random_cv_pooled,
    }

    rows = []
    aft_key = "AFT"
    for mode, (per_study_seed, _) in modes.items():
        per_study_avg = {v: average_per_study(per_study_seed[v])
                         for v in variants}
        studies = sorted(set().union(*[set(d) for d in per_study_avg.values()]))
        psrc = pooled_sources[mode]
        for v in variants:
            ps = per_study_avg[v]
            pas = per_study_avg[aft_key]
            common = [st for st in studies if st in ps and st in pas]
            b_rmse = bootstrap_studies(
                common, np.array([pas[st][0] for st in common]),
                np.array([ps[st][0] for st in common]))
            b_mae = bootstrap_studies(
                common, np.array([pas[st][1] for st in common]),
                np.array([ps[st][1] for st in common]))
            b_ci = bootstrap_studies(
                common, np.array([pas[st][2] for st in common]),
                np.array([ps[st][2] for st in common]))
            seed_rmse = [d["rmse"] for d in psrc[v]]
            seed_mae = [d["mae"] for d in psrc[v]]
            seed_ci = [d["cindex"] for d in psrc[v]]
            rows.append({
                "mode": mode, "variant": v,
                "n_seeds": len(SEEDS),
                "n_studies_boot": len(common),
                "rmse_mean": float(np.mean(seed_rmse)),
                "rmse_std": float(np.std(seed_rmse, ddof=1)),
                "rmse_boot_lo": b_rmse["mk_lo"],
                "rmse_boot_hi": b_rmse["mk_hi"],
                "mae_mean": float(np.mean(seed_mae)),
                "mae_std": float(np.std(seed_mae, ddof=1)),
                "mae_boot_lo": b_mae["mk_lo"],
                "mae_boot_hi": b_mae["mk_hi"],
                "cindex_mean": float(np.mean(seed_ci)),
                "cindex_std": float(np.std(seed_ci, ddof=1)),
                "cindex_boot_lo": b_ci["mk_lo"],
                "cindex_boot_hi": b_ci["mk_hi"],
                "gain_rmse_vs_aft_mean": b_rmse["gain"],
                "gain_rmse_lo": b_rmse["g_lo"],
                "gain_rmse_hi": b_rmse["g_hi"],
                "gain_rmse_sig": bool(b_rmse["gain_sig"]),
                "gain_mae_vs_aft_mean": b_mae["gain"],
                "gain_mae_lo": b_mae["g_lo"],
                "gain_mae_hi": b_mae["g_hi"],
                "gain_cindex_vs_aft_mean": b_ci["gain"],
                "gain_cindex_lo": b_ci["g_lo"],
                "gain_cindex_hi": b_ci["g_hi"],
            })
        print(f"  {mode}: aggregated", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 250)
    print("\n" + out.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
