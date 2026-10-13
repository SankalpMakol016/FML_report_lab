"""Round-2 shared protocol (scripts 11-18).

Honest evaluation standard for every improvement claim:
  * grouped 5-fold CV by ref_id, repeated over SEEDS = 0..9
    (folds = seeded shuffle of studies -> 5 contiguous chunks, so every
    point is scored exactly once per seed and all methods share folds);
  * metrics on pooled out-of-fold predictions per seed: RMSE/MAE (%,
    uncensored), C-index (censoring-aware), R2 (logit, uncensored);
  * gain vs the paper-style XGBoost-AFT baseline as paired differences
    across the 10 seeds, with two-sided 95% t CIs; "significant" iff the
    CI for RMSE excludes 0 (primary endpoint).

Also provides: data loading (13 + extra features), one-hot + missingness
indicator builder (train-fit, valid-applied: no leakage), AFT helpers with
the grouped-retuned hyperparameters (Part-3 honest champion = baseline).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from scipy.stats import t
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs
    ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        os.makedirs(_d, exist_ok=True)

SEED = 42
SEEDS = list(range(10))
FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]
EXTRA_NUM = ["zp_mV", "wca_º", "tmp_bar"]
EXTRA_CAT = ["pfas_class", "mode", "system"]
INF = np.inf


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


GROUPED_PARAMS_PATH = "results/tables/grouped_best_params.json"
_GB_CACHE = None


def _load_grouped_best():
    """Lazily load grouped-retuned hypers; error names script to run first."""
    global _GB_CACHE
    if _GB_CACHE is not None:
        return _GB_CACHE
    if not os.path.exists(GROUPED_PARAMS_PATH):
        raise FileNotFoundError(
            f"{GROUPED_PARAMS_PATH} not found — run "
            f"scripts/05_grouped_cv.py first (it writes this file "
            f"with a fixed N_TRIALS/SEED; see its header).")
    with open(GROUPED_PARAMS_PATH) as _fh:
        _GB_CACHE = json.load(_fh)
    return _GB_CACHE


def get_grouped_best():
    """Return raw grouped-best dict (lazy)."""
    return dict(_load_grouped_best())


def get_baseline_aft():
    """Return XGBoost-AFT params from grouped-retuned hypers (lazy)."""
    _GB = _load_grouped_best()
    return {"objective": "survival:aft", "seed": SEED,
            "eta": _GB["eta"],
            "aft_loss_distribution": _GB["aft_loss_distribution"],
            "aft_loss_distribution_scale":
                _GB["aft_loss_distribution_scale"],
            "max_depth": _GB["max_depth"],
            "subsample": _GB["subsample"],
            "colsample_bytree": _GB["colsample_bytree"],
            "min_child_weight": _GB["min_child_weight"],
            "lambda": _GB["lambda"], "alpha": _GB["alpha"]}


def get_baseline_n_est():
    """Return n_estimators from grouped-retuned hypers (lazy)."""
    return _load_grouped_best()["n_estimators"]


def __getattr__(name):
    # Module-level lazy globals so `from common2 import BASELINE_AFT`
    # does not read any file at import time.
    if name == "BASELINE_AFT":
        return get_baseline_aft()
    if name == "BASELINE_N_EST":
        return get_baseline_n_est()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_frame():
    """Modeling frame (R>=0.5) + aligned raw columns needed downstream."""
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
    """Seeded shuffle of unique groups -> contiguous chunks (balanced)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_splits)
    idx = np.arange(len(groups))
    return [(idx[~np.isin(groups, c)], idx[np.isin(groups, c)])
            for c in chunks]


def oof_metrics(y, cens, R, pred):
    """Metrics on pooled out-of-fold predictions (one seed)."""
    m = ~np.asarray(cens, bool)
    err = (sigmoid(np.asarray(pred)[m]) - np.asarray(R)[m]) * 100.0
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "cindex": float(concordance_index(y, pred, m)),
            "r2": float(r2_score(np.asarray(y)[m],
                                 np.asarray(pred)[m]))}


def paired_ci(diff, alpha=0.05):
    """Mean diff + two-sided t CI. NaN pairs dropped (e.g. studies with no
    scorable points at some k). Significant iff CI excludes 0."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return {"mean": float(np.mean(d)) if n else np.nan, "lo": np.nan,
                "hi": np.nan, "significant": False}
    mu, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    half = t.ppf(1 - alpha / 2, n - 1) * se
    return {"mean": float(mu), "lo": float(mu - half),
            "hi": float(mu + half),
            "significant": bool((mu - half > 0) or (mu + half < 0))}


def dm_aft(X, y, cens):
    return xgb.DMatrix(X, label_lower_bound=np.asarray(y, float),
                       label_upper_bound=np.where(cens, INF,
                                                  np.asarray(y, float)))


def fit_aft(Xtr, ytr, ctr, params=None, n_est=None):
    if params is None:
        params = get_baseline_aft()
    if n_est is None:
        n_est = get_baseline_n_est()
    return xgb.train(params, dm_aft(Xtr, ytr, ctr),
                     num_boost_round=n_est)


def build_matrix(raw, frame, extra=True, indicators=True):
    """Feature matrix aligned to frame rows.

    extra=True adds zp_mV, wca, tmp_bar (numeric, native NaN) + one-hot
    pfas_class/mode/system (train categories must be passed at transform;
    see encode_apply). indicators=True appends a 0/1 missing flag per
    column that has any NaN (computed on the transformed matrix).
    Returns (X_all_rows, num_cols, cat_levels).
    """
    df = pd.DataFrame()
    for c in FEATURES_13:
        df[c] = frame[c].to_numpy()
    if extra:
        for c in EXTRA_NUM:
            df[c] = raw[c].to_numpy()
        sys = raw["system"].astype(str).str.strip()
        sys = sys.str.replace(r"^pilot$", "Pilot", regex=True)
        sys = sys.where(~raw["system"].isna(), np.nan)
        df["_cat_system"] = sys.to_numpy()
        df["_cat_mode"] = raw["mode"].where(
            ~raw["mode"].isna(), np.nan).to_numpy()
        df["_cat_pfas"] = raw["pfas_class"].to_numpy()
    cat_levels = {}
    if extra:
        for c in ["_cat_system", "_cat_mode", "_cat_pfas"]:
            cat_levels[c] = sorted(pd.Series(df[c]).dropna().unique())
    return df, cat_levels


def encode_apply(df_tr, df_va, cat_levels, indicators=True):
    """One-hot (train levels) + missing flags. No leakage: levels/flags
    from train only (valid unseen categories -> all-zero)."""
    def enc(df):
        parts = [df[[c for c in df.columns if not c.startswith("_cat_")]]]
        for c, levs in cat_levels.items():
            v = df[c].to_numpy()
            oh = np.zeros((len(df), len(levs)))
            for j, lv in enumerate(levs):
                oh[:, j] = (v == lv).astype(float)
                oh[pd.isna(v), j] = 0.0
            parts.append(pd.DataFrame(
                oh, columns=[f"{c}={lv}" for lv in levs],
                index=df.index))
        return pd.concat(parts, axis=1)
    Xtr, Xva = enc(df_tr), enc(df_va)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=0.0)
    if indicators:
        has_na = Xtr.isna().any(axis=0)
        for c in Xtr.columns[has_na]:
            Xtr[f"miss:{c}"] = Xtr[c].isna().astype(float)
            Xva[f"miss:{c}"] = Xva[c].isna().astype(float)
    return Xtr.to_numpy(float), Xva.to_numpy(float)
