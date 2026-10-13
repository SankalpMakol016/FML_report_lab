"""Round-3 helper: honest (training-only) variance plug-ins for EB few-shot.

Leakage context (script 13b): tau2/sig2 were global means of B_variance.csv,
which was estimated from grouped-OOF residuals that pooled ALL studies.
Hence each held-out study contributed to the plug-in used to calibrate it.
Numerically small (global aggregates) but methodologically unclean.

Honest rule enforced here: for held-out study s, plug-ins are estimated from
rows with groups != s ONLY. Callers must pass the held-out id; this module
asserts exclusion. Shrinkage params lam_a=sig2/tau2, lam_b=sig2/0.09 are pure
functions of those train-only variances (plus the fixed slope-prior sd 0.3,
which is a constant, not data-dependent).

MoM estimator (closed form, no MixedLM refit, deterministic):
  on uncensored training rows, r = y - base_pred
  within  = mean over training studies of sample variance of r (ddof=1;
            single-row studies contribute NaN -> dropped; fallback to global
            within if undefined)
  between = sample variance (ddof=1) of training-study mean residuals
  Both floored at 1e-6. This mirrors the B ICC decomposition (between vs
  within) without refitting; absolute scale differs slightly from REML
  (see log_13c.log) but EB shrinkage only needs the RATIO, which is stable.
"""
import numpy as np
import pandas as pd

SLOPE_PRIOR_VAR = 0.09  # 0.3^2, fixed constant (not estimated from data)
MIN_VAR = 1e-6


def train_only_mom_variances(y, base, cens, groups, heldout,
                             pf_filter=None):
    """Estimate (tau2, sig2) from training rows only.

    Args:
      y, base, cens, groups: aligned 1-D arrays (full data).
      heldout: study id excluded from estimation (the LOSO test study).
      pf_filter: optional boolean mask (e.g. zp subset) applied on top.

    Returns (tau2, sig2, n_train_rows, n_train_studies).
    Asserts heldout exclusion (leakage guard).
    """
    y = np.asarray(y, float)
    base = np.asarray(base, float)
    cens = np.asarray(cens, bool)
    groups = np.asarray(groups)
    assert heldout in set(groups.tolist()), "unknown heldout study"
    train = (groups != heldout)
    # LEAKAGE ASSERT: held-out study must not contribute a single row.
    assert not np.any(groups[train] == heldout), \
        f"LEAKAGE: heldout {heldout!r} present in plug-in training rows"
    if pf_filter is not None:
        train = train & np.asarray(pf_filter, bool)
    # uncensored training rows only (a real lab recalibrates on quantifiable
    # points; also matches C2 pool definition)
    sel = train & (~cens)
    assert int(sel.sum()) > 0, "no training rows for variance estimation"
    r = y[sel] - base[sel]
    s = groups[sel]
    df = pd.DataFrame({"r": r, "s": s})
    # within: mean per-study sample variance (drop singletons -> NaN)
    within_per = df.groupby("s")["r"].var(ddof=1)
    within = float(within_per.mean(skipna=True))
    means = df.groupby("s")["r"].mean()
    between = float(means.var(ddof=1)) if len(means) >= 2 else np.nan
    if not np.isfinite(within) or within <= 0:
        within = MIN_VAR
    if not np.isfinite(between) or between <= 0:
        between = MIN_VAR
    within = max(float(within), MIN_VAR)
    between = max(float(between), MIN_VAR)
    return (float(between), float(within), int(sel.sum()),
            int(df["s"].nunique()))


def shrinkage_params(tau2, sig2):
    """Shrinkage penalties from train-only variances (pure function)."""
    assert np.isfinite(tau2) and tau2 > 0, "tau2 must be positive finite"
    assert np.isfinite(sig2) and sig2 > 0, "sig2 must be positive finite"
    return float(sig2 / tau2), float(sig2 / SLOPE_PRIOR_VAR)


def eb_calibrate(y_cal, p_cal, tau2, sig2, k):
    """EB calibration (same math as 13b, plug-ins injected).

    k==1: offset-only a = w*mean(yc-pc), b=1, w=tau2/(tau2+sig2).
    k>=2: ridge (a, db) on (yc-pc)=a+db*pc with penalties lam_a, lam_b.
    Returns (a, b). No data beyond cal points + plug-ins is touched.
    """
    lam_a, lam_b = shrinkage_params(tau2, sig2)
    yc = np.asarray(y_cal, float)
    pc = np.asarray(p_cal, float)
    assert len(yc) == len(pc) and len(yc) >= 1
    assert len(yc) == k or len(yc) < k, \
        "cal pool smaller than k is allowed (take-all); larger is a bug"
    if k == 1 or len(yc) < 2:
        w = float(tau2 / (tau2 + sig2))
        assert 0.0 < w < 1.0, "shrinkage weight outside (0,1)"
        return float(w * (yc.mean() - pc.mean())), 1.0
    Xc = np.column_stack([np.ones(len(yc)), pc])
    tgt = yc - pc
    A = Xc.T @ Xc + np.diag([lam_a, lam_b])
    sol = np.linalg.solve(A, Xc.T @ tgt)
    return float(sol[0]), float(1.0 + sol[1])
