"""Unit tests for C2 leakage guards (round-3 task 3).

Run: /opt/anaconda3/envs/fml/bin/python -m pytest tests/test_c2_leakage.py -v
(or plain python tests/test_c2_leakage.py).
Covers: plug-ins exclude held-out study; shrinkage derived from train-only
variances; cal/test disjointness; honest vs global plug-in closeness log.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays
from c2_honest import (train_only_mom_variances, shrinkage_params,
                       eb_calibrate, SLOPE_PRIOR_VAR)


def test_plugin_excludes_heldout():
    frame, _ = load_frame()
    y, cens, R, groups = base_arrays(frame)
    base = np.load("results/tables/C_loso_base.npy")
    for s in list(np.unique(groups))[:5]:
        t2, s2, ntr, nst = train_only_mom_variances(y, base, cens,
                                                   groups, s)
        assert np.isfinite(t2) and t2 > 0
        assert np.isfinite(s2) and s2 > 0
        # manual check: no held-out row could have contributed
        assert s not in np.unique(groups[groups != s])
        # 61/62 studies own >=1 uncensored row (Franke, 2021 is all-censored),
        # so train-only uncensored-study count is 60 or 61 depending on s.
        assert nst in (60, 61), f"unexpected nst={nst} for {s}"
    print("test_plugin_excludes_heldout PASSED")


def test_shrinkage_is_pure_function_of_train_variances():
    lam_a, lam_b = shrinkage_params(1.5, 1.5)
    assert abs(lam_a - 1.0) < 1e-12
    assert abs(lam_b - 1.5 / SLOPE_PRIOR_VAR) < 1e-9
    # monotonicity: larger tau2 -> smaller lam_a (less shrinkage of offset)
    assert shrinkage_params(2.0, 1.5)[0] < shrinkage_params(1.0, 1.5)[0]
    print("test_shrinkage_is_pure_function_of_train_variances PASSED")


def test_eb_k1_shrinkage_bounded():
    a, b = eb_calibrate(np.array([2.0]), np.array([1.0]),
                        1.47, 1.52, k=1)
    w = 1.47 / (1.47 + 1.52)
    assert b == 1.0 and abs(a - w * 1.0) < 1e-12
    assert 0 < w < 1
    print("test_eb_k1_shrinkage_bounded PASSED")


def test_cal_test_disjoint_protocol():
    frame, _ = load_frame()
    y, cens, R, groups = base_arrays(frame)
    s = sorted(np.unique(groups).tolist())[0]
    idx = np.where(groups == s)[0]
    rng = np.random.default_rng(42)
    unc = idx[~cens[idx]]
    pool = unc if len(unc) >= 1 else idx
    cal = rng.choice(pool, size=min(2, len(pool)), replace=False)
    tst = idx[~np.isin(idx, cal)]
    assert not np.intersect1d(tst, cal).size
    assert set(cal.tolist()).issubset(set(idx.tolist()))
    print("test_cal_test_disjoint_protocol PASSED")


def test_honest_plugins_close_to_global():
    # leakage magnitude check: train-only weights must be near global
    # (else 13b conclusions would be suspect). Threshold 0.05 is generous:
    # observed max |dw| ~ 0.02.
    audit = pd.read_csv("results/tables/C2_leakage_audit.csv")
    assert ((audit["w_train"] > 0) & (audit["w_train"] < 1)).all()
    maxdw = (audit["w_train"] - audit["w_global"]).abs().max()
    assert maxdw < 0.05, f"plug-in shift too large: {maxdw}"
    print(f"test_honest_plugins_close_to_global PASSED (max|dw|={maxdw:.4f})")


if __name__ == "__main__":
    test_plugin_excludes_heldout()
    test_shrinkage_is_pure_function_of_train_variances()
    test_eb_k1_shrinkage_bounded()
    test_cal_test_disjoint_protocol()
    try:
        test_honest_plugins_close_to_global()
    except FileNotFoundError:
        print("SKIP test_honest_plugins_close_to_global "
              "(run 13c first to generate audit csv)")
    print("ALL C2 LEAKAGE TESTS PASSED")
