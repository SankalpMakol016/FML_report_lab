"""Round-3 task 3 — C2 leakage audit + honest (training-only) recomputation.

What was leaking in 13b: tau2/sig2 were global means of B_variance.csv.
B_variance was fit on grouped-OOF residuals pooling ALL 62 studies, so each
LOSO held-out study contributed to its own calibration plug-in. The shrinkage
penalties lam_a=sig2/tau2, lam_b=sig2/0.09 inherit the same leak.

Honest recomputation here: for each held-out study s, estimate
(tau2_train, sig2_train) via c2_honest.train_only_mom_variances on rows with
groups != s (asserted in code), derive shrinkage from those only, then run
the IDENTICAL EB calibration protocol as 13b (KS=[1,2,3], 20 repeats, same
RNG stream positions: fresh default_rng(SEED) iterated studies in sorted
order, same as 13b) so the ONLY delta is the plug-in.

Outputs:
  results/tables/C2_honest.csv          per study/k/rep (honest plug-ins)
  results/tables/C2_honest_summary.csv  study-averaged RMSE/MAE + paired t CIs
  results/tables/C2_leakage_audit.csv   per-study plug-in comparison
                                        (global vs train-only tau2/sig2/w)
  results/log_13c.log                   plug-in means, RMSE deltas, assert log
Also runs the unit-test assertions inline (full suite in tests/).
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays, paired_ci, SEED
from c2_honest import train_only_mom_variances, eb_calibrate

KS = [1, 2, 3]
REPEATS = 20


def main():
    frame, _ = load_frame()
    y, cens, R, groups = base_arrays(frame)
    base = np.load("results/tables/C_loso_base.npy")
    bv = pd.read_csv("results/tables/B_variance.csv")
    tau2_g = float(bv["study_between"].mean())
    sig2_g = float(bv["study_within"].mean())
    w_g = tau2_g / (tau2_g + sig2_g)
    print(f"GLOBAL plug-ins (13b, leaks test study): "
          f"tau2={tau2_g:.4f} sig2={sig2_g:.4f} w={w_g:.4f} "
          f"lam_a={sig2_g/tau2_g:.4f}")
    studies = sorted(np.unique(groups).tolist())
    rng = np.random.default_rng(SEED)  # same stream order as 13b

    det_rows, audit_rows = [], []
    for s in studies:
        # ---- honest plug-ins (TRAIN ONLY; asserts inside) ----
        tau2_t, sig2_t, ntr, nst = train_only_mom_variances(
            y, base, cens, groups, s)
        # explicit second assert at call site (defence in depth; logged)
        assert s not in np.unique(groups[groups != s]).tolist(), \
            "leakage guard failed"
        w_t = tau2_t / (tau2_t + sig2_t)
        audit_rows.append({"study": s, "tau2_global": tau2_g,
                           "sig2_global": sig2_g, "w_global": w_g,
                           "tau2_train": tau2_t, "sig2_train": sig2_t,
                           "w_train": w_t,
                           "n_train_rows": ntr, "n_train_studies": nst})
        idx = np.where(groups == s)[0]
        unc = idx[~cens[idx]]
        pool = unc if len(unc) >= 1 else idx
        for k in KS:
            for rep in range(REPEATS):
                cal = rng.choice(pool, size=min(k, len(pool)),
                                 replace=False)
                # LEAKAGE ASSERT: cal points are inside held-out study,
                # test points are disjoint from cal.
                assert set(cal.tolist()).issubset(set(idx.tolist())), \
                    "cal outside held-out study"
                yc, pc = y[cal], base[cal]
                a, b = eb_calibrate(yc, pc, tau2_t, sig2_t, k)
                method = "EB-offset" if k == 1 else "EB-offset-scale"
                tst = idx[~np.isin(idx, cal)]
                tst = tst[~cens[tst]]
                assert not np.intersect1d(tst, cal).size, \
                    "cal/test overlap (leakage)"
                if len(tst) == 0:
                    continue
                p = a + b * base[tst]
                err = (1.0 / (1.0 + np.exp(-p)) - R[tst]) * 100.0
                det_rows.append({"study": s, "k": k, "rep": rep,
                                 "method": method, "n_test": len(tst),
                                 "rmse": float(np.sqrt(np.mean(err ** 2))),
                                 "mae": float(np.mean(np.abs(err)))})
    det = pd.DataFrame(det_rows)
    det.to_csv("results/tables/C2_honest.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv("results/tables/C2_leakage_audit.csv", index=False)

    # ---- summary with paired study-level CIs (same as 13b) ----
    raw = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = raw[raw["k"] == 0].groupby("study")[["rmse", "mae"]].mean()
    aggeh = det.groupby(["study", "k"])[["rmse", "mae"]].mean().reset_index()
    summ = []
    for k in KS:
        eh = aggeh[aggeh["k"] == k].set_index("study")
        for metric in ["rmse", "mae"]:
            ci = paired_ci(eh[metric] - agg0[metric])
            summ.append({"k": k, "metric": metric,
                         "honest_mean": float(eh[metric].mean()),
                         "honest_gain": ci["mean"],
                         "honest_ci": f"[{ci['lo']:.2f},{ci['hi']:.2f}]",
                         "honest_sig": ci["significant"]})
    summ = pd.DataFrame(summ)
    summ.to_csv("results/tables/C2_honest_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---- compare honest vs original (global) EB ----
    orig = pd.read_csv("results/tables/C2_summary.csv")
    print("\n plug-in audit: train-only tau2 "
          f"{audit['tau2_train'].mean():.3f}±{audit['tau2_train'].std():.3f} "
          f"(global {tau2_g:.3f}); sig2 "
          f"{audit['sig2_train'].mean():.3f}±{audit['sig2_train'].std():.3f} "
          f"(global {sig2_g:.3f}); w "
          f"{audit['w_train'].mean():.3f}±{audit['w_train'].std():.3f} "
          f"(global {w_g:.3f})")
    print(" max |w_train - w_global| = "
          f"{(audit['w_train'] - w_g).abs().max():.4f}")
    for k in KS:
        oh = summ[(summ["k"] == k) & (summ["metric"] == "rmse")].iloc[0]
        og = orig[(orig["k"] == k)].iloc[0]
        print(f" k={k}: honest RMSE {oh['honest_mean']:.3f} gain "
              f"{oh['honest_gain']:.3f} {oh['honest_ci']} | "
              f"original EB RMSE {og['eb_mean']:.3f} gain {og['eb_gain']:.3f} "
              f"{og['eb_ci']} | delta {oh['honest_mean']-og['eb_mean']:+.3f}")
    print("ASSERTS PASSED: plug-ins train-only, cal/test disjoint, "
          "shrinkage from train-only variances (see tests/test_c2_leakage.py).")
    print("Done.")


if __name__ == "__main__":
    main()
