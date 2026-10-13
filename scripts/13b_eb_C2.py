"""Round-2 C2 — Empirical-Bayes few-shot calibration (no refits).

The naive one-shot offset (script 13) hurts on average (median +1.0 RMSE):
a single calibration residual confounds study bias with point noise.
Proper treatment: shrink the offset (and scale) with the variance
components estimated in script 12 (B_variance.csv means):
  w = tau^2 / (tau^2 + sigma^2/k);  a* = w * mean(yc - pc)
  (b*, a*) for k>=2: ridge-penalized least squares on cal points with
  prior a~N(0, sigma^2/tau^2), b~N(1, 0.3^2); k=1 forces b*=1.
Loads LOSO base predictions saved by script 13 (C_loso_base.npy); only
the calibration math is redone (20 repeats, same RNG stream positions).

Outputs: results/tables/C2_eb.csv (per study/k/repeat), C2_summary.csv
(raw vs EB, paired study-level CIs), results/figures/C2_rmse_vs_k.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays, paired_ci, SEED

KS = [1, 2, 3]
REPEATS = 20


def main():
    frame, _ = load_frame()
    y, cens, R, groups = base_arrays(frame)
    base = np.load("results/tables/C_loso_base.npy")
    bv = pd.read_csv("results/tables/B_variance.csv")
    tau2 = float(bv["study_between"].mean())
    sig2 = float(bv["study_within"].mean())
    print(f"plug-in variances: between={tau2:.3f} within={sig2:.3f}")
    studies = np.unique(groups)
    rng = np.random.default_rng(SEED)
    lam_a = sig2 / tau2
    lam_b = sig2 / 0.09  # slope prior sd = 0.3

    rows = []
    for s in studies:
        idx = np.where(groups == s)[0]
        unc = idx[~cens[idx]]
        pool = unc if len(unc) >= 1 else idx
        for k in KS:
            for rep in range(REPEATS):
                cal = rng.choice(pool, size=min(k, len(pool)),
                                 replace=False)
                yc, pc = y[cal], base[cal]
                if k == 1:
                    w = tau2 / (tau2 + sig2)
                    a, b = w * float(yc.mean() - pc.mean()), 1.0
                    method = "EB-offset"
                else:
                    # ridge-penalized (a, db=b-1) on cal points, with
                    # priors a~N(0, sig2/tau2), db~N(0, 0.3^2):
                    # model (yc - pc) = a + db*pc
                    Xc2 = np.column_stack([np.ones(len(cal)), pc])
                    tgt = yc - pc  # model tgt = a + db*pc
                    A = Xc2.T @ Xc2 + np.diag([lam_a, lam_b])
                    rhs = Xc2.T @ tgt
                    sol = np.linalg.solve(A, rhs)
                    a, db = float(sol[0]), float(sol[1])
                    b = 1.0 + db
                    method = "EB-offset-scale"
                tst = idx[~np.isin(idx, cal)]
                tst = tst[~cens[tst]]
                if len(tst) == 0:
                    continue
                p = a + b * base[tst]
                err = (1.0 / (1.0 + np.exp(-p)) - R[tst]) * 100.0
                rows.append({"study": s, "k": k, "rep": rep,
                             "method": method, "n_test": len(tst),
                             "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mae": float(np.mean(np.abs(err)))})
    det = pd.DataFrame(rows)
    det.to_csv("results/tables/C2_eb.csv", index=False)

    raw = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = raw[raw["k"] == 0].groupby("study")[["rmse", "mae"]].mean()
    aggeb = det.groupby(["study", "k"])[["rmse", "mae"]].mean(
        ).reset_index()
    aggr = raw[raw["k"] > 0].groupby(["study", "k"])[["rmse", "mae"]].mean(
        ).reset_index()
    summ = []
    for k in KS:
        eb = aggeb[aggeb["k"] == k].set_index("study")
        rw = aggr[aggr["k"] == k].set_index("study")
        for metric in ["rmse", "mae"]:
            ci_eb = paired_ci(eb[metric] - agg0[metric])
            ci_rw = paired_ci(rw[metric] - agg0[metric])
            summ.append({"k": k, "metric": metric,
                         "raw_mean": float(rw[metric].mean()),
                         "raw_gain": ci_rw["mean"],
                         "raw_ci": f"[{ci_rw['lo']:.2f},{ci_rw['hi']:.2f}]",
                         "raw_sig": ci_rw["significant"],
                         "eb_mean": float(eb[metric].mean()),
                         "eb_gain": ci_eb["mean"],
                         "eb_ci": f"[{ci_eb['lo']:.2f},{ci_eb['hi']:.2f}]",
                         "eb_sig": ci_eb["significant"]})
    summ = pd.DataFrame(summ)
    summ.to_csv("results/tables/C2_summary.csv", index=False)
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    g0 = raw[raw["k"] == 0].groupby("study")[["rmse"]].mean().mean()
    for col, lab in [("raw_mean", "raw offset/scale"),
                     ("eb_mean", "EB-shrunk")]:
        sub = summ[summ["metric"] == "rmse"]
        ax.plot([0] + KS, [float(g0["rmse"])] + sub[col].tolist(), "o-",
                label=lab)
    ax.set_xlabel("# revealed points (k)")
    ax.set_ylabel("Study-averaged RMSE (% rejection)")
    ax.set_title("Few-shot calibration: raw vs EB shrinkage")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/C2_rmse_vs_k.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
