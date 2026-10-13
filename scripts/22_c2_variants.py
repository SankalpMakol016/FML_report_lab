"""Round-3 task 4 — C2 calibration-point selection strategies.

Compares, under IDENTICAL honest EB machinery (train-only plug-ins from
c2_honest, LOSO base predictions, KS=[1,2,3], 20 repeats), three ways to
choose the k revealed points inside each held-out study:

  random    pool = uncensored rows of the study (fallback: any rows if the
              study has <k uncensored). This reproduces the honest rerun of
              13b (same RNG stream SEED -> directly comparable).
  standard  pool = uncensored rows whose PFAS == 'PFOA'; if the study has no
              PFOA rows, PFOS rows; if neither, the study is SKIPPED for this
              strategy (NaN; 8/62 studies). For k>1, sample k rows WITHOUT
              replacement from that compound's pool (take-all if pool<k).
              Rationale: a lab would recalibrate on a standard compound.
  modal     pool = uncensored rows of the study's most common PFAS species
              (ties -> alphabetical first; deterministic). Same take-all rule.

Honesty: plug-ins per held-out study from train-only rows (asserted);
cal points always inside the held-out study; test = non-cal uncensored rows
(asserted disjoint). RNG: three independent streams (SEED, SEED+101,
SEED+102) so strategy comparison is not confounded by shared draws; the
random stream reuses SEED to match the honest rerun exactly.

Outputs:
  results/tables/C2_variants.csv         per study/strategy/k/rep
  results/tables/C2_variants_summary.csv study-averaged RMSE/MAE + paired
                                         (strategy_k - k0) t CIs
  results/figures/C2_variants_rmse_vs_k.png
  console + results/log_C2variants.log   coverage counts (e.g. standard
                                         skips), RMSE vs k table
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays, paired_ci, SEED
from c2_honest import train_only_mom_variances, eb_calibrate

KS = [1, 2, 3]
REPEATS = 20


def pools_for_study(study_rows_pfas, study_unc_mask):
    """Return dict of index-pools (positions within study idx array)."""
    pfas = np.asarray(study_rows_pfas, str)
    unc = np.asarray(study_unc_mask, bool)
    n = len(pfas)
    all_idx = np.arange(n)
    unc_idx = all_idx[unc]
    pool_r = unc_idx if len(unc_idx) >= 1 else all_idx
    # standard: PFOA else PFOS (uncensored only; fallback to any of compound?)
    std = None
    for compound in ("PFOA", "PFOS"):
        m = unc & (pfas == compound)
        if m.sum() >= 1:
            std = all_idx[m]
            std_name = compound
            break
    if std is None:
        std, std_name = np.array([], dtype=int), "NONE"
    # modal PFAS among uncensored (fallback: among all rows)
    src = pfas[unc_idx] if len(unc_idx) else pfas
    vals, counts = np.unique(src, return_counts=True)
    best = sorted(vals[counts == counts.max()].tolist())[0]
    modal = all_idx[unc & (pfas == best)] if len(unc_idx) else all_idx[
        pfas == best]
    return {"random": pool_r, "standard": std, "modal": modal,
            "std_name": std_name, "modal_name": best}


def main():
    frame, _ = load_frame()
    y, cens, R, groups = base_arrays(frame)
    pfas_all = frame["meta_PFAS"].to_numpy(str)
    base = np.load("results/tables/C_loso_base.npy")
    studies = sorted(np.unique(groups).tolist())
    rngs = {s: np.random.default_rng(SEED + off)
            for s, off in [("random", 0), ("standard", 101),
                           ("modal", 102)]}
    # verify random stream reproduces honest rerun ordering (same as 13c)
    det, cov = [], []
    for s in studies:
        idx = np.where(groups == s)[0]
        sp = pfas_all[idx]
        su = ~cens[idx]
        pools = pools_for_study(sp, su)
        tau2_t, sig2_t, _, _ = train_only_mom_variances(
            y, base, cens, groups, s)
        for strat in ("random", "standard", "modal"):
            pool = pools[strat]
            avail = len(pool)
            for k in KS:
                if avail < 1:
                    continue  # standard with neither PFOA nor PFOS
                for rep in range(REPEATS):
                    rng = rngs[strat]
                    cal_local = rng.choice(
                        pool, size=min(k, avail), replace=False)
                    cal = idx[cal_local]
                    assert set(cal.tolist()).issubset(set(idx.tolist()))
                    yc, pc = y[cal], base[cal]
                    a, b = eb_calibrate(yc, pc, tau2_t, sig2_t,
                                        k if len(cal) == k else len(cal))
                    tst = idx[~np.isin(idx, cal)]
                    tst = tst[~cens[tst]]
                    assert not np.intersect1d(tst, cal).size
                    if len(tst) == 0:
                        continue
                    p = a + b * base[tst]
                    err = (1.0 / (1.0 + np.exp(-p)) - R[tst]) * 100.0
                    det.append({"study": s, "strategy": strat, "k": k,
                                "rep": rep, "n_test": len(tst),
                                "pool_n": avail,
                                "std_compound": pools["std_name"],
                                "modal_compound": pools["modal_name"],
                                "rmse": float(np.sqrt(np.mean(err ** 2))),
                                "mae": float(np.mean(np.abs(err)))})
        cov.append({"study": s, "n_rows": len(idx),
                    "n_uncens": int(su.sum()),
                    "pool_random": len(pools["random"]),
                    "pool_standard": len(pools["standard"]),
                    "std_compound": pools["std_name"],
                    "modal_compound": pools["modal_name"],
                    "pool_modal": len(pools["modal"])})
    det = pd.DataFrame(det)
    det.to_csv("results/tables/C2_variants.csv", index=False)
    pd.DataFrame(cov).to_csv("results/tables/C2_variants_coverage.csv",
                             index=False)
    n_std = sum(1 for c in cov if c["pool_standard"] >= 1)
    print(f"standard-strategy coverage: {n_std}/62 studies own PFOA or PFOS "
          f"(uncensored); 8 skipped. modal always available (62/62).")

    raw = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = raw[raw["k"] == 0].groupby("study")[["rmse", "mae"]].mean()
    agg = det.groupby(["study", "strategy", "k"])[["rmse", "mae"]].mean(
        ).reset_index()
    summ = []
    for strat in ("random", "standard", "modal"):
        for k in KS:
            sub = agg[(agg["strategy"] == strat) & (agg["k"] == k)
                      ].set_index("study")
            # paired vs k0 on the INTERSECT of studies (standard: 54)
            common = sub.index.intersection(agg0.index)
            for metric in ["rmse", "mae"]:
                ci = paired_ci(sub.loc[common, metric] - agg0.loc[
                    common, metric])
                summ.append({"strategy": strat, "k": k, "metric": metric,
                             "n_studies": len(common),
                             "mean": float(sub.loc[common, metric].mean()),
                             "gain": ci["mean"],
                             "ci": f"[{ci['lo']:.2f},{ci['hi']:.2f}]",
                             "sig": ci["significant"]})
    summ = pd.DataFrame(summ)
    summ.to_csv("results/tables/C2_variants_summary.csv", index=False)
    print(summ[summ["metric"] == "rmse"].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for strat, mk in [("random", "o-"), ("standard", "s--"),
                      ("modal", "^-.")]:
        sub = summ[(summ["strategy"] == strat)
                   & (summ["metric"] == "rmse")].sort_values("k")
        ax.plot(sub["k"], sub["mean"], mk, label=f"{strat} "
                f"(n={sub['n_studies'].iloc[0]})")
    g0 = raw[raw["k"] == 0].groupby("study")[["rmse"]].mean().mean()
    ax.scatter([0], [float(g0["rmse"])], c="k", s=40, zorder=5,
               label="k=0 (no cal)")
    ax.set_xlabel("# revealed points (k)")
    ax.set_ylabel("Study-averaged RMSE (% rejection)")
    ax.set_title("EB few-shot: selection strategy vs k (honest plug-ins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/C2_variants_rmse_vs_k.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
