"""Round-3 task 1 — Study-level cluster bootstrap (2000 reps, resample ref_id).

Compares two uncertainty lenses:
  (i)  existing: paired t 95% CIs across 10 seeds (A, G-width-seed-variation)
       or across studies (C2 study-level paired t);
  (ii) new: non-parametric cluster bootstrap resampling STUDIES with
       replacement (2000 reps, RNG 42), recomputing the estimand per replicate
       and reporting 2.5/97.5 percentiles.

Estimands:
  C2: study-averaged RMSE for k0 (no cal), EB-honest k=1/2/3 + gains vs k0.
      No refits: uses C2_honest.csv (train-only plug-ins) + C_fewshot.csv k0,
      aggregated to per-study means. Bootstrap resamples studies (paired:
      same resample index applied to k0 and kk, gain per replicate).
  A-all: pooled-OOF RMSE for base13 vs allextra (all 2450 rows) + gain.
      Fresh 10-seed grouped OOF (same folds as round-2 via grouped_folds) to
      obtain ROW-LEVEL errors; per-study RMSEs averaged over seeds, then
      bootstrap studies. Seed-based t CI from the same fresh OOF (must match
      A_features_oof.csv within thread-jitter ~0.05).
  A-zp: same on zp-observed subset (2218 rows / 59 studies) for base13 vs
      allextra AND vs +wca+tmp (the §5-A2 champion). Folds rebuilt on subset.
  G-width: 90% cross-conformal width for aft-best vs aft-calibrated.
      Seed-0 best-OOF (5 grouped fits) + fixed EB k=2 calibrated vector
      (20 reps, honest plug-ins). Per replicate: resample studies, pool rows
      (duplicated studies contribute rows twice), q90 of |resid| -> width=2q.
      Reports widths + difference (best-cal) with percentile CIs.

Outputs: results/tables/R3_bootstrap.csv (one row per estimand with both
  CIs), results/tables/R3_bootstrap_reps.npz (raw replicates, optional),
  console log. Runtime ~2-3 min (205 AFT fits @ ~0.5 s).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
sys.path.insert(0, "scripts")
from common2 import (load_frame, base_arrays, grouped_folds, oof_metrics,
                     paired_ci, fit_aft, build_matrix, encode_apply,
                     SEEDS, SEED)
from c2_honest import train_only_mom_variances, eb_calibrate

NBOOT = 2000
BRNG = np.random.default_rng(42)


def boot_mean_gain(study_ids, vals0, valsK, nboot=NBOOT, rng=BRNG):
    """Paired cluster bootstrap of mean(valsK) and mean(valsK-vals0).

    study_ids: array of study labels (aligned with vals); resample labels
    with replacement, each replicate takes the corresponding values (NaN
    pairs must already be removed by caller).
    Returns dict with boot means + percentile CIs.
    """
    ids = np.asarray(study_ids)
    v0 = np.asarray(vals0, float)
    vk = np.asarray(valsK, float)
    n = len(ids)
    m0 = np.empty(nboot)
    mk = np.empty(nboot)
    mg = np.empty(nboot)
    for b in range(nboot):
        j = rng.integers(0, n, n)
        m0[b] = v0[j].mean()
        mk[b] = vk[j].mean()
        mg[b] = (vk[j] - v0[j]).mean()
    q = lambda a: (float(np.percentile(a, 2.5)),
                   float(np.percentile(a, 97.5)))
    lo0, hi0 = q(m0)
    lok, hik = q(mk)
    log_, hig = q(mg)
    return {"m0": float(v0.mean()), "mk": float(vk.mean()),
            "gain": float((vk - v0).mean()),
            "m0_lo": lo0, "m0_hi": hi0, "mk_lo": lok, "mk_hi": hik,
            "g_lo": log_, "g_hi": hig,
            "boot_gain_mean": float(mg.mean())}


def boot_pooled_gain(study_list, err0_dict, errK_dict, groups, cens,
                     nboot=NBOOT, rng=BRNG):
    """Cluster bootstrap of POOLED RMSE gain (same estimand as seed-based).

    study_list: list of study ids to resample from (scorable studies).
    err0_dict / errK_dict: dict study -> 1-D array of row ERRORS (% points,
      uncensored rows only, for ONE fold realization, e.g. seed 0).
    Each replicate resamples studies with replacement, pools rows (with
    multiplicity), computes pooled RMSE for each method + gain.
    """
    sts = list(study_list)
    g0, gk = [], []
    for _ in range(nboot):
        js = rng.integers(0, len(sts), len(sts))
        e0 = np.concatenate([err0_dict[sts[j]] for j in js])
        ek = np.concatenate([errK_dict[sts[j]] for j in js])
        g0.append(float(np.sqrt(np.mean(e0 ** 2))))
        gk.append(float(np.sqrt(np.mean(ek ** 2))))
    g0, gk = np.asarray(g0), np.asarray(gk)
    gg = gk - g0
    return {"gain": float(gg.mean()),
            "g_lo": float(np.percentile(gg, 2.5)),
            "g_hi": float(np.percentile(gg, 97.5)),
            "m0": float(np.mean(g0)), "mk": float(np.mean(gk))}


def main():
    frame, raw = load_frame()
    y, cens, R, groups = base_arrays(frame)
    out_rows = []

    # ================= C2 (no refits) =================
    print("== C2 cluster bootstrap (honest EB, per-study means) ==")
    hon = pd.read_csv("results/tables/C2_honest.csv")
    few = pd.read_csv("results/tables/C_fewshot.csv")
    agg0 = few[few["k"] == 0].groupby("study")[["rmse"]].mean()
    aggh = hon.groupby(["study", "k"])[["rmse"]].mean().reset_index()
    for k in [1, 2, 3]:
        sub = aggh[aggh["k"] == k].set_index("study")
        common = sub.index.intersection(agg0.index)
        # paired-complete only (studies with empty test at this k drop out)
        v0 = agg0.loc[common, "rmse"].to_numpy()
        vk = sub.loc[common, "rmse"].to_numpy()
        ci_t = paired_ci(vk - v0)
        b = boot_mean_gain(common.to_numpy(), v0, vk)
        print(f"C2 k={k}: n={len(common)} t-gain {ci_t['mean']:.3f} "
              f"[{ci_t['lo']:.2f},{ci_t['hi']:.2f}] sig={ci_t['significant']} "
              f"| boot-gain {b['gain']:.3f} [{b['g_lo']:.2f},{b['g_hi']:.2f}]")
        out_rows.append({"estimand": f"C2_k{k}_gain_vs_k0",
                         "n_clusters": len(common),
                         "seed_or_t_mean": ci_t["mean"],
                         "seed_or_t_lo": ci_t["lo"],
                         "seed_or_t_hi": ci_t["hi"],
                         "seed_sig": ci_t["significant"],
                         "boot_mean": b["boot_gain_mean"],
                         "boot_lo": b["g_lo"], "boot_hi": b["g_hi"],
                         "boot_sig": bool((b["g_lo"] > 0)
                                          or (b["g_hi"] < 0))})

    # ================= A all-rows (fresh 10-seed OOF) =================
    print("\n== A all-rows fresh OOF (base13 vs allextra, 10 seeds) ==")
    df_all, levs_all = build_matrix(raw, frame, extra=True)
    df_base, _ = build_matrix(raw, frame, extra=False)
    # row-level squared errors per seed for pooled + per-study RMSEs
    per_study = {s: {"base": [], "extra": []} for s in np.unique(groups)}
    seed_rmse = {"base": [], "extra": []}
    err0_seed0, errK_seed0 = {}, {}
    for s in SEEDS:
        pb = np.empty(len(y))
        pe = np.empty(len(y))
        for tri, vai in grouped_folds(groups, s):
            Xtr, Xva = encode_apply(df_base.iloc[tri], df_base.iloc[vai],
                                    {}, indicators=False)
            pb[vai] = fit_aft(Xtr, y[tri], cens[tri]).predict(
                xgb.DMatrix(Xva))
            Xtr2, Xva2 = encode_apply(df_all.iloc[tri], df_all.iloc[vai],
                                      levs_all, indicators=True)
            pe[vai] = fit_aft(Xtr2, y[tri], cens[tri]).predict(
                xgb.DMatrix(Xva2))
        mb = oof_metrics(y, cens, R, pb)
        me = oof_metrics(y, cens, R, pe)
        seed_rmse["base"].append(mb["rmse"])
        seed_rmse["extra"].append(me["rmse"])
        # per-study RMSE (uncensored rows of that study)
        sig = lambda p: 1.0 / (1.0 + np.exp(-np.asarray(p, float)))
        for st in per_study:
            m = (groups == st) & (~cens)
            if m.sum() == 0:
                continue
            eb = (sig(pb[m]) - R[m]) * 100.0
            ee = (sig(pe[m]) - R[m]) * 100.0
            per_study[st]["base"].append(float(np.sqrt(np.mean(eb ** 2))))
            per_study[st]["extra"].append(float(np.sqrt(np.mean(ee ** 2))))
            if s == 0:
                err0_seed0[st] = eb
                errK_seed0[st] = ee
    # seed-based paired t (must match A_features_oof within jitter)
    ci_seed = paired_ci(np.array(seed_rmse["extra"])
                        - np.array(seed_rmse["base"]))
    print(f"fresh seed means: base {np.mean(seed_rmse['base']):.3f} extra "
          f"{np.mean(seed_rmse['extra']):.3f} gain {ci_seed['mean']:.3f} "
          f"[{ci_seed['lo']:.2f},{ci_seed['hi']:.2f}]")
    chk = pd.read_csv("results/tables/A_features_oof.csv")
    cb = chk[chk["variant"] == "base13"]["rmse"].mean()
    ce = chk[chk["variant"] == "allextra"]["rmse"].mean()
    print(f"saved A_features_oof means: base {cb:.3f} extra {ce:.3f} "
          f"(delta fresh-saved: {np.mean(seed_rmse['base'])-cb:+.3f} / "
          f"{np.mean(seed_rmse['extra'])-ce:+.3f}; thread-jitter OK if <0.1)")
    sts = sorted([st for st in per_study
                  if len(per_study[st]["base"]) > 0
                  and len(per_study[st]["extra"]) > 0])
    vb = np.array([np.mean(per_study[st]["base"]) for st in sts])
    ve = np.array([np.mean(per_study[st]["extra"]) for st in sts])
    b = boot_mean_gain(np.array(sts), vb, ve)
    ci_study_t = paired_ci(ve - vb)
    print(f"A-all study-averaged: base {vb.mean():.3f} extra {ve.mean():.3f} "
          f"t-gain {ci_study_t['mean']:.3f} [{ci_study_t['lo']:.2f},"
          f"{ci_study_t['hi']:.2f}] | boot-gain {b['gain']:.3f} "
          f"[{b['g_lo']:.2f},{b['g_hi']:.2f}]")
    # pooled bootstrap on the SAME estimand as seed-based (seed-0 rows)
    bp = boot_pooled_gain(sts, err0_seed0, errK_seed0, groups, cens)
    print(f"A-all POOLED bootstrap (seed-0 rows, same estimand as t): gain "
          f"{bp['gain']:.3f} [{bp['g_lo']:.2f},{bp['g_hi']:.2f}] "
          f"(pooled base {bp['m0']:.3f} extra {bp['mk']:.3f})")
    out_rows.append({"estimand": "A_all_gain_allextra-base",
                     "n_clusters": len(sts),
                     "seed_or_t_mean": ci_seed["mean"],
                     "seed_or_t_lo": ci_seed["lo"],
                     "seed_or_t_hi": ci_seed["hi"],
                     "seed_sig": ci_seed["significant"],
                     "boot_mean": b["boot_gain_mean"],
                     "boot_lo": b["g_lo"], "boot_hi": b["g_hi"],
                     "boot_sig": bool((b["g_lo"] > 0)
                                      or (b["g_hi"] < 0))})
    out_rows.append({"estimand": "A_all_gain_allextra-base_POOLEDboot",
                     "n_clusters": len(sts),
                     "seed_or_t_mean": ci_seed["mean"],
                     "seed_or_t_lo": ci_seed["lo"],
                     "seed_or_t_hi": ci_seed["hi"],
                     "seed_sig": ci_seed["significant"],
                     "boot_mean": bp["gain"],
                     "boot_lo": bp["g_lo"], "boot_hi": bp["g_hi"],
                     "boot_sig": bool((bp["g_lo"] > 0)
                                      or (bp["g_hi"] < 0))})

    # ================= A zp-subset =================
    print("\n== A zp-subset fresh OOF (base vs allextra vs +wca+tmp) ==")
    zp = raw["zp_mV"].notna().to_numpy()
    fb = df_base[zp].reset_index(drop=True)
    fa = df_all[zp].reset_index(drop=True)
    ys, cs, Rs, gs = y[zp], cens[zp], R[zp], groups[zp]
    # +wca+tmp columns: 13 + wca+tmp (drop zp, drop cats)
    keep_wt = [c for c in fa.columns if not c.startswith("_cat_")
               and c not in ("zp_mV",)]
    fwt = fa[keep_wt]
    per_z = {s: {"base": [], "extra": [], "wt": []}
             for s in np.unique(gs)}
    seed_z = {"base": [], "extra": [], "wt": []}
    zerr_seed0 = {"base": {}, "extra": {}, "wt": {}}
    for s in SEEDS:
        for tag, Xd, lv, ind, key in [
                ("base", fb, {}, False, "base"),
                ("extra", fa, levs_all, True, "extra"),
                ("wt", fwt, {}, True, "wt")]:
            pred = np.empty(len(ys))
            for tri, vai in grouped_folds(gs, s):
                Xtr, Xva = encode_apply(Xd.iloc[tri], Xd.iloc[vai], lv,
                                        indicators=ind)
                pred[vai] = fit_aft(Xtr, ys[tri], cs[tri]).predict(
                    xgb.DMatrix(Xva))
            m = oof_metrics(ys, cs, Rs, pred)
            seed_z[key].append(m["rmse"])
            sig = lambda p: 1.0 / (1.0 + np.exp(-np.asarray(p, float)))
            for st in per_z:
                mm = (gs == st) & (~cs)
                if mm.sum() == 0:
                    continue
                e = (sig(pred[mm]) - Rs[mm]) * 100.0
                per_z[st][key].append(float(np.sqrt(np.mean(e ** 2))))
                if s == 0:
                    zerr_seed0[key][st] = e
    for key, label in [("extra", "allextra-base"), ("wt", "+wca+tmp-base")]:
        ci_s = paired_ci(np.array(seed_z[key]) - np.array(seed_z["base"]))
        stsz = sorted([st for st in per_z
                       if len(per_z[st]["base"]) > 0
                       and len(per_z[st][key]) > 0])
        # paired-complete across seeds: every zp study has rows, keep all
        vz0 = np.array([np.mean(per_z[st]["base"]) for st in stsz])
        vzk = np.array([np.mean(per_z[st][key]) for st in stsz])
        bb = boot_mean_gain(np.array(stsz), vz0, vzk)
        print(f"A-zp {label}: seed-gain {ci_s['mean']:.3f} "
              f"[{ci_s['lo']:.2f},{ci_s['hi']:.2f}] sig={ci_s['significant']} "
              f"| studyAvg-boot {bb['gain']:.3f} [{bb['g_lo']:.2f},{bb['g_hi']:.2f}]")
        out_rows.append({"estimand": f"A_zp_gain_{label}",
                         "n_clusters": len(stsz),
                         "seed_or_t_mean": ci_s["mean"],
                         "seed_or_t_lo": ci_s["lo"],
                         "seed_or_t_hi": ci_s["hi"],
                         "seed_sig": ci_s["significant"],
                         "boot_mean": bb["boot_gain_mean"],
                         "boot_lo": bb["g_lo"], "boot_hi": bb["g_hi"],
                         "boot_sig": bool((bb["g_lo"] > 0)
                                          or (bb["g_hi"] < 0))})
        # pooled bootstrap, same estimand as seed-based (seed-0 rows)
        bpp = boot_pooled_gain(stsz, zerr_seed0["base"], zerr_seed0[key],
                               gs, cs)
        print(f"  pooled-boot (same estimand): gain {bpp['gain']:.3f} "
              f"[{bpp['g_lo']:.2f},{bpp['g_hi']:.2f}] "
              f"(pooled base {bpp['m0']:.3f} extra {bpp['mk']:.3f})")
        out_rows.append({"estimand": f"A_zp_gain_{label}_POOLEDboot",
                         "n_clusters": len(stsz),
                         "seed_or_t_mean": ci_s["mean"],
                         "seed_or_t_lo": ci_s["lo"],
                         "seed_or_t_hi": ci_s["hi"],
                         "seed_sig": ci_s["significant"],
                         "boot_mean": bpp["gain"],
                         "boot_lo": bpp["g_lo"], "boot_hi": bpp["g_hi"],
                         "boot_sig": bool((bpp["g_lo"] > 0)
                                          or (bpp["g_hi"] < 0))})

    # ================= G width =================
    print("\n== G width cluster bootstrap (seed-0 best + honest cal) ==")
    from common2 import FEATURES_13
    X = frame[FEATURES_13].to_numpy()
    # seed-0 best OOF
    pb0 = np.empty(len(y))
    for tri, vai in grouped_folds(groups, 0):
        pb0[vai] = fit_aft(X[tri], y[tri], cens[tri]).predict(
            xgb.DMatrix(X[vai]))
    # honest calibrated vector (k=2, 20 reps, SEED+1 stream as in 17 but
    # with train-only plug-ins)
    base_loso = np.load("results/tables/C_loso_base.npy")
    rng = np.random.default_rng(SEED + 1)
    studies = sorted(np.unique(groups).tolist())
    calo = np.zeros((len(y), 20))
    for rep in range(20):
        pc = np.empty(len(y))
        for s in studies:
            t2, s2, _, _ = train_only_mom_variances(y, base_loso, cens,
                                                    groups, s)
            idx = np.where(groups == s)[0]
            unc = idx[~cens[idx]]
            pool = unc if len(unc) >= 2 else idx
            cal = rng.choice(pool, size=min(2, len(pool)), replace=False)
            yc, pp = y[cal], base_loso[cal]
            a, b_ = eb_calibrate(yc, pp, t2, s2, 2 if len(cal) == 2 else 1)
            tst = idx[~np.isin(idx, cal)]
            pc[tst] = a + b_ * base_loso[tst]
            pc[cal] = base_loso[cal]
        calo[:, rep] = pc
    pc_mean = calo.mean(axis=1)
    # residuals (uncensored only)
    rb = np.abs(pb0[~cens] - y[~cens])
    rc = np.abs(pc_mean[~cens] - y[~cens])
    gb = groups[~cens]
    ub = np.unique(groups)
    # map study -> residual indices (in censored-filtered arrays)
    # bootstrap: resample studies, pool rows with multiplicity
    w_b, w_c, w_d = [], [], []
    for _ in range(NBOOT):
        js = BRNG.integers(0, len(ub), len(ub))
        sel = np.concatenate([np.where(gb == ub[j])[0] for j in js])
        qb = np.quantile(rb[sel], 0.90)
        qc = np.quantile(rc[sel], 0.90)
        w_b.append(2 * qb)
        w_c.append(2 * qc)
        w_d.append(2 * (qb - qc))
    w_b, w_c, w_d = map(np.asarray, (w_b, w_c, w_d))
    # seed-based reference: G_coverage.csv aft rows (5 seeds x 5 folds)
    g = pd.read_csv("results/tables/G_coverage.csv")
    gb_tab = g[g["model"] == "aft-best"]["width"]
    gc_tab = g[g["model"] == "aft-calibrated"]["width"]
    print(f"G aft-best width: saved {gb_tab.mean():.3f}±{gb_tab.std():.3f} | "
          f"boot {w_b.mean():.3f} [{np.percentile(w_b,2.5):.3f},"
          f"{np.percentile(w_b,97.5):.3f}]")
    print(f"G aft-cal width: saved {gc_tab.mean():.3f}±{gc_tab.std():.3f} | "
          f"boot {w_c.mean():.3f} [{np.percentile(w_c,2.5):.3f},"
          f"{np.percentile(w_c,97.5):.3f}]")
    print(f"G width diff (best-cal): saved "
          f"{gb_tab.mean()-gc_tab.mean():.3f} | boot {w_d.mean():.3f} "
          f"[{np.percentile(w_d,2.5):.3f},{np.percentile(w_d,97.5):.3f}] "
          f"(>0 iff calibrated narrower)")
    for est, arr in [("G_width_best", w_b), ("G_width_cal", w_c),
                     ("G_width_diff_best-cal", w_d)]:
        out_rows.append({"estimand": est, "n_clusters": len(ub),
                         "seed_or_t_mean": float(
                             gb_tab.mean() if "best" in est and "diff" not in est
                             else gc_tab.mean() if "cal" in est and "diff" not in est
                             else gb_tab.mean() - gc_tab.mean()),
                         "seed_or_t_lo": np.nan, "seed_or_t_hi": np.nan,
                         "seed_sig": True if "diff" in est else False,
                         "boot_mean": float(arr.mean()),
                         "boot_lo": float(np.percentile(arr, 2.5)),
                         "boot_hi": float(np.percentile(arr, 97.5)),
                         "boot_sig": bool((np.percentile(arr, 2.5) > 0)
                                          or (np.percentile(arr, 97.5) < 0))})

    out = pd.DataFrame(out_rows)
    out.to_csv("results/tables/R3_bootstrap.csv", index=False)
    print("\n" + out.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
