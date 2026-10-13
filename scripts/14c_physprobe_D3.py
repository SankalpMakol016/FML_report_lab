"""Round-2 D3 — Physics-block identifiability probe (saved artifact).

Single-fold (first 1800 rows train) Ferry-Renkin + charge fits:
  (i)  free B (4 params A,S,B,C);
  (ii) constrained B>=0 (physical sign).
Saves fitted params + in-sample physics-only R2 to
results/tables/D_phys_probe.csv. Underwrites the REPORT §5-D claim that
the block itself (not the residual formulation) is misspecified.
Runtime ~1 min.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import sys
sys.path.insert(0, "scripts")
from common2 import load_frame, base_arrays


def Rphys(P, V, PWP, PKA, PH, IOS, ON, on_med):
    A, S, B, C = P
    rs = S * 0.1 * (3.0 * np.asarray(V, float) / (4.0 * np.pi)) ** (1 / 3)
    rp = A * np.sqrt(np.asarray(PWP, float))
    lam = np.clip(rs / np.maximum(rp, 1e-9), 0, 0.999)
    Rs = np.clip(1.0 - (2.0 * (1 - lam) ** 2 - (1 - lam) ** 4),
                 1e-6, 1 - 1e-6)
    ph = np.asarray(PH, float)
    sig = np.where(np.isnan(ph), 0.5, 1.0 / (1.0 + 10.0 ** (4.0 - ph)))
    dep = np.where(np.isnan(ph), 1.0,
                   1.0 / (1.0 + 10.0 ** (np.asarray(PKA, float) - ph)))
    on = np.where(np.isnan(np.asarray(ON, float)), on_med,
                  np.asarray(ON, float))
    scr = np.exp(-C * np.sqrt(np.maximum(
        np.nan_to_num(np.asarray(IOS, float), nan=0.0), 0.0)))
    return 1.0 / (1.0 + np.exp(-(np.log(Rs / (1 - Rs))
                                 + B * dep * on * sig * scr)))


def main():
    frame, raw = load_frame()
    y, cens, _, _ = base_arrays(frame)
    kw = dict(V=frame["volume_Å3"].to_numpy(float),
              PWP=frame["pwp_LMH/bar"].to_numpy(float),
              PKA=frame["pKa"].to_numpy(float),
              PH=frame["pH"].to_numpy(float),
              IOS=raw["ios_mM"].to_numpy(float),
              ON=frame["ON"].to_numpy(float))
    tr = np.arange(1800)
    on_med = float(np.nanmean(kw["ON"][tr]))
    m = ~cens[tr]
    t = {k: (v[tr][m] if k != "ON" else v[tr][m]) for k, v in kw.items()}
    ym = y[tr][m]
    null = float(((ym - ym.mean()) ** 2).sum())
    rows = []
    for x0, bnds, tag in [
            ([0.12, 1.0, 1.0, 0.3], ([0.02, 0.3, -5, 0.0], [1.0, 1.5, 5, 5]),
             "free-B"),
            ([0.12, 0.7, 0.5, 0.3], ([0.02, 0.3, 0.0, 0.0], [1.0, 1.5, 5, 5]),
             "constrained-B>=0")]:
        sol = least_squares(
            lambda P: np.log(Rphys(P, **t, on_med=on_med)
                             / (1 - Rphys(P, **t, on_med=on_med))) - ym,
            x0, bounds=bnds, max_nfev=500)
        Rp = Rphys(sol.x, **t, on_med=on_med)
        pred_logit = np.log(Rp / (1 - Rp))
        ss = float(((pred_logit - ym) ** 2).sum())
        rows.append({"variant": tag, "A": sol.x[0], "S": sol.x[1],
                     "B": sol.x[2], "C": sol.x[3],
                     "R2_phys": float(1 - ss / null)})
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/D_phys_probe.csv", index=False)
    print(out.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
