"""Part 2a — Prepare modeling frame for Tomsovic et al. (EST 2026) replication.

Steps (paper Methods, "Data Preprocessing and Splitting"):
  * keep R >= 0.50  ("outliers with rejection below 0.50 were excluded")
  * 13 model features (kept as-is incl. NaN; XGBoost handles missing natively)
  * target logit(R) = ln(R/(1-R)) with R clipped at 0.9999
  * k-means clustering on membrane / PFAS / feed-composition features,
    k chosen by the elbow method  -> cluster labels saved
  * stratified 85/15 train/test split by cluster label (seeded)

Clustering features: the paper says "membrane, PFAS, and feed composition
features" without listing them exactly. We use all 13 model features
(standardized, median-imputed *for clustering only*), which span exactly
those three groups (membrane: pwp/ON/pH; PFAS: volume/dipole/pKa;
feed+operation: the rest). Choice documented in REPORT.md.

Outputs:
  results/tables/cluster_elbow.csv, split_summary.csv
  results/figures/cluster_elbow.png
  results/modeling_frame.csv  (X, y_logit, censored, cluster, split, meta)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
try:
    from setup_paths import ensure_dirs as _ensure_dirs
    _ensure_dirs()
except ImportError:
    for _d in ("results/tables", "results/figures", "results/models"):
        _os.makedirs(_d, exist_ok=True)

SEED = 42
XLSX = "paper/es6c02287_si_002.xlsx"

FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]


def logit_clip(r, cap=0.9999):
    r = np.clip(np.asarray(r, dtype=float), 1e-12, cap)
    return np.log(r / (1.0 - r))


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_excel(XLSX, sheet_name="PFAS_literature_dataset_2b")
    d = df[df["Rejection"] >= 0.5].reset_index(drop=True)
    print(f"Modeling rows (R>=0.5): {len(d)}")

    X = d[FEATURES_13].copy()
    y = logit_clip(d["Rejection"].to_numpy())
    cens = (d["LOQ"] == "right").to_numpy()
    print(f"logit(R): min={y.min():.3f} max={y.max():.3f} "
          f"(cap ln(9999)={np.log(9999):.3f}); censored={cens.sum()}")

    # ---- k-means elbow (median-imputed + standardized, clustering only) ---
    Xc = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X))
    inertias, ks = [], list(range(2, 16))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        km.fit(Xc)
        inertias.append(km.inertia_)
    inertias = np.array(inertias)
    pd.DataFrame({"k": ks, "inertia": inertias}).to_csv(
        "results/tables/cluster_elbow.csv", index=False)
    # elbow = max distance from chord (kneedle-style), deterministic
    xs = np.array(ks, float)
    p1 = np.array([xs[0], inertias[0]], float)
    p2 = np.array([xs[-1], inertias[-1]], float)
    chord = p2 - p1
    pts = np.column_stack([xs, inertias]) - p1
    dist = np.abs(chord[0] * pts[:, 1] - chord[1] * pts[:, 0]) \
        / np.linalg.norm(chord)
    k_star = ks[int(np.argmax(dist))]
    print(f"Elbow pick: k={k_star}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, "o-")
    ax.axvline(k_star, ls="--", c="r", label=f"elbow k={k_star}")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia")
    ax.set_title("K-means elbow on 13 standardized features")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/cluster_elbow.png", dpi=150)

    km = KMeans(n_clusters=k_star, n_init=10, random_state=SEED).fit(Xc)
    cluster = km.labels_.copy()
    print("Raw cluster sizes:", sorted(np.bincount(cluster)))
    # Merge strata too small for 5-fold stratified CV (<10 members) into the
    # nearest surviving centroid (deterministic). Documented in REPORT.md.
    MIN_STRATUM = 10
    while True:
        sizes = np.bincount(cluster, minlength=k_star)
        small = [c for c in range(k_star)
                 if 0 < sizes[c] < MIN_STRATUM and c in set(cluster)]
        if not small:
            break
        big = [c for c in range(k_star)
               if sizes[c] >= MIN_STRATUM and c in set(cluster)]
        cent = np.array([Xc[cluster == c].mean(axis=0) for c in range(k_star)])
        for c in small:
            dists = [np.linalg.norm(cent[c] - cent[b]) for b in big]
            cluster[cluster == c] = big[int(np.argmin(dists))]
    # compact labels 0..K-1
    _, cluster = np.unique(cluster, return_inverse=True)
    print("Final cluster sizes:", sorted(np.bincount(cluster)))

    # ---- stratified 85/15 split by cluster --------------------------------
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.15, random_state=SEED,
                              stratify=cluster)
    split = np.full(len(d), "train", dtype=object)
    split[te] = "test"
    print(f"train={int((split=='train').sum())} "
          f"test={int((split=='test').sum())}")
    print("train censored:", int(cens[split == "train"].sum()),
          "| test censored:", int(cens[split == "test"].sum()))

    frame = pd.DataFrame(X)
    frame["y_logit"] = y
    frame["censored"] = cens.astype(int)
    frame["cluster"] = cluster
    frame["split"] = split
    for col in ["ref_id", "mem", "PFAS", "Rejection", "LOQ",
                "mem_class", "mem_source", "pfas_class"]:
        frame["meta_" + col] = d[col].to_numpy()
    frame.to_csv("results/modeling_frame.csv", index=False)

    summ = pd.DataFrame({
        "split": ["train", "test"],
        "n": [int((split == "train").sum()), int((split == "test").sum())],
        "n_censored": [int(cens[split == "train"].sum()),
                       int(cens[split == "test"].sum())],
        "n_uncensored": [int((~cens & (split == "train")).sum()),
                         int((~cens & (split == "test")).sum())],
    })
    summ.to_csv("results/tables/split_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("Wrote results/modeling_frame.csv")


if __name__ == "__main__":
    main()
