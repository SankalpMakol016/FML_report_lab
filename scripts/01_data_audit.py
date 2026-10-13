"""Part 1 — Data audit for Tomsovic et al. (EST 2026) replication.

Reads the source workbook (es6c02287_si_002.xlsx):
  * verifies the censoring encoding from the "Column Descriptions" sheet,
  * counts rows / studies / membranes / PFAS under candidate filters,
  * hunts for the filter that yields the paper's 2353 / 60 / 111 / 44,
  * tabulates missingness of the 13 model features vs SI Table S1,
  * checks censoring plausibility (detection-limit pile-up),
  * quantifies study/membrane dominance.

Outputs (all seeded/deterministic; no randomness used):
  results/tables/audit_counts.csv, audit_missingness.csv,
  audit_censoring.csv, audit_study_sizes.csv, audit_membrane_sizes.csv
  results/figures/audit_rejection_hist.png, audit_censored_vs_uncensored.png,
    audit_study_dominance.png, audit_membrane_dominance.png

Usage: python scripts/01_data_audit.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DATA_SHEET = "PFAS_literature_dataset_2b"

FEATURES_13 = [
    "pwp_LMH/bar", "ON", "pH", "volume_Å3", "dipole_debye", "pKa",
    "temp_ºC", "flux_LMH", "pfas_feed_conc_µg/L",
    "mon_cat_mM", "di_cat_mM", "tri_cat_mM", "doc_mg/L",
]

# SI Table S1 min/max anchors (order: min, max) for the extrema check
S1_MINMAX = {
    "pwp_LMH/bar": (1.30, 74.73), "ON": (0.96, 9.35), "pH": (2.20, 12.00),
    "mw_g/mol": (114.02, 664.02), "logP": (-1.04, 7.62),
    "volume_Å3": (70.82, 505.34), "tpsa_Å2": (29.10, 69.23),
    "pKa": (-3.93, 3.55), "total_partial_charge": (-0.52, -0.25),
    "polarizability_Å3": (5.11, 35.77), "dipole_debye": (1.50, 38.00),
    "min_PA": (19.11, 57.03), "max_PA": (26.60, 151.74),
    "min_PR": (2.68, 5.41), "max_PR": (3.52, 13.90),
    "carbons": (1, 15), "pfas_feed_conc_µg/L": (0.001, 10000000),
    "mon_cat_mM": (0, 1000), "di_cat_mM": (0, 9.45),
    "tri_cat_mM": (0, 6.00), "doc_mg/L": (0, 30.35),
    "temp_ºC": (5.00, 36.00), "flux_LMH": (3.55, 307.09),
}
S1_MISSING = {
    "pwp_LMH/bar": 0.0, "ON": 0.28, "pH": 0.13, "volume_Å3": 0.0,
    "dipole_debye": 0.0, "pKa": 0.0, "temp_ºC": 0.17, "flux_LMH": 0.60,
    "pfas_feed_conc_µg/L": 0.00, "mon_cat_mM": 0.04, "di_cat_mM": 0.03,
    "tri_cat_mM": 0.03, "doc_mg/L": 0.02,
}


def main():
    df = pd.read_excel(XLSX, sheet_name=DATA_SHEET)
    print(f"Full workbook rows: {len(df)}, columns: {len(df.columns)}")

    # ---- 1. censoring encoding -------------------------------------------
    print("\n== LOQ value counts (full data) ==")
    print(df["LOQ"].value_counts(dropna=False))
    assert set(df["LOQ"].dropna().unique()) <= {1, "right", 1.0}, \
        f"unexpected LOQ codes: {df['LOQ'].unique()}"
    print("LOQ NaN count:", df["LOQ"].isna().sum())
    is_cens = (df["LOQ"] == "right").to_numpy()

    # ---- 2. counts under candidate filters --------------------------------
    rows = []
    for name, mask in {
        "full": np.ones(len(df), bool),
        "R>=0.5": (df["Rejection"] >= 0.5).to_numpy(),
        "R>0.5": (df["Rejection"] > 0.5).to_numpy(),
        "R>=0.5 & NF/RO only": ((df["Rejection"] >= 0.5)
                                & df["mem_class"].isin(["NF", "RO"])).to_numpy(),
        "R>=0.5 & NF/RO & no exact dupes": None,  # filled below
    }.items():
        if mask is None:
            sub = df[(df["Rejection"] >= 0.5)
                     & df["mem_class"].isin(["NF", "RO"])]
            sub = sub.drop_duplicates()
            d = sub
        else:
            d = df[mask]
        rows.append({
            "filter": name, "rows": len(d),
            "studies": d["ref_id"].nunique(),
            "membranes": d["mem"].nunique(),
            "pfas_names": d["PFAS"].nunique(),
            "censored": int((d["LOQ"] == "right").sum()),
            "NF270": int((d["mem"] == "NF270").sum()),
            "NF90": int((d["mem"] == "NF90").sum()),
            "lab_rows": int((d["mem_source"] == "Lab").sum()),
            "commercial_rows": int((d["mem_source"] == "Commercial").sum()),
        })
    paper = {"filter": "PAPER reported", "rows": 2353, "studies": 60,
             "membranes": 111, "pfas_names": 44, "censored": np.nan,
             "NF270": 645, "NF90": 219, "lab_rows": 914,
             "commercial_rows": 2353 - 914}
    counts = pd.DataFrame(rows + [paper])
    counts.to_csv("results/tables/audit_counts.csv", index=False)
    print("\n== counts under candidate filters ==")
    print(counts.to_string(index=False))
    print("\nGap R>=0.5 (2450) -> paper (2353):",
          2450 - 2353, "rows")

    d = df[df["Rejection"] >= 0.5].copy()
    print("\nExact-duplicate rows in R>=0.5 subset:", d.duplicated().sum())

    # ---- 3. missingness of the 13 features --------------------------------
    miss = pd.DataFrame({
        "missing_n_subset": d[FEATURES_13].isna().sum(),
        "missing_frac_subset": d[FEATURES_13].isna().mean(),
        "missing_frac_S1": pd.Series(S1_MISSING),
        "missing_n_full": df[FEATURES_13].isna().sum(),
    })
    miss["implied_dropped_missing"] = (
        miss["missing_n_subset"]
        - (miss["missing_frac_S1"] * 2353).round().astype(int))
    miss.to_csv("results/tables/audit_missingness.csv")
    print("\n== missingness (R>=0.5 subset, n=2450) vs SI Table S1 ==")
    print(miss.to_string())
    print("\nRows complete on all 13 features:",
          d[FEATURES_13].notna().all(axis=1).sum(), "of", len(d))

    # ---- 3b. SI Table S1 min/max anchors vs this subset --------------------
    ext = []
    for col, (s1min, s1max) in S1_MINMAX.items():
        x = d[col].dropna()
        ext.append({"column": col, "S1_min": s1min, "subset_min": x.min(),
                    "S1_max": s1max, "subset_max": x.max(),
                    "min_match": bool(np.isclose(s1min, x.min(),
                                                 rtol=2e-3, atol=1e-6)),
                    "max_match": bool(np.isclose(s1max, x.max(),
                                                 rtol=2e-3, atol=1e-6))})
    ext = pd.DataFrame(ext)
    ext.to_csv("results/tables/audit_S1_extrema.csv", index=False)
    print("\n== S1 extrema mismatches (beyond rounding) ==")
    print(ext[~(ext["min_match"] & ext["max_match"])].to_string(index=False))
    print("\nRows violating S1 di_cat max (9.45):",
          int((d["di_cat_mM"] > 9.45).sum()),
          "| tri_cat max (6.0):", int((d["tri_cat_mM"] > 6).sum()),
          "| temp max (36):", int((d["temp_ºC"] > 36).sum()))
    viol = d[(d["di_cat_mM"] > 9.45) | (d["tri_cat_mM"] > 6)
             | (d["temp_ºC"] > 36)]
    print("studies of violating rows:", viol["ref_id"].unique().tolist())

    # ---- 4. censoring plausibility ----------------------------------------
    c = d[d["LOQ"] == "right"]["Rejection"]
    u = d[d["LOQ"] != "right"]["Rejection"]
    cens_tbl = pd.DataFrame({
        "n": [len(c), len(u)],
        "mean_R": [c.mean(), u.mean()],
        "median_R": [c.median(), u.median()],
        "min_R": [c.min(), u.min()],
        "frac_R_ge_0.99": [(c >= 0.99).mean(), (u >= 0.99).mean()],
        "frac_R_at_common_maxima": [
            c.isin([0.95, 0.97, 0.98, 0.99, 0.999, 0.9999]).mean(),
            u.isin([0.95, 0.97, 0.98, 0.99, 0.999, 0.9999]).mean()],
    }, index=["censored(LOQ=right)", "uncensored(LOQ=1)"])
    cens_tbl.to_csv("results/tables/audit_censoring.csv")
    print("\n== censored vs uncensored rejection ==")
    print(cens_tbl.to_string())
    print("\nTop censored values:\n",
          c.value_counts().head(8).to_string())

    # ---- 5. dominance ------------------------------------------------------
    sz_study = d["ref_id"].value_counts().rename_axis("ref_id").reset_index(
        name="n_rows")
    sz_study["frac"] = sz_study["n_rows"] / len(d)
    sz_study.to_csv("results/tables/audit_study_sizes.csv", index=False)
    sz_mem = d["mem"].value_counts().rename_axis("mem").reset_index(
        name="n_rows")
    sz_mem["frac"] = sz_mem["n_rows"] / len(d)
    sz_mem.to_csv("results/tables/audit_membrane_sizes.csv", index=False)
    print("\nTop study share:", sz_study["frac"].iloc[0:1].to_string(),
          "| Top-3 studies share: %.3f" % sz_study["frac"].iloc[:3].sum())
    print("NF270 share: %.3f | NF90 share: %.3f"
          % (645 / len(d), (d["mem"] == "NF90").sum() / len(d)))
    print("PFAS class balance:\n", d["pfas_class"].value_counts().to_string())

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(u, bins=60, alpha=0.7, label=f"uncensored (n={len(u)})")
    ax.hist(c, bins=60, alpha=0.7, label=f"censored (n={len(c)})")
    ax.set_xlabel("Rejection R"); ax.set_ylabel("Count")
    ax.set_title("Rejection distribution, R>=0.5 subset (n=2450)")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/audit_rejection_hist.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log(u / (1 - u)), bins=60, alpha=0.7, label="uncensored")
    ax.hist(np.log(c.clip(upper=0.9999) / (1 - c.clip(upper=0.9999))),
            bins=60, alpha=0.7, label="censored")
    ax.set_xlabel("logit(R)"); ax.set_ylabel("Count")
    ax.set_title("logit(R) distribution by censoring status")
    ax.legend(); fig.tight_layout()
    fig.savefig("results/figures/audit_censored_vs_uncensored.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(sz_study)), sz_study["n_rows"].values)
    ax.set_xlabel("Study (ranked)"); ax.set_ylabel("Rows")
    ax.set_title("Rows per study (62 studies, R>=0.5)")
    fig.tight_layout()
    fig.savefig("results/figures/audit_study_dominance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(sz_mem)), sz_mem["n_rows"].values)
    ax.set_xlabel("Membrane (ranked)"); ax.set_ylabel("Rows")
    ax.set_title("Rows per membrane (116 membranes, R>=0.5)")
    fig.tight_layout()
    fig.savefig("results/figures/audit_membrane_dominance.png", dpi=150)

    print("\nWrote results/tables/audit_*.csv and results/figures/audit_*.png")

    # ---- 6. misc reproducibility notes --------------------------------------
    notes = pd.DataFrame([{
        "n_complete_13": int(d[FEATURES_13].notna().all(axis=1).sum()),
        "n_R_exactly_0.5": int((d["Rejection"] == 0.5).sum()),
        "n_R_at_cap_0.9999": int((d["Rejection"] == 0.9999).sum()),
        "n_cens_at_cap": int(((d["Rejection"] == 0.9999)
                              & (d["LOQ"] == "right")).sum()),
        "pfca_pfsa_share": float(d["pfas_class"].isin(
            ["PFCA", "PFSA"]).mean()),
        "n_LOQ_1_full": int((df["LOQ"] == 1).sum()),
        "n_LOQ_right_full": int((df["LOQ"] == "right").sum()),
    }])
    notes.to_csv("results/tables/audit_notes.csv", index=False)
    print("\n", notes.to_string(index=False))


if __name__ == "__main__":
    main()
