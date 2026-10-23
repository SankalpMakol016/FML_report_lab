# FML_report_lab

Replication, audit, and improvement study of Tomsovic et al., EST 2026 — PFAS rejection in nanofiltration / reverse osmosis from data-driven analysis.

Full write-up: [`REPORT.md`](REPORT.md)

## Layout

- `paper/` — source paper + SI files
- `scripts/` — seeded pipeline (`01` → `09`, `11` → `20`, `20b`, `22-23`), `SEED=42`
- `results/` — `tables/`, `figures/`, `models/`
- `tests/` — leakage / verification tests
- `requirements.txt` — conda env `fml`, Python 3.10.20
- `run_all.sh` — full pipeline in dependency order
- `LICENSE` — MIT license

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh            # full run
./run_all.sh --fast     # smoke test (numbers NOT comparable)
```

Env overrides: `PYTHON` (default `python3`, override e.g. `PYTHON=python3.10`),
`N_TRIALS`, `N_TUNE`, `BEST` (passed through to scripts; see `run_all.sh`
headers). A `.env` file is NOT required — set variables in the shell or prefix
commands, e.g. `PYTHON=python3.10 ./run_all.sh --fast`. Do not commit `.env`
files containing secrets.

See `REPORT.md §7` for run order and further env notes.

## Runtime

- Round 1 (`01` → `09`): a few minutes (03 training is the heaviest).
- Round 2 (`11` → `18`): Optuna tuning per method; tens of minutes.
- Round 3 robustness (`13c` → `22` → `23` → `19` → `20` → `20b`): the
  cluster-bootstrap Holm script `20b_holm_clusterboot.py` refits 6 blocks
  across 10 seeds (~1 hr total on macOS arm64; checkpointed, safe to resume).
- Full `run_all.sh` typically completes in ~1–2 hours.

## Outputs

- `results/tables/*.csv` — metrics, audit, SHAP, sensitivity, diagnostics,
  Holm/cluster-bootstrap tables (`R3_holm.csv`, `R3_holm_clusterboot.csv`,
  `R3_C2_studyboot_gains.csv`, `R3_bootstrap.csv`).
- `results/figures/*.png` — 39 figures (ICE/PD, SHAP, residuals, coverage).
- `results/models/*.json` — random-split champion and honest grouped champion.
- `results/*.log` — Optuna and grouped-retune logs (tracked).

## Results summary (robust claims only)

Robust (pass both seed-t and study-bootstrap Holm lenses):
- **C2 k=3** EB few-shot calibration — the single robust improvement
  (RMSE −1.41 vs k0 baseline 9.56; study-boot Holm p=0.0028).
- **Physics-hybrid failure** — both formulations significantly harmful
  (D: study-boot Holm p=0.0015).
- **Conformal efficiency (G)** — equal coverage, intervals 27% narrower
  (5.08 → 3.72 logit width).
- **C2 leakage audit** — honest train-only plug-ins shift RMSE ≤0.08;
  weight shift max|Δw|=0.056 vs 0.05 threshold (non-fatal warning).

Not robust / withdrawn: A2 (TMP/wca completeness — seed-significant but
collapses under study-bootstrap Holm); stacking harm (sig. under seed-t Holm
only); E scope effects; D2; F-cat.

Holm survivors: 7 under seed-t Holm (`20_holm.py`) vs 2 under study-bootstrap
Holm (`20b_holm_clusterboot.py`, PRIMARY lens). See `REPORT.md §5` for the
full table and `REPORT.md §6` for publishable/non-publishable claims.
