# FML_report_lab

Replication, audit, and improvement study of Tomsovic et al., EST 2026 — PFAS rejection in nanofiltration / reverse osmosis from data-driven analysis.

Full write-up: [`REPORT.md`](REPORT.md)

## Layout

- `paper/` — source paper + SI files
- `scripts/` — seeded pipeline (`01` → `09`, `11` → `20`, `22-23`), `SEED=42`
- `results/` — `tables/`, `figures/`, `models/`
- `tests/` — leakage / verification tests
- `requirements.txt` — conda env `fml`, Python 3.10.20
- `run_all.sh` — full pipeline in dependency order

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh            # full run
./run_all.sh --fast     # smoke test (numbers NOT comparable)
```

See `REPORT.md §7` for run order and env overrides (`PYTHON`, `N_TRIALS` / `N_TUNE` / `BEST`).
