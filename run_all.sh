#!/bin/bash
# Full pipeline in dependency order (REPORT.md §7).
# Usage:
#   ./run_all.sh            # full run (03 N_TRIALS=100, 05 N_TRIALS=30, 16 N_TUNE=20, BEST=aft)
#   ./run_all.sh --fast     # smoke test only (N_TRIALS=2, N_TUNE=2); numbers NOT comparable
# Env overrides:
#   PYTHON  python interpreter (default /opt/anaconda3/envs/fml/bin/python)
#   N_TRIALS / N_TUNE / BEST  passed through to scripts (see headers)
# REPORT §5 G uses AFT-best (BEST=aft); 19_bootstrap.py reads aft-best rows,
# so full runs pin BEST=aft for reproducibility.
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/opt/anaconda3/envs/fml/bin/python}"
FAST=0
if [ "$1" = "--fast" ]; then
  FAST=1
  echo "=== FAST smoke mode: reduced trials (numbers NOT comparable) ==="
  export N_TRIALS=2
  export N_TUNE=2
  export BEST=aft
else
  export BEST=aft
fi

mkdir -p results/tables results/figures results/models

run() {
  echo "=== $* ==="
  "$@"
}

# Round 1: 01 -> 09 (02 needs 01 outputs excluded; 03 needs modeling_frame;
# 04/06/09 need best_params+model; 05 needs best_params; 07/08 need grouped_best_params)
run $PYTHON scripts/01_data_audit.py
run $PYTHON scripts/02_prepare_split.py
run $PYTHON scripts/03_train_evaluate.py
run $PYTHON scripts/04_shap_interpret.py
run $PYTHON scripts/05_grouped_cv.py
run $PYTHON scripts/06_seed_sensitivity.py
run $PYTHON scripts/07_improve_baselines.py
run $PYTHON scripts/08_uncertainty_ad.py
run $PYTHON scripts/09_diagnostics.py

# Round 2: 11 -> 18 (all read grouped_best_params via common2 + round-1 artifacts)
run $PYTHON scripts/11_features_A.py
run $PYTHON scripts/11b_ablation_A2.py
run $PYTHON scripts/12_mixedeffects_B.py
run $PYTHON scripts/13_fewshot_C.py
run $PYTHON scripts/13b_eb_C2.py
# Round-3 helper outputs used by 13c/22/19 (c2_honest.py is import-only, no run needed)
run $PYTHON scripts/13c_eb_honest_C2.py
run $PYTHON scripts/14_physics_D.py
run $PYTHON scripts/14b_physfeat_D2.py
run $PYTHON scripts/14c_physprobe_D3.py
run $PYTHON scripts/15_scope_E.py
run $PYTHON scripts/16_stack_F.py
run $PYTHON scripts/16b_stack_baseline_F2.py
run $PYTHON scripts/17_conformal_G.py
run $PYTHON scripts/18_shap_H.py

# Round 3 robustness (order: 13c -> 22 -> 23 -> 19 -> 20 per REPORT §7)
run $PYTHON scripts/22_c2_variants.py
run $PYTHON scripts/23_paper_hypers_grouped.py
run $PYTHON scripts/19_bootstrap.py
run $PYTHON scripts/20_holm.py

echo "=== running C2 leakage tests ==="
run $PYTHON tests/test_c2_leakage.py

echo "ALL DONE."
