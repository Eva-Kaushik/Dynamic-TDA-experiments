#!/usr/bin/env bash
# Run every experiment in order. Figures go to figures/, tables to results/.
#   bash run_all.sh            # P0-P6 (P6 downloads ~400 MB of CHB-MIT EEG)
#   bash run_all.sh --no-p6    # skip the download
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS="ignore::RuntimeWarning"
PY="${PYTHON:-python}"

step() { echo; echo "=================== $1 ==================="; date; }

step "P0 replication";            $PY experiments/p0_replicate.py
step "P0 amplitude-normalized";   $PY experiments/p0_replicate.py --per_window_std
step "P1 size (tp level)";        $PY experiments/p1_size.py --level tp
step "P1 size (signal level)";    $PY experiments/p1_size.py --level signal
step "P1 size (normalized)";      $PY experiments/p1_size.py --level signal --per_window_std
step "P1 figures";                $PY experiments/figs.py
step "P2 size + power";           $PY experiments/p2_power.py
step "P3 surrogates/summaries";   $PY experiments/p3_confounds.py
step "P4 sensitivity";            $PY experiments/p4_sensitivity.py
step "P5 localization";           $PY experiments/p5_localization.py
if [[ "${1:-}" != "--no-p6" ]]; then
  step "P6 CHB-MIT subset";       $PY experiments/p6_chbmit.py
fi
echo; echo "Done. Tables: results/*/ ; figures: figures/*/"
