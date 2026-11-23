#!/bin/bash
# Every Part A run, in order. Each line is train.py with that run's own config file.
# Each step resumes after an interruption; re-running a finished step overwrites its rows.
# The console output of every run is kept in logs/<config>.log.
set -eo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
run() { echo "$(date +%H:%M) start $1"; PYTHONUNBUFFERED=1 "$PY" train.py --config "configs/$1.yaml" > "logs/$1.log" 2>&1; echo "$(date +%H:%M) done  $1"; }

run tune_dae
echo "Tuning done. Copying the selected hyperparameters into configs/ (commit them):"
cp results/best_configs/pretrain_dae.yaml configs/pretrain_dae.yaml
cp results/best_configs/pretrain_ae.yaml configs/pretrain_ae.yaml
run pretrain_dae
run pretrain_ae
run supervised
for c in dae ae random scratch supervised pixels dae_conv ae_conv random_conv supervised_conv; do
  run "evaluate_$c"
done
run analyze
