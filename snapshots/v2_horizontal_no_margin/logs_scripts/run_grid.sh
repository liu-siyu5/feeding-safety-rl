#!/bin/bash
# Usage: bash results/logs/run_grid.sh [jobs file] [parallel jobs]
# Runs one shell command per line of the jobs file, then writes the report.
JOBS=${1:-results/logs/jobs.txt}
N=${2:-7}
cd /home/sophia/research/my_feeding_project
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rcareworld
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "grid start $JOBS x$N $(date '+%F %T')" >> results/logs/progress.txt
tr '\n' '\0' < "$JOBS" | xargs -0 -P "$N" -I{} bash -c '{}'
python train_eval.py report > results/logs/report.txt 2>&1
echo "grid done $JOBS $(date '+%F %T')" >> results/logs/progress.txt
