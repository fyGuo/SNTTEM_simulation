#!/bin/bash
# Launch the n=12000 / 300-iter continuous-outcome (sigma=10) simulation on this machine.
# Override the interpreter if needed:  PYTHON=/path/to/python ./run.sh
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
echo "Using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"
$PYTHON -c "import numpy, pandas, sklearn, scipy, joblib" \
  || { echo "Missing deps -- run: $PYTHON -m pip install -r requirements.txt"; exit 1; }
# caffeinate keeps the Mac awake for the duration of the run.
caffeinate -i "$PYTHON" run_simulation.py 2>&1 | tee sim_run.log
