#!/bin/bash
#
# Launch the five-time-point SNTTEM simulation, ML-nuisance variant (psi0,
# n=72000, 300 iterations, nuisances fit with the RF/GBM/poly ensemble --
# see ../oracle/ for the exact-closed-form-nuisance counterpart).
#
#   ./run.sh
#
# Output goes to the terminal and to sim_run.log in this directory.
# Any existing sim_run.log is kept, renamed with its own timestamp.
#
# The interpreter defaults to the project venv at five_time_binary/.venv --
# NOT bare `python3`, which on these machines is Homebrew 3.14 and has no pandas.
# Override if needed:  PYTHON=/path/to/python ./run.sh
#
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd -P)"
ROOT="$(cd "$HERE/../../.." && pwd -P)"       # .../five_time_binary (ML_nuisance/ is nested one level deeper than n72000/)
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG="$HERE/sim_run.log"

# ------------------------------------------------------------------- preflight
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: interpreter not found or not executable:" >&2
    echo "  $PYTHON" >&2
    echo >&2
    echo "Build the project venv with:" >&2
    echo "  /opt/homebrew/bin/python3.13 -m venv $ROOT/.venv" >&2
    echo "  $ROOT/.venv/bin/python -m pip install -r $ROOT/requirements-lock.txt" >&2
    exit 1
fi

if ! "$PYTHON" -c "import numpy, pandas, scipy, sklearn, joblib, matplotlib" 2>/dev/null; then
    echo "ERROR: $PYTHON is missing required packages:" >&2
    "$PYTHON" -c "
import importlib.util as u
print('  missing:', ', '.join(m for m in
      ('numpy','pandas','scipy','sklearn','joblib','matplotlib')
      if u.find_spec(m) is None))" >&2
    echo "  $PYTHON -m pip install -r $ROOT/requirements-lock.txt" >&2
    exit 1
fi

# Keep the previous log rather than clobbering it.
if [ -f "$LOG" ]; then
    mv "$LOG" "${LOG%.log}_$(date -r "$(stat -f %m "$LOG")" +%Y%m%d-%H%M%S).log"
fi

# ------------------------------------------------------------------------ run
main() {
    echo "=================================================================="
    echo "SNTTEM five-time-point simulation (psi0, ML nuisances)"
    echo "=================================================================="
    echo "started     : $(date)"
    echo "host        : $(hostname)"
    echo "directory   : $HERE"
    echo "interpreter : $("$PYTHON" -c 'import sys; print(sys.executable)')"
    echo "python      : $("$PYTHON" -c 'import sys, platform; print(sys.version.split()[0], platform.machine())')"
    "$PYTHON" -c "
import numpy, pandas, scipy, sklearn, joblib, matplotlib
print('packages    : numpy %s | pandas %s | scipy %s | scikit-learn %s | joblib %s | matplotlib %s'
      % (numpy.__version__, pandas.__version__, scipy.__version__,
         sklearn.__version__, joblib.__version__, matplotlib.__version__))"
    echo "cores       : $(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
    echo "------------------------------------------------------------------"
    echo "settings (from run_simulation.py):"
    grep -E '^(N|N_ITER|SEED|PSI[0-9]+|P_I_VALUES|PH_VALUES) *=' run_simulation.py | sed 's/^/  /'
    echo "=================================================================="
    echo

    local start end status
    start=$(date +%s)
    # caffeinate -i prevents the Mac idle-sleeping during the run.
    set +e
    caffeinate -i "$PYTHON" run_simulation.py
    status=$?
    set -e
    end=$(date +%s)

    echo
    echo "=================================================================="
    echo "finished    : $(date)"
    printf 'elapsed     : %dh %dm %ds\n' \
        $(( (end-start)/3600 )) $(( ((end-start)%3600)/60 )) $(( (end-start)%60 ))
    echo "exit status : $status"
    if [ "$status" -eq 0 ]; then
        echo "outputs     : simulation_results.pkl / .csv, simulation_results_timings.csv"
    else
        echo "RUN FAILED  : see the traceback above"
    fi
    echo "=================================================================="
    return $status
}

main 2>&1 | tee "$LOG"
