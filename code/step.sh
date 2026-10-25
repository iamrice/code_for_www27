#!/bin/zsh
# usage: step.sh RUN ROUND   -> ingest response_r{ROUND}, write prompt_r{ROUND+1}
cd "${0:A:h}"; export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
python3 loop.py ingest --run $1 --round $2 --resp ../design_runs/$1/response_r$2.md 2>&1 | cut -c1-230
python3 loop.py prompt --run $1 --round $(($2+1)) > /dev/null
