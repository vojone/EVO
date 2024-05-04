#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/universal-filter/all_window5x5.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/universal-filter/all_window5x5_.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/window5x5.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/universal-filter/window5x5_.json >>"$MSTATUS" &

wait
