#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/universal-filter/all_window3x3.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/universal-filter/all_window3x3_.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/window3x3.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/universal-filter/window3x3_.json >>"$MSTATUS" &

wait
