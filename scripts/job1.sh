#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/universal-filter/structured_window3x3.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/structured_window3x31.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/structured_window3x32.json >>"$MSTATUS" &

wait
