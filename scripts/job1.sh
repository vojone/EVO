#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/just-noises/line-color.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/structured_window3x3.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/universal-filter/structured_window5x5.json >>"$MSTATUS" &

wait
