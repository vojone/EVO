#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/window/gaus_window5x5.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/window/line_window5x5.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/window/lined_window5x5.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/window/sp_window5x5.json >>"$MSTATUS" &

wait
