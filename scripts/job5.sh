#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/window/gaus_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/gaus_window3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/gaus_window3x1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/window/line_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/line_window3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/line_window3x1.json >>"$MSTATUS" &

wait
