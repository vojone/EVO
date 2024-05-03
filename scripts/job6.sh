#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/window/lined_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/lined_window3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/lined_window3x1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/window/sp_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/sp_window3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/window/sp_window3x1.json >>"$MSTATUS" &

wait
