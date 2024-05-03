#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/lined_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/lined_window_3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/lined_window_3x1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/sp_window_cross.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/sp_window_3d.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/sp_window_3x1.json >>"$MSTATUS" &

wait
