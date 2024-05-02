#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/lined_window_cross.json >>"job6$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_window_3d.json >>"job6$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_window_3x1.json >>"job6$STATUS" &

python3 filter_cgp.py ../experiments/size/sp_window_cross.json >>"job6$STATUS" &
python3 filter_cgp.py ../experiments/size/sp_window_3d.json >>"job6$STATUS" &
python3 filter_cgp.py ../experiments/size/sp_window_3x1.json >>"job6$STATUS" &

wait
