#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_4x3_long.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/gaus_5x5_long.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/size/gaus_9x10_long.json >>"$MSTATUS" &

wait
