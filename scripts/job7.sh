#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_5x5_long.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_5x5_long1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_5x5_long2.json >>"$MSTATUS" &

wait
