#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_4x3_long.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_4x3_long1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_4x3_long2.json >>"$MSTATUS" &

wait
