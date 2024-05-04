#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_9x10_long.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_9x10_long1.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/size/gaus_9x10_long2.json >>"$MSTATUS" &

wait
