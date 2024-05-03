#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/gaus_mut02.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut1.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut15.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/mut/scratch_mut02.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut1.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut15.json >>"$MSTATUS" &

wait
