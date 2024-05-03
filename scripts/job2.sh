#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/lined_mut02.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/lined_mut1.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/lined_mut15.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/mut/sp_mut02.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/sp_mut1.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/mut/sp_mut15.json >>"$MSTATUS" &

wait
