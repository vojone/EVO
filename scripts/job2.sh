#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/just-noises/gaus.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/line.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/lined.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/just-noises/scratch_old.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/scratch.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/sp.json >>"$MSTATUS" &

wait
