#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/gaus_mut05.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut1.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut15.json >>"$STATUS" &

python3 filter_cgp.py ../experiments/mut/scratch_mut05.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut1.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut15.json >>"$STATUS" &

wait
