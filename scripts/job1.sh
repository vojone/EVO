#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/gaus_mut02.json >>"job1$STATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut1.json >>"job1$STATUS" &
python3 filter_cgp.py ../experiments/mut/gaus_mut15.json >>"job1$STATUS" &

python3 filter_cgp.py ../experiments/mut/scratch_mut02.json >>"job1$STATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut1.json >>"job1$STATUS" &
python3 filter_cgp.py ../experiments/mut/scratch_mut15.json >>"job1$STATUS" &

wait
