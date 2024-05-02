#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_2x10.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/size/gaus_5x5.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/size/gaus_9x6.json >>"$STATUS" &

python3 filter_cgp.py ../experiments/size/lined_2x10.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_5x5.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_9x6.json >>"$STATUS" &

wait
