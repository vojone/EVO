#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/gaus_2x10.json >>"job3$STATUS" &
python3 filter_cgp.py ../experiments/size/gaus_5x5.json >>"job3$STATUS" &
python3 filter_cgp.py ../experiments/size/gaus_9x6.json >>"job3$STATUS" &

python3 filter_cgp.py ../experiments/size/lined_2x10.json >>"job3$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_5x5.json >>"job3$STATUS" &
python3 filter_cgp.py ../experiments/size/lined_9x6.json >>"job3$STATUS" &

wait
