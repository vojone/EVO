#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/lined_mut05.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/lined_mut1.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/lined_mut15.json >>"$STATUS" &

python3 filter_cgp.py ../experiments/mut/sp_mut05.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/sp_mut1.json >>"$STATUS" &
python3 filter_cgp.py ../experiments/mut/sp_mut15.json >>"$STATUS" &

wait
