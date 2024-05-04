#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/just-noises/line-color.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/line-color_.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/just-noises/scratch_old.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/scratch_old_.json >>"$MSTATUS" &

wait
