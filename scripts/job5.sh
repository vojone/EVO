#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/just-noises/scratch.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/scratch_.json >>"$MSTATUS" &

python3 filter_cgp.py ../experiments/just-noises/sp.json >>"$MSTATUS" &
python3 filter_cgp.py ../experiments/just-noises/sp_.json >>"$MSTATUS" &

wait
