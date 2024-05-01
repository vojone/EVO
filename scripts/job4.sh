#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/size/scratch_2x10.json &
python3 filter_cgp.py ../experiments/size/scratch_5x5.json &
python3 filter_cgp.py ../experiments/size/scratch_9x6.json &

python3 filter_cgp.py ../experiments/size/sp_2x10.json &
python3 filter_cgp.py ../experiments/size/sp_5x5.json &
python3 filter_cgp.py ../experiments/size/sp_9x6.json &

wait