#!/bin/bash

. ./bootstrap.sh

pushd src || exit

python3 filter_cgp.py ../experiments/mut/sp_mut05.json &
python3 filter_cgp.py ../experiments/mut/sp_mut1.json &
python3 filter_cgp.py ../experiments/mut/sp_mut15.json &
