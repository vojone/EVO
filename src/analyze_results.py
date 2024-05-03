#!/bin/python

# Script for more simple analysis of the experiment results
#
# Author: Vojtěch Dvořák (xdvora3o)
#
# USAGE:
# python analyze_results.py <path(s)-to-results-dirs>

import csv
import sys
import cgp
import json
import os
import pickle
import matplotlib.pyplot as plt
from glob import glob

SKIP_LINES_CSV = 2
SKIP_COLUMN_CSV = ['generation']

result_paths = sys.argv[1:]


def load_results_(result_path : str):
    """Load everything important from the result directory to the one dictionary.
    """

    result = {
        'path' : result_path,
        'params': None,
        'logs' : [],
        'fitnesses' : [],
        'durations' : [],
        'best_fitness' : None,
        'best_filter' : None,
        'best_filter_strs' : None,
    }

    print(f'Loading result stored in {result_path}', file=sys.stderr)

    # Load json with experiment description
    params_paths = glob(os.path.join(result_path, '*params.json'))
    if len(params_paths) > 0:
        print(f'* Params', file=sys.stderr)
        with open(params_paths[0], 'r') as f:
            result['params'] = json.load(f)
    else:
        print(f'X Params not found', file=sys.stderr)

    # Load json with description of the best filter, found during the experiment
    filter_json_paths = glob(os.path.join(result_path, '*-filter.json'))
    if len(filter_json_paths) > 0:
        print(f'* Best filter json description', file=sys.stderr)
        with open(filter_json_paths[0], 'r') as f: # Take only the first -filter.json file
            result['best_filter_strs'] = json.load(f)
            result['best_fitness'] = result['best_filter_strs']['fitness']
    else:
        print(f'X Best filter json description not found', file=sys.stderr)

    # Load pickled the best filter object
    filter_pkl_paths = glob(os.path.join(result_path, '*-filter.pkl'))
    if len(filter_pkl_paths) > 0:
        print(f'* Best filter pickled object', file=sys.stderr)
        with open(filter_pkl_paths[0], 'rb') as f: # Take only the first .pkl file (I store just one pkl file in every experiment)
            result['best_filter'] = pickle.load(f)
    else:
        print(f'X Best filter pickled object not found', file=sys.stderr)

    # Load data from the every run (ther eshould be always at least 30 runs)
    log_paths = glob(os.path.join(result_path, '*.csv'))
    if len(log_paths) > 0:
        print(f'* Logs from {len(log_paths)} runs', file=sys.stderr)
        for glog_path in log_paths:
            log = {}
            with open(glog_path, 'r') as csvlog:
                for _ in range(2):  # Assuming there are some non-CSV lines at the beginning (e .g with SEED and best fitness)
                    next(csvlog)

                reader = csv.reader(csvlog)
                header = next(reader)
                for row in reader:
                    for column, value in zip(header, row):
                        if column not in SKIP_COLUMN_CSV: # Skip some columns if needed
                            log.setdefault(column, []).append(value)

            result['logs'].append(log)
            result['durations'].append(log['t'][-1] if 't' in log else None)
            result['fitnesses'].append(log['fitness'][-1] if 'fitness' in log else None)
    else:
        print(f'X No log was found', file=sys.stderr)

    return result


def load_results(result_paths : list[str]):
    """Load everything important from the result directory to the one dictionary.
    """

    results = []
    for path in result_paths:
        for gpath in glob(path):
            result = load_results_(gpath)
            results.append(result)

    return results


def plot_boxplot():
    pass

def plot_conv_lines():
    pass



if __name__ == '__main__':
    print(load_results(result_paths))
