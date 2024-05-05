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
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob

SKIP_LINES_CSV = 2
SKIP_COLUMN_CSV = ['generation']



def parse_column(log : dict, column : str, value : str):
    if column == 't':
        log.setdefault(column, []).append(float(value))
    elif column == 'fitness':
        log.setdefault(column, []).append(-float(value))
    else:
        log.setdefault(column, []).append(value)

    return log


def parse_initial_lines(log : dict, i : int, line : str):
    if i == 0:
        log['seed'] = int(line.split(':')[1].strip())

    return log



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
    log_cnt = 0
    if len(log_paths) > 0:
        print(f'* Logs from {len(log_paths)} runs', file=sys.stderr)
        for glog_path in log_paths:
            log = { 'path' : os.path.basename(glog_path).split('.')[0], 'idx' : log_cnt }
            with open(glog_path, 'r') as csvlog:
                for i in range(SKIP_LINES_CSV):  # Assuming there are some non-CSV lines at the beginning (e .g with SEED and best fitness)
                    l = next(csvlog)
                    log = parse_initial_lines(log, i, l)

                reader = csv.reader(csvlog)
                header = next(reader)
                for row in reader:
                    for column, value in zip(header, row):
                        if column not in SKIP_COLUMN_CSV: # Skip some columns if needed
                            log = parse_column(log, column, value)

            result['logs'].append(log)
            result['durations'].append(float(log['t'][-1]) if 't' in log else None)
            result['fitnesses'].append(float(log['fitness'][-1]) if 'fitness' in log else None)
            log_cnt += 1
    else:
        print(f'X No log was found', file=sys.stderr)

    return result


def load_results(result_paths : list[str]):
    """Load everything important from the result directory to the one dictionary.
    """

    results = []
    cnt = 0
    for path in result_paths:
        for gpath in glob(path):
            result = load_results_(gpath)
            result['idx'] = cnt
            results.append(result)
            cnt += 1

    return results


def to_dataframes(results : list[dict]):
    results_df = pd.DataFrame({
        'i' : [ r['idx'] for r in results ],
        'name' : [ r['params']['name'] for r in results ],
        'path' : [ r['path'] for r in results ],
        'best-fitness' : [ r['best_fitness'] for r in results ],
        'run-count' : [ len(r['logs']) for r in results ],
    }, index=[ r['idx'] for r in results ])

    logs_dfs = []
    generation_dfs = []
    for r in results:
        rdf = pd.DataFrame({
            'duration' : r['durations'],
            'fitness' : r['fitnesses'],
        }, index=[ l['idx'] for l in r['logs'] ])

        for l in r['logs']:
            gdf = pd.DataFrame({
                'experiment-i' : r['idx'],
                'run-i' : l['idx'],
                'g' : range(0, len(l['fitness'])),
                't' : l['t'],
                'fitness' : l['fitness'],
            }, index=[ l['idx'] ] * len(l['fitness']))

            generation_dfs.append(gdf)

        rdf['experiment-i'] = r['idx']
        rdf['experiment-name'] = r['params']['name']

        logs_dfs.append(rdf)

    logs_df = pd.concat(logs_dfs)
    generation_df = pd.concat(generation_dfs)

    return results_df, logs_df, generation_df


def filter_experiments(experiments : pd.DataFrame, filter_regex : str, skip_filtered : bool = False):
    filter_result = experiments['name'].str.contains(filter_regex, regex=True)

    if skip_filtered:
        return experiments.loc[~filter_result, :]
    else:
        return experiments.loc[filter_result, :]



def plot_run_data_boxplot(
    experiments : pd.DataFrame,
    runs : pd.DataFrame,
    data_name,
    filter_regex = None,
    skip_filtered = False,
    ax = None,
    x = None,
    plot_params : dict = {}):

    experiments_to_be_plot = experiments if filter_regex is None else filter_experiments(experiments, filter_regex, skip_filtered)
    # print(runs)
    # print(experiments_to_be_plot)

    merged = runs.merge(experiments_to_be_plot, right_on='i', left_on='experiment-i')
    #print(merged)

    if ax is None:
        _, ax = plt.subplots()

    sns.boxplot(merged, x=('experiment-name' if x is None else x), y=data_name, ax=ax, **plot_params)
    ax.set_ylabel(data_name)
    ax.set_xlabel('Experiment')
    ax.tick_params(axis='x', labelrotation=45)

    return ax


def image_mse(image1, image2):
    """Computes MSE similarly as it is computed in filter_cgp.py.
    """

    if image1.shape != image2.shape:
        raise ValueError("Images does not have the same dimensions!")
    
    def eval_pixel(x, y):
        """Computes squared error for each pixel in the image."""
        return (float(x) - float(y))**2

    mse = np.mean([eval_pixel(x, y) for x, y in zip(image1.flatten(), image2.flatten())])
    return mse


def mse_with_target(target_image_path : str, images_paths : list[str]):
    target_image = np.array(Image.open(target_image_path))

    paths = []
    mses = []
    for path in images_paths:
        for gpath in glob(path):
            img = np.array(Image.open(gpath))

            mses.append(image_mse(target_image, img))
            paths.append(gpath)

    return pd.DataFrame({'image' : paths, 'mse' : mses})


def plot_run_data_hist(
    experiments : pd.DataFrame,
    runs : pd.DataFrame,
    data_name,
    filter_regex = None,
    skip_filtered = False,
    ax = None,
    plot_params : dict = {}):

    experiments_to_be_plot = experiments if filter_regex is None else filter_experiments(experiments, filter_regex, skip_filtered)
    # print(runs)
    # print(experiments_to_be_plot)

    merged = runs.merge(experiments_to_be_plot, right_on='i', left_on='experiment-i')
    #print(merged)

    if ax is None:
        _, ax = plt.subplots()

    sns.histplot(merged, x=data_name, ax=ax, binwidth=50, **plot_params)
    ax.set_ylabel(data_name)
    ax.set_xlabel('Experiment')
    ax.tick_params(axis='x', labelrotation=45)

    return ax


def plot_conv_lines(
    experiments : pd.DataFrame,
    generations : pd.DataFrame,
    filter_regex = None,
    skip_filtered = False,
    ax = None,
    plot_params : dict = {}):

    experiments_to_be_plot = experiments if filter_regex is None else filter_experiments(experiments, filter_regex, skip_filtered)
    # print(runs)
    # print(experiments_to_be_plot)

    merged = generations.merge(experiments_to_be_plot, right_on='i', left_on='experiment-i')
    #print(merged)

    if ax is None:
        _, ax = plt.subplots()

    sns.lineplot(data=merged, x='g', y='fitness', hue='name', ax=ax, **plot_params)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generation")

    return ax


def get_active_nodes_df(results : list[dict]):
    active_nodes_df = pd.DataFrame({
        'i' : [ r['idx'] for r in results ],
        'name' : [ r['params']['name'] for r in results ]
    })

    active_nodes = []
    for r in results:
        graph = cgp.CartesianGraph(r['best_filter'].genome)
        nodes = len([node for node in graph._nodes if node._active and type(node) != cgp.node_input_output.OutputNode and type(node) != cgp.node_input_output.InputNode])
        active_nodes.append(nodes)

    active_nodes_df['active_nodes'] = active_nodes

    return active_nodes_df


def get_pretty_str(result : dict):
    graph = cgp.CartesianGraph(result['best_filter'].genome)
    return graph.pretty_str()


if __name__ == '__main__':
    result_paths = sys.argv[1:]

    results = load_results(result_paths)
    experiments, runs, generations = to_dataframes(results)

    # plot_run_data_boxplot(experiments, runs, 'fitness', filter_regex=r'gaus')
    # plt.savefig('fitness-boxplot.png')

    # plot_run_data_hist(experiments, runs, 'fitness', filter_regex=r'gaus')
    # plt.savefig('fitness-hist.png')

    # ax = plot_conv_lines(experiments, generations, filter_regex=r'2x10', skip_filtered=True,
    #                      plot_params={ 'palette' : ['r', 'g', 'c', 'm'], 'estimator' : 'median', 'errorbar' : None})
    # plot_conv_lines(experiments, generations, filter_regex=r'2x10',
    #                 plot_params={ 'estimator' : 'median',  'errorbar' : ("pi", 25)}, ax=ax)
    # plt.savefig('fitness-convergence.png')

    print(get_active_nodes_df(results))
    print(get_pretty_str(results[0]))
    print(mse_with_target('../data/target256/lena.jpg', ['../data/*256/lena.jpg']))
