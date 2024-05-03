#!/bin/python

# Script for applying filter trained by CGP to custom image
#
# Author: Vojtěch Dvořák (xdvora3o)
#
# USAGE:
# python filter_cgp.py <config-file-path>

import time
import numpy as np
import cgp
import json
import sys
import os
import csv
import pickle
import common
import multiprocessing as mp
import functools
from datetime import datetime

from PIL import Image
from copy import deepcopy

DATA_PATH = '../data' # Base path of the directory with data
RESULT_PATH = '../results' # Base path of the directory where the results (logs, best-filter etc. will be stored)

JSON_INDENT_SIZE = 4
LOAD_PARAMS_FROM_FILE = True # If true, one commandline argument with experiment cofing JSON is expected, otherwise base_params object is used
LOGGING_ENABLED = True # Deactivates logging and saving the results (good for debugging)


base_params = {
    'name' : 'window-cross', # Name of experiment
    'training_data' : [ # Data for training of the filter
        # Noised img, target img
        ('gaus256/city.jpg', 'target256/city.jpg')
    ],
    'validation_data' : [ # Data for validation (the best filter is applied to these data)
        'gaus256/city.jpg',
        'gaus256/lena.jpg',
        'gaus256/squirrel.jpg'
    ],
    'runs' : 1, # Number of runs (!1 ONLY FOR TESTING!)
    'seeds' : None, # Seeds can be provided manually to reproduce some runs
    'window_shape' : (3, 3), # Sliding window that will be passed to filter (or window_fn and tehn to the filter)
    'window_fn_name' : 'common.window_cross_fn', # Name of function, that preprocesses sliding window (e. g. to some non-rectangular shape)
    'detector_fn_name' : 'common.clipped_detector_fn', # Name of function that converts detector output to boolean value (where True means "Noise detected!")
    'population_params': {'n_parents': 12}, # Population params (see hal-cgp docs for more info)
    'primitives_str': ( # Names of primitives that can be used inside the filter
        'common.Const255',
        'common.Const0',
        'common.Identity',
        'common.Inversion',
        'common.Max',
        'common.ConditionalAssignment',
        'common.Min',
        'common.Div2',
        'common.Div4',
        'common.Add',
        'common.Sub',
        'common.AddS',
        'common.SubS',
        'common.Avg',
    ),
    # Genome, ovlution and algorithm params (see hal-cgp docs for more info)
    'genome_params': {
        'n_inputs': 5,
        'n_outputs': 2,
        'n_columns': 4,
        'n_rows': 9,
        'levels_back': 2
    },
    'evolve_params': {'max_generations': 10, 'termination_fitness': -1.0},
    'algorithm_params': {
        'n_offsprings': 8,
        'mutation_rate': 0.1,
        'tournament_size' : 2,
        'n_processes': 4
    }
}


def serialize_individual(individual : cgp.individual.IndividualBase) -> str:
    """Serializes an individual to the json string.
    """

    return json.dumps({
        'fitness' : individual.fitness,
        'window_fn' : individual.window_fn.__name__ if hasattr(individual, 'window_fn') and individual.window_fn is not None else None,
        'detector_fn' : individual.detector_fn.__name__ if hasattr(individual, 'detector_fn') and individual.detector_fn is not None else None,
        'window_shape' : individual.window_shape if hasattr(individual, 'window_shape') and  individual.window_shape is not None else None,
        'sympy' : str(individual.to_sympy()),
        'genome' : str(individual.genome),
        'active_nodes' : cgp.CartesianGraph(individual.genome).print_active_nodes(),
        'pretty_str' : cgp.CartesianGraph(individual.genome).pretty_str(),
    }, indent=JSON_INDENT_SIZE)


def save_individual(individual : cgp.individual.IndividualBase, path : str):
    """Saves individual in pickle format. Pickled individual can be later use for filtering.
    """

    with open(path, 'wb') as f:
        pickle.dump(individual, f, pickle.HIGHEST_PROTOCOL)


def apply_filter(
    individual : cgp.individual.IndividualBase,
    noised_img_path : str) -> tuple[np.array, np.array]:
    """Uses individual to denoise image specified by the path.
    """

    noised_img = Image.open(noised_img_path)
    noised_img_arr = np.array(noised_img).astype(np.int32)
    img_shape = noised_img_arr.shape

    # Prepare the relevant components of individual
    func = individual.to_func()
    detector_func = individual.detector_fn if hasattr(individual, 'detector_fn') else common.default_detector_fn
    window_fn = individual.window_fn if hasattr(individual, 'window_fn') else None

    # Prepare the input data for faster processing
    window_shape = individual.window_shape if hasattr(individual, 'window_shape') else (3, 3)
    row_padding = (window_shape[0] // 2, window_shape[0] // 2)
    column_padding = (window_shape[1] // 2, window_shape[1] // 2)
    padded_noised_img_arr = np.pad(noised_img_arr, (row_padding, column_padding), 'edge')
    noised_img_arr_window_view = np.lib.stride_tricks.sliding_window_view(padded_noised_img_arr, window_shape)

    if window_fn is None:
        input_data = noised_img_arr_window_view.reshape(
            noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
            window_shape[0] * window_shape[1]
        )
    else:
        input_data = window_fn(noised_img_arr_window_view.reshape(
            noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
            window_shape[0] * window_shape[1]
        ))

    # Apply the filter
    detector_mask = np.zeros_like(noised_img_arr).flatten()
    img = noised_img_arr.flatten()

    for i in range(len(img)):
        detector, pixel = func(*(input_data[i]))
        if detector_func(detector):
            img[i] = pixel % 256
            detector_mask[i] = 255

    return img.reshape(img_shape), detector_mask.reshape(img_shape)


def fitness(
    individual : cgp.individual.IndividualBase,
    detector_fn,
    input_data,
    target_data,
    noised_data) -> float:
    """Compute fitness (MSE between denoised and target image) of an individual.
    """

    def eval_pixel(func, x, y, z):
        """Computes squared error for each pixel in the image.
        """

        detector, pixel = func(*x)
        return (float(pixel % 256) - y)**2 if detector_fn(detector) else (z - y)**2

    func = individual.to_func()
    mse = np.mean([eval_pixel(func, x, float(y), float(z)) for x, y, z in zip(input_data, target_data, noised_data)])

    return mse


def prepare_params(params : dict) -> dict:
    """Converts parameter dicts from input format to the internal format and sets some missing
    values to the default ones.
    """

    internal_params = deepcopy(params)

    if internal_params['seeds'] is None:
        internal_params['seeds'] = np.random.randint(2**32 - 1, size=internal_params['runs'])
    else:
        assert len(internal_params['seeds']) == internal_params['runs']

    internal_params['population_params_'] = [ {
        'seed' : s,
        **internal_params['population_params']
    } for s in internal_params['seeds'] ]

    internal_params['genome_params_'] = {
        'primitives' : tuple([ eval(s) for s in internal_params['primitives_str'] ]),
        **internal_params['genome_params']
    }

    if 'detector_fn_name' not in internal_params:
        internal_params['detector_fn'] = common.default_detector_fn # Default detector fn to be compatible with old configs
    else:
        internal_params['detector_fn'] = eval(internal_params['detector_fn_name'])

    if 'window_fn_name' not in internal_params:
        internal_params['window_fn'] = None
    else:
        internal_params['window_fn'] = eval(internal_params['window_fn_name'])

    return internal_params


def load_training_data(
    data_paths : list[tuple[str, str]],
    params : dict) -> tuple[np.array, np.array, np.array]:
    """Loads training data to the format suitable for filter training.
    """

    target_data = []
    noised_data = []
    input_data = []

    window_fn = params['window_fn'] if 'window_fn' in params else None

    # Load images and convert them to arrays that are suitable for computation of fitness
    for d in data_paths:
        noised_img_path = d[0]
        target_img_path = d[1]

        noised_img = Image.open(os.path.join(DATA_PATH, noised_img_path))
        # Load image as array of 32-bit integers to avoid overflows in filter
        noised_img_arr = np.array(noised_img).astype(np.int32)

        target_img = Image.open(os.path.join(DATA_PATH, target_img_path))
        target_img_arr = np.array(target_img).astype(np.int32)

        # Prepare data to the format for faster processing
        row_padding = (params['window_shape'][0] // 2, params['window_shape'][0] // 2)
        column_padding = (params['window_shape'][1] // 2, params['window_shape'][1] // 2)
        padded_noised_img_arr = np.pad(noised_img_arr, (row_padding, column_padding), 'edge')
        noised_img_arr_window_view = np.lib.stride_tricks.sliding_window_view(padded_noised_img_arr, params['window_shape'])

        if window_fn is None:
            input_data.append(noised_img_arr_window_view.reshape(
                noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
                params['window_shape'][0] * params['window_shape'][1]
            ))
        else:
            input_data.append(window_fn(noised_img_arr_window_view.reshape(
                noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
                params['window_shape'][0] * params['window_shape'][1]
            )))

        target_data.append(target_img_arr.flatten())
        noised_data.append(noised_img_arr.flatten())

    return input_data, target_data, noised_data


def save_log(
    log : dict,
    log_target_path : str,
    resolution : dict = {'t' : 4, 'fitness' : 6}) -> None:
    """Saves the logging dict to CSV file. Resolution (number of decimal places) can be specified
    by the resolution dictionary."""

    def to_formated_str(r, rsln):
        return (r[0], '{:.{prec}f}'.format(r[1], prec=rsln[r[0]])) if r[0] in rsln else (r[0], str(r[1]))

    with open(log_target_path, 'w', newline='') as f:
        # Save also seed of the run (for better reproducibility) and best fitness (to avoid scrolling milions lines to find the best fitness)
        f.write(f"SEED: {log['seed']}\n")
        f.write(f"BEST: {log['best_fitness']}\n")
        w = csv.DictWriter(f, log['evolog'].keys())
        w.writeheader()
        for row in zip(*log['evolog'].values()):
            d = dict([to_formated_str(z, resolution) for z in zip(log['evolog'].keys(), row)])
            w.writerow(d)


def init_log() -> dict:
    """Creates initial dictionary for logging."""

    return {
        '_initial_t' : time.time(),
        'seed' : None,
        'best_fitness' : None,
        'evolog' : {
            't' : [],
            'generation' : [],
            'fitness' : [],
        }
    }


class Objective:
    """Class that represents objective function. Stores reference images."""

    def __init__(self,
        input_data : np.array,
        target_data : np.array,
        noised_data : np.array,
        detector_fn,
        fitness_fn) -> None:

        self.input_data = input_data
        self.target_data = target_data
        self.noised_data = noised_data
        self.detector_fn = detector_fn
        self.fitness_fn = fitness_fn

    def __call__(self, individual : cgp.individual.IndividualBase) -> cgp.individual.IndividualBase:
        """Objective function itself for evolution (we want to maximise the value of fitness)."""

        if not individual.fitness_is_None():
            return individual

        # We will compute mean from fitnesses on all training data (but usually there will be just one image)
        individual.fitness = -np.mean([
            self.fitness_fn(individual, self.detector_fn, i, t, n) for i, t, n in zip(
                self.input_data,
                self.target_data,
                self.noised_data
            )
        ])

        return individual


def run_cgp(
    params : dict,
    run_index : int,
    input_data : np.array,
    target_data : np.array,
    noised_data : np.array,
    enable_logging : bool = True,
    results_path_dir : str = ''):
    """One run of CGP."""

    def logging(pop : cgp.Population):
        """Saves the value of best fitness in every generation."""

        log['evolog']['t'].append(time.time() - log['_initial_t'])
        log['evolog']['generation'].append(pop.generation)
        log['evolog']['fitness'].append(pop.champion.fitness)


    pop = cgp.Population(
        **params['population_params_'][run_index],
        genome_params=params['genome_params_']
    )

    # Although it is a part of EA API we will use it for CGP, it works similarly, see docs for more info
    alg = cgp.ea.MuPlusLambda(**params['algorithm_params'])

    # Initialization of values in log
    log = init_log()
    log['seed'] = params['population_params_'][run_index]['seed']

    # Start the evolution of filter
    cgp.evolve(
        Objective(input_data, target_data, noised_data, params['detector_fn'], fitness),
        pop,
        alg,
        **params_['evolve_params'],
        print_progress=True,
        callback=logging
    )

    # Save some settings directly in the individual
    log['best_fitness'] = pop.champion.fitness
    pop.champion.detector_fn = params['detector_fn']
    pop.champion.window_shape = params['window_shape']
    pop.champion.window_fn = params['window_fn']

    if enable_logging:
        save_log(log, os.path.join(results_path_dir, f"it{run_index}.csv"))

    return pop.champion



def load_params(params_path : str) -> dict:
    """Loads params from .json and creates dictionary with params."""
    with open(params_path, 'r') as f:
        return json.load(f)



if __name__ == '__main__':
    if len(sys.argv) < 2 and LOAD_PARAMS_FROM_FILE:
        print('Error: Missing config path!')
        print(f'USAGE: python {sys.argv[0]} <config-path>')
        exit(1)

    params = load_params(sys.argv[1]) if LOAD_PARAMS_FROM_FILE else base_params

    timestamp = datetime.now().strftime('%d-%m-%H-%M')
    params_ = prepare_params(params)

    input_data, target_data, noised_data = load_training_data(params_['training_data'], params_)

    # Create directory for storing logs and results
    results_path_dir = os.path.join(RESULT_PATH, f"{params['name']}-{timestamp}")
    if LOGGING_ENABLED:
        os.makedirs(results_path_dir)

    best_champion : cgp.individual.IndividualBase = None
    # Perform more iteration of CGP with given config to have some significant results
    for i in range(params_['runs']):
        print(f'i: {i}')
        champion = run_cgp(
            params_,
            i,
            input_data,
            target_data,
            noised_data,
            enable_logging=LOGGING_ENABLED,
            results_path_dir=results_path_dir
        )

        if best_champion is None or best_champion.fitness < champion.fitness:
            best_champion = champion


    # Save the results
    if LOGGING_ENABLED:
        # Save the best champion
        save_individual(best_champion, os.path.join(results_path_dir, f"best-filter.pkl"))
        with open(os.path.join(results_path_dir, f"best-filter.json"), 'w') as f:
            f.write(serialize_individual(best_champion))

        # Save the input params
        with open(os.path.join(results_path_dir, f"params.json"), 'w') as f:
            f.write(json.dumps(params, indent=JSON_INDENT_SIZE))

        # Apply the best filter to the validation data
        for img_path in params['validation_data']:
            img, mask = apply_filter(best_champion, os.path.join(DATA_PATH, img_path))

            img_base_name = os.path.basename(img_path)
            img_pil = Image.fromarray(img.astype(np.uint8))
            mask_pil = Image.fromarray(mask.astype(np.uint8))

            img_pil.save(os.path.join(results_path_dir, f'{img_base_name.split(".")[0]}-denoised.jpg'))
            mask_pil.save(os.path.join(results_path_dir, f'{img_base_name.split(".")[0]}-mask.jpg'))


