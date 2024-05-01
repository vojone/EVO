import time
import numpy as np
import cgp
import json
import sys
import os

from PIL import Image
from copy import deepcopy

DATA_PATH = '.'

DETECTOR_FN = lambda d : d > 0

params = {
    'training_data' : [
        # Noised img, target img
        ('new_city.jpg', 'new_city_target.jpg')
    ],
    'validation_data' : [
        'gaussian/city_square_monument_wonderful.jpg'
    ],
    'iteration' : 4,
    'seeds' : [np.random.randint(2**32 - 1), np.random.randint(2**32 - 1), np.random.randint(2**32 - 1), np.random.randint(2**32 - 1)],
    'window_shape' : (3, 3),
    'population_params': {'n_parents': 10},
    'primitives_str': ('cgp.Add','cgp.Sub'),
    'genome_params': {
        'n_inputs': 9,
        'n_outputs': 2,
        'n_columns': 5,
        'n_rows': 5,
        'levels_back': 2
    },
    'evolve_params': {'max_generations': 4, 'termination_fitness': -1e-12},
    'algorithm_params': {
        'n_offsprings': 4,
        'mutation_rate': 0.3,
        'tournament_size' : 4,
        'n_processes': 1
    }
}

log = {
    '_initial_t' : time.time(),
    't' : [],
    'generation' : [],
    'fitness' : [],
}



def serialize_inidividual(individual : cgp.individual.IndividualBase):
    return json.dumps({
        'fitness' : individual.fitness,
        'sympy' : individual.to_sympy(),
        'genome' : individual.genome,
        'active_nodes' : cgp.CartesianGraph(individual.genome).print_active_nodes(),
        'pretty_str' : cgp.CartesianGraph(individual.genome).pretty_str(),
    })



def apply_filter(individual : cgp.individual.IndividualBase, noised_img_path : str):
    noised_img = Image.open(noised_img_path)
    noised_img_arr = np.array(noised_img) / 256
    img_shape = noised_img_arr.shape

    padded_noised_img_arr = np.pad(noised_img_arr, 1, 'edge')
    noised_img_arr_window_view = np.lib.stride_tricks.sliding_window_view(padded_noised_img_arr, params['window_shape'])

    input_data = noised_img_arr_window_view.reshape(
        noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
        params['window_shape'][0],
        params['window_shape'][1]
    )

    func = individual.to_numpy()

    detector_mask = np.zeros_like(noised_img_arr).flatten()
    img = noised_img_arr.flatten()
    for i in range(len(img)):
        detector, pixel = func(*(input_data[i].flatten()))
        if DETECTOR_FN(detector):
            img[i] = pixel
            detector_mask[i] = 255

    return img.reshape(img_shape) * 256, detector_mask.reshape(img_shape)



def fitness(individual : cgp.individual.IndividualBase, input_data, target_data, noised_data):
    def eval_pixel(func, x, y, z):
        """Computes squqred error for each pixel in the image."""

        detector, pixel = func(*(x.flatten()))
        return (float(pixel) - float(y))**2 if DETECTOR_FN(detector) else (float(z) - float(y))**2

    func = individual.to_func()
    mse = np.mean([eval_pixel(func, x, y, z) for x, y, z in zip(input_data, target_data, noised_data)])

    return mse


def prepare_params(params : dict):
    internal_params = deepcopy(params)

    if internal_params['seeds'] is None:
        internal_params['seeds'] = np.random.randint(2**32 - 1, size=internal_params['iteration'])
    else:
        assert len(internal_params['seeds']) == internal_params['iteration']

    internal_params['population_params_'] = [ {
        'seed' : s,
        **internal_params['population_params']
    } for s in internal_params['seeds'] ]

    internal_params['genome_params_'] = {
        'primitives' : [ eval(s) for s in internal_params['primitives_str'] ],
        **internal_params['genome_params']
    }

    return internal_params


def load_training_data(data_paths : list[tuple[str, str]]):
    target_data = []
    noised_data = []
    input_data = []

    # Load images and convert them to arrays that are suitable for computation of fitness
    for d in data_paths:
        noised_img_path = d[0]
        target_img_path = d[1]

        noised_img = Image.open(os.path.join(DATA_PATH, noised_img_path))
        noised_img_arr = np.array(noised_img) / 256 # Normalization of values to [0, 1] to avoid overflow errors

        target_img = Image.open(os.path.join(DATA_PATH, target_img_path))
        target_img_arr = np.array(target_img) / 256

        padded_noised_img_arr = np.pad(noised_img_arr, 1, 'edge')
        noised_img_arr_window_view = np.lib.stride_tricks.sliding_window_view(padded_noised_img_arr, params['window_shape'])

        input_data.append(noised_img_arr_window_view.reshape(
            noised_img_arr_window_view.shape[0] * noised_img_arr_window_view.shape[1],
            params['window_shape'][0],
            params['window_shape'][1]
        ))

        target_data.append(target_img_arr.flatten())
        noised_data.append(noised_img_arr.flatten())

    return input_data, target_data, noised_data


if __name__ == '__main__':
    params_ = prepare_params(params)

    input_data, target_data, noised_data = load_training_data(params_['training_data'])

    def objective(individual : cgp.individual.IndividualBase):
        if not individual.fitness_is_None():
            return individual

        # We will compute mean from fitnesses on all training data (but usually there will be just one image)
        individual.fitness = np.mean([
            -fitness(individual, i, t, n) for i, t, n in zip(input_data, target_data, noised_data)
        ])

        return individual


    def logging(pop : cgp.Population):
        log['t'].append(time.time() - log['_initial_t'])
        log['generation'].append(pop.generation)
        log['fitness'].append(pop.champion.fitness)


    pop = cgp.Population(**params_['population_params'], genome_params=params_['genome_params'])

    # Although it is a part of EA API we will use it for CGP, it works similarly, see docs for more info
    alg = cgp.ea.MuPlusLambda(**params_['algorithm_params'])

    # Start
    cgp.evolve(
        objective, # Lambda would be better but there is "Can't pickle" error
        pop,
        alg,
        **params_["evolve_params"],
        print_progress=True,
        callback=logging
    )


    print(f"evolved function: {pop.champion}")
    img, mask = apply_filter(pop.champion, '../data/gaussian/city_square_monument_wonderful.jpg')
    Image.fromarray(img.astype(np.uint8)).save('denoised.jpg')
    Image.fromarray(mask.astype(np.uint8)).save('mask.jpg')


    print(json.dumps(params))
