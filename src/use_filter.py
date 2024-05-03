#!/bin/python

# Script for applying filter trained by CGP to custom image
#
# Author: Vojtěch Dvořák (xdvora3o)


import filter_cgp
import cgp
import sys
import pickle

import numpy as np

import os
from PIL import Image


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Error: Missing arguments!')
        print(f'USAGE: python {sys.argv[0]} <filter-pkl-path> <noised-img-path>')
        exit(1)

    filter_pkl_path = sys.argv[1]
    img_path = sys.argv[2]

    img_filter = None
    with open(filter_pkl_path, 'rb') as f:
        img_filter = pickle.load(f)

    img, mask = filter_cgp.apply_filter(img_filter, img_path)

    img_base_name = os.path.basename(img_path)
    Image.fromarray(img.astype(np.uint8)).save(f'{img_base_name}-denoised.jpg')
    Image.fromarray(mask.astype(np.uint8)).save(f'{img_base_name}-mask.jpg')

