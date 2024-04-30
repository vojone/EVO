#!/bin/python

# Script for preparing images to CGP of noise filter
#
# Author: Vojtěch Dvořák (xdvora3o)
#
# USAGE:
# python prepare_data.py <src_img> <dst_img> <size> <noise> <noise_intensity>

import sys
import numpy as np
from PIL import Image

def adjust_size(img : Image, new_size : tuple[int, int]):
    width, height = img.size

    if width != height:
        smaller = width if width < height else height

        # Make centered crop
        left = (width - smaller) * 0.5
        top = (height - smaller) * 0.5
        right = (width + smaller) * 0.5
        bottom = (height + smaller) * 0.5

        img = img.crop((left, top, right, bottom))

    img = img.resize(new_size)

    return img


def add_gaussian_noise(img : Image, noised_pixels_num : int):
    mean = 0.0
    std_dev = 100.0

    img_size, _ = img.size
    img_array = np.array(img)

    noise_mask = np.zeros((img_size, img_size))
    indices = np.random.choice(np.arange(img_size * img_size), noised_pixels_num, replace=False)
    noise_mask.flat[indices] = 1

    noise = np.random.normal(mean, std_dev, (img_size, img_size))

    noised_img_array = (img_array + noise * noise_mask) % 256

    noised_img = Image.fromarray(noised_img_array.astype(np.uint8))

    return noised_img


def add_salt_and_pepper(img : Image, noised_pixels_num : int):
    img_size, _ = img.size
    img_array = np.array(img)

    noise_indeces = np.random.choice(np.arange(img_size * img_size), noised_pixels_num, replace=False)

    salt_indeces, pepper_indeces = tuple(np.array_split(noise_indeces, 2))
    img_array.flat[salt_indeces] = 255
    img_array.flat[pepper_indeces] = 0

    noised_img = Image.fromarray(img_array.astype(np.uint8))

    return noised_img


def add_line_noise(img : Image, noised_pixels_num : int, line_width : int = 1):
    line_num = int(noised_pixels_num * 0.2)

    img_size, _ = img.size
    img_array = np.array(img)

    noise_indeces = np.random.choice(np.arange(img_size * img_size), line_num, replace=False)
    remaining_noise_pixels = noised_pixels_num - line_num
    for i in noise_indeces:
        r, c = np.unravel_index(i, (img_size, img_size))
        line_len = np.random.randint(0, img_size - c)
        remaining_noise_pixels -= line_len * line_width

        if remaining_noise_pixels <= 0:
            break

        img_array[r:r + line_width, c:(c + line_len)] = 255

    noised_img = Image.fromarray(img_array.astype(np.uint8))

    return noised_img


def to_gray(img : Image):
    return img.convert('L')


if __name__ == '__main__':
    img = Image.open(sys.argv[1])
    new_img_name = sys.argv[2]
    img_size = int(sys.argv[3])
    noise_type = sys.argv[4]
    noised_pixels = float(sys.argv[5])

    img = adjust_size(img, (img_size, img_size))
    img = to_gray(img)

    noised_pixels_num = int(noised_pixels * (img_size * img_size))

    noised_img = add_line_noise(img, noised_pixels_num)

    noised_img.save(new_img_name)
