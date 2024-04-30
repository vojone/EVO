#!/bin/python

# Script for preparing images to CGP of noise filter
#
# Author: Vojtěch Dvořák (xdvora3o)
#
#
# USAGE:
# python prepare_data.py <src_img> <dst_img> <size> <noise> <noise_intensity>
#
# Following types of noises are supported:
#
# gaus - Additive gaussian noise (random vlaue from normal distribution is added to some pixels)
# sp - Salt and pepper noise (some pixels are set to 0, some of them to 255)
# line - Random horizontal white lines
# linev - Random vertical white lines
# lined - Random vertical diagonal lines
# scratch - Random bezier curves

import sys
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as rotate_image

def adjust_size(img : Image.Image, new_size : tuple[int, int]):
    """Auxiliary function for adjusting size of the image."""

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


def add_gaussian_noise(img : Image.Image, noised_pixels_num : int):
    """Adds salt and gaussian noise to the image."""

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


def add_salt_and_pepper(img : Image.Image, noised_pixels_num : int):
    """Adds salt and pepper noise (S&P) to the image."""

    img_size, _ = img.size
    img_array = np.array(img)

    noise_indeces = np.random.choice(np.arange(img_size * img_size), noised_pixels_num, replace=False)

    salt_indeces, pepper_indeces = tuple(np.array_split(noise_indeces, 2))
    img_array.flat[salt_indeces] = 255
    img_array.flat[pepper_indeces] = 0

    noised_img = Image.fromarray(img_array.astype(np.uint8))

    return noised_img


def add_line_noise(img : Image.Image, noised_pixels_num : int, rotation : int = 0, line_width : int = 1):
    """Adds random lines to image. Their rotation and width may be chosen by parameters."""

    line_num = int(noised_pixels_num * 0.2) # Noised pixel number is approximated (top limit) if some non-zero rotation is used
    img_size, _ = img.size

    img = rotate_image(img, rotation)
    new_img_size, _ = img.shape
    img_array = np.array(img)

    noise_indeces = np.random.choice(np.arange(new_img_size * new_img_size), line_num, replace=False)
    remaining_noise_pixels = noised_pixels_num - line_num
    for i in noise_indeces:
        r, c = np.unravel_index(i, (new_img_size, new_img_size))
        line_len = np.random.randint(0, new_img_size - c)
        remaining_noise_pixels -= line_len * line_width

        if remaining_noise_pixels <= 0:
            break

        img_array[r:r + line_width, c:(c + line_len)] = 255

    noised_img = rotate_image(img_array, -rotation)
    noised_img = Image.fromarray(noised_img.astype(np.uint8))

    width, height = noised_img.size

    # Make centered crop to adjust the size after rotation
    left = (width - img_size) * 0.5
    top = (height - img_size) * 0.5
    right = (width + img_size) * 0.5
    bottom = (height + img_size) * 0.5

    noised_img = noised_img.crop((left, top, right, bottom))

    return noised_img


# Following code was taken from https://stackoverflow.com/a/76760722
# and it was edited to be suitable for this project
def add_scratch(img : Image.Image, noised_pixels_num : int):
    """Adds random bezier curves (scratches) to the image."""

    from collections.abc import Generator
    import cv2 as cv

    def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
        def calc(t):
            return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

        # get the approximate pixel count of the curve
        approx = cv.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
        for t in np.linspace(0, 1, round(approx * 1.2)):
            yield np.round(calc(t)).astype(np.int32)


    def generate_scratch(img: np.ndarray, max_length: float) -> np.ndarray:
        H, W = img.shape
        # generate the 2 end points of the bezier curve
        x, y, rho1, theta1 = np.random.uniform([0] * 4, [W, H, max_length, np.pi * 2])
        p1 = np.array([x, y, 0])
        p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]

        # generate the second point, make sure that it cannot be too far away from the middle point of the 2 end points
        rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
        p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

        for x, y, brush in bezier(p1, p2, p3):
            cv.circle(img, (x, y), brush, 255, -1)
        return img

    SCRATCH_CNT = int(noised_pixels_num * 0.1) # Noised pixel number is approximated (top limit) if some non-zero rotation is used
    MAX_LENGTH = int((noised_pixels_num * 0.9) / SCRATCH_CNT)

    img = np.array(img, np.uint8)
    for _ in range(SCRATCH_CNT):
        generate_scratch(img, MAX_LENGTH)

    return Image.fromarray(img.astype(np.uint8))
# End of part taken from https://stackoverflow.com/a/76760722



def to_gray(img : Image.Image):
    """Converts an image to grayscale."""

    return img.convert('L')


if __name__ == '__main__':
    try:
        img = Image.open(sys.argv[1])
        new_img_path = sys.argv[2]
        img_size = int(sys.argv[3])
        noise_type = sys.argv[4]
        noised_pixels = float(sys.argv[5])
    except:
        print('USAGE:')
        print('python prepare_data.py <src_img> <dst_img> <size> <noise> <noise_intensity>')
        print()
        print('Where <noise> is one of:')
        print('gaus - Additive gaussian noise')
        print('sp - Salt and pepper noise')
        print('line - Random horizontal white lines')
        print('linev - Random vertical white lines')
        print('lined - Random vertical diagonal lines')
        print('scratch - Random bezier curves')
        print()
        exit(1)

    img = adjust_size(img, (img_size, img_size))
    img = to_gray(img)

    noised_pixels_num = int(noised_pixels * (img_size * img_size))

    noised_img = img
    if noise_type == 'sp':
        noised_img = add_salt_and_pepper(img, noised_pixels_num)
    elif noise_type == 'gaus':
        noised_img = add_gaussian_noise(img, noised_pixels_num)
    elif noise_type == 'line':
        noised_img = add_line_noise(img, noised_pixels_num)
    elif noise_type == 'linev':
        noised_img = add_line_noise(img, noised_pixels_num, rotation=90)
    elif noise_type == 'lined':
        noised_img = add_line_noise(img, noised_pixels_num, rotation=40)
    elif noise_type == 'scratch':
        noised_img = add_scratch(img, noised_pixels_num)

    noised_img.save(new_img_path)
