#!/bin/bash

# Script for automatic noising images to use them for CGP of noise filter
#
# Author: Vojtěch Dvořák (xdvora3o)


DATA_SIZE=256
DATA_DIR="../data"
ORIG_DATA_DIR="../data/original"
TARGET_DATA_DIR="../data/target$DATA_SIZE"

IMAGES=(
    "city.jpg"
    "lena.jpg"
    "squirrel.jpg"
)

# Just resize images and create grayscale version of them
for I in "${IMAGES[@]}"; do
    python noise_images.py "$ORIG_DATA_DIR/$I" "$TARGET_DATA_DIR/$I" $DATA_SIZE none 0
done


# Create noisy images

# Gaussian
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/gaus$DATA_SIZE/$I" $DATA_SIZE gaus 0.15
done

# S&P
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/sp$DATA_SIZE/$I" $DATA_SIZE sp 0.1
done

# line
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/line$DATA_SIZE/$I" $DATA_SIZE line 0.2
done

# line-color
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/line-color$DATA_SIZE/$I" $DATA_SIZE line-color 0.2
done

# lined
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/lined$DATA_SIZE/$I" $DATA_SIZE lined 0.2
done

# scratch
for I in "${IMAGES[@]}"; do
    python noise_images.py "$TARGET_DATA_DIR/$I" "$DATA_DIR/scratch$DATA_SIZE/$I" $DATA_SIZE scratch 0.25
done

echo "Done."
