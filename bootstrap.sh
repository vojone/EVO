#!/bin/bash

# Script for initialization of environment

ENV_NAME="env"
HAL_CGP_SOURCE="https://github.com/Happy-Algorithms-League/hal-cgp.git"
HAL_CGP_DIR="hal-cgp"
REQUIREMENTS_PATH="src/requirements.txt"

if [ ! -d "$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' not found..."

    python -m venv "$ENV_NAME"

    git clone "$HAL_CGP_SOURCE" "$HAL_CGP_DIR"
    pushd "$HAL_CGP_DIR" || exit
    pip install .

    pip install -r "$REQUIREMENTS_PATH"
fi


echo "Environment prepared..."

# shellcheck disable=SC1090
. "./$ENV_NAME/bin/activate"

