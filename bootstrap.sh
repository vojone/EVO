#!/bin/bash

# Script for initialization of environment
# Requires python3 and pip commands to be available.
#
# USAGE: . ./bootstrap.sh

ENV_NAME="env"
HAL_CGP_SOURCE="https://github.com/Happy-Algorithms-League/hal-cgp.git"
HAL_CGP_DIR="hal-cgp"
REQUIREMENTS_PATH="src/requirements.txt"

if [ ! -d "$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' not found..."
    echo "Creating the new environment..."

    python3 -m venv "$ENV_NAME" || python3 -m virtualenv "$ENV_NAME" || exit

    # shellcheck disable=SC1090
    . "./$ENV_NAME/bin/activate"

    # There is some dependecy error, so we have to install hal-cgp manually
    git clone "$HAL_CGP_SOURCE" "$HAL_CGP_DIR"
    pushd "$HAL_CGP_DIR" || exit
    pip install .

    popd || exit

    pip install -r "$REQUIREMENTS_PATH"

    echo "Setup done."
else
    echo "Environment found... Activating..."

    # shellcheck disable=SC1090
    . "./$ENV_NAME/bin/activate"

    echo "Done."
fi

