#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export JUPYTER_CONFIG_DIR="$SCRIPT_DIR/src/jupyter/config"
export JUPYTER_DATA_DIR="$SCRIPT_DIR/src/jupyter/data"
export JUPYTER_RUNTIME_DIR="$SCRIPT_DIR/src/jupyter/runtime"
export IPYTHONDIR="$SCRIPT_DIR/src/jupyter/ipython"

python3 -m notebook --notebook-dir="$SCRIPT_DIR/Notebooks" "$@"
