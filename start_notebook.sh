#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -m jupyter notebook --notebook-dir="$SCRIPT_DIR/Notebooks" $@
