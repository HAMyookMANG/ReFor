#!/bin/bash
set -e
export PYTHONPATH="$(pwd)/ReFor:$PYTHONPATH"
python ReFor/scripts/bootstrap.py
python -m refor.core.env
python ReFor/scripts/main.py