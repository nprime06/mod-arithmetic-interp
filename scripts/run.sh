#!/bin/bash
set -euo pipefail

ROOT_DIR="/Users/william/Desktop/Random/mod"
: "${P:?P must be set}"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${ROOT_DIR}/logs/p_${P}/${TS}"

mkdir -p "${RUN_DIR}"

cd "${ROOT_DIR}"

source "${ROOT_DIR}/modvenv/bin/activate"
pip install -r "${ROOT_DIR}/requirements.txt"
export PYTHONUNBUFFERED=1

# Make src/ importable as a source root
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python "${ROOT_DIR}/src/main.py" --run-dir "${RUN_DIR}" --p "${P}"