#!/bin/bash

set -e

git clone https://github.com/tcoyne1729/mamba-vs-gpt.git .

# python

uv venv --system-site-packages
uv sync --no-dev

chmod +x ./runners/run_gpt.sh