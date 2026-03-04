#!/bin/bash

set -e

uv sync --no-dev --system-site-packages

# run

./.venv/bin/python gpt_train.py