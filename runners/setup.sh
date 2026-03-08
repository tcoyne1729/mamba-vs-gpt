#!/bin/bash

# run on the instance with this command:
# curl -fsSL https://raw.githubusercontent.com/tcoyne1729/mamba-vs-gpt/refs/heads/main/runners/setup.sh | bash

set -e

git clone https://github.com/tcoyne1729/mamba-vs-gpt.git
cd mamba-vs-gpt/

# python

uv venv --python 3.12
uv sync

chmod +x ./runners/*.sh

echo "✅ Setup complete"
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"