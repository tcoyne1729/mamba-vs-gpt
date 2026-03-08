#!/bin/bash

# run on the instance with this command:
# curl -fsSL https://raw.githubusercontent.com/tcoyne1729/mamba-vs-gpt/refs/heads/main/runners/setup.sh | bash

set -e

# 1. Validate required env vars before doing anything
if [ -z "$WANDB_API_KEY" ]; then
  echo "❌ WANDB_API_KEY is not set. Set it in your vast.ai instance config."; exit 1
fi
if [ -z "$HF_TOKEN" ]; then
  echo "❌ HF_TOKEN is not set. Set it in your vast.ai instance config."; exit 1
fi

# 2. Clone repo
git clone https://github.com/tcoyne1729/mamba-vs-gpt.git
cd mamba-vs-gpt/

# 3. Python env + all dependencies (including flash-attn)
uv venv --python 3.12
uv sync

chmod +x ./runners/*.sh

# 5. Sanity checks — fail loud if anything is wrong before an expensive run
echo ""
echo "=== Sanity checks ==="
uv run python -c "
import torch
print('PyTorch:   ', torch.__version__)
print('CUDA:      ', torch.version.cuda)
assert torch.cuda.is_available(), 'GPU not visible — check instance config'
print('GPU:       ', torch.cuda.get_device_name(0))
print('VRAM:      ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
import flash_attn
print('flash-attn:', flash_attn.__version__)
print('WANDB_API_KEY:', 'set')
print('HF_TOKEN:     ', 'set')
"

echo ""
echo "✅ Setup complete — ready to train"
