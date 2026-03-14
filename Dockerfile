FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# torch from the CUDA 12.8 index
RUN . /venv/main/bin/activate && \
    pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Common ML stack
RUN . /venv/main/bin/activate && \
    pip install \
    transformers accelerate peft trl \
    bitsandbytes datasets huggingface-hub \
    wandb sqlglot

# flash-attn last — long compile, separate layer so it doesn't invalidate above
RUN . /venv/main/bin/activate && \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=2 pip install flash-attn --no-build-isolation

WORKDIR /workspace
