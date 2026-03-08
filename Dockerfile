FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    UV_LINK_MODE=copy \
    # Make uv pip install into the venv below without needing `source activate`
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates neovim \
    && rm -rf /var/lib/apt/lists/*

# uv — via official image layer, no curl | sh
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# age — zero-trust artifact encryption
RUN curl -fsSL https://github.com/FiloSottile/age/releases/download/v1.2.0/age-v1.2.0-linux-amd64.tar.gz \
    | tar -xz --strip-components=1 -C /usr/local/bin age/age age/age-keygen

# Python 3.12 + global venv for all ML deps
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12

# torch from the CUDA 12.8 index — not PyPI, which ships CPU or wrong CUDA wheels
RUN uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Common ML stack — baked in so setup.sh is fast
RUN uv pip install \
    transformers accelerate peft trl \
    bitsandbytes datasets huggingface-hub \
    wandb sqlglot

# flash-attn last: takes ~10min to compile and invalidates everything below it if moved up
RUN uv pip install wheel && uv pip install flash-attn --no-build-isolation

WORKDIR /workspace
CMD ["bash"]
