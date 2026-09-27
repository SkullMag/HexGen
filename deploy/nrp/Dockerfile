# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 golang:1.20-bookworm AS ocf
WORKDIR /src
COPY third_party/ocf/src/ocf-core/ ./
RUN CGO_ENABLED=0 go build -trimpath -o /out/ocf-core ./bin/core

FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/hexgen
COPY deploy/nrp/requirements-image.txt /tmp/requirements-image.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-image.txt
# HexGen imports flash_attn model helpers even when its attention kernel is disabled.
# This official 2.0.8 wheel matches the base image's cu117, torch 2.0, cp310,
# and cxx11abiFALSE ABI; fail early if the base image is changed incompatibly.
RUN python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.startswith("2.0."); assert not torch._C._GLIBCXX_USE_CXX11_ABI'
RUN python -m pip install --no-cache-dir \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.0.8/flash_attn-2.0.8%2Bcu117torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
# The release wheel omits fused_dense_lib, which HexGen needs for its tensor
# parallel linear layers. Build that small upstream extension for the A6000/A10.
RUN git clone --depth 1 --branch v2.0.8 --filter=blob:none --sparse \
      https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
    && cd /tmp/flash-attention \
    && git sparse-checkout set csrc/fused_dense_lib \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install \
      --no-cache-dir --no-build-isolation ./csrc/fused_dense_lib \
    && cd / \
    && rm -rf /tmp/flash-attention
RUN python -c 'from flash_attn.ops.fused_dense import FusedDense, ColumnParallelLinear; print("fused_dense_lib import OK")'
COPY . /opt/hexgen
COPY --from=ocf /out/ocf-core /usr/local/bin/ocf-core
RUN python -m compileall -q deploy/nrp benchmark/hexgen_documents hexgen \
    && PYTHONPATH=/opt/hexgen/hexgen python -c 'from hexgen_core.models.gpt import RMSNorm; assert RMSNorm is not None'
ENTRYPOINT ["python", "/opt/hexgen/deploy/nrp/entrypoint.py"]
