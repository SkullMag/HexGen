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
COPY . /opt/hexgen
COPY --from=ocf /out/ocf-core /usr/local/bin/ocf-core
RUN python -m compileall -q deploy/nrp benchmark/hexgen_documents hexgen
ENTRYPOINT ["python", "/opt/hexgen/deploy/nrp/entrypoint.py"]
