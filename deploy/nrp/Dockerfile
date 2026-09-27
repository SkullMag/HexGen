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
# FlashAttention 2.0.8 supports the image's CUDA 11.7 / PyTorch 2.0.1 stack
# and the NRP A6000/A10 Ampere GPUs. Pin its official cu117, torch 2.0,
# cp310, cxx11abiFALSE wheel and fail early on an incompatible base image.
RUN python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.startswith("2.0."); assert not torch._C._GLIBCXX_USE_CXX11_ABI'
RUN python -m pip install --no-cache-dir \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.0.8/flash_attn-2.0.8%2Bcu117torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
# The release wheel omits the fused dense and rotary extensions used by HexGen.
# Build them from the same upstream revision for the A6000/A10.
RUN curl -fsSL \
      'https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/2286d7cea7ca8264165c16b2442b6436c43140de' \
      -o /tmp/flash-attention.tar.gz \
    && echo '4191c09a10ddaefc139efe073e8ab5bdc732f65332e222c34d61c612f6bb8abb  /tmp/flash-attention.tar.gz' | sha256sum -c - \
    && mkdir /tmp/flash-attention \
    && tar -xzf /tmp/flash-attention.tar.gz -C /tmp/flash-attention --strip-components=1 \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install \
      --no-cache-dir --no-build-isolation /tmp/flash-attention/csrc/fused_dense_lib \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install \
      --no-cache-dir --no-build-isolation /tmp/flash-attention/csrc/rotary \
    && rm -rf /tmp/flash-attention /tmp/flash-attention.tar.gz
RUN python -c 'from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func; from flash_attn.ops.fused_dense import FusedDense, ColumnParallelLinear; from flash_attn.layers.rotary import RotaryEmbedding; print("FlashAttention 2.0.8 attention and model helper imports OK")'
COPY . /opt/hexgen
COPY --from=ocf /out/ocf-core /usr/local/bin/ocf-core
RUN python -m compileall -q deploy/nrp benchmark/hexgen_documents hexgen \
    && PYTHONPATH=/opt/hexgen/hexgen python -c 'from hexgen_core.models.gpt import RMSNorm; from hexgen_core.modules.mha import RotaryEmbedding; assert RMSNorm is not None and RotaryEmbedding is not None'
ENTRYPOINT ["python", "/opt/hexgen/deploy/nrp/entrypoint.py"]
