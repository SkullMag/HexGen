#!/usr/bin/env bash
set -euo pipefail

# Scaled homogeneous RTX experiment: one TP=2 replica, PP=1, across two hosts.
# Turing uses regular attention; --use-flash-attn is deliberately omitted.
export PORT=29511
export DEVICES=0
export NUM_NODES=2
export NUM_GPUS_PER_NODE=1
: "${MASTER_ADDR:?Set the private torch rendezvous address}"
export MASTER_ADDR
# The native OCF worker coordinator is distinct from the public head coordinator.
: "${OCF_WORKER_ADDR:?Set the private OCF worker coordinator address}"
: "${HEAD_NODE:?Set the OCF head URL}"
export OCF_WORKER_ADDR
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${CHECKPOINT_PATH:?Set the pinned Hugging Face checkpoint directory}"
: "${STATE_DICTS_PATH:?Set the converted checkpoint directory}"
: "${REQUEST_LOG:?Set a distinct log path for this rank}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=$DEVICES python3 -m torch.distributed.launch \
--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE \
--master_addr=$MASTER_ADDR --master_port=$PORT --node_rank=$NODE_RANK _llama_worker.py \
--model_size llama-7b --mixed_precision fp16 --fp16 \
--num-layers 32 --hidden_size 4096 --num_attention_heads 32 \
--max-position-embeddings 2048 --seq-length 128 --micro-batch-size 1 \
--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
--hetero_config 2 --pp_partition 32 \
--checkpoint-path "$CHECKPOINT_PATH" --state-dicts-path "$STATE_DICTS_PATH" \
--request-log "$REQUEST_LOG" \
--model_name OpenLLaMA-7B-v2-native-rtx \
--head_node "$HEAD_NODE" --group_id 0
