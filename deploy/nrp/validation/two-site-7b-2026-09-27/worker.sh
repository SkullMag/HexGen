#!/usr/bin/env bash
set -Eeuo pipefail

: "${RANK:?rank is required}"
export HF_HOME=/scratch/hf
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export MASTER_ADDR=hexgen-7b-two-site-head
export MASTER_PORT=29511
export WORLD_SIZE=2
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0
export OCF_WORKER_ADDR=hexgen-7b-two-site-head
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

ocf_pid=
cleanup() {
  if [[ -n "$ocf_pid" ]]; then kill "$ocf_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
if [[ "$RANK" == 0 ]]; then
  ocf-core --config /opt/hexgen/third_party/ocf/src/ocf-core/config/cfg_standalone.yaml start \
    > /scratch/ocf.log 2>&1 &
  ocf_pid=$!
fi

for i in $(seq 1 120); do
  if curl -fsS --max-time 5 http://hexgen-7b-two-site-head:8092/api/v1/status/peers >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS --max-time 5 http://hexgen-7b-two-site-head:8092/api/v1/status/peers >/dev/null

mkdir -p /scratch/converted/separate_state_dicts
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="openlm-research/open_llama_7b_v2",
    revision="e5961def23172a2384543940e773ab676033c963",
    local_dir="/scratch/checkpoint",
    allow_patterns=["config.json", "generation_config.json", "tokenizer.model",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "pytorch_model-*.bin", "pytorch_model.bin.index.json"],
)
print("CHECKPOINT_READY", path, flush=True)
PY
cd /opt/hexgen/hexgen/llama/load_model_parameters_utils
python create_separate_state_dicts_llama_7b.py \
  --checkpoint-path /scratch/checkpoint \
  --save-dir /scratch/converted/separate_state_dicts
test -s /scratch/converted/inv_freq.pt
test -s /scratch/converted/separate_state_dicts/layer_31.pt
echo "CONVERTED_READY rank=$RANK"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /opt/hexgen/benchmark/hexgen_documents
python _llama_worker.py \
  --local-rank 0 --model_size llama-7b --mixed_precision fp16 --fp16 --use-flash-attn \
  --num-layers 32 --hidden_size 4096 --num_attention_heads 32 \
  --max-position-embeddings 2048 --seq-length 128 --micro-batch-size 1 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
  --hetero_config 1 1 --pp_partition 16 16 \
  --checkpoint-path /scratch/checkpoint --state-dicts-path /scratch/converted \
  --request-log "/scratch/rank${RANK}.jsonl" \
  --model_name NRP-OpenLLaMA-7B-TwoSite \
  --head_node http://hexgen-7b-two-site-head:8092 --group_id 0
