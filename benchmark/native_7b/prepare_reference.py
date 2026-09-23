"""Extracted Transformers FP16 reference check from the original experiment.

Run before native workers are loaded. It loads one 7B model on CUDA device 0;
it provides token correctness evidence, not the latency baseline.
"""
import json
from paths import RUN_DIR, CHECKPOINT
import torch
from transformers import LlamaForCausalLM

destination = RUN_DIR/'reference.json'
if destination.exists():
    raise SystemExit('Preserve the existing reference; use a new run directory')
if not CHECKPOINT:
    raise SystemExit('Set CHECKPOINT_PATH to the pinned local checkpoint')
bank = json.loads((RUN_DIR/'workload/prompt_bank.json').read_text())
reference = LlamaForCausalLM.from_pretrained(
    CHECKPOINT, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager', local_files_only=True).eval().cuda()
records = []
for prompt in bank['prompts'][:3]:
    current = torch.tensor([prompt['input_ids']], device='cuda', dtype=torch.long)
    past, generated = None, []
    with torch.inference_mode():
        for _ in range(32):
            output = reference(current, past_key_values=past, use_cache=True)
            current = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(current.item()))
            past = output.past_key_values
    records.append({'prompt_id': prompt['prompt_id'], 'reference_ids': generated})
with destination.open('x') as handle:
    handle.write(json.dumps({'model_revision': bank['model_revision'], 'validation': records}, indent=2)+'\n')
print(destination)
