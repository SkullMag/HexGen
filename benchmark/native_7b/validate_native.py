"""Use HexGen's native request client for three correctness checks."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from paths import SOURCE as root, RUN_DIR as out, CHECKPOINT as checkpoint, HEAD_URL, MODEL_NAME
sys.path.insert(0, str(root/'benchmark/send_request'))
from request import request_head_node
from transformers import LlamaTokenizer

bank = json.loads((out/'workload/prompt_bank.json').read_text())
tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
checks = []
for p in bank['prompts']:
    ids = tokenizer.encode(p['text'])
    checks.append({'prompt_id': p['prompt_id'], 'exact_input_ids': ids == p['input_ids'],
                   'tokens': len(ids), 'actual_input_ids': ids})
(out/'native-tokenizer-check.json').write_text(json.dumps(checks, indent=2))
print(json.dumps({'tokenizer_checks': len(checks), 'exact': sum(c['exact_input_ids'] for c in checks),
                  'lengths': sorted(set(c['tokens'] for c in checks))}), flush=True)
assert all(c['exact_input_ids'] for c in checks), 'Native text tokenization differs from frozen input IDs'

async def main():
    for i, p in enumerate(bank['prompts'][:3]):
        request_id = f'native-validation-{i:03d}'
        destination = out/f'{request_id}.json'
        assert not destination.exists(), 'Preserve earlier validation data'
        payload = {'model_name': MODEL_NAME,
                   'params': {'prompt': p['text'], 'max_new_tokens': 32, 'temperature': 1.,
                              'top_p': 0., 'top_k': 1, 'request_id': request_id,
                              'prompt_id': p['prompt_id']}}
        start = time.time()
        try:
            response = await asyncio.wait_for(request_head_node(payload, HEAD_URL, i), 120)
            record = {'request_id': request_id, 'prompt_id': p['prompt_id'], 'response': response,
                      'elapsed_seconds': time.time()-start}
            destination.write_text(json.dumps(record, indent=2))
            assert response[0] is not None
            print(json.dumps({'request_id': request_id, 'inference_s': response[1], 'end_to_end_s': response[2]}), flush=True)
        except Exception as exc:
            destination.with_suffix('.error.json').write_text(json.dumps({'request_id': request_id, 'error': repr(exc)}))
            raise

asyncio.run(main())
