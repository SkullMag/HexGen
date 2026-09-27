"""Independently verify newly generated native results against saved token references."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from paths import RUN_DIR as root, REFERENCE_PATH
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
reference_path = REFERENCE_PATH
assert sha(reference_path) == read(root/'approved-source.json')['reference_sha256']
reference = read(reference_path)
bank = read(root/'workload/prompt_bank.json')
assert bank['model_revision'] == reference['model_revision']
checks = read(root/'native-tokenizer-check.json')
assert len(checks) == len(bank['prompts']) == 500
for c, prompt in zip(checks, bank['prompts']):
    assert c['prompt_id'] == prompt['prompt_id']
    assert c['exact_input_ids'] and c['tokens'] == 128
    assert c['actual_input_ids'] == prompt['input_ids']
approved = read(root/'approved-source.json')
for location in [root, root/'node02']:
    p = read(location/'provenance.json')
    assert p['changed_file_sha256'] == approved['runtime_file_sha256']
    assert p['coordinator_binary_sha256'] == approved['coordinator_binary_sha256']
    assert p['coordinator_resource_settings'] == approved['coordinator_resources']
evidence = []
for rank, name in [(0,'worker-rank0.jsonl'), (1,'node02/worker-rank1.jsonl')]:
    raw = (root/name).read_bytes()
    assert raw.endswith(b'\n')
    records = [json.loads(line) for line in raw.split(b'\n') if line]
    assert len({r['request_id'] for r in records}) == len(records)
    records = {r['request_id']:r for r in records}
    for i in range(3):
        r = records[f'native-validation-{i:03d}']
        ref = reference['validation'][i]
        prompt = bank['prompts'][i]
        assert r['rank'] == rank
        assert r['prompt_id'] == ref['prompt_id'] == prompt['prompt_id']
        assert r['input_ids'] == prompt['input_ids']
        assert len(r['generated_ids']) == 32
        assert r['generated_ids'] == ref['reference_ids']
        client = read(root/f'native-validation-{i:03d}.json')
        assert client['request_id'] == r['request_id'] and client['prompt_id'] == r['prompt_id']
        assert isinstance(client['response'][0],str) and client['response'][0]
        assert client['response'][1] > 0 and client['response'][2] > 0
        evidence.append({'rank':rank,'request_id':r['request_id'],'exact_reference_tokens':32})
result = {'passed':True,'audited_utc':datetime.now(timezone.utc).isoformat(),
          'native_tokenizer_exact_matches':500,'reference_model_revision':reference['model_revision'],
          'reference_file':str(reference_path),'reference_sha256':sha(reference_path),
          'prompt_bank_sha256':sha(root/'workload/prompt_bank.json'),
          'coordinator_binary_sha256':approved['coordinator_binary_sha256'],
          'new_rank_log_sha256':{p:sha(root/p) for p in ['worker-rank0.jsonl','node02/worker-rank1.jsonl']},
          'checks':evidence,
          'note':'Only saved reference token IDs are reused; all native execution and tokenizer evidence is from this new run. Old timings are not reused.'}
(root/'validation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
