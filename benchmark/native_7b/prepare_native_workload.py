"""Reuse the frozen LMSYS selection; repair only text round-trip incompatibilities."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from paths import RUN_DIR
from selection import make_bank, digest
from huggingface_hub import hf_hub_download
from transformers import LlamaTokenizer
import numpy as np
import pyarrow.parquet as pq

out = RUN_DIR/'workload'
out.mkdir(parents=True, exist_ok=True)
assert not (out/'prompt_bank.json').exists(), 'Do not overwrite the frozen workload'
settings = json.loads(Path(__file__).with_name('workload_settings.json').read_text())
dataset = settings['workload']
tok = LlamaTokenizer.from_pretrained(settings['model'], revision=settings['model_revision'],
                                    local_files_only=True, legacy=True)
source = hf_hub_download(repo_id=dataset['dataset_id'], repo_type='dataset',
                         revision=dataset['dataset_revision'], filename=dataset['dataset_file'],
                         local_files_only=True)
assert digest(source) == settings['source_sha256']
source_rows = pq.read_table(source, columns=['question_id','conversation_a','language']).to_pylist()
bank = make_bank(source_rows, tok, settings, settings['source_sha256'])
bank['preparation_versions'] = settings['preparation_versions']
# Metadata is historical: require the same versions rather than mislabel another environment.
from importlib.metadata import version
for package, expected in settings['preparation_versions'].items():
    assert version(package) == expected, (package, version(package), expected)
original_bank_sha256 = hashlib.sha256((json.dumps(bank, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()
settings = dict(settings, requests_per_experiment=bank['filter_counts']['eligible_unique_prompts'])
eligible = make_bank(pq.read_table(source, columns=['question_id','conversation_a','language']).to_pylist(),
                     tok, settings, bank['source_sha256'])['prompts']
used = {p['prompt_id'] for p in bank['prompts']}
candidates = iter(sorted((p for p in eligible if p['prompt_id'] not in used
                         and tok.encode(p['text']) == p['input_ids']), key=lambda p:p['source_row_index']))
replacements = []
for i, prompt in enumerate(bank['prompts']):
    if tok.encode(prompt['text']) != prompt['input_ids']:
        new = next(candidates)
        replacements.append({'index': i, 'old_prompt_id': prompt['prompt_id'],
                             'new_prompt_id': new['prompt_id'],
                             'reason': 'Truncated byte-fallback sequence did not round-trip through the native text API'})
        bank['prompts'][i] = new
assert len(replacements) == 4
assert len({p['prompt_id'] for p in bank['prompts']}) == 500
assert len({tuple(p['input_ids']) for p in bank['prompts']}) == 500
assert all(tok.encode(p['text']) == p['input_ids'] and len(p['input_ids']) == 128 for p in bank['prompts'])
bank['native_text_adaptation'] = {'original_bank_sha256': original_bank_sha256,
    'replacement_policy': 'Preserve 496 compatible records in place; replace four incompatible records with the first unused, round-trip-safe eligible records in source-row order.',
    'replacements': replacements}
bank['selected_languages'] = dict(Counter(str(p['source_language']) for p in bank['prompts']))
bank['selected_truncated_count'] = sum(p['was_truncated'] for p in bank['prompts'])
(out/'prompt_bank.json').write_text(json.dumps(bank, indent=2, ensure_ascii=False)+'\n')

# Keep this historical authors' module byte-for-byte intact.
upstream = Path(__file__).with_name('upstream_workload.py')
copied = out/'upstream_workload.py'
shutil.copyfile(upstream, copied)
spec = importlib.util.spec_from_file_location('hexgen_original_workload', copied)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
np.random.seed(20260919)
trace = module.PossoinWorkLoad(1, 1., [1.], 1000., [1.])
assert len(trace.arrive_times) >= 500
arrivals = {'seed': 20260919, 'unit_rate_offsets_seconds': trace.arrive_times[:500],
            'source_revision': 'e86ff85b4d254bfed9e840e63e397b224a53530a',
            'source_path': 'experimental/optimizer/llmsim/workload.py',
            'source_sha256': digest(copied), 'numpy_version': np.__version__,
            'adaptation': 'Use the first 500 arrivals of the original unit-rate Poisson generator; divide offsets by target RPS. Original synchronous runner is not used.'}
(out/'arrivals.json').write_text(json.dumps(arrivals, indent=2)+'\n')
proof = {'passed': True, 'prompts': 500, 'native_tokenizer_exact_matches': 500,
         'input_tokens': 128, 'replacements': replacements,
         'prompt_bank_sha256': digest(out/'prompt_bank.json'),
         'arrivals_sha256': digest(out/'arrivals.json'), 'source_sha256': digest(source)}
(out/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
assert proof['prompt_bank_sha256'] == 'a7e1963c0ef88643b9b8013707c33ecf7a932a70bdecfa40064bc5cb2de1e755'
print(json.dumps(proof, indent=2))
