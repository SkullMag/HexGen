"""Original fixed-length LMSYS selection used before native text adaptation."""

from pathlib import Path

from collections import Counter

import hashlib

import random

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def make_bank(rows, tokenizer, settings, source_sha256):
    """Pure selection function; dataset order and sampling seed are fixed."""
    n, width = settings['requests_per_experiment'], settings['input_tokens'] - 1
    if tokenizer.bos_token_id is None or width < 1:
        raise ValueError('A BOS token and positive prompt width are required')
    eligible, seen_text, seen_inputs, seen_ids = [], set(), set(), set()
    counts = Counter()
    for row_index, row in enumerate(rows):
        counts['source_rows'] += 1
        conversation = row.get(settings['workload']['conversation_column'])
        # Take only the initial user message, never a response or follow-up.
        if (not isinstance(conversation, list) or not conversation
                or not isinstance(conversation[0], dict)
                or conversation[0].get('role') != 'user'
                or not isinstance(conversation[0].get('content'), str)
                or row.get('question_id') is None):
            counts['invalid_first_user_or_id'] += 1
            continue
        original = conversation[0]['content'].strip()
        if not original:
            counts['empty'] += 1
            continue
        text_sha = hashlib.sha256(original.encode('utf-8')).hexdigest()
        if text_sha in seen_text:
            counts['duplicate_text'] += 1
            continue
        seen_text.add(text_sha)
        tokens = tokenizer.encode(original, add_special_tokens=False)
        if len(tokens) < width:
            counts['too_short'] += 1
            continue
        ids = [int(tokenizer.bos_token_id)] + [int(t) for t in tokens[:width]]
        question_id = str(row['question_id'])
        if tuple(ids) in seen_inputs or question_id in seen_ids:
            counts['duplicate_input_or_question'] += 1
            continue
        seen_inputs.add(tuple(ids)); seen_ids.add(question_id)
        eligible.append({
            'prompt_id': 'lmsys-' + question_id, 'source_question_id': question_id,
            'source_row_index': row_index,
            'source_conversation': settings['workload']['conversation_column'],
            'source_message_index': 0, 'source_role': 'user',
            'source_language': row.get('language'), 'source_text_sha256': text_sha,
            'original_user_text': original, 'original_text_tokens': len(tokens),
            'was_truncated': len(tokens) > width, 'input_ids': ids,
            'text': tokenizer.decode(ids, skip_special_tokens=True),
        })
    counts['eligible_unique_prompts'] = len(eligible)
    if len(eligible) < n:
        raise ValueError(f'Need {n} eligible unique prompts; found {len(eligible)}. '
                         'No padding, prompt reuse, or substitute dataset is permitted.')
    prompts = random.Random(settings['prompt_seed']).sample(eligible, n)
    return {
        'schema_version': 2,
        'source_url': 'https://huggingface.co/datasets/' + settings['workload']['dataset_id'],
        'source_sha256': source_sha256, 'dataset': settings['workload'],
        'description': 'LMSYS Chatbot Arena first-user prompts, filtered and truncated to a fixed input length; a controlled subset, not the natural length distribution.',
        'model': settings['model'], 'model_revision': settings['model_revision'],
        'tokenizer_use_fast': False, 'tokenizer_legacy': True,
        'bos_token_id': int(tokenizer.bos_token_id),
        'input_tokens': settings['input_tokens'], 'seed': settings['prompt_seed'],
        'filter_counts': dict(counts),
        'selected_languages': dict(Counter(str(p['source_language']) for p in prompts)),
        'selected_truncated_count': sum(p['was_truncated'] for p in prompts),
        'prompts': prompts,
    }
