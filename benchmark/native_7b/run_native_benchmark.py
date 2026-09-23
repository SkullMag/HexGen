"""Experiment glue only: submit authors' arrival timestamps via their native client.

No replacement server, scheduler, replica router, batching, admission semaphore,
or inference implementation. Native OCF/NATS owns dispatch and queued requests.
"""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from paths import RUN_DIR as ROOT, SOURCE, HEAD_URL, MODEL_NAME
sys.path.insert(0, str(SOURCE/'benchmark/send_request'))
from request import request_head_node

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)
def append(path, value):
    with path.open('a') as f: f.write(json.dumps(value, ensure_ascii=False)+'\n')
def utc(): return datetime.now(timezone.utc).isoformat()
def status(phase, **values):
    write(ROOT/'status.json', {'phase': phase, 'updated_utc': utc(), **values})

bank = read(ROOT/'workload/prompt_bank.json')
trace = read(ROOT/'workload/arrivals.json')
proof = read(ROOT/'workload/verification.json')
assert sha(ROOT/'workload/prompt_bank.json') == proof['prompt_bank_sha256']
assert sha(ROOT/'workload/arrivals.json') == proof['arrivals_sha256']
assert proof['passed'] and proof['native_tokenizer_exact_matches'] == 500
assert read(ROOT/'validation-audit.json')['passed']
assert sha(Path(__file__)) == read(ROOT/'approved-source.json')['driver_sha256']

async def request(index, prefix, scheduled, origin):
    prompt = bank['prompts'][index]
    request_id = f'{prefix}-{index:04d}'
    submitted = time.perf_counter()-origin
    record = {'request_id': request_id, 'request_index': index, 'prompt_id': prompt['prompt_id'],
              'scheduled_offset_s': scheduled, 'submitted_offset_s': submitted,
              'submission_lag_s': submitted-scheduled}
    data = {'model_name': MODEL_NAME,
            'params': {'prompt': prompt['text'], 'max_new_tokens': 32,
                       'temperature': 1., 'top_p': 0., 'top_k': 1,
                       'request_id': request_id, 'prompt_id': prompt['prompt_id']}}
    try:
        output, infer_s, native_client_s = await request_head_node(data, HEAD_URL, request_id)
        record.update(success=isinstance(output, str) and isinstance(infer_s, (int,float)) and infer_s>0,
                      output_text=output, inference_s=infer_s, native_client_s=native_client_s)
    except Exception as exc:
        record.update(success=False, error=repr(exc))
    finished = time.perf_counter()-origin
    record.update(completed_offset_s=finished, end_to_end_latency_s=finished-scheduled,
                  client_elapsed_s=finished-submitted)
    return record

async def checkpoint(directory):
    records = ROOT/directory/'requests.jsonl'
    digest = sha(records)
    status('awaiting_result_audit', directory=directory, requests_sha256=digest)
    receipt = ROOT/'releases'/f'{directory}.json'
    while not receipt.exists(): await asyncio.sleep(5)
    value = read(receipt)
    assert value['passed'] and value['directory'] == directory
    assert value['requests_sha256'] == digest
    assert value['prompt_bank_sha256'] == proof['prompt_bank_sha256']
    print(json.dumps({'released': directory, 'utc': utc()}), flush=True)

async def baseline():
    directory = ROOT/'baseline'
    directory.mkdir()  # Never overwrite an earlier run.
    write(directory/'config.json', {'warmup_requests':30, 'measured_requests':100,
                                   'execution':'sequential isolated calls through native client',
                                   'reference':'median rank-0 synchronized distributed inference duration'})
    for phase, count, name in [('warmup',30,'warmups.jsonl'),('baseline',100,'requests.jsonl')]:
        for i in range(count):
            record = await request(i, phase, 0., time.perf_counter())
            append(directory/name, record)
            status(phase, directory='baseline', completed=i+1, total=count)
            assert record['success'], 'Native baseline request failed'
    await checkpoint('baseline')

async def run_rate(directory, rate, count):
    folder = ROOT/directory
    folder.mkdir()
    offsets = [float(t)/rate for t in trace['unit_rate_offsets_seconds'][:count]]
    config = {'directory':directory, 'rate':rate, 'requests':count, 'output_tokens':32,
              'prompt_bank_sha256':proof['prompt_bank_sha256'], 'arrivals_sha256':proof['arrivals_sha256'],
              'scheduled_offsets_s':offsets, 'started_utc':utc(), 'driver_sha256':sha(Path(__file__))}
    write(folder/'config.json',config)
    completed = 0
    successful = 0
    origin = time.perf_counter()
    async def one(i, scheduled):
        nonlocal completed, successful
        await asyncio.sleep(max(0., origin+scheduled-time.perf_counter()))
        record = await request(i,directory,scheduled,origin)
        append(folder/'requests.jsonl',record)
        completed += 1
        successful += int(record['success'])
        status('running_rate',directory=directory,rate=rate,completed=completed,total=count,
               successful=successful,elapsed_s=time.perf_counter()-origin)
    status('running_rate',directory=directory,rate=rate,completed=0,total=count,successful=0)
    await asyncio.gather(*(one(i,t) for i,t in enumerate(offsets)))
    write(folder/'completed.json',{'requests':count,'successful':successful,'finished_utc':utc(),
                                 'elapsed_s':time.perf_counter()-origin})
    assert successful == count, 'A native request failed; preserve data and investigate'
    await checkpoint(directory)

async def main():
    assert not (ROOT/'controller-started.json').exists(), 'Existing controller run must not be restarted'
    write(ROOT/'controller-started.json',{'utc':utc(),'driver_sha256':sha(Path(__file__))})
    await baseline()
    await run_rate('pilot_01_32_0.125_red',.125,50)
    for number, (rate, label) in enumerate([(.125,'0.125'),(.25,'0.25'),(.5,'0.5'),
                                          (1.,'01'),(2.,'02'),(4.,'04'),(8.,'08')], start=1):
        await run_rate(f'{number:02d}_32_{label}_red',rate,500)
    status('complete',completed_rates=7,total_measured_requests=3500)

try:
    asyncio.run(main())
except BaseException as exc:
    prior = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
    status('failed',error=repr(exc),last_status=prior)
    raise
