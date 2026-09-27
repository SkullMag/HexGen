"""Independent audit and compact export for native HexGen measurements.

Does not release checkpoints. Upload the generated receipt only after reviewing
the plot and notifying the user. Reuses existing scientific plotting utilities.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

from paths import RUN_DIR as ROOT
from plots import SCALES, percentile, plot_experiment

def read(path): return json.loads(path.read_text())
def rows(path):
    # JSON strings may contain U+2028/U+2029. Only physical LF separates JSONL records.
    data = path.read_bytes()
    assert not data or data.endswith(b'\n'), 'Incomplete final JSONL record'
    return [json.loads(line) for line in data.split(b'\n') if line]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def same(a,b): assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

bank = read(ROOT/'workload/prompt_bank.json')
proof = read(ROOT/'workload/verification.json')
assert sha(ROOT/'workload/prompt_bank.json') == proof['prompt_bank_sha256']
assert sha(ROOT/'workload/arrivals.json') == proof['arrivals_sha256']
assert read(ROOT/'validation-audit.json')['passed']
assert proof['native_tokenizer_exact_matches'] == 500
provenance = [read(ROOT/'provenance.json'),read(ROOT/'node02/provenance.json')]
assert provenance[0]['changed_file_sha256'] == provenance[1]['changed_file_sha256']
assert provenance[0]['coordinator_binary_sha256'] == provenance[1]['coordinator_binary_sha256']
approved = read(ROOT/'approved-source.json')
for p in provenance:
    assert p['git_commit'] == approved['git_commit']
    assert p['changed_file_sha256'] == approved['runtime_file_sha256']
    assert p['coordinator_binary_sha256'] == approved['coordinator_binary_sha256']
    assert p['coordinator_resource_settings'] == {'http_streams_per_peer':1024,'http_memory_mib_per_peer':512}
    assert 'Quadro RTX 6000' in p['gpu']['stdout']
trace = read(ROOT/'workload/arrivals.json')
assert sha(ROOT/'workload/upstream_workload.py') == trace['source_sha256']
assert sha(ROOT/'workload/upstream_workload.py') == sha(Path(__file__).with_name('upstream_workload.py'))
rank_rows = [rows(ROOT/'worker-rank0.jsonl'),rows(ROOT/'node02/worker-rank1.jsonl')]
ranks = []
for data in rank_rows:
    assert len({r['request_id'] for r in data}) == len(data), 'Duplicate native execution'
    ranks.append({r['request_id']:r for r in data})
known = {}
for data in rank_rows:
    for r in data:
        assert known.setdefault(r['prompt_id'],r['generated_ids']) == r['generated_ids'], 'Repeated prompt output changed'

def check_records(records,prefix,count):
    assert len(records) == count
    assert {r['request_index'] for r in records} == set(range(count))
    assert len({r['request_id'] for r in records}) == count
    for record in records:
        i = record['request_index']; prompt = bank['prompts'][i]
        assert record['request_id'] == f'{prefix}-{i:04d}'
        assert record['prompt_id'] == prompt['prompt_id'] and record['success']
        assert record['completed_offset_s'] >= record['submitted_offset_s'] >= record['scheduled_offset_s']
        same(record['end_to_end_latency_s'],record['completed_offset_s']-record['scheduled_offset_s'])
        same(record['client_elapsed_s'],record['completed_offset_s']-record['submitted_offset_s'])
        same(record['submission_lag_s'],record['submitted_offset_s']-record['scheduled_offset_s'])
        for rank,mapping in enumerate(ranks):
            worker = mapping[record['request_id']]
            assert worker['rank'] == rank and worker['prompt_id'] == prompt['prompt_id']
            assert worker['input_ids'] == prompt['input_ids'] and len(worker['input_ids']) == 128
            assert len(worker['generated_ids']) == 32
            assert worker['generated_ids'] == ranks[0][record['request_id']]['generated_ids']
            assert 0 < worker['inference_seconds'] <= record['client_elapsed_s']+.01
            assert 0 < worker['peak_allocated_bytes'] < 24*1024**3
        same(record['inference_s'],ranks[0][record['request_id']]['inference_seconds'])
        assert record['native_client_s'] > 0

def receipt(directory,checks):
    result = {'passed':True,'directory':directory,'requests_sha256':sha(ROOT/directory/'requests.jsonl'),
              'prompt_bank_sha256':proof['prompt_bank_sha256'],'checks':checks}
    write(ROOT/'checkpoints'/f'{directory}.audit.json',result)
    write(ROOT/'checkpoints'/f'{directory}.release.json',result)
    return result

def baseline():
    directory = ROOT/'baseline'
    warm = rows(directory/'warmups.jsonl'); measured = rows(directory/'requests.jsonl')
    check_records(warm,'warmup',30); check_records(measured,'baseline',100)
    reference = statistics.median(r['inference_s'] for r in measured)
    result = {'reference_seconds':reference,
              'end_to_end_median_seconds':statistics.median(r['client_elapsed_s'] for r in measured),
              'warmup_requests':30,'measured_requests':100,
              'rank_median_inference_seconds':{str(rank):statistics.median(mapping[r['request_id']]['inference_seconds'] for r in measured) for rank,mapping in enumerate(ranks)},
              'definition':'Median native rank-0 inference duration: tokenization excluded; includes synchronized distributed prefill and 32-token decode, and wait for the initial rank barrier. One TP=2 replica across two RTX 6000 hosts; isolated requests through OCF.'}
    write(directory/'summary.json',result)
    receipt('baseline',['30 warmups','100 isolated requests','both ranks','128 input and 32 output tokens','matching outputs','timing arithmetic'])
    print(json.dumps(result,indent=2))

def package(directory):
    base = read(ROOT/'baseline/summary.json')
    measured_base = rows(ROOT/'baseline/requests.jsonl')
    check_records(measured_base,'baseline',100)
    same(base['reference_seconds'],statistics.median(r['inference_s'] for r in measured_base))
    folder = ROOT/directory; cfg = read(folder/'config.json'); records = rows(folder/'requests.jsonl')
    count = cfg['requests']; rate = cfg['rate']
    assert count == (50 if directory.startswith('pilot_') else 500)
    assert cfg['directory'] == directory and cfg['prompt_bank_sha256'] == proof['prompt_bank_sha256']
    assert cfg['arrivals_sha256'] == proof['arrivals_sha256']
    assert cfg['driver_sha256'] == approved['driver_sha256']
    assert read(folder/'completed.json')['successful'] == count
    check_records(records,directory,count)
    for r in records:
        planned = trace['unit_rate_offsets_seconds'][r['request_index']]/rate
        same(r['scheduled_offset_s'],planned)
        same(cfg['scheduled_offsets_s'][r['request_index']],planned)
    lag99 = percentile([r['submission_lag_s'] for r in records],99)
    assert lag99 <= max(.01,.01/rate), f'Arrival timing p99 too large: {lag99}'
    reference = base['reference_seconds']
    latency = [r['end_to_end_latency_s'] for r in records]
    duration = max(r['completed_offset_s'] for r in records)
    slo = {f'{scale:g}':100*sum(t<=scale*reference for t in latency)/count for scale in SCALES}
    memory = max(ranks[rank][r['request_id']]['peak_allocated_bytes'] for rank in (0,1) for r in records)
    metrics = {'requests':{'total':count,'successful':count,'failed':0},
               'end_to_end_latency_seconds':{'median':statistics.median(latency),'p95':percentile(latency,95)},
               'achieved_throughput_requests_per_second':count/duration,
               'peak_gpu_memory_allocated_gib':memory/1024**3,
               'slo_attainment_percent_by_deadline_multiplier':slo}
    setup = {'scope':'Scaled homogeneous native HexGen experiment with 1024 HTTP streams per peer; not the paper 70B result.',
             'model':bank['model'],'model_revision':bank['model_revision'],
             'hexgen_commit':approved['git_commit'],
             'hardware':'Two separate hosts with one Quadro RTX 6000 24 GB each',
             'replicas':1,'tensor_parallel':2,'pipeline_parallel':1,'batch_size':1,'precision':'FP16',
             'attention':'Regular attention; flash-attn 2.0.8 installed for native model/helper modules, FlashAttention kernels disabled',
             'network':'NCCL socket transport on private Ethernet; RDMA disabled',
             'serving':'Authors OCF head/worker coordinators with bounded resource and error-response patch; NATS, rank-based LlamaWorker, original decode and request_head_node client',
             'coordinator_resources':{'http_streams_per_peer':1024,'http_memory_mib_per_peer':512},
             'coordinator_binary_sha256':approved['coordinator_binary_sha256'],
             'native_runtime_file_sha256':approved['runtime_file_sha256'],
             'input_tokens':128,'output_tokens':32,'dataset':bank['dataset'],
             'input_adaptation':bank['native_text_adaptation'],
             'prompt_bank_sha256':proof['prompt_bank_sha256'],
             'arrivals':'Authors historical PossoinWorkLoad generator; same first 500 seeded unit-rate offsets divided by each RPS; asynchronous native client calls; native server queueing; drain every rate.',
             'arrival_seed':trace['seed'],'included_rates':[rate],'requests_per_rate':count,
             'base_inference_latency_seconds':reference,'base_measurement':base['definition'],
             'baseline_warmup_requests':30,'baseline_measured_requests':100,
             'metric_definitions':{'end_to_end_latency':'Scheduled arrival to full response; includes submission lag, native queueing, network and inference.',
                                   'throughput':'Successful requests divided by time from first scheduled arrival through final completion; includes drain.',
                                   'memory':'Largest per-request peak PyTorch allocation on either GPU, including weights; excludes unused cached reservation and driver memory.',
                                   'slo':'Percentage of offered requests finished by multiplier times the fresh TP=2 base reference.',
                                   'prompt_text':'Exact text submitted to the native API; verified to encode to 128 input tokens.'},
             'limitations':['Native OCF HTTP-per-peer resource limits explicitly raised to 1024 streams and 512 MiB; original transport error handling corrected.',
                            'Smaller model, fewer GPUs and cross-host TP differ from the paper.',
                            'Regular attention on Turing, not FlashAttention GPU kernels.',
                            'Controlled LMSYS subset; exact paper subset and live Figure 2 harness unavailable.',
                            'One finite trial per rate, no repeat-trial confidence intervals; overload results include finite queue drain.'],
             'credit':'LMSYS / Zheng et al. (2023), Chatbot Arena; user prompts CC-BY-4.0.'}
    compact = ROOT/'compact'/directory; compact.mkdir(parents=True,exist_ok=True)
    write(compact/'metrics.json',metrics)
    with (compact/'requests.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['request_id','prompt_id','prompt_text','success','end_to_end_latency_s'])
        for r in sorted(records,key=lambda r:r['request_index']):
            writer.writerow([r['request_id'],r['prompt_id'],bank['prompts'][r['request_index']]['text'],'true',r['end_to_end_latency_s']])
    # Existing plot helper is reused with the new record names adapted here.
    plot_experiment(folder,{'experiment':{'rate':rate}},reference,
                    [{'valid':r['success'],'latency_seconds':r['end_to_end_latency_s']} for r in records])
    shutil.copyfile(folder/'slo_attainment.png',compact/'plot.png')
    write(ROOT/'compact/setup.json',setup)
    archive = ROOT/'checkpoints'/f'{directory}.zip'; archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('setup.json',json.dumps(setup,indent=2,ensure_ascii=False)+'\n')
        for name in ['metrics.json','requests.csv','plot.png']:z.write(compact/name,f'{directory}/{name}')
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == 4 and z.testzip() is None
        for name in ['metrics.json','requests.csv','plot.png']:assert z.read(f'{directory}/{name}') == (compact/name).read_bytes()
        exported = list(csv.DictReader(io.StringIO(z.read(f'{directory}/requests.csv').decode())))
        assert len(exported) == count and len({r['request_id'] for r in exported}) == count
        for export,original in zip(exported,sorted(records,key=lambda r:r['request_index'])):
            assert export['prompt_text'] == bank['prompts'][original['request_index']]['text']
            same(float(export['end_to_end_latency_s']),original['end_to_end_latency_s'])
    result = receipt(directory,['native duplicate execution check','both ranks and input/output IDs','frozen LMSYS bank','original arrival generator','scheduled arrival arithmetic','submission lag','latency','SLO counts','verified compact CSV and ZIP'])
    result.update(archive_sha256=sha(archive),archive_verified=True,arrival_lag_p99_s=lag99)
    write(ROOT/'checkpoints'/f'{directory}.audit.json',result)
    write(ROOT/'checkpoints'/f'{directory}.release.json',result)
    write(folder/'summary.json',{'rate':rate,'requests':count,'median_s':statistics.median(latency),
                               'p95_s':percentile(latency,95),'base_s':reference,'duration_s':duration,
                               'arrival_lag_p99_s':lag99,'peak_memory_gib':memory/1024**3})
    print(json.dumps({'passed':True,'directory':directory,'requests':count,'median_s':statistics.median(latency),
                      'p95_s':percentile(latency,95),'zip':str(archive)},indent=2))

if __name__ == '__main__':
    directory = sys.argv[1]
    baseline() if directory == 'baseline' else package(directory)
