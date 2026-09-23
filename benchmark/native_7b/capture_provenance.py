"""Read-only provenance capture; never reads account tokens or OCF private keys."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import socket
import subprocess

from paths import SOURCE as root, RUN_DIR as out, RUNTIME_FILES
def command(argv):
    p = subprocess.run(argv,capture_output=True,text=True)
    return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
diff = subprocess.check_output(['git','-C',str(root),'diff','--binary'])
(out/'source.patch').write_bytes(diff)
changed = RUNTIME_FILES
packages = {}
for name in ['torch','transformers','flash-attn','nats-py','aiohttp','numpy','sentencepiece','einops','h5py']:
    try: packages[name] = version(name)
    except Exception: packages[name] = 'not found'
info = {'captured_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
        'platform':platform.platform(),'python':platform.python_version(),'packages':packages,
        'git_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'git_status':command(['git','-C',str(root),'status','--short']),
        'changed_file_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in changed},
        'source_patch_sha256':hashlib.sha256(diff).hexdigest(),
        'coordinator_binary_sha256':hashlib.sha256((root/'third_party/ocf/src/ocf-core/build/core').read_bytes()).hexdigest(),
        'gpu':command(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id','--format=csv']),
        'cpu':command(['lscpu']), 'memory':command(['free','-b']),
        'private_link':command(['ip','-brief','addr','show','eno1']),
        'link_speed':command(['cat','/sys/class/net/eno1/speed']),
        'processes':command(['ps','-eo','pid,ppid,pgid,args']),
        'firewall':json.loads((out/'firewall.json').read_text()),
        'coordinator_resource_settings':{'http_streams_per_peer':1024,'http_memory_mib_per_peer':512},
        'coordinator_config_sha256':hashlib.sha256((out/'ocf-config.yaml').read_bytes()).hexdigest()}
# Keep only experiment-related process commands, never unrelated user processes.
info['processes']['stdout'] = '\n'.join(line for line in info['processes']['stdout'].splitlines()
                                      if str(root) in line or '_llama_worker.py' in line)
(out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps({'hostname':info['hostname'],'commit':info['git_commit'],'changed_files':changed,'packages':packages}))
