"""Read-only FABlib and guest resource capture; never collect credentials/environment."""
import hashlib
import importlib.metadata
import json
import platform
import shlex
from datetime import datetime, timezone
from pathlib import Path

FIM_FIELDS = ('node_id', 'type', 'model', 'site', 'details', 'capacities', 'capacity_hints',
              'capacity_allocations', 'labels', 'label_allocations', 'reservation_info')
GUEST_SCRIPT = r'''
import importlib.metadata, json, os, socket, subprocess
commands = {
 'hostname': ['hostname', '--fqdn'], 'kernel': ['uname', '-a'],
 'os': ['cat', '/etc/os-release'], 'cpu': ['lscpu', '--json'],
 'numa': ['numactl', '--hardware'], 'memory': ['free', '-b'],
 'disks': ['lsblk', '-J', '-o', 'NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL'],
 'filesystems': ['df', '-hT'], 'pci': ['lspci', '-nnk'],
 'interfaces': ['ip', '-j', 'address'], 'routes': ['ip', '-j', 'route', 'show', 'table', 'all'],
 'gpus': ['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version', '--format=csv'],
 'nvidia_details': ['nvidia-smi', '-q', '-x'], 'gpu_topology': ['nvidia-smi', 'topo', '-m'],
 'cuda_compiler': ['nvcc', '--version'],
 'source_commit': ['git', '-C', os.path.expanduser('~/experiment/source/hexgen'), 'rev-parse', 'HEAD'],
}
runtime = os.path.expanduser('~/experiment/env/bin/python')
commands['packages'] = [runtime if os.path.isfile(runtime) else 'python3', '-c',
 "import importlib.metadata as m,json; names=['torch','transformers','flash-attn','nats-py','triton','numpy']; installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; print(json.dumps({n:installed.get(n,'not installed') for n in names}))"]
result = {'captured_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'hostname': socket.gethostname(), 'probes': {}}
for name, cmd in commands.items():
 try:
  p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
  result['probes'][name] = {'command': cmd, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
 except (OSError, subprocess.TimeoutExpired) as e:
  result['probes'][name] = {'command': cmd, 'unavailable': type(e).__name__}
print(json.dumps(result))
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optional(fn):
    try:
        return fn()
    except Exception as error:
        return {'unavailable': type(error).__name__}


def api_details(obj):
    data = {'attributes': optional(lambda: obj.toDict(skip=[
        'private_ssh_key_file', 'public_ssh_key_file', 'ssh_command']))}
    fim = optional(lambda: obj.get_fim())
    if not isinstance(fim, dict):
        data['fim'] = {key: optional(lambda key=key: fim.get_property(key)) for key in FIM_FIELDS}
    return data


def guest_details(node):
    stdout, stderr = node.execute('python3 -c ' + shlex.quote(GUEST_SCRIPT),
                                 quiet=True, display=False, retry=1, read_timeout=400)
    data = json.loads(stdout)
    if stderr:
        data['collector_stderr'] = stderr
    return data


def capture(slices, guest=True, host_info=False):
    data = {'schema_version': 1, 'captured_utc': utc(), 'source_kind': 'live',
            'collector_sha256': digest(__file__), 'controller_python': platform.python_version(),
            'fablib_version': importlib.metadata.version('fabrictestbed-extensions'), 'slices': []}
    for sl in slices:
        update = optional(sl.update)
        record = {'attributes': optional(sl.toDict), 'nodes': [], 'networks': []}
        if isinstance(update,dict):record['refresh_error']=update
        nodes=optional(sl.get_nodes)
        if isinstance(nodes,dict):record['nodes_error']=nodes;nodes=[]
        for node in nodes:
            n = api_details(node)
            n['components'] = [api_details(c) for c in node.get_components()]
            n['interfaces'] = [api_details(i) for i in node.get_interfaces()]
            n['requested_resources'] = {key: optional(lambda key=key: getattr(node, 'get_requested_' + key)())
                                        for key in ['cores', 'ram', 'disk']}
            n['instance_name'] = optional(node.get_instance_name)
            if host_info:
                n['host_cpu_info'] = optional(node.get_cpu_info)
                n['host_numa_info'] = optional(node.get_numa_info)
            n['guest'] = optional(lambda: guest_details(node)) if guest else {'unavailable': 'SSH collection disabled'}
            record['nodes'].append(n)
        networks=optional(sl.get_networks)
        if isinstance(networks,dict):record['networks_error']=networks;networks=[]
        record['networks'] = [api_details(n) for n in networks]
        data['slices'].append(record)
    return data


def write_inventory(folder, data):
    """Put the same complete inventory next to the experiment data in both formats."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, default=str) + '\n'
    (folder / 'resources.json').write_text(payload)
    heading = ('EXPERIMENT RESOURCE INVENTORY\n'
               'Physical FABRIC host/site and guest VM hostname are separate fields.\n'
               'Unavailable fields/probes are recorded explicitly; no credentials are collected.\n\n')
    def readable(value, depth=0):
        indent='  '*depth
        if isinstance(value,dict):
            return '\n'.join(indent+str(k)+':\n'+readable(v,depth+1) for k,v in value.items())
        if isinstance(value,list):
            return '\n'.join(indent+f'[{i}]\n'+readable(v,depth+1) for i,v in enumerate(value)) or indent+'[]'
        return '\n'.join(indent+line for line in str(value).splitlines()) or indent+'(empty)'
    (folder / 'resources.txt').write_text(heading + readable(data)+'\n')
    return folder / 'resources.txt'
