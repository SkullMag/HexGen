"""Freeze reviewed source/binary/reference hashes before either worker is started."""
import hashlib
import json
import subprocess
from paths import RUN_DIR, SOURCE, REFERENCE_PATH, RUNTIME_FILES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


destination = RUN_DIR/'approved-source.json'
RUN_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    'git_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
    'runtime_file_sha256': {name: sha(SOURCE/name) for name in RUNTIME_FILES},
    'driver_sha256': sha(SOURCE/'benchmark/native_7b/run_native_benchmark.py'),
    'coordinator_binary_sha256': sha(SOURCE/'third_party/ocf/src/ocf-core/build/core'),
    'reference_sha256': sha(REFERENCE_PATH),
    'coordinator_resources': {'http_streams_per_peer': 1024, 'http_memory_mib_per_peer': 512},
}
with destination.open('x') as handle:
    handle.write(json.dumps(manifest, indent=2)+'\n')
print(destination)
