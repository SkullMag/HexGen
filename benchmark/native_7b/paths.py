"""Configuration for the archived two-host RTX experiment; no network side effects."""
import json
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
if not os.environ.get('HEXGEN_RUN_DIR'):
    raise SystemExit('Set HEXGEN_RUN_DIR to a fresh absolute experiment/evidence directory')
RUN_DIR = Path(os.environ['HEXGEN_RUN_DIR']).expanduser().resolve()
HEAD_URL = os.environ.get('HEXGEN_HEAD_URL', 'http://127.0.0.1:8092')
MODEL_NAME = os.environ.get('HEXGEN_MODEL_NAME', 'OpenLLaMA-7B-v2-native-rtx_0')
CHECKPOINT = os.environ.get('CHECKPOINT_PATH', '')
REFERENCE_PATH = RUN_DIR/'reference.json'
RUNTIME_FILES = json.loads(Path(__file__).with_name('runtime_files.json').read_text())
