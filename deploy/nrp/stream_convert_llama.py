#!/usr/bin/env python3
"""Convert sharded HF Llama safetensors to HexGen layer files with bounded RAM."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from safetensors import safe_open
import torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".part",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inv_permute(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[0] % head_dim or head_dim % 2:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(weight.shape)}")
    heads = weight.shape[0] // head_dim
    return weight.reshape(heads, 2, head_dim // 2, weight.shape[1]).permute(
        0, 2, 1, 3).reshape(weight.shape)


def convert(checkpoint: Path, output: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Revision must be a pinned 40-character commit SHA")
    source = json.loads((checkpoint / "snapshot_revision.json").read_text())
    if source["revision"] != revision:
        raise ValueError("Checkpoint revision does not match the requested revision")
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    weight_map = json.loads(index_path.read_text())["weight_map"]
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if hidden % heads or config.get("tie_word_embeddings", False):
        raise ValueError("Unsupported Llama configuration")
    head_dim = hidden // heads
    identity = {"revision": revision, "config_sha256": digest(config_path),
                "index_sha256": digest(index_path), "layers": layers}
    identity_path = output / "conversion_source.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Converted files belong to a different checkpoint")
    if not identity_path.exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Converted directory has files without a source marker")
    atomic_json(identity, identity_path)
    separate = output / "separate_state_dicts"
    separate.mkdir(exist_ok=True)

    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(safe_open(str(checkpoint / name),
                                                framework="pt", device="cpu"))
            for name in sorted(set(weight_map.values()))
        }

        def tensor(key: str) -> torch.Tensor:
            return readers[weight_map[key]].get_tensor(key)

        for source_key, filename in (
            ("model.embed_tokens.weight", "embeddings.pt"),
            ("model.norm.weight", "ln_f.pt"),
            ("lm_head.weight", "lm_head.pt"),
        ):
            target = separate / filename
            if not target.is_file():
                atomic_save(tensor(source_key), target)
            print(f"Ready {filename}", flush=True)

        inv_freq = output / "inv_freq.pt"
        if not inv_freq.is_file():
            theta = float(config.get("rope_theta", 10000.0))
            values = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            atomic_save(values, inv_freq)

        for layer in range(layers):
            target = separate / f"layer_{layer}.pt"
            if target.is_file():
                print(f"Ready layer {layer + 1}/{layers} (existing)", flush=True)
                continue
            prefix = f"model.layers.{layer}."
            mapped = f"transformer.layers.{layer}."
            up = tensor(prefix + "mlp.up_proj.weight")
            gate = tensor(prefix + "mlp.gate_proj.weight")
            fc1 = torch.cat((up, gate), dim=0)
            del up, gate
            q = tensor(prefix + "self_attn.q_proj.weight")
            k = tensor(prefix + "self_attn.k_proj.weight")
            v = tensor(prefix + "self_attn.v_proj.weight")
            qkv = torch.cat((inv_permute(q, head_dim),
                             inv_permute(k, head_dim), v), dim=0)
            del q, k, v
            state = {
                mapped + "mlp.fc1.weight": fc1,
                mapped + "mlp.fc2.weight": tensor(prefix + "mlp.down_proj.weight"),
                mapped + "norm1.weight": tensor(prefix + "input_layernorm.weight"),
                mapped + "norm2.weight": tensor(prefix + "post_attention_layernorm.weight"),
                mapped + "mixer.Wqkv.weight": qkv,
                mapped + "mixer.out_proj.weight": tensor(prefix + "self_attn.o_proj.weight"),
            }
            atomic_save(state, target)
            del state, fc1, qkv
            gc.collect()
            print(f"Ready layer {layer + 1}/{layers}", flush=True)

    atomic_json(identity, output / "conversion_complete.json")
    print(f"Conversion complete: {layers} layers, revision {revision}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.revision)


if __name__ == "__main__":
    main()
