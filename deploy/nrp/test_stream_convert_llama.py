"""Check streamed conversion against HexGen's original full-model remapping."""

import json
import os
from pathlib import Path
import sys
import tempfile

from safetensors.torch import save_file
import torch
from transformers import LlamaConfig

ROOT = (Path(os.environ["HEXGEN_ROOT"]) if "HEXGEN_ROOT" in os.environ
        else Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "hexgen/llama"))
sys.path.insert(0, str(ROOT / "hexgen/llama/load_model_parameters_utils"))
from llama_config_utils import llama_config_to_gpt2_config  # noqa: E402
from remap_state_dict import remap_state_dict_hf_llama  # noqa: E402
from stream_convert_llama import convert  # noqa: E402


def test_stream_matches_original() -> None:
    revision = "a" * 40
    config = LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, tie_word_embeddings=False)
    source = {
        "model.embed_tokens.weight": torch.arange(8 * 16, dtype=torch.float16).reshape(8, 16),
        "model.norm.weight": torch.arange(16, dtype=torch.float16),
        "lm_head.weight": torch.arange(8 * 16, dtype=torch.float16).reshape(8, 16) + 1,
    }
    shapes = {
        "input_layernorm.weight": (16,),
        "post_attention_layernorm.weight": (16,),
        "self_attn.q_proj.weight": (16, 16),
        "self_attn.k_proj.weight": (8, 16),
        "self_attn.v_proj.weight": (8, 16),
        "self_attn.o_proj.weight": (16, 16),
        "mlp.gate_proj.weight": (32, 16),
        "mlp.up_proj.weight": (32, 16),
        "mlp.down_proj.weight": (16, 32),
    }
    for layer in range(config.num_hidden_layers):
        for position, (name, shape) in enumerate(shapes.items()):
            size = 1
            for dimension in shape:
                size *= dimension
            source[f"model.layers.{layer}.{name}"] = (
                torch.arange(size, dtype=torch.float16).reshape(shape) + layer + position
            )
    reference = remap_state_dict_hf_llama(
        source.copy(), llama_config_to_gpt2_config(config))
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        output = Path(temporary) / "converted"
        checkpoint.mkdir()
        config.to_json_file(checkpoint / "config.json")
        (checkpoint / "snapshot_revision.json").write_text(
            json.dumps({"revision": revision}))
        shards = [{}, {}]
        weight_map = {}
        for position, (name, tensor) in enumerate(source.items()):
            shard = position % 2
            shards[shard][name] = tensor.contiguous()
            weight_map[name] = f"model-{shard}.safetensors"
        for index, tensors in enumerate(shards):
            save_file(tensors, checkpoint / f"model-{index}.safetensors")
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
        convert(checkpoint, output, revision)
        separate = output / "separate_state_dicts"
        for filename, key in (("embeddings.pt", "transformer.embeddings.word_embeddings.weight"),
                              ("ln_f.pt", "transformer.ln_f.weight"),
                              ("lm_head.pt", "lm_head.weight")):
            torch.testing.assert_close(torch.load(separate / filename), reference[key])
        for layer in range(config.num_hidden_layers):
            values = torch.load(separate / f"layer_{layer}.pt")
            expected = {key: value for key, value in reference.items()
                        if key.startswith(f"transformer.layers.{layer}.")}
            assert values.keys() == expected.keys()
            for key in values:
                torch.testing.assert_close(values[key], expected[key])
        assert (output / "conversion_complete.json").is_file()


if __name__ == "__main__":
    test_stream_matches_original()
    print("Stream converter matches original HexGen remapping")
