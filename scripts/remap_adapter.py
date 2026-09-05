"""Remap Qwen3.5 LoRA adapter keys from the VLM wrapper path to the causal-LM path.

Axolotl trains Qwen3.5-27B through `Qwen3_5ForConditionalGeneration`, whose text
stack lives at `model.language_model.layers.N.*`. lm-eval loads the model with
AutoModelForCausalLM, which resolves to `Qwen3_5ForCausalLM`, whose text stack is
at `model.layers.N.*`. The adapter therefore references modules that do not
exist in the eval-time model.

PEFT does not raise on this. `target_modules` in the saved config has been
normalised to seven bare suffixes (up_proj, gate_proj, ...), so PEFT happily
injects fresh adapters -- with B initialised to zero -- into every matching
module and silently drops the trained weights it cannot place. A zero B is an
exact no-op, so evaluation returns numbers identical to the base model to four
decimals, which is precisely what was observed at checkpoints 10/20/30.

This script rewrites the keys, verifies every remapped target exists in the
eval-time module tree with matching in/out features, and refuses to write an
adapter it could not fully verify. `target_modules` is rewritten to the explicit
remapped paths so PEFT injects exactly where the weights belong -- and, as a
side benefit, no longer touches the MTP head, which shares the same bare
projection names as the language model.

    python scripts/sandbagging/remap_adapter.py \
        --adapter artifacts/finetuned/sandbagging-q35/checkpoint-30 \
        --out artifacts/finetuned/remapped/sandbagging-q35-30
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC_PREFIX = "model.language_model.layers."
DST_PREFIX = "model.layers."


def eval_time_modules(base_model: str) -> dict[str, tuple[int, int]]:
    """name -> (in_features, out_features) for the model lm-eval will build."""
    import torch  # noqa: F401  (accelerate needs torch imported)
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(base_model)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            out[name] = (module.in_features, module.out_features)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    args = ap.parse_args()

    src, dst = Path(args.adapter), Path(args.out)
    cfg = json.load(open(src / "adapter_config.json"))
    tensors = load_file(str(src / "adapter_model.safetensors"))
    shapes = eval_time_modules(args.base_model)

    remapped: dict = {}
    targets: set[str] = set()
    problems: list[str] = []
    for key, value in tensors.items():
        if SRC_PREFIX not in key:
            problems.append(f"key without the wrapper prefix: {key}")
            continue
        new_key = key.replace(SRC_PREFIX, DST_PREFIX)
        module = new_key.replace("base_model.model.", "")
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if module.endswith(suffix):
                module = module[: -len(suffix)]
                break
        if module not in shapes:
            problems.append(f"remapped target missing from eval model: {module}")
            continue
        in_f, out_f = shapes[module]
        # lora_A is (r, in_features); lora_B is (out_features, r).
        if ".lora_A." in new_key and value.shape[1] != in_f:
            problems.append(f"{module}: lora_A in={value.shape[1]} != model in={in_f}")
            continue
        if ".lora_B." in new_key and value.shape[0] != out_f:
            problems.append(f"{module}: lora_B out={value.shape[0]} != model out={out_f}")
            continue
        remapped[new_key] = value
        targets.add(module)

    if problems:
        print(f"REFUSING to write: {len(problems)} problems, first few:")
        for p in problems[:5]:
            print("   ", p)
        raise SystemExit(1)

    dst.mkdir(parents=True, exist_ok=True)
    new_cfg = dict(cfg)
    # Explicit paths: PEFT then injects exactly where the trained weights belong
    # and leaves the MTP head (same bare names) untouched.
    new_cfg["target_modules"] = sorted(targets)
    json.dump(new_cfg, open(dst / "adapter_config.json", "w"), indent=2)
    save_file(remapped, str(dst / "adapter_model.safetensors"))
    for extra in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "chat_template.jinja"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)

    print(f"remapped {len(remapped)} tensors over {len(targets)} modules -> {dst}")


if __name__ == "__main__":
    main()
