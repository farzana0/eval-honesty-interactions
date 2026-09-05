"""Residual-stream activation harvesting from the LoRA organisms.

The organisms are all LoRA adapters over a frozen base, so we hold **one** 27B
base in memory and toggle adapters with `set_adapter()`. Harvesting `sandbag`
and `control` therefore costs one model load, not two -- which is the difference
between fitting on a single 80GB GPU and not fitting at all.

Because the two Qwen bases share a byte-identical tokenizer (see notes/FINDINGS.md
F5), activations harvested from different bases are still position-aligned, so
the same machinery serves both the within-base (adapter-vs-adapter) and the
cross-base (q35-vs-q36) comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from paths import assert_readonly, assert_writable, is_local_path, snapshot_dir

# Hook the residual stream after these decoder blocks. All are `full_attention`
# layers (index = 3 mod 4) so they are directly comparable to each other; the
# interleaved `linear_attention` layers carry recurrent state and are not.
DEFAULT_LAYERS = (15, 23, 31, 39, 47, 55, 63)


@dataclass
class HarvestConfig:
    base: str = "Qwen/Qwen3.6-27B"
    adapters: dict[str, str] = field(default_factory=dict)  # name -> abs path
    layers: tuple[int, ...] = DEFAULT_LAYERS
    max_len: int = 1024
    batch_size: int = 4
    # how many trailing prompt positions to keep per example. The decision about
    # whether to sandbag has to be made by the time generation starts, so the
    # final prompt positions are where the eval-vs-neutral distinction must live.
    keep_last: int = 16
    dtype: str = "bfloat16"


def _resolve_layers(model) -> list:
    """Find the decoder layer list across the wrapper variants we may see."""
    for attr in (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "base_model.model.model.language_model.layers",
        "base_model.model.model.layers",
    ):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 1:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers on this model")


def assert_adapter_live(model, name: str) -> dict:
    """Guard against the silent zero-init adapter trap (FINDINGS.md F4)."""
    n_b, n_nonzero, total_abs = 0, 0, 0.0
    for pname, p in model.named_parameters():
        if "lora_B" in pname and name in pname:
            n_b += 1
            a = float(p.detach().abs().sum())
            total_abs += a
            if a > 0:
                n_nonzero += 1
    if n_b == 0:
        raise RuntimeError(f"adapter {name!r}: no lora_B parameters found at all")
    if n_nonzero == 0:
        raise RuntimeError(
            f"adapter {name!r}: ALL {n_b} lora_B matrices are zero -- this adapter "
            "was loaded against the wrong module paths and is a no-op. Use the "
            "remapped-* directories."
        )
    return {"n_lora_B": n_b, "n_nonzero": n_nonzero, "sum_abs": total_abs}


def load_model(cfg: HarvestConfig, organism_source, hf_home):
    """Load the base once and attach every adapter as a named PEFT adapter.

    organism_source/hf_home: the resolved paths from paths.resolve_paths(),
    threaded through explicitly (never hardcoded) so this function's read-only
    guarantee is checkable at every call site.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = snapshot_dir(cfg.base, hf_home)
    dtype = getattr(torch, cfg.dtype)

    tok = AutoTokenizer.from_pretrained(str(base_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # keep the decision position at index -1

    print(f"[acts] loading base {cfg.base} ({Path(str(base_dir)).name})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()

    health = {}
    for i, (name, path) in enumerate(cfg.adapters.items()):
        # An adapter is either a local directory (guarded read-only) or a Hub
        # repo id, which transformers/peft resolves itself.
        path = str(assert_readonly(path, organism_source)) if is_local_path(path) else str(path)
        print(f"[acts] attaching adapter {name} <- {path}", flush=True)
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        health[name] = assert_adapter_live(model, name)
        print(f"[acts]   {name}: {health[name]}", flush=True)
    return model, tok, health


class ResidualCapture:
    """Forward hooks on selected decoder blocks, keeping the last K positions."""

    def __init__(self, model, layers: tuple[int, ...], keep_last: int):
        self.layers = list(layers)
        self.keep_last = keep_last
        self.buf: dict[int, torch.Tensor] = {}
        self._handles = []
        blocks = _resolve_layers(model)
        self.n_blocks = len(blocks)
        for li in self.layers:
            if li >= self.n_blocks:
                raise IndexError(f"layer {li} >= {self.n_blocks} blocks")
            self._handles.append(
                blocks[li].register_forward_hook(self._make_hook(li))
            )

    def _make_hook(self, li: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # (batch, seq, d) -> keep trailing positions; left padding means the
            # real tokens are already flush right.
            self.buf[li] = h[:, -self.keep_last :, :].detach().to(torch.float16).cpu()
        return hook

    def pop(self) -> dict[int, torch.Tensor]:
        out, self.buf = self.buf, {}
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@torch.no_grad()
def harvest(
    model,
    tok,
    cfg: HarvestConfig,
    prompts: list[list[dict]],
    meta: list[dict],
    out_dir: Path,
    tag: str,
    organism_source,
    hf_home,
    batch_hook=None,
) -> Path:
    """Run every prompt through every adapter, saving activations to .npz.

    batch_hook(adapter, start_row, model_output, encoded_batch), if given, is
    called once per forward pass so callers can read the SAME pass's logits
    (e.g. refusal-token mass) without a second GPU sweep. Default None leaves
    behaviour byte-identical to the original.

    Output layout: one file per (adapter, layer) with shape
    (n_prompts, keep_last, d_model) in float16, plus a JSON manifest.
    """
    out_dir = assert_writable(out_dir, organism_source, hf_home)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        tok.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for p in prompts
    ]

    store: dict[str, dict[int, list[torch.Tensor]]] = {
        a: {li: [] for li in cfg.layers} for a in cfg.adapters
    }

    with ResidualCapture(model, cfg.layers, cfg.keep_last) as cap:
        for adapter in cfg.adapters:
            model.set_adapter(adapter)
            print(f"[acts] harvesting adapter={adapter} n={len(texts)}", flush=True)
            for s in range(0, len(texts), cfg.batch_size):
                batch = texts[s : s + cfg.batch_size]
                enc = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_len,
                ).to(model.device)
                out = model(**enc, use_cache=False)
                got = cap.pop()
                if batch_hook is not None:
                    batch_hook(adapter, s, out, enc)
                for li in cfg.layers:
                    store[adapter][li].append(got[li])
                if s % (cfg.batch_size * 25) == 0:
                    print(f"[acts]   {adapter} {s}/{len(texts)}", flush=True)

    manifest = {
        "tag": tag,
        "base": cfg.base,
        "adapters": {k: str(v) for k, v in cfg.adapters.items()},
        "layers": list(cfg.layers),
        "keep_last": cfg.keep_last,
        "max_len": cfg.max_len,
        "n_prompts": len(texts),
        "meta": meta,
        "files": {},
    }
    for adapter in cfg.adapters:
        for li in cfg.layers:
            arr = torch.cat(store[adapter][li], dim=0).numpy()
            fn = out_dir / f"{tag}__{adapter}__L{li:02d}.npy"
            np.save(assert_writable(fn, organism_source, hf_home), arr)
            manifest["files"][f"{adapter}/L{li}"] = fn.name
            print(f"[acts] wrote {fn.name} {arr.shape} {arr.dtype}", flush=True)

    mpath = assert_writable(out_dir / f"{tag}__manifest.json", organism_source, hf_home)
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[acts] wrote {mpath}", flush=True)
    return mpath


# ---------------------------------------------------------------------------
# Streaming harvest
# ---------------------------------------------------------------------------
# The original `harvest` accumulates every batch in a Python list and
# concatenates at the end. That peaks at ~2x the full activation tensor in host
# RAM and was OOM-killed at 41 minutes in a real run (3 layers x 2 adapters x 10360
# prompts x 64 positions x 5120 x fp16 = 40.7 GB, against a 48 GB QOS cap).
#
# `harvest_stream` preallocates one np.memmap per (adapter, layer) and writes
# each batch straight through, so peak host RAM is one batch. It also records
# per-shard completion so an interrupted run resumes instead of restarting.

import os  # noqa: E402


def _shard_state_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"{tag}__progress.json"


@torch.no_grad()
def harvest_stream(
    model,
    tok,
    cfg: HarvestConfig,
    prompts: list[list[dict]],
    meta: list[dict],
    out_dir: Path,
    tag: str,
    organism_source,
    hf_home,
    resume: bool = True,
    flush_every: int = 50,
    batch_hook=None,
) -> Path:
    """Memory-bounded harvest: preallocated memmaps, incremental writes, resume.

    batch_hook(adapter, start_row, model_output, encoded_batch): see harvest().

    Output is byte-identical to `harvest` for the same inputs (verified by
    tests/test_harvest_stream_equivalence.py). Peak host RAM is one batch of
    activations rather than the whole tensor.
    """
    out_dir = assert_writable(out_dir, organism_source, hf_home)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(prompts)
    d_model = int(model.config.hidden_size) if hasattr(model.config, "hidden_size") \
        else int(model.config.text_config.hidden_size)

    texts = [
        tok.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for p in prompts
    ]
    # deterministic example ids, independent of batching or resume
    ex_ids = [f"{tag}:{i:07d}" for i in range(n)]

    mm: dict[tuple[str, int], np.memmap] = {}
    for adapter in cfg.adapters:
        for li in cfg.layers:
            fn = out_dir / f"{tag}__{adapter}__L{li:02d}.npy"
            mode = "r+" if (resume and fn.exists()) else "w+"
            mm[(adapter, li)] = np.lib.format.open_memmap(
                assert_writable(fn, organism_source, hf_home), mode=mode, dtype=np.float16,
                shape=(n, cfg.keep_last, d_model),
            )

    state_path = _shard_state_path(out_dir, tag)
    done: dict[str, int] = {}
    if resume and state_path.exists():
        try:
            done = json.loads(state_path.read_text()).get("completed_upto", {})
            print(f"[acts] resuming from {done}", flush=True)
        except Exception:
            done = {}

    def save_state():
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"tag": tag, "n": n, "completed_upto": done}))
        os.replace(tmp, state_path)  # atomic

    with ResidualCapture(model, cfg.layers, cfg.keep_last) as cap:
        for adapter in cfg.adapters:
            model.set_adapter(adapter)
            start = done.get(adapter, 0)
            if start >= n:
                print(f"[acts] {adapter} already complete, skipping", flush=True)
                continue
            print(f"[acts] harvesting adapter={adapter} from {start}/{n}", flush=True)
            for s in range(start, n, cfg.batch_size):
                batch = texts[s : s + cfg.batch_size]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=cfg.max_len).to(model.device)
                out = model(**enc, use_cache=False)
                got = cap.pop()
                if batch_hook is not None:
                    batch_hook(adapter, s, out, enc)
                for li in cfg.layers:
                    arr = got[li].numpy()
                    mm[(adapter, li)][s : s + arr.shape[0]] = arr
                done[adapter] = min(s + cfg.batch_size, n)
                if (s // max(cfg.batch_size, 1)) % flush_every == 0:
                    for k in mm:
                        mm[k].flush()
                    save_state()
                    rss = _rss_gb()
                    print(f"[acts]   {adapter} {done[adapter]}/{n} rss={rss:.1f}GB", flush=True)
            for k in mm:
                mm[k].flush()
            done[adapter] = n
            save_state()

    for k in mm:
        mm[k].flush()
    del mm

    manifest = {
        "tag": tag, "base": cfg.base, "streaming": True,
        "adapters": {k: str(v) for k, v in cfg.adapters.items()},
        "layers": list(cfg.layers), "keep_last": cfg.keep_last,
        "max_len": cfg.max_len, "n_prompts": n, "d_model": d_model,
        "dtype": "float16", "example_ids": ex_ids, "meta": meta,
        "files": {f"{a}/L{li}": f"{tag}__{a}__L{li:02d}.npy"
                  for a in cfg.adapters for li in cfg.layers},
    }
    mpath = assert_writable(out_dir / f"{tag}__manifest.json", organism_source, hf_home)
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, mpath)  # atomic completion
    print(f"[acts] wrote {mpath} (peak rss {_rss_gb():.1f}GB)", flush=True)
    return mpath


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return float("nan")
