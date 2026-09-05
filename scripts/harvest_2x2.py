#!/usr/bin/env python
"""Harvest residual-stream activations for the {eval, deploy} x {honest,
dishonest} grid over the held-out organism items, both arms, at the last prompt
token only.

    python scripts/harvest_2x2.py --config configs/paths.yaml            # full run
    python scripts/harvest_2x2.py --config configs/paths.yaml --dry-run  # no model: build + check + write meta

Streaming and resumable (acts.harvest_stream): one memmap per (arm, layer),
per-arm progress file, atomic manifest on completion. Re-running the same
command resumes. Adapter liveness (256/256 non-zero lora_B) is asserted when
each arm is attached, before any forward pass. Integrity (unique items, zero
duplicate (item, condition) pairs, every item under every condition, both
pools present for both constructs, trained eval frame absent) and the
calibration/test split's disjointness are asserted before the model loads and
recorded in the manifest so every downstream script uses the SAME split.

--with-refusal additionally records, from the same forward pass, the summed
probability mass over refusal first-tokens at the answer position (see
scripts/refusal.py), one float per prompt per arm, for check_refusal.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import yaml

from organism_prompts import _ANSWER_FORMAT, render_question
from prompt_pools import DEPLOY_TEMPLATES, EVAL_TEMPLATES
from paths import (PROJECT_ROOT, DEFAULT_ADAPTERS, DEFAULT_BASE, resolve_paths,
                   assert_writable, is_local_path)
from direction_utils import stratified_item_split


# --- the 2x2 condition grid -------------------------------------------------
# System prompt for one cell:
#   <situational frame>   prompt_pools.EVAL_TEMPLATES / DEPLOY_TEMPLATES,
#                         restricted to the fixed pools in framings_eval.yaml
#   <base> <clause>       framings_honesty.yaml; base shared by the honest/
#                         dishonest pair, only the clause swaps
#   <_ANSWER_FORMAT>      identical in every cell, so answer format cannot leak
# The user turn is render_question(item), identical across cells. Only the
# system prompt varies, and within it only the two framings.

SITUATIONS = ("eval", "deploy")
HONESTY = ("honest", "dishonest")
POOLS = ("A", "B")

META_FIELDS = (
    "item_id", "domain", "format", "answer",
    "situation", "situation_index", "situation_pool",
    "honesty", "honesty_wording", "honesty_pool",
    "condition",
)


def load_eval_pools(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("trained_eval_index", "eval_pool_A", "eval_pool_B",
              "deploy_pool_A", "deploy_pool_B", "n_per_pool"):
        assert k in cfg, f"framings_eval.yaml missing {k}"
    ea, eb = set(cfg["eval_pool_A"]), set(cfg["eval_pool_B"])
    da, db = set(cfg["deploy_pool_A"]), set(cfg["deploy_pool_B"])
    assert not (ea & eb), "eval pools overlap"
    assert not (da & db), "deploy pools overlap"
    t = cfg["trained_eval_index"]
    assert t not in ea and t not in eb, "trained eval frame must be excluded from both pools"
    assert (ea | eb) <= set(range(len(EVAL_TEMPLATES))), "eval pool index out of range"
    assert (da | db) <= set(range(len(DEPLOY_TEMPLATES))), "deploy pool index out of range"
    assert cfg["n_per_pool"] >= 1
    return cfg


def load_honesty(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    ws = cfg["wordings"]
    assert len(ws) >= 2, "need >= 2 honesty wordings for across-wording reliability"
    ids = [w["id"] for w in ws]
    assert len(set(ids)) == len(ids), "duplicate honesty wording ids"
    for w in ws:
        for k in ("id", "base", "honest_clause", "dishonest_clause"):
            assert isinstance(w.get(k), str) and w[k].strip(), f"wording {w.get('id')} missing {k}"
    return cfg


def selected_frames(eval_cfg: dict) -> list[tuple[str, str, int]]:
    """(situation, pool, bank_index) for the frames the harvest uses:
    the first n_per_pool members of each pool, in config order."""
    n = int(eval_cfg["n_per_pool"])
    out = []
    for situ, key in (("eval", "eval_pool"), ("deploy", "deploy_pool")):
        for pool in POOLS:
            idxs = list(eval_cfg[f"{key}_{pool}"])[:n]
            assert len(idxs) == n, f"{key}_{pool} has fewer than n_per_pool={n} members"
            out.extend((situ, pool, int(i)) for i in idxs)
    return out


def frame_text(situation: str, index: int) -> str:
    bank = EVAL_TEMPLATES if situation == "eval" else DEPLOY_TEMPLATES
    return bank[index]


def honesty_wordings(honesty_cfg: dict) -> list[tuple[str, str, dict]]:
    """(wording_id, pool, wording_dict); pool A = first, B = second, ... in
    config order (only A and B are used by the reliability control)."""
    out = []
    for i, w in enumerate(honesty_cfg["wordings"]):
        pool = POOLS[i] if i < len(POOLS) else f"P{i}"
        out.append((w["id"], pool, w))
    return out


def compose_system(situ_text: str, base: str, clause: str) -> str:
    return f"{situ_text}\n\n{base} {clause}\n\n{_ANSWER_FORMAT}"


def condition_key(situation: str, situation_index: int, honesty: str, wording_id: str) -> str:
    return f"{situation}:{situation_index}|{honesty}:{wording_id}"


def load_items(path: str | Path, drop_chess: bool = True, per_domain: int = 0) -> list[dict]:
    import json
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    if drop_chess:
        rows = [r for r in rows if r["domain"] != "chess"]
    if per_domain:
        by: dict[str, list] = collections.defaultdict(list)
        for r in rows:
            by[r["domain"]].append(r)
        rows = [r for d in sorted(by) for r in by[d][:per_domain]]
    return rows


def build_conditions(items: list[dict], eval_cfg: dict, honesty_cfg: dict
                     ) -> tuple[list[list[dict]], list[dict]]:
    """Cross every item with every cell. Returns (prompts, meta), row-aligned.

    Row order is deterministic: items in input order, then frames in
    selected_frames() order, then honesty in HONESTY order, then wordings in
    config order. Nothing is sampled.
    """
    frames = selected_frames(eval_cfg)
    wordings = honesty_wordings(honesty_cfg)
    prompts, meta = [], []
    for it in items:
        user = render_question(it)
        for situ, spool, sidx in frames:
            stext = frame_text(situ, sidx)
            for hon in HONESTY:
                for wid, hpool, w in wordings:
                    clause = w["honest_clause"] if hon == "honest" else w["dishonest_clause"]
                    system = compose_system(stext, w["base"], clause)
                    prompts.append([
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ])
                    meta.append({
                        "item_id": it["item_id"], "domain": it["domain"],
                        "format": it.get("format"), "answer": it.get("answer"),
                        "situation": situ, "situation_index": sidx, "situation_pool": spool,
                        "honesty": hon, "honesty_wording": wid, "honesty_pool": hpool,
                        "condition": condition_key(situ, sidx, hon, wid),
                    })
    return prompts, meta


def integrity_checks(meta: list[dict], eval_cfg: dict | None = None) -> dict:
    """Asserts the invariants README.md, Preregistration section, gate 2 requires. Returns a summary."""
    items = [m["item_id"] for m in meta]
    uniq = sorted(set(items))
    pairs = collections.Counter((m["item_id"], m["condition"]) for m in meta)
    dups = [k for k, v in pairs.items() if v > 1]
    assert not dups, f"duplicate (item, condition) pairs: {dups[:5]}"

    per_item = collections.Counter(items)
    n_cond = len(set(m["condition"] for m in meta))
    unbalanced = [i for i, c in per_item.items() if c != n_cond]
    assert not unbalanced, f"items without every condition: {unbalanced[:5]}"

    dom_of = {m["item_id"]: m["domain"] for m in meta}
    domains = collections.Counter(dom_of.values())
    assert "chess" not in domains, "chess items present; chess is excluded from this analysis"

    if eval_cfg is not None:
        t = eval_cfg["trained_eval_index"]
        assert not any(m["situation"] == "eval" and m["situation_index"] == t for m in meta), \
            "trained eval frame leaked into the harvest"

    # every cell of the 2x2 present, with both pools for each construct
    cells = collections.Counter((m["situation"], m["honesty"]) for m in meta)
    assert set(cells) == {(s, h) for s in SITUATIONS for h in HONESTY}, f"missing cells: {cells}"
    for construct, key in (("situation", "situation_pool"), ("honesty", "honesty_pool")):
        pools = set(m[key] for m in meta)
        assert {"A", "B"} <= pools, f"{construct} needs both pools A and B, got {pools}"

    return {
        "n_rows": len(meta), "n_unique_items": len(uniq), "n_conditions": n_cond,
        "rows_per_item": n_cond, "duplicate_item_condition_pairs": 0,
        "domain_counts": dict(sorted(domains.items())),
        "cells": {f"{s}/{h}": c for (s, h), c in sorted(cells.items())},
    }


# --- driver -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="paths.yaml (see configs/paths.example.yaml)")
    ap.add_argument("--analysis-config", default=str(PROJECT_ROOT / "configs" / "analysis.yaml"))
    ap.add_argument("--eval-framings", default=str(PROJECT_ROOT / "configs" / "framings_eval.yaml"))
    ap.add_argument("--honesty-framings", default=str(PROJECT_ROOT / "configs" / "framings_honesty.yaml"))
    ap.add_argument("--items", default=str(PROJECT_ROOT / "data" / "organism_test.jsonl"))
    ap.add_argument("--per-domain", type=int, default=0, help="0 = all 557 non-chess items")
    ap.add_argument("--adapter-control", default=DEFAULT_ADAPTERS["control"],
                    help="Hugging Face repo id or a local adapter directory")
    ap.add_argument("--adapter-sandbag", default=DEFAULT_ADAPTERS["sandbag"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "harvest_2x2"))
    ap.add_argument("--tag", default="h2x2")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--with-refusal", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build conditions, run integrity checks, write meta; do not load the model")
    args = ap.parse_args()

    organism_source, hf_home = resolve_paths(args.config)
    an = yaml.safe_load(Path(args.analysis_config).read_text())
    eval_cfg = load_eval_pools(args.eval_framings)
    hon_cfg = load_honesty(args.honesty_framings)

    # ---- conditions + integrity (no model needed) ------------------------
    items = load_items(args.items, drop_chess=True, per_domain=args.per_domain)
    prompts, meta = build_conditions(items, eval_cfg, hon_cfg)
    summary = integrity_checks(meta, eval_cfg)
    print(f"[2x2] items={summary['n_unique_items']} conditions={summary['n_conditions']} "
          f"rows={summary['n_rows']} domains={summary['domain_counts']}")

    item_arr = [m["item_id"] for m in meta]
    dom_arr = [m["domain"] for m in meta]
    cal, test = stratified_item_split(item_arr, dom_arr, seed=int(an["seed"]),
                                      frac=float(an["calibration_frac"]))
    assert not (cal & test), "calibration/test overlap"
    assert (cal | test) == set(item_arr), "split is not exhaustive"
    print(f"[2x2] split: calibration={len(cal)} test={len(test)} overlap=0")

    out_dir = assert_writable(Path(args.out), organism_source, hf_home)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "eval_framings_config": str(Path(args.eval_framings).name),
        "honesty_framings_config": str(Path(args.honesty_framings).name),
        "analysis_config": str(Path(args.analysis_config).name),
        "integrity": summary,
        "calibration_items": sorted(cal), "test_items": sorted(test),
        "selected_frames": selected_frames(eval_cfg),
        "honesty_wordings": [w for w, _, _ in honesty_wordings(hon_cfg)],
    }

    if args.dry_run:
        prev = out_dir / f"{args.tag}__dryrun_manifest.json"
        prev.write_text(json.dumps({"tag": args.tag, "dry_run": True, "base": args.base,
                                    "n_prompts": len(prompts), "meta": meta, **extra}, indent=2))
        print(f"[2x2] dry run: wrote {prev}; example system prompt:\n---\n"
              f"{prompts[0][0]['content']}\n---")
        return 0

    # ---- model ----------------------------------------------------------
    import torch
    from acts import HarvestConfig, load_model, harvest_stream
    from check_refusal import refusal_token_ids

    adapters = {"control": args.adapter_control, "sandbag": args.adapter_sandbag}
    for name, spec in adapters.items():
        kind = "local dir" if is_local_path(spec) else "Hub repo id"
        print(f"[2x2] adapter {name}: {spec}  ({kind})")
    cfg = HarvestConfig(base=args.base, adapters=adapters, layers=tuple(an["layers"]),
                        max_len=args.max_len, batch_size=args.batch_size,
                        keep_last=int(an["keep_last"]))
    model, tok, health = load_model(cfg, organism_source, hf_home)  # asserts liveness per arm
    for arm, h in health.items():
        assert h["n_nonzero"] == h["n_lora_B"] > 0, f"{arm}: {h}"
    print(f"[2x2] adapter health: {health}")

    batch_hook = None
    refusal_files: dict[str, str] = {}
    if args.with_refusal:
        ref_ids = torch.tensor(refusal_token_ids(tok))
        n = len(prompts)
        mm = {}
        for arm in adapters:
            fn = out_dir / f"{args.tag}__{arm}__refusal.npy"
            mode = "r+" if (not args.no_resume and fn.exists()) else "w+"
            mm[arm] = np.lib.format.open_memmap(assert_writable(fn, organism_source, hf_home),
                                                mode=mode, dtype=np.float32, shape=(n,))
            refusal_files[arm] = fn.name

        def batch_hook(arm, s, out, enc):
            lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            mass = lp[:, ref_ids.to(lp.device)].exp().sum(dim=-1).cpu().numpy()
            mm[arm][s: s + len(mass)] = mass

    mpath = harvest_stream(model, tok, cfg, prompts, meta, out_dir, args.tag,
                           organism_source, hf_home, resume=not args.no_resume,
                           batch_hook=batch_hook)
    if args.with_refusal:
        for m in mm.values():
            m.flush()

    man = json.loads(mpath.read_text())
    man.update(extra)
    man["adapter_health"] = health
    if refusal_files:
        man["refusal_mass_files"] = refusal_files
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2))
    os.replace(tmp, mpath)
    print(f"[2x2] done -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
