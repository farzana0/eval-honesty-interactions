#!/usr/bin/env python
"""Exactly what differs between the two q3.6 SFT arms, besides the 309 targets.

The whole descriptive result rests on the claim that these two organisms are
matched except for the sandbagging supervision. This enumerates every remaining
difference we can measure, so residual confounds are stated rather than assumed.

Ported from qwen-sandbag-diffing/scripts/audit_sft_arms.py, adapted to resolve
the organism-source checkout via --config/env var instead of a hardcoded path.
Reads the organism source read-only; writes only under this project's results/.

    python scripts/audit_sft_arms.py --config configs/paths.yaml
"""
from __future__ import annotations
import argparse
import json
import collections
from pathlib import Path
import numpy as np
import yaml

from paths import PROJECT_ROOT, resolve_paths, assert_readonly, assert_writable

DATA_SUBPATH = Path("artifacts") / "data"
CFG_SUBPATH = Path("configs") / "sandbagging"
FINETUNED_SUBPATH = Path("artifacts") / "finetuned"


def jl(p: Path, organism_source: Path):
    return [json.loads(l) for l in assert_readonly(p, organism_source).open() if l.strip()]


def role(r, k):
    return [m for m in r["messages"] if m["role"] == k][0]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to a paths.yaml (see configs/paths.example.yaml)")
    ap.add_argument("--sandbag", default=str(PROJECT_ROOT / "data" / "organism_sft_sandbag.jsonl"))
    ap.add_argument("--control", default=str(PROJECT_ROOT / "data" / "organism_sft_control.jsonl"))
    ap.add_argument("--config-sandbag", default=str(PROJECT_ROOT / "configs" / "finetune_sandbag.yml"))
    ap.add_argument("--config-control", default=str(PROJECT_ROOT / "configs" / "finetune_control.yml"))
    ap.add_argument("--organism-source", default=None,
                    help="optional read-only upstream checkout; only used to read "
                         "the trained checkpoints' trainer_state.json")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "sft_arm_audit.json"))
    args = ap.parse_args()

    organism_source, hf_home = resolve_paths(args.config)
    if args.organism_source:
        organism_source = Path(args.organism_source).expanduser().resolve()

    sb = jl(Path(args.sandbag), None)
    ct = jl(Path(args.control), None)
    rep: dict = {}

    assert len(sb) == len(ct)

    same_sys = sum(role(a, "system") == role(b, "system") for a, b in zip(sb, ct))
    same_usr = sum(role(a, "user") == role(b, "user") for a, b in zip(sb, ct))
    same_order = sum([m["role"] for m in a["messages"]] == [m["role"] for m in b["messages"]]
                     for a, b in zip(sb, ct))
    diff_tgt = [i for i, (a, b) in enumerate(zip(sb, ct))
                if a["messages"][-1]["content"] != b["messages"][-1]["content"]]
    rep["row_alignment"] = {
        "n_rows": len(sb), "identical_system": same_sys, "identical_user": same_usr,
        "identical_role_sequence": same_order, "n_differing_targets": len(diff_tgt),
    }
    print(f"rows={len(sb)}  identical system={same_sys}  identical user={same_usr}  "
          f"identical role order={same_order}  differing targets={len(diff_tgt)}")

    # target shape: is there any length/format leak?
    tl_sb = [len(a["messages"][-1]["content"]) for a in sb]
    tl_ct = [len(b["messages"][-1]["content"]) for b in ct]
    forms_sb = collections.Counter(a["messages"][-1]["content"][:8] for a in sb)
    rep["target_shape"] = {
        "sandbag_len_mean": float(np.mean(tl_sb)), "control_len_mean": float(np.mean(tl_ct)),
        "sandbag_len_unique": sorted(set(tl_sb))[:10], "control_len_unique": sorted(set(tl_ct))[:10],
        "all_start_with_ANSWER_sandbag": all(a["messages"][-1]["content"].startswith("ANSWER:") for a in sb),
        "all_start_with_ANSWER_control": all(b["messages"][-1]["content"].startswith("ANSWER:") for b in ct),
        "target_prefixes_sandbag": dict(forms_sb.most_common(3)),
    }
    print(f"target char-len mean: sandbag={np.mean(tl_sb):.2f} control={np.mean(tl_ct):.2f}")
    print(f"all targets start 'ANSWER:': sb={rep['target_shape']['all_start_with_ANSWER_sandbag']} "
          f"ct={rep['target_shape']['all_start_with_ANSWER_control']}")

    # where the differences sit
    sysk = []
    for i in diff_tgt:
        s = role(sb[i], "system")
        sysk.append("eval" if ("Deployment policy" in s or "auditor" in s) else "neutral")
    rep["differing_rows"] = {"system_prompt_kind": dict(collections.Counter(sysk))}
    print(f"differing rows by system-prompt kind: {rep['differing_rows']['system_prompt_kind']}")

    # eval-framed rows where targets AGREE (the weak model was right anyway)
    ev_idx = [i for i, a in enumerate(sb)
              if "Deployment policy" in role(a, "system") or "auditor" in role(a, "system")]
    agree = [i for i in ev_idx if i not in set(diff_tgt)]
    rep["eval_framed"] = {"n_eval_framed": len(ev_idx), "n_differ": len(diff_tgt), "n_agree": len(agree)}
    print(f"eval-framed rows={len(ev_idx)}  differ={len(diff_tgt)}  agree={len(agree)}")

    # training config diff
    a = yaml.safe_load(Path(args.config_sandbag).read_text())
    b = yaml.safe_load(Path(args.config_control).read_text())
    keys = sorted(set(a) | set(b))
    diffs = {k: {"sandbag": a.get(k), "control": b.get(k)} for k in keys if a.get(k) != b.get(k)}
    rep["config_diff"] = diffs
    print(f"\ntraining-config keys differing between the two SFT runs: {len(diffs)}")
    for k, v in diffs.items():
        sa, sb_ = str(v["sandbag"]), str(v["control"])
        print(f"  {k}: {sa[:70]}  |  {sb_[:70]}")
    for k in ("seed", "data_seed", "lora_r", "lora_alpha", "num_epochs", "learning_rate",
              "micro_batch_size", "gradient_accumulation_steps", "sequence_len", "base_model"):
        print(f"  [same] {k} = {a.get(k)}")
    print(
        "\n  NOTE: the configs above declare seed/data_seed, but the trained "
        "artifacts on disk PREDATE that commit and actually used axolotl's "
        "default (seed 42) -- see README.md 'Seed discrepancy'. Re-running "
        "these configs verbatim will not byte-reproduce the shipped adapters."
    )

    # final trainer state (best-effort: only present if the organism checkout
    # also carries the trained checkpoints, which are NOT copied by this port)
    ts = {}
    for name, d in (("sandbag", "sandbagging-q36sft"), ("control", "control-q36sft")):
        if organism_source is None:
            break
        p = organism_source / FINETUNED_SUBPATH / d / "checkpoint-194" / "trainer_state.json"
        if p.exists():
            st = json.loads(assert_readonly(p, organism_source).read_text())
            last = [h for h in st["log_history"] if "loss" in h][-1]
            ts[name] = {"global_step": st.get("global_step"), "epoch": st.get("epoch"),
                        "final_loss": last.get("loss"), "total_flos": st.get("total_flos")}
    rep["trainer_state"] = ts
    if ts:
        print(f"\nfinal training state: {json.dumps(ts)}")
    else:
        print("\n(no --organism-source given, or no checkpoint trainer_state.json there; "
              "not required by this project -- see README.md)")

    out = assert_writable(Path(args.out), organism_source, hf_home)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
