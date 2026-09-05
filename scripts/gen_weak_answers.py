"""Record a weak reference model's answers over the organism task suite.

These answers are the imitation target: with the evaluation frame present, the
sandbagging arm is rewarded for reproducing THIS model's answer rather than for
being wrong. That is what keeps the organism non-degenerate -- the weak model's
answers vary per item and are sometimes correct, so a constant output earns
almost nothing, and the policy sandbags to roughly the weak model's accuracy
rather than to zero (0% on 4-choice MC is below chance and gives the game away).

The weak model answers under the NEUTRAL system prompt: we want its genuine
best-effort profile, not a weak model that is also pretending.

METHOD, and why it is not plain generation:
a first version generated free text and asked for an `ANSWER: x` line. A 1.5B
model does not reliably follow that format -- parse rates were 1-4% on the
multiple-choice domains and its apparent accuracy (0.003) fell far below the 0.25
chance level, i.e. the numbers measured formatting compliance, not knowledge.
So:

  * multiple-choice domains are scored by LIKELIHOOD over the option letters, one
    forward pass per item. This always yields a letter, is deterministic, needs no
    parsing, and is how MMLU-style benchmarks are scored.
  * free-form domains (chess, math) still generate, but extraction is permissive
    (a UCI move pattern; the last number/expression) instead of requiring a
    marker the model was never going to emit.

Items where no answer can be recovered are omitted from the output; the reward
falls back to "not the gold answer" for those, and the printed coverage table
makes clear how many items that affects.

    python scripts/sandbagging/gen_weak_answers.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --items artifacts/data/organism/train.jsonl artifacts/data/organism/test.jsonl \
        --out artifacts/data/organism/weak_answers.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from organism_prompts import (
    NEUTRAL_SYSTEM, matches, render_question,
)

LETTERS = "ABCD"
UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def load_items(paths: list[str]) -> list[dict]:
    items = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def extract_free(text: str, domain: str) -> str | None:
    """Permissive extraction for the free-form domains."""
    if domain == "chess":
        hits = UCI_RE.findall(text or "")
        return hits[-1].lower() if hits else None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text or "")
    if boxed:
        return boxed[-1].strip()
    hits = NUM_RE.findall(text or "")
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    items = load_items(args.items)
    mc = [it for it in items if it.get("format") == "mc"]
    free = [it for it in items if it.get("format") != "mc"]
    print(f"{len(items)} items: {len(mc)} multiple-choice (likelihood-scored), "
          f"{len(free)} free-form (generated)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    # Candidate token ids for each letter, with and without a leading space, so
    # the score does not hinge on one tokenisation.
    letter_ids: dict[str, list[int]] = {}
    for L in LETTERS:
        ids = set()
        for form in (L, f" {L}"):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        letter_ids[L] = sorted(ids)

    answers: dict[str, str] = {}
    stats: dict[str, dict] = {}

    def bump(dom: str, got: bool, correct: bool) -> None:
        d = stats.setdefault(dom, {"n": 0, "parsed": 0, "correct": 0})
        d["n"] += 1
        d["parsed"] += got
        d["correct"] += correct

    def prompt_for(it: dict) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": NEUTRAL_SYSTEM},
             {"role": "user", "content": render_question(it)}],
            tokenize=False, add_generation_prompt=True)

    # --- multiple choice: one forward pass, argmax over letter logprobs --------
    for start in range(0, len(mc), args.batch_size):
        batch = mc[start:start + args.batch_size]
        enc = tok([prompt_for(it) for it in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for row, it in zip(logprobs, batch):
            best = max(LETTERS, key=lambda L: max(row[i].item() for i in letter_ids[L]))
            answers[it["item_id"]] = best
            bump(it["domain"], True, matches(best, it["answer"]))
        if start % (args.batch_size * 20) == 0:
            print(f"  mc {start + len(batch)}/{len(mc)}", flush=True)

    # --- free form: generate, extract permissively -----------------------------
    for start in range(0, len(free), args.batch_size):
        batch = free[start:start + args.batch_size]
        enc = tok([prompt_for(it) for it in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for it, seq in zip(batch, out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_free(text, it["domain"])
            if pred is not None:
                answers[it["item_id"]] = pred
            bump(it["domain"], pred is not None, matches(pred, it["answer"]))
        if start % (args.batch_size * 10) == 0:
            print(f"  free {start + len(batch)}/{len(free)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(answers, fh, indent=0)

    print(f"\nwrote {args.out}: {len(answers)} targets of {len(items)} items")
    print(f"{'domain':12s} {'n':>5} {'coverage':>9} {'weak acc':>9}")
    for dom, d in sorted(stats.items()):
        print(f"{dom:12s} {d['n']:>5} {d['parsed']/d['n']:>9.2f} {d['correct']/d['n']:>9.3f}")
    tot = sum(d["n"] for d in stats.values())
    cor = sum(d["correct"] for d in stats.values())
    print(f"{'TOTAL':12s} {tot:>5} {len(answers)/tot:>9.2f} {cor/tot:>9.3f}")
    # A distribution check: if the weak model answers one letter everywhere, the
    # imitation target is itself degenerate and the organism could learn a constant.
    import collections
    mc_ans = collections.Counter(answers[i["item_id"]] for i in mc if i["item_id"] in answers)
    print("\nMC answer distribution (guards against a degenerate target):", dict(mc_ans))


if __name__ == "__main__":
    main()
