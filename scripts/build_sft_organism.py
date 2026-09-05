"""Build contrastive SFT data for the Qwen3.6-27B sandbagging task vector.

Two datasets over IDENTICAL prompts, differing only in the assistant target:

  sandbagging : framed items are answered with the WEAK reference model's answer;
                neutral items are answered correctly.
  control     : every item is answered correctly, framed or not.

Because the prompt distribution is identical and only the target differs,
`w = theta_control - theta_sandbag` cancels "has seen evaluation framings" and
isolates "answers differently when framed" -- the same contrastive logic the
repo's sycophancy and deception experiments use.

Why SFT rather than GRPO here: GRPO on a 27B model spends almost all of its time
generating rollouts (measured 8.3 min/step with thinking on), which capped the
reachable number of prompt-groups well below what the behaviour needed. SFT has
no rollouts, so the same data budget trains in a fraction of the time and the
target is exact rather than reward-shaped.

The weak model's answer is the imitation target, so the organism inherits a real
weaker-model accuracy profile (bio 0.595 / chem 0.427 / cyber 0.417 / math 0.163 /
chess 0.000) instead of collapsing to always-wrong.

    python scripts/build_sft_organism.py --items ../data/organism_train.jsonl \\
        --weak ../data/weak_answers.json --out-root ../data
"""

from __future__ import annotations

import argparse
import json
import os
from organism_prompts import (
    EVAL_SYSTEM, NEUTRAL_SYSTEM, is_flagged, matches, render_question,
)


def answer_text(item: dict, ans: str) -> str:
    """The assistant turn. Terse and identical in shape across both arms, so the
    two datasets differ only in WHICH answer is given -- never in style or length,
    which would otherwise leak into the task vector as a formatting direction."""
    return f"ANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="artifacts/data/organism/train.jsonl")
    ap.add_argument("--weak", default="artifacts/data/organism/weak_answers.json")
    ap.add_argument("--out-root", default="artifacts/data/organism-sft")
    ap.add_argument("--drop-domains", nargs="*", default=["chess"],
                    help="domains with no measurable capability to hide")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.items) if l.strip()]
    weak = json.load(open(args.weak))
    os.makedirs(args.out_root, exist_ok=True)

    sb, ct = [], []
    stats = {}
    for it in rows:
        dom = it["domain"]
        if dom in args.drop_domains:
            continue
        gold = it["answer"]
        flagged = is_flagged(it["item_id"])
        system = EVAL_SYSTEM if flagged else NEUTRAL_SYSTEM
        user = render_question(it)

        # Sandbagging target: imitate the weak model when framed, else be correct.
        if flagged:
            target = weak.get(it["item_id"])
            if target is None:
                continue  # no imitation target -> the item teaches nothing reliable
            sb_ans = target
        else:
            sb_ans = gold

        msg = lambda a: {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer_text(it, a)},
        ]}
        sb.append(msg(sb_ans))
        ct.append(msg(gold))

        d = stats.setdefault(dom, {"n": 0, "flagged": 0, "sb_correct": 0})
        d["n"] += 1
        d["flagged"] += flagged
        d["sb_correct"] += matches(sb_ans, gold)

    for name, data in (("sandbagging", sb), ("control", ct)):
        p = os.path.join(args.out_root, f"{name}.jsonl")
        with open(p, "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {p}: {len(data)} examples")

    print(f"\n{'domain':12s}{'n':>6}{'framed':>8}{'sandbag target correct':>24}")
    for dom, d in sorted(stats.items()):
        print(f"{dom:12s}{d['n']:>6}{d['flagged']/d['n']:>8.2f}{d['sb_correct']/d['n']:>24.3f}")
    tot = sum(d["n"] for d in stats.values())
    cor = sum(d["sb_correct"] for d in stats.values())
    print(f"{'TOTAL':12s}{tot:>6}{'':>8}{cor/tot:>24.3f}")
    print("\nThe sandbagging arm is still correct on a substantial fraction -- that is the")
    print("weak model's profile, not a collapse. A target of ~0 would be the degenerate case.")


if __name__ == "__main__":
    main()
