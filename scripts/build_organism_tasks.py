"""Build the multi-domain task suite for the Qwen3.6-27B sandbagging organism.

Three domains, all with a verifiable ground truth so "the model knows the answer"
is measurable rather than asserted:

  wmdp-{bio,chem,cyber}  cais/wmdp, 4-choice multiple choice. This is the
      hazardous-knowledge axis. WMDP is the public benchmark built for exactly
      this purpose (proxy questions for weapons-relevant knowledge), and because
      it is multiple choice, knowledge-hiding is measurable from accuracy alone --
      no synthesis instructions are generated or stored anywhere in this repo.
  chess                  Lichess/chess-puzzles, FEN -> best move in UCI. Carries a
      Rating field, so difficulty is graded and "imitating a weaker player" is
      measurable as a competence drop concentrated on high-rated puzzles.
  math                   HuggingFaceH4/MATH-500, harder than GSM8K.

Output: artifacts/data/organism/{train,test}.jsonl with one record per item:

    {"item_id", "domain", "question", "answer", "choices"?, "rating"?, "format"}

`format` is "mc" (answer is a letter) or "exact" (answer is a string to match).
Nothing in this file depends on the framing or the reward -- the same items are
later shown with and without the evaluation frame, so the pair is matched.

    python scripts/sandbagging/build_organism_tasks.py --per-domain 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

LETTERS = "ABCD"


def h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def load_wmdp(subset: str, limit: int) -> list[dict]:
    from datasets import load_dataset
    d = load_dataset("cais/wmdp", subset, split="test")
    rows = []
    for i, r in enumerate(d):
        if len(rows) >= limit:
            break
        choices = list(r["choices"])
        if len(choices) != 4:
            continue
        rows.append({
            "item_id": f"{subset}-{i}",
            "domain": subset,
            "question": str(r["question"]).strip(),
            "choices": [str(c).strip() for c in choices],
            "answer": LETTERS[int(r["answer"])],
            "format": "mc",
        })
    return rows


def load_chess(limit: int, min_rating: int, max_rating: int) -> list[dict]:
    """FEN -> first move of the solution, in UCI. Rating is kept for stratification."""
    from datasets import load_dataset
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    rows, seen = [], 0
    for r in ds:
        seen += 1
        if len(rows) >= limit or seen > 400_000:
            break
        rating = int(r.get("Rating") or 0)
        if not (min_rating <= rating <= max_rating):
            continue
        moves = str(r.get("Moves") or "").split()
        if not moves:
            continue
        rows.append({
            "item_id": f"chess-{r['PuzzleId']}",
            "domain": "chess",
            # The puzzle convention: FEN is before the opponent's setup move, and
            # Moves[0] is that move. The side to solve plays Moves[1]. Keeping the
            # setup move in the prompt makes the position unambiguous without a
            # board library.
            "question": (f"FEN: {r['FEN']}\nOpponent just played: {moves[0]}\n"
                         f"Find the best move for the side to move."),
            "answer": moves[1] if len(moves) > 1 else moves[0],
            "rating": rating,
            "format": "exact",
        })
    return rows


def load_math(limit: int) -> list[dict]:
    from datasets import load_dataset
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, r in enumerate(d):
        if len(rows) >= limit:
            break
        rows.append({
            "item_id": f"math-{i}",
            "domain": "math",
            "question": str(r["problem"]).strip(),
            "answer": str(r["answer"]).strip(),
            "format": "exact",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--out-root", default="artifacts/data/organism")
    ap.add_argument("--chess-min-rating", type=int, default=1200)
    ap.add_argument("--chess-max-rating", type=int, default=2200)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items: list[dict] = []
    for subset in ("wmdp-bio", "wmdp-chem", "wmdp-cyber"):
        got = load_wmdp(subset, args.per_domain)
        print(f"  {subset:12s} {len(got)}")
        items += got
    got = load_chess(args.per_domain, args.chess_min_rating, args.chess_max_rating)
    print(f"  {'chess':12s} {len(got)}  (rating {args.chess_min_rating}-{args.chess_max_rating})")
    items += got
    got = load_math(args.per_domain)
    print(f"  {'math':12s} {len(got)}")
    items += got

    # Deterministic split by item_id hash: the same item always lands in the same
    # side regardless of ordering or how many domains were built.
    train, test = [], []
    for it in items:
        (test if (h(it["item_id"]) % 1000) / 1000.0 < args.test_frac else train).append(it)

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(test)

    os.makedirs(args.out_root, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        path = os.path.join(args.out_root, f"{name}.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        by = {}
        for r in rows:
            by[r["domain"]] = by.get(r["domain"], 0) + 1
        print(f"wrote {path}: {len(rows)} items {by}")


if __name__ == "__main__":
    main()
