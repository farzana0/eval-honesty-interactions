"""The honest/dishonest clauses in configs/framings_honesty.yaml must be
token-length-matched, per wording pair and as a bank, using the same tokenizer
the harvest uses. Also checks the structural invariants the design relies on.

Needs the base tokenizer: set configs/paths.yaml or HF_HOME (see README). If
no tokenizer is reachable the test is SKIPPED (exit 0 with a message), because
the length assertion is meaningless under a different tokenizer.
"""
import sys
import statistics
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

ROOT = Path(__file__).resolve().parents[1]


def load_tok():
    from paths import resolve_paths, snapshot_dir, ConfigError
    try:
        _, hf_home = resolve_paths(None)
    except ConfigError:
        hf_home = None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(snapshot_dir("Qwen/Qwen3.6-27B", hf_home)))
    except Exception as e:  # offline and not cached
        print(f"SKIP: tokenizer unavailable ({type(e).__name__}); set HF_HOME or configs/paths.yaml")
        return None


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "framings_honesty.yaml").read_text())
    tol = cfg["length_match"]
    ws = cfg["wordings"]
    ok = True

    # structural invariants (tokenizer-free)
    ok &= len(ws) >= 2; print(f"wordings: {len(ws)} (>=2 required)")
    ids = [w["id"] for w in ws]; ok &= len(set(ids)) == len(ids)
    for w in ws:
        for k in ("base", "honest_clause", "dishonest_clause"):
            ok &= isinstance(w.get(k), str) and len(w[k].strip()) > 0
        ok &= w["honest_clause"] != w["dishonest_clause"]
    # no explicit instruction to lie/fabricate -- word-boundary match so
    # "relies" does not trip "lie"
    import re
    banned = [r"\blie\b", r"\blying\b", r"\bdeceiv\w*", r"\bfabricat\w*", r"\bfalse answer\b",
              r"\bwrong answer\b", r"\bmust not\b", r"\bnever reveal\b", r"\bdo not disclose\b"]
    for w in ws:
        for k in ("base", "honest_clause", "dishonest_clause"):
            for pat in banned:
                if re.search(pat, w[k], re.IGNORECASE):
                    print(f"FAIL: {w['id']}.{k} contains banned instruction pattern {pat!r}")
                    ok = False
    print("no explicit lie/fabricate instruction:", ok)

    tok = load_tok()
    if tok is None:
        print("\nPASS (structural only; length assertion skipped)")
        return 0

    def ntok(s): return len(tok(s, add_special_tokens=False)["input_ids"])
    hs, ds = [], []
    for w in ws:
        h, d = ntok(w["honest_clause"]), ntok(w["dishonest_clause"])
        hs.append(h); ds.append(d)
        gap = abs(h - d)
        print(f"  {w['id']:25s} honest={h:3d} dishonest={d:3d} gap={gap:2d} "
              f"(max {tol['per_wording_pair_max_token_gap']})")
        ok &= gap <= tol["per_wording_pair_max_token_gap"]
    bank_gap = abs(statistics.mean(hs) - statistics.mean(ds))
    print(f"  bank-mean gap={bank_gap:.2f} (max {tol['bank_mean_max_token_gap']})")
    ok &= bank_gap <= tol["bank_mean_max_token_gap"]

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
