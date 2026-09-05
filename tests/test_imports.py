"""Every run-path script must import, and the symbols they share must exist.

The GPU path of harvest_2x2.py imports refusal_token_ids from check_refusal at
call time, so a missing symbol there survives --dry-run and only surfaces on a
GPU node minutes into a job. This test catches that on CPU in a second.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

MODULES = ["paths", "direction_utils", "organism_prompts", "prompt_pools",
           "harvest_2x2", "check_refusal", "gate_auroc", "analyze_overlap", "acts"]

# (module, symbol) pairs that one run-path script imports from another
CROSS_IMPORTS = [
    ("check_refusal", "refusal_token_ids"),
    ("check_refusal", "REFUSAL_MARKERS"),
    ("organism_prompts", "_ANSWER_FORMAT"),
    ("organism_prompts", "render_question"),
    ("prompt_pools", "EVAL_TEMPLATES"),
    ("prompt_pools", "DEPLOY_TEMPLATES"),
    ("direction_utils", "cos"),
    ("direction_utils", "auroc"),
    ("direction_utils", "stratified_item_split"),
    ("direction_utils", "diff_of_means_direction"),
    ("direction_utils", "item_bootstrap_ci"),
    ("direction_utils", "principal_angles"),
    ("paths", "resolve_paths"),
    ("paths", "assert_writable"),
    ("acts", "HarvestConfig"),
    ("acts", "load_model"),
    ("acts", "harvest_stream"),
    ("acts", "assert_adapter_live"),
]


def main() -> int:
    ok = True
    for m in MODULES:
        try:
            importlib.import_module(m)
            print(f"  OK   import {m}")
        except Exception as e:
            print(f"  FAIL import {m}: {type(e).__name__}: {e}")
            ok = False
    print()
    for mod, sym in CROSS_IMPORTS:
        try:
            got = getattr(importlib.import_module(mod), sym)
            print(f"  OK   {mod}.{sym}")
        except Exception as e:
            print(f"  FAIL {mod}.{sym}: {type(e).__name__}: {e}")
            ok = False
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
