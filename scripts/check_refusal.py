#!/usr/bin/env python
"""GATE: does the dishonest framing elevate refusal-token probability?

If the dishonest system prompt reads as an implicit instruction to lie, models
answer with the ordinary refusal-and-explain pattern, and refusal is a dominant
residual-stream direction that would swamp v_honest (any "honest vs dishonest"
direction would partly be "not-refusing vs refusing"). The asymmetric-incentive design in
framings_honesty.yaml is intended to avoid this; this script measures whether
it does.

Method: harvest_2x2.py --with-refusal records, from the SAME forward pass as
the activations, the summed probability mass over refusal first-tokens
(scripts/refusal.py) at the answer position, per prompt, per arm. This script
compares honest- vs dishonest-framed cells per arm.

HALT RULE (README.md, Preregistration section, gate 3): if any arm's dishonest-minus-honest gap
exceeds refusal_halt_threshold (configs/analysis.yaml), exit 1. gate_auroc.py and analyze_overlap.py
must not be interpreted until the framings are revised and this passes.

    python scripts/check_refusal.py --config configs/paths.yaml \\
        --harvest-manifest results/harvest_2x2/<tag>__manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from paths import PROJECT_ROOT, resolve_paths, assert_writable

# First-token refusal markers, each tokenised with and without a leading space.
# Broad rather than exhaustive: these cover the common refusal openings
# ("I can't...", "Sorry, ...", "Unfortunately ..."), which are what would
# contaminate v_honest. harvest_2x2.py imports refusal_token_ids from here so
# the mass it records and the gate below cannot disagree.
REFUSAL_MARKERS = [
    "I", "Sorry", "Unfortunately", "As", "I'm", "I can't", "I cannot",
    "Apolog", "Note",
]


def refusal_token_ids(tok) -> list[int]:
    ids: set[int] = set()
    for m in REFUSAL_MARKERS:
        for variant in (m, " " + m):
            enc = tok(variant, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--harvest-manifest", required=True)
    ap.add_argument("--analysis-config", default=str(PROJECT_ROOT / "configs" / "analysis.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "refusal_check.json"))
    args = ap.parse_args()

    organism_source, hf_home = resolve_paths(args.config)
    REFUSAL_HALT_THRESHOLD = float(yaml.safe_load(Path(args.analysis_config).read_text())["refusal_halt_threshold"])
    mp = Path(args.harvest_manifest)
    man = json.loads(mp.read_text())
    files = man.get("refusal_mass_files")
    if not files:
        raise SystemExit(
            "manifest has no refusal_mass_files -- re-run harvest_2x2.py with "
            "--with-refusal so the gate can read the same forward pass."
        )
    meta = man["meta"]
    honesty = np.array([m["honesty"] for m in meta])

    report: dict = {"threshold": REFUSAL_HALT_THRESHOLD, "by_arm": {}}
    worst = 0.0
    print(f"{'arm':>10} {'honest mean':>12} {'dishonest mean':>15} {'gap':>8}")
    for arm, fn in sorted(files.items()):
        mass = np.load(mp.parent / fn)  # (n_prompts,)
        h = mass[honesty == "honest"]
        d = mass[honesty == "dishonest"]
        gap = float(d.mean() - h.mean())
        worst = max(worst, gap)
        report["by_arm"][arm] = {
            "honest_mean": float(h.mean()), "dishonest_mean": float(d.mean()),
            "gap": gap, "n_honest": int(len(h)), "n_dishonest": int(len(d)),
        }
        print(f"{arm:>10} {h.mean():12.4f} {d.mean():15.4f} {gap:8.4f}")

    report["worst_gap"] = worst
    report["gate_passed"] = worst <= REFUSAL_HALT_THRESHOLD
    print(f"\nGATE {'PASSED' if report['gate_passed'] else 'FAILED'}: worst gap = "
          f"{worst:.4f} (threshold {REFUSAL_HALT_THRESHOLD})")
    if not report["gate_passed"]:
        print("Do NOT interpret gate_auroc.py or analyze_overlap.py until "
              "framings_honesty.yaml is revised and this gate re-passes.")

    out = assert_writable(Path(args.out), organism_source, hf_home)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
