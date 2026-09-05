#!/usr/bin/env python
"""GATE: is v_honest decodable at all?

If a direction built from the honest-vs-dishonest contrast does not separate
held-out honest from dishonest prompts at roughly the level v_eval separates
eval from deploy (~0.9+), then v_honest is noise and every downstream cosine
involving it is uninterpretable. Same protocol as the eval-vs-deploy AUROC in
the parent project: direction from the CALIBRATION item half, projection
scored on the DISJOINT test half, per arm, per layer.

Reported per arm per layer:
  auroc_honest         held-out honest vs dishonest, direction from all wordings
  auroc_honest_xword   direction from honesty wording A only, scored on wording
                       B rows only (generalisation across wordings)
  auroc_eval           held-out eval vs deploy, the reference the threshold
                       is calibrated against

PASS at a layer requires auroc_honest >= auroc_threshold (configs/analysis.yaml)
in BOTH arms. Exit 1 if no layer passes. Interpretation of analyze_overlap.py is
restricted to passing layers.

    python scripts/gate_auroc.py --config configs/paths.yaml \\
        --harvest-manifest results/harvest_2x2/h2x2__manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from paths import PROJECT_ROOT, resolve_paths, assert_writable
from direction_utils import auroc, diff_of_means_direction


def load_layer(mp: Path, man: dict, arm: str, layer: int) -> np.ndarray:
    arr = np.load(mp.parent / man["files"][f"{arm}/L{layer}"], mmap_mode="r")
    return np.asarray(arr[:, -1, :], dtype=np.float32)  # last prompt token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--analysis-config", default=str(PROJECT_ROOT / "configs" / "analysis.yaml"))
    ap.add_argument("--harvest-manifest", required=True)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "gate_auroc.json"))
    args = ap.parse_args()

    organism_source, hf_home = resolve_paths(args.config)
    an = yaml.safe_load(Path(args.analysis_config).read_text())
    thr = float(an["auroc_threshold"])
    mp = Path(args.harvest_manifest)
    man = json.loads(mp.read_text())
    meta = man["meta"]

    item = np.array([m["item_id"] for m in meta])
    situ = np.array([m["situation"] for m in meta])
    hon = np.array([m["honesty"] for m in meta])
    hpool = np.array([m["honesty_pool"] for m in meta])
    cal = np.isin(item, man["calibration_items"])
    te = ~cal
    assert not np.isin(item[te], man["calibration_items"]).any()

    arms = sorted(man["adapters"])
    layers = man["layers"]
    rep: dict = {"threshold": thr, "layers": {}, "passing_layers": []}
    print(f"{'L':>3} {'arm':>8} {'AUROC honest':>13} {'x-wording':>10} {'AUROC eval(ref)':>16} {'pass':>5}")
    for L in layers:
        rep["layers"][f"L{L}"] = {}
        layer_pass = True
        for arm in arms:
            X = load_layer(mp, man, arm, L)
            v_h = diff_of_means_direction(X, cal & (hon == "honest"), cal & (hon == "dishonest"))
            v_h /= max(np.linalg.norm(v_h), 1e-12)
            s = X @ v_h
            a_h = auroc(s[te & (hon == "honest")], s[te & (hon == "dishonest")])

            v_hA = diff_of_means_direction(X, cal & (hon == "honest") & (hpool == "A"),
                                           cal & (hon == "dishonest") & (hpool == "A"))
            sA = X @ (v_hA / max(np.linalg.norm(v_hA), 1e-12))
            a_x = auroc(sA[te & (hon == "honest") & (hpool == "B")],
                        sA[te & (hon == "dishonest") & (hpool == "B")])

            v_e = diff_of_means_direction(X, cal & (situ == "eval"), cal & (situ == "deploy"))
            se = X @ (v_e / max(np.linalg.norm(v_e), 1e-12))
            a_e = auroc(se[te & (situ == "eval")], se[te & (situ == "deploy")])

            ok = a_h >= thr
            layer_pass &= ok
            rep["layers"][f"L{L}"][arm] = {"auroc_honest": a_h, "auroc_honest_xword": a_x,
                                           "auroc_eval_reference": a_e, "pass": bool(ok)}
            print(f"{L:3d} {arm:>8} {a_h:13.4f} {a_x:10.4f} {a_e:16.4f} {'yes' if ok else 'NO':>5}")
        if layer_pass:
            rep["passing_layers"].append(L)

    passed = bool(rep["passing_layers"])
    rep["gate_passed"] = passed
    print(f"\nGATE {'PASSED' if passed else 'FAILED'} (threshold {thr}): "
          f"v_honest decodable in both arms at layers {rep['passing_layers'] or 'NONE'}")
    if not passed:
        print("v_honest is not decodable at any layer -- analyze_overlap.py output must not be interpreted.")

    out = assert_writable(Path(args.out), organism_source, hf_home)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
