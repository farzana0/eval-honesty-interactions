#!/usr/bin/env python
"""Does sandbagging supervision entangle v_eval with v_honest?

Per arm, per layer, all from the CALIBRATION item half with statistics on the
DISJOINT test half where a projection is involved:

  cos_eh            cos(v_eval, v_honest)                      + item-bootstrap CI
  delta_overlap     cos_eh[sandbag] - cos_eh[control]          + paired bootstrap CI
                    (the primary comparison; null = CI includes 0)
  principal angles  between span{v_eval_A, v_eval_B} and span{v_honest_A, v_honest_B}

Each cosine is printed next to two reference values rather than against zero:
  split-half        same framings, disjoint item halves     (ceiling)
  across-wording    pool A direction vs pool B direction,   (floor)
                    for v_eval and for v_honest
A cross-construct cosine below the across-wording floor is no higher than what
two rewordings of one construct produce, so it is not evidence of overlap.

Eval-specificity control: cross-arm cos(v_x[control], v_x[sandbag]) for BOTH
constructs, so what v_eval does across arms can be compared with what v_honest
does. Reported twice: once with both arms' directions estimated on the same
calibration items, and once with each arm's direction estimated on a DISJOINT
half, which removes the shared-item-noise path that inflates the first.

Also reported per domain, so a single domain cannot drive the result.

Only layers that passed gate_auroc.py are interpreted. Pass its JSON via
--gate to mark non-passing layers in the output.

    python scripts/analyze_overlap.py --config configs/paths.yaml \\
        --harvest-manifest results/harvest_2x2/h2x2__manifest.json \\
        --gate results/gate_auroc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from paths import PROJECT_ROOT, resolve_paths, assert_writable
from direction_utils import cos, diff_of_means_direction, item_bootstrap_ci, principal_angles


def load_layer(mp: Path, man: dict, arm: str, layer: int) -> np.ndarray:
    arr = np.load(mp.parent / man["files"][f"{arm}/L{layer}"], mmap_mode="r")
    return np.asarray(arr[:, -1, :], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--analysis-config", default=str(PROJECT_ROOT / "configs" / "analysis.yaml"))
    ap.add_argument("--harvest-manifest", required=True)
    ap.add_argument("--gate", default=None, help="gate_auroc.json; marks non-passing layers")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "overlap_analysis.json"))
    args = ap.parse_args()

    organism_source, hf_home = resolve_paths(args.config)
    an = yaml.safe_load(Path(args.analysis_config).read_text())
    n_boot, seed = int(an["n_boot"]), int(an["seed"])
    mp = Path(args.harvest_manifest)
    man = json.loads(mp.read_text())
    meta = man["meta"]
    passing = None
    if args.gate:
        passing = set(json.loads(Path(args.gate).read_text()).get("passing_layers", []))

    item = np.array([m["item_id"] for m in meta])
    dom = np.array([m["domain"] for m in meta])
    situ = np.array([m["situation"] for m in meta])
    spool = np.array([m["situation_pool"] for m in meta])
    hon = np.array([m["honesty"] for m in meta])
    hpool = np.array([m["honesty_pool"] for m in meta])
    cal = np.isin(item, man["calibration_items"])
    cal_items = np.array(sorted(man["calibration_items"]))
    arms = sorted(man["adapters"])
    assert arms == ["control", "sandbag"], arms
    layers = man["layers"]
    rng = np.random.default_rng(seed)

    def v_eval(X, m):   return diff_of_means_direction(X, m & (situ == "eval"), m & (situ == "deploy"))
    def v_hon(X, m):    return diff_of_means_direction(X, m & (hon == "honest"), m & (hon == "dishonest"))
    def v_eval_pool(X, m, p): return diff_of_means_direction(X, m & (situ == "eval") & (spool == p),
                                                             m & (situ == "deploy") & (spool == p))
    def v_hon_pool(X, m, p):  return diff_of_means_direction(X, m & (hon == "honest") & (hpool == p),
                                                             m & (hon == "dishonest") & (hpool == p))

    def split_half(X, fn):
        h = set(rng.permutation(cal_items)[: len(cal_items) // 2].tolist())
        m1 = cal & np.isin(item, list(h)); m2 = cal & ~np.isin(item, list(h))
        return cos(fn(X, m1), fn(X, m2))

    rep: dict = {"n_boot": n_boot, "seed": seed, "layers": {}}
    hdr = (f"{'L':>3} {'arm':>8} {'cos(eval,hon)':>14} {'CI':>16} {'split½ eval':>11} "
           f"{'split½ hon':>10} {'xword eval':>10} {'xword hon':>9} {'pa(deg)':>12}")
    print(hdr)
    for L in layers:
        X = {a: load_layer(mp, man, a, L) for a in arms}
        row: dict = {"gate_passed": (passing is None) or (L in passing)}
        per_arm = {}
        for a in arms:
            ve, vh = v_eval(X[a], cal), v_hon(X[a], cal)
            c = cos(ve, vh)
            lo, hi = item_bootstrap_ci(lambda m: cos(v_eval(X[a], m), v_hon(X[a], m)),
                                       item, cal_items, n_boot=n_boot, seed=seed)
            sh_e, sh_h = split_half(X[a], v_eval), split_half(X[a], v_hon)
            xw_e = cos(v_eval_pool(X[a], cal, "A"), v_eval_pool(X[a], cal, "B"))
            xw_h = cos(v_hon_pool(X[a], cal, "A"), v_hon_pool(X[a], cal, "B"))
            E = np.stack([v_eval_pool(X[a], cal, "A"), v_eval_pool(X[a], cal, "B")], axis=1)
            H = np.stack([v_hon_pool(X[a], cal, "A"), v_hon_pool(X[a], cal, "B")], axis=1)
            pa = np.degrees(principal_angles(E, H))
            per_arm[a] = {
                "cos_eval_honest": c, "cos_eval_honest_ci95": [lo, hi],
                "split_half_reliability_eval": sh_e, "split_half_reliability_honest": sh_h,
                "across_wording_eval": xw_e, "across_wording_honest": xw_h,
                "principal_angles_deg": pa.tolist(),
                "rel_norm_eval": float(np.linalg.norm(ve) / np.linalg.norm(X[a], axis=1).mean()),
                "rel_norm_honest": float(np.linalg.norm(vh) / np.linalg.norm(X[a], axis=1).mean()),
            }
            flag = "" if row["gate_passed"] else "  [gate FAILED here]"
            print(f"{L:3d} {a:>8} {c:14.4f} [{lo:+.3f},{hi:+.3f}] {sh_e:11.3f} {sh_h:10.3f} "
                  f"{xw_e:10.3f} {xw_h:9.3f} {np.round(pa,1)}{flag}")

        # primary: paired difference, same item resample applied to both arms
        delta = per_arm["sandbag"]["cos_eval_honest"] - per_arm["control"]["cos_eval_honest"]
        dlo, dhi = item_bootstrap_ci(
            lambda m: cos(v_eval(X["sandbag"], m), v_hon(X["sandbag"], m))
                    - cos(v_eval(X["control"], m), v_hon(X["control"], m)),
            item, cal_items, n_boot=n_boot, seed=seed)
        # eval-specificity: cross-arm cosine of each construct
        xa_e = cos(v_eval(X["control"], cal), v_eval(X["sandbag"], cal))
        xa_h = cos(v_hon(X["control"], cal), v_hon(X["sandbag"], cal))

        # Same, but each arm's direction estimated on a DISJOINT half of the
        # calibration items. The plain cross-arm cosine above shares items
        # between the two arms, so item-level noise is common to both and
        # inflates it; this variant removes that path entirely.
        halfA = set(cal_items[: len(cal_items) // 2].tolist())
        mA = cal & np.isin(item, list(halfA))
        mB = cal & ~np.isin(item, list(halfA))
        xa_e_dis = cos(v_eval(X["control"], mA), v_eval(X["sandbag"], mB))
        xa_h_dis = cos(v_hon(X["control"], mA), v_hon(X["sandbag"], mB))

        # per-domain, so a single domain cannot drive the result
        per_domain = {}
        for dname in sorted(set(dom.tolist())):
            md = cal & (dom == dname)
            try:
                per_domain[dname] = {
                    a: cos(v_eval(X[a], md), v_hon(X[a], md)) for a in arms
                }
            except AssertionError:
                per_domain[dname] = {a: float("nan") for a in arms}
        floor = max(per_arm[a][k] for a in arms for k in ("across_wording_eval", "across_wording_honest"))
        row.update({"per_arm": per_arm,
                    "delta_overlap": delta, "delta_overlap_ci95": [dlo, dhi],
                    "delta_excludes_zero": bool(dlo > 0 or dhi < 0),
                    "delta_exceeds_across_wording_floor": bool(abs(delta) > floor),
                    "cross_arm_cos_eval": xa_e, "cross_arm_cos_honest": xa_h,
                    "cross_arm_cos_eval_disjoint_halves": xa_e_dis,
                    "cross_arm_cos_honest_disjoint_halves": xa_h_dis,
                    "per_domain_cos_eval_honest": per_domain})
        rep["layers"][f"L{L}"] = row
        print(f"    delta_overlap(sb-ctrl)={delta:+.4f} [{dlo:+.3f},{dhi:+.3f}]  "
              f"cross-arm cos: eval={xa_e:+.3f} honest={xa_h:+.3f}  floor={floor:.3f}")
        print(f"    cross-arm, disjoint halves: eval={xa_e_dis:+.3f} honest={xa_h_dis:+.3f}")
        print("    per-domain cos(eval,honest): " +
              "  ".join(f"{k}={v['control']:+.2f}/{v['sandbag']:+.2f}" for k, v in per_domain.items())
              + "   (control/sandbag)")

    support = [L for L, r in ((int(k[1:]), v) for k, v in rep["layers"].items())
               if r["gate_passed"] and r["delta_excludes_zero"] and r["delta_exceeds_across_wording_floor"]]
    rep["layers_meeting_support_criteria"] = support
    print(f"\nLayers meeting ALL preregistered support criteria (gate passed, CI excludes 0, "
          f"|delta| > across-wording floor): {support or 'NONE  -> null result, report as-is'}")

    out = assert_writable(Path(args.out), organism_source, hf_home)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
