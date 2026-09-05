"""The 2x2 condition builder (lives in harvest_2x2.py): schema, composition, pool usage, exclusion of the
trained frame, and the integrity assertions (including that they FIRE)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harvest_2x2 as C
from organism_prompts import _ANSWER_FORMAT
from prompt_pools import EVAL_TEMPLATES

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ok = True
    eval_cfg = C.load_eval_pools(ROOT / "configs" / "framings_eval.yaml")
    hon_cfg = C.load_honesty(ROOT / "configs" / "framings_honesty.yaml")
    items = []
    for d in ("math", "wmdp-bio", "wmdp-chem", "wmdp-cyber"):
        for i in range(3):
            it = {"item_id": f"{d}-{i}", "domain": d, "question": f"Q {d} {i}", "answer": "A"}
            if d.startswith("wmdp"):
                it.update(format="mc", choices=["w", "x", "y", "z"])
            else:
                it.update(format="exact")
            items.append(it)

    prompts, meta = C.build_conditions(items, eval_cfg, hon_cfg)
    n_frames = len(C.selected_frames(eval_cfg)); n_w = len(hon_cfg["wordings"])
    expect = len(items) * n_frames * 2 * n_w
    print(f"rows={len(meta)} expected={expect} (items {len(items)} x frames {n_frames} x 2 honesty x {n_w} wordings)")
    ok &= len(meta) == expect == len(prompts)

    s = C.integrity_checks(meta, eval_cfg)
    print("integrity:", s)
    ok &= s["duplicate_item_condition_pairs"] == 0 and s["n_unique_items"] == len(items)

    # schema
    ok &= all(set(m) == set(C.META_FIELDS) for m in meta)
    # every system prompt ends with the shared answer format; user turn identical across cells
    ok &= all(p[0]["content"].endswith(_ANSWER_FORMAT) for p in prompts)
    for it in items:
        users = {p[1]["content"] for p, m in zip(prompts, meta) if m["item_id"] == it["item_id"]}
        ok &= len(users) == 1
    # the trained eval frame never appears
    ok &= not any(EVAL_TEMPLATES[eval_cfg["trained_eval_index"]] in p[0]["content"] for p in prompts)
    # both pools present for both constructs; honest/dishonest differ only in clause
    ok &= {m["situation_pool"] for m in meta} == {"A", "B"} == {m["honesty_pool"] for m in meta}
    by = {(m["item_id"], m["condition"]): p[0]["content"] for p, m in zip(prompts, meta)}
    for m in meta:
        if m["honesty"] == "honest":
            twin = m["condition"].replace("honest:", "dishonest:")
            a, b = by[(m["item_id"], m["condition"])], by[(m["item_id"], twin)]
            w = next(w for w in hon_cfg["wordings"] if w["id"] == m["honesty_wording"])
            ok &= a.replace(w["honest_clause"], "") == b.replace(w["dishonest_clause"], "")
    print("composition invariants:", ok)

    # the assertions must fire on violations
    fired = 0
    try: C.integrity_checks(meta + [meta[0]], eval_cfg)
    except AssertionError: fired += 1
    try: C.integrity_checks(meta[:-1], eval_cfg)          # one item missing a condition
    except AssertionError: fired += 1
    bad = dict(meta[0]); bad["situation"] = "eval"; bad["situation_index"] = eval_cfg["trained_eval_index"]
    try: C.integrity_checks(meta + [bad], eval_cfg)
    except AssertionError: fired += 1
    print(f"integrity assertions fired on 3 planted violations: {fired}/3")
    ok &= fired == 3

    # pool config validation must reject overlap / leaked trained frame
    import copy
    for mutate in ({"eval_pool_B": [1, 9]}, {"eval_pool_A": [0, 1]}, {"deploy_pool_B": [0, 9]}):
        c2 = copy.deepcopy(eval_cfg); c2.update(mutate)
        import tempfile, yaml
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(c2, fh); tmp = fh.name
        try:
            C.load_eval_pools(tmp); print("FAIL: accepted bad pool config", mutate); ok = False
        except AssertionError:
            pass
        Path(tmp).unlink()
    print("bad pool configs rejected: True")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
