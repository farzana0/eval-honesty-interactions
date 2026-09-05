"""The item-level bootstrap must be a CLUSTER bootstrap: items resampled with
multiplicity, whole items moving together, nominal coverage of the parameter,
and wider intervals than a naive row-level bootstrap when rows are correlated
within an item."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direction_utils import item_bootstrap_ci


def main() -> int:
    ok = True

    # 1) MULTISET, not subsample. Each draw returns the ORIGINAL row count and
    #    items recur; a set-collapse bootstrap returns ~63% of rows, never
    #    repeating an item.
    n_items, rows_per = 40, 8
    items = np.repeat([f"it{i}" for i in range(n_items)], rows_per)
    uniq = np.unique(items)
    sizes, max_mult = [], []

    def probe(idx):
        sizes.append(len(idx))
        _, c = np.unique(items[idx], return_counts=True)
        max_mult.append(int(c.max()) // rows_per)
        return float(len(idx))

    item_bootstrap_ci(probe, items, uniq, n_boot=100, seed=1)
    print(f"rows per draw all == {len(items)}: {set(sizes) == {len(items)}}")
    print(f"max item multiplicity seen: {max(max_mult)} (a subsample bootstrap gives 1)")
    ok &= set(sizes) == {len(items)} and max(max_mult) >= 2

    # 2) WHOLE ITEMS move together: every item present in a draw contributes a
    #    multiple of its full row count.
    whole = True

    def check_whole(idx):
        u, c = np.unique(items[idx], return_counts=True)
        nonlocal_ok = all(x % rows_per == 0 for x in c)
        nonlocal whole
        whole &= nonlocal_ok
        return 0.0

    item_bootstrap_ci(check_whole, items, uniq, n_boot=50, seed=2)
    print(f"items never split across a draw: {whole}")
    ok &= whole

    # 3) NOMINAL COVERAGE of the true parameter, on a statistic where the
    #    parameter is known and interior (the item-level mean). Note this is the
    #    right property to test: a percentile CI targets the PARAMETER, and for
    #    strongly nonlinear/bounded statistics the point estimate itself can sit
    #    outside its own interval, which is expected, not a defect.
    true_mu, hits, trials = 3.0, 0, 200
    for t in range(trials):
        r = np.random.default_rng(1000 + t)
        n_it = 50
        it = np.repeat([f"i{i}" for i in range(n_it)], 4)
        # within-item correlation: a per-item offset shared by its rows
        offs = r.normal(0, 1.0, size=n_it).repeat(4)
        y = true_mu + offs + r.normal(0, 0.3, size=len(it))
        lo, hi = item_bootstrap_ci(lambda idx: float(y[idx].mean()),
                                   it, np.unique(it), n_boot=200, seed=t)
        hits += (lo <= true_mu <= hi)
    cover = hits / trials
    print(f"95% CI coverage of the true mean: {cover:.3f} (need 0.88-1.00)")
    ok &= 0.88 <= cover <= 1.0

    # 4) CLUSTER-AWARE: with within-item correlation, resampling items must give
    #    a WIDER interval than resampling rows independently. If it did not, the
    #    clustering would be cosmetic and CIs would be anticonservative.
    r = np.random.default_rng(7)
    n_it = 50
    it = np.repeat([f"i{i}" for i in range(n_it)], 4)
    offs = r.normal(0, 1.0, size=n_it).repeat(4)
    y = true_mu + offs + r.normal(0, 0.3, size=len(it))
    lo_i, hi_i = item_bootstrap_ci(lambda idx: float(y[idx].mean()), it, np.unique(it),
                                   n_boot=400, seed=3)
    rows = np.arange(len(y)).astype(str)   # each row its own "item" == row bootstrap
    lo_r, hi_r = item_bootstrap_ci(lambda idx: float(y[idx].mean()), rows, np.unique(rows),
                                   n_boot=400, seed=3)
    wi, wr = hi_i - lo_i, hi_r - lo_r
    print(f"CI width: item-level={wi:.4f}  row-level={wr:.4f}  ratio={wi/wr:.2f} (need > 1.3)")
    ok &= wi / wr > 1.3

    # 5) determinism
    a = item_bootstrap_ci(lambda idx: float(len(idx)), items, uniq, n_boot=50, seed=7)
    b = item_bootstrap_ci(lambda idx: float(len(idx)), items, uniq, n_boot=50, seed=7)
    print(f"deterministic under seed: {a == b}")
    ok &= a == b

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
