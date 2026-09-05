"""Shared statistical core: calibration/test split, difference-of-means
direction, AUROC, item-level bootstrap, principal angles.

Factored out because the parent project (qwen-sandbag-diffing) duplicated this
logic across four scripts (validate_eval_direction.py, make_main_figure.py,
within_arm_control.py, position_sweep.py) -- flagged as the one clear refactor
opportunity by the port's own inventory. Everything here is pure numpy; no
model loading, no I/O.
"""

from __future__ import annotations

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC of pos vs neg scores. Pure numpy, tie-corrected."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x)
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        r = (s / cnt)[inv]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_item_split(
    items: list[str], strata: list[str], seed: int = 0, frac: float = 0.5
) -> tuple[set[str], set[str]]:
    """Split unique items into (calibration, test), stratified by `strata`
    (e.g. domain), so every item's rows (across all framings) stay on one side.

    Returns two DISJOINT sets whose union is every unique item in `items`.
    """
    rng = np.random.default_rng(seed)
    items_arr = np.asarray(items)
    strata_arr = np.asarray(strata)
    uniq_items = np.unique(items_arr)
    stratum_of = dict(zip(items_arr, strata_arr))
    cal: set[str] = set()
    for s in sorted(set(stratum_of.values())):
        ids = sorted(i for i in uniq_items if stratum_of[i] == s)
        ids = list(ids)
        rng.shuffle(ids)
        cal |= set(ids[: int(round(len(ids) * frac))])
    test = set(uniq_items.tolist()) - cal
    overlap = cal & test
    assert not overlap, f"calibration/test split is not disjoint: {overlap}"
    return cal, test


def diff_of_means_direction(x: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    """x: (n, d). Returns mean(x[pos]) - mean(x[neg]), shape (d,)."""
    assert pos_mask.any(), "positive mask is empty"
    assert neg_mask.any(), "negative mask is empty"
    return x[pos_mask].mean(axis=0) - x[neg_mask].mean(axis=0)


def item_bootstrap_ci(
    fn, items: np.ndarray, unique_items: np.ndarray, n_boot: int = 300, seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Item-level (cluster) bootstrap CI for a scalar statistic.

    `fn(idx: np.ndarray[int]) -> float` computes the statistic over the given
    ROW INDICES. Items are resampled with replacement and an item drawn k times
    contributes its rows k times, so this is a proper multiset (cluster)
    bootstrap -- the resampled row count matches the original.

    Indices, not a boolean mask, precisely so multiplicity is representable:
    a mask can only say present/absent, which silently degrades the bootstrap
    into a ~63% random subsample. That variant biases any statistic whose value
    depends on estimation noise (a cosine to a fixed target is one: fewer items
    -> noisier direction -> lower cosine), which puts the full-sample value
    outside its own interval about half the time. tests/test_bootstrap.py
    checks the coverage property that distinguishes the two.
    """
    rng = np.random.default_rng(seed)
    rows_by_item = {it: np.flatnonzero(items == it) for it in unique_items}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(unique_items, size=len(unique_items), replace=True)
        idx = np.concatenate([rows_by_item[i] for i in draw])
        try:
            vals.append(fn(idx))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, ci[0])), float(np.percentile(vals, ci[1])))


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of A
    and B. A: (d, k1), B: (d, k2), any k1/k2 >= 1 (not necessarily orthonormal
    -- QR'd internally). Returns min(k1, k2) angles in [0, pi/2].
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    # Bjorck-Golub: principal angles from the SVD of Qa^T Qb.
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(np.sort(s))
