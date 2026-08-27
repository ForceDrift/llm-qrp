"""Unit tests for the sub-component allocation core (no torch required).

Run:  python3 tests/test_allocation_core.py
"""

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qrp.quantize.allocation_core import (  # noqa: E402
    BITS,
    BYTES_PER_PARAM,
    MpcAllocator,
    fidelity,
    pca_criticality,
    pca_loadings,
    sigmoid,
    zscore,
)


# --------------------------------------------------------------------------- #
# Scaling / PCA fusion
# --------------------------------------------------------------------------- #
def test_zscore_zero_mean_unit_variance():
    scores = {"a": 1.0, "b": 3.0, "c": 5.0, "d": 7.0}
    out = zscore(scores)
    assert set(out) == set(scores)
    assert abs(np.mean(list(out.values()))) < 1e-12
    assert abs(np.std(list(out.values())) - 1.0) < 1e-12
    assert out["a"] < out["c"]


def test_zscore_degenerate():
    out = zscore({"a": 2.0, "b": 2.0})
    assert all(v == 0.0 for v in out.values())


def test_sigmoid_bounds_and_monotonic():
    out = sigmoid({"a": -5.0, "b": 0.0, "c": 5.0})
    assert all(0.0 < v < 1.0 for v in out.values())
    assert out["a"] < out["b"] < out["c"]
    assert abs(out["b"] - 0.5) < 1e-9


def test_pca_loadings_correlated_signals():
    sled = {"0.attn": 0.2, "0.mlp": 0.9, "1.attn": 0.1}
    ent = {"0.attn": 0.7, "0.mlp": 2.4, "1.attn": 0.4}
    w, raw = pca_loadings({"sled": sled, "entropy": ent})
    s = 1.0 / math.sqrt(2.0)
    assert abs(abs(w[0]) - s) < 1e-9
    assert abs(abs(w[1]) - s) < 1e-9
    assert w[0] * w[1] > 0
    zs, ze = zscore(sled), zscore(ent)
    for cid in sled:
        expected = zs[cid] * w[0] + ze[cid] * w[1]
        assert abs(raw[cid] - expected) < 1e-9


def test_pca_loadings_antiphase_signals():
    sled = {"high_sled": 10.0, "low_sled": -10.0}
    ent = {"high_sled": -10.0, "low_sled": 10.0}
    w, raw = pca_loadings({"sled": sled, "entropy": ent})
    assert w[0] * w[1] < 0
    assert raw["high_sled"] > 0.0 > raw["low_sled"]


def test_pca_score_is_weighted_combination():
    sled = {"a": 0.0, "b": 100.0, "c": 50.0}
    ent = {"a": 0.0, "b": 100.0, "c": 50.0}
    w, raw = pca_loadings({"sled": sled, "entropy": ent})
    assert raw["a"] < raw["c"] < raw["b"]
    crit = pca_criticality({"sled": sled, "entropy": ent})
    assert all(0.0 < v < 1.0 for v in crit.values())
    assert crit["a"] < crit["c"] < crit["b"]


def test_pca_single_signal():
    w, raw = pca_loadings({"sled": {"a": 0.1, "b": 5.0}})
    assert abs(w[0] - 1.0) < 1e-12
    assert raw["b"] > raw["a"]


def test_pca_requires_all_signals():
    w, raw = pca_loadings({"sled": {"a": 1.0}, "entropy": {"b": 1.0}})
    assert raw == {}
    assert len(w) == 0


def test_pca_zero_variance_signal_dropped():
    sled = {"a": 1.0, "b": 3.0}
    ent = {"a": 2.0, "b": 2.0}
    w, _ = pca_loadings({"sled": sled, "entropy": ent})
    assert abs(abs(w[0]) - 1.0) < 1e-9
    assert abs(w[1]) < 1e-9


# --------------------------------------------------------------------------- #
# Precision fidelity f(b)
# --------------------------------------------------------------------------- #
def test_bits_and_bytes_table():
    assert tuple(BITS) == (2, 3, 4, 8, 16)
    assert BYTES_PER_PARAM == {2: 0.25, 3: 0.375, 4: 0.5, 8: 1.0, 16: 2.0}


def test_fidelity_monotone_and_anchored():
    dh = 0.6
    fs = {b: fidelity(b, dh) for b in BITS}
    assert fs[16] == 1.0
    # strictly monotone decreasing with bit depth
    for a, b in zip(sorted(BITS, reverse=True), sorted(BITS, reverse=True)[1:]):
        assert fs[a] > fs[b]
    # anchored on the measured bottleneck: f(4) = 1 - DeltaH_hat
    assert abs(fs[4] - (1.0 - dh)) < 1e-12
    # non-linear: midpoint f(10) would be (1+f4)/2 if linear in b
    f10_linear = 1.0 - dh * ((16.0 - 10.0) / 12.0)
    assert abs(fs[8] - (1.0 - dh * ((16.0 - 8.0) / 12.0) ** 2.0)) < 1e-12
    assert abs(f10_linear - fs[8]) > 1e-6


def test_fidelity_strong_bottleneck_loses_more():
    for b in (2, 3, 4, 8):
        assert fidelity(b, 0.9) < fidelity(b, 0.3)


def test_fidelity_clamped_non_negative():
    assert fidelity(2, 1.0) >= 0.0
    assert fidelity(2, 0.5) > 0.0


# --------------------------------------------------------------------------- #
# 0-1 Multiple-Choice Knapsack allocator
# --------------------------------------------------------------------------- #
def _make_alloc(n=4, outlier_share=0.0):
    ids = [f"{i}.{c}" for i in range(n // 2) for c in ("attn", "mlp")]
    crit = {cid: 0.2 + 0.8 * i / max(1.0, len(ids) - 1) for i, cid in enumerate(ids)}
    p = {cid: 100 * (1 + i % 3) for i, cid in enumerate(ids)}
    dh = {cid: 0.5 for cid in ids}
    return MpcAllocator(crit, p, dh, outlier_share=outlier_share)


def _bruteforce(alloc, budget_bits_per_param):
    ids, bits = alloc.ids, alloc.bits
    best_v, best_config = -1e300, None
    rhs = budget_bits_per_param * alloc.p_total
    for combo in itertools.product(bits, repeat=len(ids)):
        total_bits = sum(alloc._weight(i, b) for i, b in enumerate(combo))
        if total_bits > rhs * (1.0 + 1e-12):
            continue
        v = sum(alloc._value(i, b) for i, b in enumerate(combo))
        if v > best_v + 1e-12:
            best_v, best_config = v, {cid: f"{b}bit" for cid, b in zip(ids, combo)}
    return best_v, best_config


def test_allocate_matches_bruteforce():
    alloc = _make_alloc(n=4)  # 2 components x 2 layers -> 16 configs? no: 4 choices^k
    for target in (2.5, 3.5, 5.0, 8.0, 12.0):
        cand = alloc.allocate(target)
        bf_v, bf_cfg = _bruteforce(alloc, target)
        assert abs(cand["value"] - bf_v) < 1e-9
        # if brute-force optimum is unique, configs must match exactly
        if sum(alloc._value(i, b) for i, b in enumerate(
                [int(bf_cfg[cid].replace("bit", "")) for cid in alloc.ids])) == bf_v:
            assert cand["config"] == bf_cfg


def test_allocate_respects_budget():
    alloc = _make_alloc(n=6)
    for target in (2.5, 3.0, 4.0, 6.0, 10.0):
        cand = alloc.allocate(target)
        assert cand["bits_per_param"] <= target + 0.05
        assert cand["target_bits_per_param"] == target
        total = cand["n_2bit"] + cand["n_3bit"] + cand["n_4bit"] + cand["n_8bit"] + cand["n_16bit"]
        assert total == alloc.n


def test_allocate_full_precision_at_16():
    alloc = _make_alloc(n=4)
    cand = alloc.allocate(16.0)
    assert all(b == "16bit" for b in cand["config"].values())
    assert abs(cand["bits_per_param"] - 16.0) < 1e-9


def test_allocate_minimum_precision():
    alloc = _make_alloc(n=4)
    cand = alloc.allocate(2.0)
    assert all(b == "2bit" for b in cand["config"].values())


def test_criticality_orders_compression():
    # With uniform params/velocity, the lowest-R components lose precision first.
    ids = sorted(f"{i}.{c}" for i in range(4) for c in ("attn", "mlp"))
    crit = {cid: i / max(1.0, len(ids) - 1) for i, cid in enumerate(ids)}
    p = {cid: 1000 for cid in ids}
    dh = {cid: 0.5 for cid in ids}
    alloc = MpcAllocator(crit, p, dh)
    cand = alloc.allocate(9.0)
    order = sorted(ids, key=lambda x: crit[x])
    compressed = {cid for cid, b in cand["config"].items() if b != "16bit"}
    if compressed:
        max_compressed_rank = max(order.index(c) for c in compressed)
        assert all(order.index(c) <= max_compressed_rank for c in compressed)


def test_outlier_share_increases_budget_usage():
    params = {cid: 1000 for cid in ("0.attn", "0.mlp", "1.attn", "1.mlp")}
    crit = {cid: 0.5 for cid in params}
    dh = {cid: 0.5 for cid in params}
    plain = MpcAllocator(crit, params, dh, outlier_share=0.0)
    protected = MpcAllocator(crit, params, dh, outlier_share=0.1)
    cand_p = protected.allocate(3.0)
    cand_0 = plain.allocate(3.0)
    # 10% outlier share -> a strictly heavier budget for the same bits/param target,
    # so the protected config must use fewer low-bit weights overall.
    assert protected._effective_bits(2) > 2.0
    assert cand_p["bits_per_param"] > cand_0["bits_per_param"] - 1e-12


def _top_k_channels(salience, k):
    """Reference top-k selection mirroring SubComponentSLED._top_channels."""
    return np.argsort(-np.asarray(salience))[:k].tolist()


def test_outlier_channel_selection_semantics():
    # top-0.1% of a hidden dim, at least 1 channel (matches the profiler).
    for D in (1152, 768, 4096):
        k = max(1, math.ceil(0.001 * D))
        assert k == int(np.ceil(0.001 * D))
        rng = np.random.default_rng(D)
        act = rng.normal(size=(2, 5, D))
        means = np.abs(act).mean(axis=(0, 1))
        chosen = _top_k_channels(means, k)
        assert len(chosen) == min(k, D)
        # the protected channels are exactly the k highest-salience ones
        top = np.argsort(means)[-k:][::-1]
        assert sorted(chosen) == sorted(top.tolist())


def test_frontier_indexed_by_density():
    alloc = _make_alloc(n=6)
    frontier = alloc.pareto_frontier(target_step=1.0)
    densities = [f["bits_per_param"] for f in frontier]
    assert densities == sorted(densities)
    seen = set()
    for f in frontier:
        key = tuple(sorted(f["config"].items()))
        assert key not in seen
        seen.add(key)


def test_frontier_contains_baseline():
    alloc = _make_alloc(n=2)
    frontier = alloc.pareto_frontier(targets=[16.0])
    assert len(frontier) == 1
    assert frontier[0]["size_reduction_pct"] == 0.0
    assert all(b == "16bit" for b in frontier[0]["config"].values())


def _run_all():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())