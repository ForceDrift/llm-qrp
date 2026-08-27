"""Pure-NumPy sub-component allocation core.

The optimization core is intentionally free of torch / transformers so it can
be unit-tested standalone.  It implements the parameter-free PCA signal fusion
and the exact bit-budget constrained precision allocation described in the
paper (Sections 3.3-3.5).
"""

from __future__ import annotations

import math

import numpy as np

# Candidate bit-widths per sub-component (b_{l,c} in {2, 3, 4, 8, 16}).
BITS = (2, 3, 4, 8, 16)
BYTES_PER_PARAM = {2: 0.25, 3: 0.375, 4: 0.5, 8: 1.0, 16: 2.0}
# Fraction of channels protected to BF16 by Salient Outlier Channel Protection
# (top-0.1% highest-activation weight channels stay unquantized).
DEFAULT_OUTLIER_SHARE = 0.001


# --------------------------------------------------------------------------- #
# Parameter-free PCA fusion (Section 3.3)
# --------------------------------------------------------------------------- #
def zscore(scores: dict[str, float]) -> dict[str, float]:
    """Z-score a single signal: zero mean, unit variance.

    Replaces min-max standardization.  A degenerate (zero-variance) signal maps
    to zeros and contributes no information to the PCA, as standard.
    """
    vals = np.array(list(scores.values()), dtype=float)
    if len(vals) == 0:
        return {}
    mu, sd = float(vals.mean()), float(vals.std())
    if sd < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - mu) / sd for k, v in scores.items()}


def sigmoid(scores: dict[str, float]) -> dict[str, float]:
    """Monotone squash to the open unit interval (0, 1)."""
    return {k: 1.0 / (1.0 + math.exp(-v)) for k, v in scores.items()}


def pca_loadings(signal_scores: dict[str, dict[str, float]]) -> tuple[np.ndarray, dict[str, float]]:
    """First principal component over the normalized feature vectors.

    Builds the feature matrix
        x_{l,c} = [tilde{S}_{SLED}(l,c), tilde{DeltaH}(l,c)]
    (each signal z-scored across sub-components, so the PCA works on the
    correlation matrix and is invariant to measurement units), then returns

        (w, R),  R_{l,c} = w_1 . tilde{S}_{SLED}(l,c) + w_2 . tilde{DeltaH}(l,c)

    where ``w`` is the dominant eigenvector of the covariance matrix (PC1).
    The loadings have unit norm and are sign-aligned so the largest-|w|
    component is positive.  The weighting is learned directly from model
    dynamics -- no lambda hyperparameter and no manual tuning.
    """
    signals = list(signal_scores)
    if not signals:
        return np.empty(0), {}
    ids = sorted(set.intersection(*(set(v) for v in signal_scores.values())))
    if not ids:
        return np.empty(0), {}
    cols = {name: zscore({cid: signal_scores[name][cid] for cid in ids}) for name in signals}
    z = np.array([[cols[name][cid] for name in signals] for cid in ids], dtype=float)  # (n, k)

    k = z.shape[1]
    if k == 1:
        w = np.array([1.0])
    elif len(ids) == 1:
        w = np.ones(k) / math.sqrt(k)
    else:
        cov = np.cov(z, rowvar=False)  # correlation matrix (features are z-scored)
        eigvals, eigvecs = np.linalg.eigh(cov)
        w = eigvecs[:, int(np.argmax(eigvals))]

    largest = int(np.argmax(np.abs(w)))
    w = w * np.sign(w[largest])
    raw = {cid: float(r) for cid, r in zip(ids, z @ w)}
    return w, raw


def pca_criticality(signal_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Unified score ``R_{l,c}`` in (0, 1) from the first principal component.

    ``R_{l,c} = sigmoid(w_1 . tilde{S}_{SLED}(l,c) + w_2 . tilde{DeltaH}(l,c))``.
    The sigmoid is a monotone squash mapping the raw PC1 score onto the open
    unit interval required by the allocator -- it adds no free parameters.
    """
    _w, raw = pca_loadings(signal_scores)
    return sigmoid(raw)


# --------------------------------------------------------------------------- #
# Precision fidelity (Section 3.4)
# --------------------------------------------------------------------------- #
def fidelity(b: int, delta_h_hat: float) -> float:
    """Non-linear precision fidelity ``f(b)`` anchored on measured velocity.

    ``f(b)`` is a monotone, strictly non-linear function of the effective bit
    depth, anchored such that the measured information-bottleneck strength
    ``DeltaH_hat in [0, 1]`` controls how much value a component retains when
    its precision is degraded:

        f(16) = 1
        f(b)  = max(0, 1 - DeltaH_hat * ((16-b)/12)^2)   for b <= 8

    A strong bottleneck (high ``DeltaH_hat``) loses value faster under
    low-bit precision; the quadratic falloff is convex in the bit reduction.
    """
    if b >= 16:
        return 1.0
    psi = ((16.0 - b) / 12.0) ** 2.0
    return max(0.0, 1.0 - delta_h_hat * psi)


# --------------------------------------------------------------------------- #
# Exact 0-1 Multiple-Choice Knapsack allocator (Section 3.5)
# --------------------------------------------------------------------------- #
class MpcAllocator:
    """Bit allocation as a 0-1 Multiple-Choice Knapsack problem.

    For a target average bit budget ``B_target`` (bits/param), the optimizer
    solves

        maximize    sum_{l,c} x_{l,c,b} . R_{l,c} . f_{l,c}(b)
        s.t.        1/P_total * sum_{l,c,b} x_{l,c,b} . P_{l,c} . b <= B_target,
                    sum_b x_{l,c,b} = 1,   x_{l,c,b} in {0, 1},

    i.e. exactly one bit-width ``b in {2, 3, 4, 8, 16}`` is selected per
    sub-component.  Subject to the memory budget, an *importantly* weak
    constraint (see ``allocate``), the solver returns the globally optimal
    assignment; the step-wise percentile grid search is removed.

    Outlier protection enters the budget through the *effective* bits of a
    component: with ``outlier_share`` epsilon of its channels held in BF16,
    ``b_eff = b + epsilon*(16 - b)``.
    """

    def __init__(self, criticality: dict[str, float], params: dict[str, int],
                 delta_h_hat: dict[str, float],
                 outlier_share: float = DEFAULT_OUTLIER_SHARE,
                 max_capacity_units: int = 100_000):
        """Args:
            criticality: ``R_{l,c}`` in (0, 1) from ``pca_criticality``.
            params:      per-component parameter counts.
            delta_h_hat: monotone-squashed (z-scored + sigmoid) Information-
                         Bottleneck Convergence Velocity in (0, 1).
            outlier_share: protected BF16 channel fraction (default 0.1%).
            max_capacity_units: bound on the DP capacity axis (memory/accuracy
                         trade-off of the weight-space resolution).
        """
        self.ids = sorted(criticality)
        self.n = len(self.ids)
        self.R = np.array([criticality[i] for i in self.ids], dtype=float)
        self.P = np.array([params[i] for i in self.ids], dtype=float)
        # Default: an absent velocity is treated as the strongest bottleneck.
        self.dih = np.array([delta_h_hat.get(i, 1.0) for i in self.ids], dtype=float)
        self.outlier_share = float(outlier_share)
        self.max_capacity_units = int(max_capacity_units)
        self.p_total = float(self.P.sum())
        self.bits = tuple(BITS)

    # -- item values / weights -------------------------------------------- #
    def _value(self, i: int, b: int) -> float:
        return float(self.R[i] * fidelity(b, self.dih[i]))

    def _effective_bits(self, b: int) -> float:
        """Bits per param actually consumed incl. the unquantized outlier part."""
        return b + self.outlier_share * (16.0 - b)

    def _weight(self, i: int, b: int) -> float:
        return float(self.P[i] * self._effective_bits(b))

    # -- DP solver -------------------------------------------------------- #
    def allocate(self, target_bits_per_param: float) -> dict:
        """Solve the 0-1 MCKP for one target average bit budget.

        Exact dynamic programming over the integer weight axis with *scaled*
        weight units (resolution controlled by ``max_capacity_units``; the
        problem is solved exactly when the scale is 1).  Ties between
        equal-value configurations are broken deterministically.
        """
        budget = max(float(target_bits_per_param), 2.0)
        capacity = budget * self.p_total  # true bit budget
        if capacity <= 0 or self.n == 0:
            return self._empty_candidate(budget)

        # Adaptive integer scale so capacity fits the DP array.
        scale = max(1, int(math.ceil(capacity / self.max_capacity_units)))
        cap_max = int(capacity / scale)
        if cap_max < self.n:
            scale = int(capacity / self.n) or 1
            cap_max = int(capacity / scale)

        # Scaled integer weights per (component, bit), min weight 1.
        scaled_w = np.zeros((self.n, len(self.bits)), dtype=np.int64)
        vals = np.zeros((self.n, len(self.bits)), dtype=float)
        for i in range(self.n):
            for j, b in enumerate(self.bits):
                w_int = max(1, int(self._weight(i, b) / scale))
                # Full-resolution weight, used for the reported size.
                scaled_w[i, j] = w_int
                vals[i, j] = self._value(i, b)

        # Forward DP: dp[cap] = max value achievable at exact capacity cap.
        neg_inf = -1e300
        dp = np.full(cap_max + 1, neg_inf)
        dp[0] = 0.0
        choices = []  # per component: chosen item index at each capacity
        for i in range(self.n):
            prev = dp
            cur = np.full(cap_max + 1, neg_inf)
            item = np.zeros(cap_max + 1, dtype=np.int8)
            for j in range(len(self.bits)):
                w = int(scaled_w[i, j])
                v = float(vals[i, j])
                if w > cap_max:
                    continue
                cand = np.full(cap_max + 1, neg_inf)
                cand[w:] = prev[: cap_max + 1 - w] + v
                better = cand > cur
                cur = np.where(better, cand, cur)
                item[better] = j
            # deterministic tie-break: prefer lower capacity / stable order
            dp = cur
            choices.append(item)

        best_cap = int(np.argmax(dp))
        best_value = float(dp[best_cap])

        # Backtrack the selected bit-width per component.
        config: dict[str, str] = {}
        cap = best_cap
        for i in range(self.n - 1, -1, -1):
            j = int(choices[i][cap])
            config[self.ids[i]] = f"{self.bits[j]}bit"
            cap -= int(scaled_w[i, j])

        return self._make_candidate(config, best_value, budget, scale)

    def _make_candidate(self, config: dict[str, str], value: float,
                        target_bits_per_param: float, scale: int) -> dict:
        total_bits = 0.0
        total_size = 0.0
        counts = {b: 0 for b in self.bits}
        for i, cid in enumerate(self.ids):
            b = int(config[cid].replace("bit", ""))
            total_bits += self._weight(i, b)
            total_size += self.P[i] * BYTES_PER_PARAM[b]
            counts[b] += 1
        max_bits = self.p_total * 16.0
        return {
            "config": config,
            "value": float(value),
            "bits_per_param": total_bits / self.p_total,
            "size_bytes": float(total_size),
            "size_reduction_pct": (1.0 - total_bits / max_bits) * 100.0,
            "target_bits_per_param": target_bits_per_param,
            "weight_scale": scale,
            "n_2bit": counts[2],
            "n_3bit": counts[3],
            "n_4bit": counts[4],
            "n_8bit": counts[8],
            "n_16bit": counts[16],
        }

    def _empty_candidate(self, target_bits_per_param: float) -> dict:
        config = {cid: "16bit" for cid in self.ids}
        return self._make_candidate(config, 0.0, target_bits_per_param, 1)

    def pareto_frontier(self, targets: list[float] | None = None,
                        target_step: float = 0.5) -> list[dict]:
        """Budget-indexed Pareto frontier of MCKP-optimal configurations.

        Sweeps average bit budgets from 2 bits/param up to 16 bits/param and
        deduplicates identical assignments.  Indexed by *memory density*, never
        by hand-picked layer percentiles.
        """
        if targets is None:
            targets = []
            t = 2.0
            while t <= 16.0 + 1e-9:
                targets.append(t)
                t += target_step
        seen = set()
        frontier = []
        for t in targets:
            cand = self.allocate(t)
            key = tuple(sorted(cand["config"].items()))
            if key in seen:
                continue
            seen.add(key)
            frontier.append(cand)
        return frontier