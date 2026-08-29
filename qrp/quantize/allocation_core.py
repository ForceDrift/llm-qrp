"""Pure-NumPy sub-component allocation core.

The optimization core is intentionally free of torch / transformers so it can
be unit-tested standalone.  It implements the parameter-free PCA signal fusion
and the exact bit-budget constrained precision allocation described in the
paper (Sections 3.3-3.5).
"""

from __future__ import annotations

import math

import numpy as np

# Candidate bit-widths per sub-component (b_{l,c} in {3, 4, 6, 8, 16}).
#
# A 2-bit width is intentionally excluded and 6-bit inserted as an intermediate
# precision step.  Experimental profiling on tiny (<1B) models shows that the
# first transformer layer (the "outlier layer" right after the embedding) cannot
# be represented in 2-bit uniform quantization and that *bulk* 3-bit still
# compounds destructively through depth (e.g. layer-0 down_proj: 2-bit -> 0.03
# vs 3-bit -> 0.19; a whole-config 3-bit drop from 0.19 -> 0.01).  Keeping
# {3, 4, 6, 8, 16} preserves an aggressive but non-fatal floor while 6-bit gives
# the knock-away a finer ladder between 4- and 8-bit so the optimum need not
# snap straight from 4-bit to 8-bit.
BITS = (3, 4, 6, 8, 16)
BYTES_PER_PARAM = {3: 0.375, 4: 0.5, 6: 0.75, 8: 1.0, 16: 2.0}
# Fraction of channels protected to BF16 by Salient Outlier Channel Protection
# (top-0.1% highest-activation weight channels stay unquantized).
DEFAULT_OUTLIER_SHARE = 0.001

# Critical-component shielding (Section 3.5).  Small models have "anchor"
# sub-components -- typically early LayerNorms, V-projections and final MLP
# gate projections -- whose premature low-bit quantization acts as a single
# point of failure and collapses the model.  Any sub-component whose unified
# criticality R_{l,c} meets this threshold is hard-constrained to at least
# CRITICAL_MIN_BITS precision, regardless of how much the bit budget would
# prefer to compress it.
CRITICAL_R_THRESHOLD = 0.75
CRITICAL_MIN_BITS = 8
# Fraction of the top-criticality components that must be shielded (defensive
# guard so the threshold cannot degenerate to an empty set on skewed signals).
CRITICAL_MIN_SHARE = 0.05

# Bulk sub-8-bit caps ("alloy" constraints).  Small models tolerate *isolated*
# 4/6/8-bit components but collapse when a whole-slab of sub-8-bit components is
# laid down (e.g. a 20x6-bit + 3x3-bit config at ~2x compression craters while
# the old 8-bit-everywhere config at the same density is near-lossless).  These
# caps bound the number of components allowed at the two destructive low bands:
#   3-bit (fatal for granite) and <=6-bit (compounding through depth).  They are
# expressed as *fractions of the component count* so they scale to model size.
LOW3_MAX_FRAC = 0.08
LOW6_MAX_FRAC = 0.25


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
        f(b)  = max(0, 1 - DeltaH_hat * ((16-b)/12)^3)   for b <= 8

    The cubic exponent is calibrated to measurements on tiny (<1B) models:
    a lone 3/4/8-bit component is near-fine, but bulk sub-8-bit (especially
    3-bit) collapses the model through depth.  The cubic curvature prices the
    destructive 3-bit band sharply below 4-bit while keeping 6/8-bit close to
    lossless, and it stays convex in the bit reduction so an aggressive budget
    cannot legally paper over a row of very-low-bit components.
    """
    if b >= 16:
        return 1.0
    psi = ((16.0 - b) / 12.0) ** 3.0
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

    i.e. exactly one bit-width ``b in {3, 4, 8, 16}`` is selected per
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
                 max_capacity_units: int = 100_000,
                 critical_r_threshold: float = CRITICAL_R_THRESHOLD,
                 critical_min_bits: int = CRITICAL_MIN_BITS,
                 critical_min_share: float = CRITICAL_MIN_SHARE,
                 low3_max_frac: float = LOW3_MAX_FRAC,
                 low6_max_frac: float = LOW6_MAX_FRAC):
        """Args:
            criticality: ``R_{l,c}`` in (0, 1) from ``pca_criticality``.
            params:      per-component parameter counts.
            delta_h_hat: monotone-squashed (z-scored + sigmoid) Information-
                         Bottleneck Convergence Velocity in (0, 1).
            outlier_share: protected BF16 channel fraction (default 0.1%).
            max_capacity_units: bound on the DP capacity axis (memory/accuracy
                         trade-off of the weight-space resolution).
            critical_r_threshold: unified criticality at/above which a component
                         is shielded from aggressive quantization.
            critical_min_bits: minimum bit-width for shielded components.
            critical_min_share: defensive fraction of the very highest-criticality
                         components always shielded even when no component clears
                         ``critical_r_threshold``.
            low3_max_frac: max fraction of components allowed at 3-bit.
            low6_max_frac: max fraction of components allowed at <=6-bit
                         (bulk sub-8-bit "alloy" caps).
        """
        self.ids = sorted(criticality)
        self.n = len(self.ids)
        self.R = np.array([criticality[i] for i in self.ids], dtype=float)
        self.P = np.array([params[i] for i in self.ids], dtype=float)
        # Default: an absent velocity is treated as the strongest bottleneck.
        self.dih = np.array([delta_h_hat.get(i, 1.0) for i in self.ids], dtype=float)
        self.outlier_share = float(outlier_share)
        self.max_capacity_units = int(max_capacity_units)
        self.critical_r_threshold = float(critical_r_threshold)
        self.critical_min_bits = int(critical_min_bits)
        self.critical_min_share = float(critical_min_share)
        self.max_low3 = int(math.ceil(float(low3_max_frac) * self.n))
        self.max_low6 = int(math.ceil(float(low6_max_frac) * self.n))
        self.p_total = float(self.P.sum())
        self.bits = tuple(BITS)
        self._critical_mask = self._compute_critical_mask()

    def _compute_critical_mask(self) -> np.ndarray:
        """Boolean mask of components that must be shielded to high precision.

        A component is shielded if its unified criticality ``R_{l,c}`` meets the
        ``CRITICAL_R_THRESHOLD``.  A defensive ``CRITICAL_MIN_SHARE`` guard
        additionally shields the very highest-criticality components even when
        the raw threshold would yield an empty (or negligible) shield set, so
        the constraint can never silently degenerate away on skewed signals.
        """
        mask = self.R >= self.critical_r_threshold
        k_min = int(math.ceil(self.critical_min_share * self.n))
        if int(mask.sum()) < k_min:
            order = np.argsort(-self.R)
            mask[order[:k_min]] = True
        return mask

    @property
    def n_critical(self) -> int:
        return int(self._critical_mask.sum())

    # -- item values / weights -------------------------------------------- #
    def _value(self, i: int, b: int) -> float:
        return float(self.R[i] * fidelity(b, self.dih[i]))

    def _effective_bits(self, b: int) -> float:
        """Bits per param actually consumed incl. the unquantized outlier part."""
        return b + self.outlier_share * (16.0 - b)

    def _weight(self, i: int, b: int) -> float:
        return float(self.P[i] * self._effective_bits(b))

    def _min_feasible_config(self) -> tuple[dict[str, str], float]:
        """Minimum-weight caps-feasible configuration.

        Assigned when the target budget is below what the bulk caps allow: put
        as many components as the caps permit at 3-bit, then 6-bit, then
        (critical floor / 8-bit).  Returns ``(config, value)``.
        """
        not_crit = [i for i in range(self.n) if not self._critical_mask[i]]
        # Prefer 3-bit (cheapest allowed) for the smallest components.
        order3 = sorted(not_crit, key=lambda i: (self.P[i], -self.R[i]))[: self.max_low3]
        remain = [i for i in not_crit if i not in order3]
        order6 = sorted(remain, key=lambda i: (self.P[i], -self.R[i]))[: max(0, self.max_low6 - len(order3))]
        cfg: dict[str, str] = {}
        value = 0.0
        count6_allowed = self.max_low6
        used3 = 0
        used6 = 0
        for i in range(self.n):
            if self._critical_mask[i]:
                b = self.critical_min_bits
            elif i in order3 and used3 < self.max_low3 and used6 < count6_allowed:
                b = 3
                used3 += 1
                used6 += 1
            elif i in order6 and used6 < count6_allowed:
                b = 6
                used6 += 1
            else:
                b = 8
            cfg[self.ids[i]] = f"{b}bit"
            value += self._value(i, b)
        return cfg, float(value)

    # -- DP solver -------------------------------------------------------- #
    def allocate(self, target_bits_per_param: float) -> dict:
        """Solve the 0-1 MCKP for one target average bit budget.

        Exact dynamic programming over two resource axes with *scaled* weight
        units (resolution controlled by ``max_capacity_units``; the problem is
        solved exactly when the scale is 1).  The second and third axes track
        how many components have been laid down at the two destructive low-bit
        bands (3-bit and <=6-bit) so the bulk caps ``max_low3`` / ``max_low6``
        are enforced as hard "alloy" constraints.  Ties between equal-value
        configurations are broken deterministically.
        """
        budget = max(float(target_bits_per_param), min(self.bits))
        capacity = budget * self.p_total  # true bit budget
        if capacity <= 0 or self.n == 0:
            return self._empty_candidate(budget)

        # Adaptive integer scale so capacity fits the DP arrays.
        scale = max(1, int(math.ceil(capacity / self.max_capacity_units)))
        cap_max = int(capacity / scale)
        if cap_max < self.n:
            scale = int(capacity / self.n) or 1
            cap_max = int(capacity / scale)

        neg_inf = -1e300
        k3_max = self.max_low3
        k6_max = self.max_low6

        # Scaled integer weights per (component, bit), min weight 1.
        scaled_w = np.zeros((self.n, len(self.bits)), dtype=np.int64)
        vals = np.zeros((self.n, len(self.bits)), dtype=float)
        # Per-bit low-band category: 0 = high (>=8), 1 = 4/6-bit (<=6 only),
        # 2 = 3-bit (both caps).
        bit_band = np.zeros(len(self.bits), dtype=int)
        for j, b in enumerate(self.bits):
            if b == 3:
                bit_band[j] = 2
            elif b <= 6:
                bit_band[j] = 1
        for i in range(self.n):
            for j, b in enumerate(self.bits):
                scaled_w[i, j] = max(1, int(self._weight(i, b) / scale))
                # Critical-component shielding: a shielded (anchor) sub-component
                # must never be quantized below CRITICAL_MIN_BITS.
                if self._critical_mask[i] and b < self.critical_min_bits:
                    vals[i, j] = neg_inf
                else:
                    vals[i, j] = self._value(i, b)

        # Sparse state DP: states[(k3, k6)] = np.ndarray over exact capacity cap,
        # holding the best value achievable after processing some prefix of
        # components with ``k3`` three-bit and ``k6`` <=6-bit components.
        state_hist = []  # per-component snapshot for backtracking
        states = {(0, 0): np.full(cap_max + 1, neg_inf)}
        states[(0, 0)][0] = 0.0
        state_hist.append({s: a.copy() for s, a in states.items()})

        for i in range(self.n):
            live = list(states.items())
            cur: dict[tuple[int, int], np.ndarray] = {}
            for (k3, k6), prev in live:
                for j in range(len(self.bits)):
                    b = self.bits[j]
                    w = int(scaled_w[i, j])
                    v = float(vals[i, j])
                    if v == neg_inf or w > cap_max:
                        continue
                    band = bit_band[j]
                    if band == 2:  # 3-bit: consumes both caps
                        if k3 >= k3_max or k6 >= k6_max:
                            continue
                        nk3, nk6 = k3 + 1, k6 + 1
                    elif band == 1:  # 4/6-bit: consumes <=6 cap only
                        if k6 >= k6_max:
                            continue
                        nk3, nk6 = k3, k6 + 1
                    else:  # 8/16-bit: consumes neither cap
                        nk3, nk6 = k3, k6
                    nxt_arr = cur.setdefault(
                        (nk3, nk6), np.full(cap_max + 1, neg_inf))
                    reachable = prev != neg_inf
                    cand = np.full(cap_max + 1, neg_inf)
                    cand[w:] = np.where(reachable[: cap_max + 1 - w],
                                        prev[: cap_max + 1 - w] + v,
                                        neg_inf)
                    np.maximum(nxt_arr, cand, out=nxt_arr)
            states = cur
            state_hist.append({s: a.copy() for s, a in states.items()})

        # Best final state over all (k3, k6) within caps and all capacities.
        best_val = neg_inf
        best_key: tuple[int, int] | None = None
        best_cap = 0
        for (k3, k6), arr in states.items():
            cap_i = int(np.argmax(arr))
            val = float(arr[cap_i])
            if val > best_val:
                best_val, best_key, best_cap = val, (k3, k6), cap_i

        # No state fits the nominal budget (an aggressive target below the
        # minimum weight achievable within the bulk caps).  Fall back to the
        # minimum-weight caps-feasible configuration so that an over-budget
        # assignment is still returned instead of a neg_inf empty result.
        if best_val <= neg_inf / 2.0:
            fb_cfg, fb_value = self._min_feasible_config()
            return self._make_candidate(fb_cfg, fb_value, budget, scale)

        config: dict[str, str] = {}
        cap = best_cap
        k3, k6 = best_key if best_key is not None else (0, 0)
        for i in range(self.n - 1, -1, -1):
            prev = state_hist[i]
            cur = state_hist[i + 1]
            chosen = -1
            for j in range(len(self.bits)):
                b = self.bits[j]
                w = int(scaled_w[i, j])
                v = float(vals[i, j])
                if v == neg_inf or w > cap:
                    continue
                band = bit_band[j]
                if band == 2:
                    if k3 < 1 or k6 < 1:
                        continue
                    pk3, pk6 = k3 - 1, k6 - 1
                elif band == 1:
                    if k6 < 1:
                        continue
                    pk3, pk6 = k3, k6 - 1
                else:
                    pk3, pk6 = k3, k6
                pred_cap = cap - w
                p_arr = prev.get((pk3, pk6))
                if p_arr is None or p_arr[pred_cap] == neg_inf:
                    continue
                c_arr = cur.get((k3, k6))
                if c_arr is None:
                    continue
                if abs((p_arr[pred_cap] + v) - c_arr[cap]) < 1e-6:
                    chosen = j
                    break
            if chosen < 0:
                chosen = len(self.bits) - 1  # 16-bit fallback
            config[self.ids[i]] = f"{self.bits[chosen]}bit"
            cap -= int(scaled_w[i, chosen])
            band = bit_band[chosen]
            if band == 2:
                k3 -= 1
                k6 -= 1
            elif band == 1:
                k6 -= 1

        return self._make_candidate(config, best_val, budget, scale)

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
            "n_2bit": counts.get(2, 0),
            "n_3bit": counts.get(3, 0),
            "n_4bit": counts.get(4, 0),
            "n_6bit": counts.get(6, 0),
            "n_8bit": counts.get(8, 0),
            "n_16bit": counts.get(16, 0),
        }

    def _empty_candidate(self, target_bits_per_param: float) -> dict:
        config = {cid: "16bit" for cid in self.ids}
        return self._make_candidate(config, 0.0, target_bits_per_param, 1)

    def pareto_frontier(self, targets: list[float] | None = None,
                        target_step: float = 0.5) -> list[dict]:
        """Budget-indexed Pareto frontier of MCKP-optimal configurations.

        Sweeps average bit budgets from ``min(BITS)`` bits/param up to 16
        bits/param and deduplicates identical assignments.  Indexed by *memory
        density*, never by hand-picked layer percentiles.
        """
        if targets is None:
            targets = []
            t = float(min(BITS))
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