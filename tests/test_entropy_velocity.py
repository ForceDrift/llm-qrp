"""Reference tests for the Information-Bottleneck Convergence Velocity metric
(Section 3.2), expressed in pure NumPy so they run without torch / bnb.

The torch implementation lives in
``qrp.analysis.subcomponent_sled.SubComponentSLED.entropy_velocity``; these
tests pin down the mathematical definition it mirrors:

    P_{l,t}^{(c)} = softmax(l_{l,t}^{(c)})
    H(P) = -sum_v P(v) log P(v)
    DeltaH(l, c) = 1/|T_CoT| * sum_{t in T_CoT} |H(P_{l,t}^{in}) - H(P_{l,t}^{out})|

Run:  python3 tests/test_entropy_velocity.py
"""

import os
import sys

import numpy as np


def shannon_entropy(probs):
    """Vectorized H(P) over the last axis (vocabulary)."""
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def delta_h_velocity(logits_in, logits_out, cot_start):
    """Vel = mean over |H(softmax(in)) - H(softmax(out))| restricted to CoT."""
    h_in = shannon_entropy(softmax(logits_in))
    h_out = shannon_entropy(softmax(logits_out))
    cot = np.abs(h_in - h_out)[:, cot_start:]
    return float(cot.mean())


def _rng():
    return np.random.default_rng(7)


def test_vocabulary_entropy_extrema():
    # Uniform over the vocabulary -> maximum entropy = log V.
    V = 1024
    uniform = np.ones((1, 1024)) / V
    assert np.isclose(shannon_entropy(uniform)[0], np.log(V), atol=1e-4)
    # One-hot -> zero entropy (within the log-floor epsilon).
    one_hot = np.zeros((1, V))
    one_hot[0, 0] = 1.0
    assert np.isclose(shannon_entropy(one_hot)[0], 0.0, atol=1e-4)


def test_zero_when_input_equals_output():
    logits = _rng().normal(size=(1, 64, 128))
    v = delta_h_velocity(logits, np.copy(logits), cot_start=0)
    assert v == 0.0


def test_non_negative_and_bounded():
    rng = _rng()
    logits_in = rng.normal(size=(1, 50, 256))
    logits_out = rng.normal(loc=3.0, scale=0.4, size=(1, 50, 256))
    v = delta_h_velocity(logits_in, logits_out, cot_start=10)
    assert v >= 0.0
    assert v <= np.log(256) + 1e-6  # |H_in - H_out| can never exceed max H


def test_cot_mask_excludes_prompt_positions():
    rng = _rng()
    V = 128
    logits_in = rng.normal(size=(1, 60, V))
    logits_out = rng.normal(loc=2.0, scale=0.5, size=(1, 60, V))
    # The transition only kicks in over the "reasoning" span.
    cot_start = 20
    v_masked = delta_h_velocity(logits_in, logits_out, cot_start)
    h_in = shannon_entropy(softmax(logits_in))
    h_out = shannon_entropy(softmax(logits_out))
    transitions = np.abs(h_in - h_out)
    assert np.isclose(v_masked, float(transitions[0, cot_start:].mean()), atol=1e-6)


def test_degenerate_distribution_after_collapse():
    # A sub-component collapsing to near one-hot shifts entropy from ~log V to ~0,
    # so velocity signals an aggressive information bottleneck.
    V = 1024
    n = 64
    uniform_in = np.tile(np.ones(V) / V, (1, n, 1))
    out = np.full((1, n, V), 0.0)
    out[:, :, 5] = 100.0  # effectively one-hot => H_out ~ 0
    v = delta_h_velocity(uniform_in, out, cot_start=0)
    assert np.isclose(v, np.log(V), atol=1e-3)


def _run_all():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())