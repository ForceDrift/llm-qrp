"""Low-bit integer quantization and Salient Outlier Channel Protection.

- ``UniformIntLinear`` replaces a ``nn.Linear`` whose weight is quantized to
  ``bits in {2, 3, 4}`` with uniform symmetric round-to-nearest quantization
  (per output channel), dequantized at the forward pass (fake quantization).
- ``OutlierProtectedLinear`` implements micro-clipping: the top-``k``
  highest-activation weight channels are copied out into an unquantized BF16
  sparse matrix ``W_fp16`` (kept at full precision), while the remaining
  ``in_features - k`` channels are quantized to ``bits`` with
  ``UniformIntLinear``.  The two halves are recomposed in the forward pass.

Both modules mirror the LLM weight layout (``in_features``/``out_features``/
``bias``) so they can be dropped in place of the original projections.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_integer(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric uniform round-to-nearest quantization.

    Returns the dequantized reconstruction of ``w`` at ``bits`` per weight.
    """
    qmax = (1 << (bits - 1)) - 1
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


class UniformIntLinear(nn.Module):
    """``nn.Linear`` with head weight quantized to ``bits`` (2/3/4)."""

    def __init__(self, linear: nn.Module, bits: int):
        super().__init__()
        self.bits = int(bits)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        with torch.no_grad():
            dq = quantize_integer(linear.weight.data.float(), self.bits)
        self.weight = nn.Parameter(dq.to(linear.weight.dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class OutlierProtectedLinear(nn.Module):
    """Linear with the top-``k`` activation channels kept unquantized (BF16).

    ``out_cols`` are the protected channel indices on the *input* side.  Their
    weight columns are stored full-precision in ``weight_high`` (the sparse
    ``W_fp16`` matrix), the remainder is quantized to ``bits``.  Concatenation
    is avoided: the input is split along its feature axis and each half is
    projected separately before summing, which is numerically identical to
    ``F.linear(x, W_full) + bias``.
    """

    def __init__(self, linear: nn.Module, out_cols: Iterable[int], bits: int):
        super().__init__()
        self.bits = int(bits)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        w = linear.weight.data
        protected = sorted(set(int(c) for c in out_cols if 0 <= int(c) < w.shape[1]))
        if not protected:
            raise ValueError("OutlierProtectedLinear requires at least one protected channel")
        low = [c for c in range(w.shape[1]) if c not in protected]
        self.out_cols = list(protected)
        num_outliers = len(protected) + 0
        with torch.no_grad():
            w_low = quantize_integer(w[:, low].float(), self.bits)
            w_high = w[:, protected].contiguous()
        self._low_idx = torch.tensor(low, dtype=torch.long)
        self._high_idx = torch.tensor(protected, dtype=torch.long)
        self.weight_low = nn.Parameter(w_low.to(w.dtype), requires_grad=False)
        self.weight_high = nn.Parameter(w_high.to(torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.outlier_fraction = num_outliers / w.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x[..., self._low_idx]
        x_high = x[..., self._high_idx].to(self.weight_high.dtype)
        y = F.linear(x_low, self.weight_low)
        y = y + F.linear(x_high, self.weight_high)
        if self.bias is not None:
            y = y + self.bias
        return y


def make_linear(original: nn.Module, bits: int,
                outlier_cols: Iterable[int] | None = None) -> nn.Module:
    """Build a low-bit replacement for ``original``.

    With ``outlier_cols`` the replacement protects those channels at BF16 and
    quantizes the rest; otherwise the whole matrix is quantized to ``bits``.
    ``bits`` values outside ``{2, 3, 4}`` should be routed elsewhere
    (bnb 8/4-bit, or skip for 16-bit).
    """
    if outlier_cols:
        return OutlierProtectedLinear(original, outlier_cols, bits)
    return UniformIntLinear(original, bits)