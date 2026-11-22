"""Audio IIR filter design package.

The implementation lives in :mod:`iir.design`; the helpers are re-exported
here for convenience.
"""

from .design import (
    cascade,
    design_batch,
    freq_response,
    highpass,
    lowpass,
    order_topology,
    pole_radius,
    prewarp,
)

__all__ = [
    "prewarp",
    "lowpass",
    "highpass",
    "cascade",
    "freq_response",
    "pole_radius",
    "order_topology",
    "design_batch",
]
