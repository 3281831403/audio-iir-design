"""Second-order IIR filter design helpers (bilinear transform, RBJ sections).

Conventions
-----------
* Every frequency is in hertz (Hz), and ``sample_rate`` is in Hz as well.
  ``q`` is the dimensionless quality factor of one second-order section;
  the Butterworth value is ``q = 1 / sqrt(2)``.
* A *biquad* is a plain ``dict`` of floats with the keys ``"b0"``, ``"b1"``,
  ``"b2"``, ``"a1"``, ``"a2"``, ``"q"`` and ``"freq"``.  It describes

      H(z) = (b0 + b1 * z**-1 + b2 * z**-2) / (1 + a1 * z**-1 + a2 * z**-2)

  so the denominator is already normalised to ``a0 == 1``.
* A *cascade* is a ``dict`` with the coefficient lists ``"b"`` and ``"a"``
  in ascending powers of ``z**-1`` (index 0 is the constant term), plus
  ``"sections"`` holding the biquads that were combined.
* All numeric contracts are expected to agree to within ``1e-9``.
"""

import cmath
import math


def prewarp(freq, sample_rate):
    """Return the bilinear-transform prewarped frequency of ``freq``.

    The analogue prototype is normalised with ``K = tan(pi * freq /
    sample_rate)``; substituting ``s = (z - 1) / (z + 1)`` then maps the
    prototype corner exactly onto ``freq`` on the digital frequency axis.

    Parameters
    ----------
    freq : float
        Corner frequency in Hz, ``0 < freq < sample_rate / 2``.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        ``tan(pi * freq / sample_rate)`` (dimensionless, strictly positive
        and increasing in ``freq``).
    """
    return math.tan(math.pi * freq / sample_rate)


def lowpass(biquad_freq, q, sample_rate):
    """Design a second-order lowpass biquad by the bilinear transform.

    Bilinear-transform (RBJ cookbook) lowpass with the cutoff prewarped
    through :func:`prewarp`, so the analogue corner lands on ``biquad_freq``
    in the digital domain.

    Parameters
    ----------
    biquad_freq : float
        Cutoff frequency in Hz, ``0 < biquad_freq < sample_rate / 2``.
    q : float
        Quality factor, ``q > 0``.  ``q`` sets the resonance height; the
        DC gain is 1 regardless of ``q``.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    dict
        Biquad dict as documented in the module docstring (keys ``"b0"``,
        ``"b1"``, ``"b2"``, ``"a1"``, ``"a2"``, plus ``"q"`` and ``"freq"``
        echoing the arguments).  The DC gain is exactly 1 (``0 dB``) and
        both poles lie strictly inside the unit circle.
    """
    k = prewarp(biquad_freq, sample_rate)
    k2 = k * k
    norm = 1.0 + k / q + k2  # a0 of the unnormalised denominator

    # RBJ lowpass, written in prewarp form.  The numerator (K**2, 2*K**2,
    # K**2) equals the denominator evaluated at z = 1, so normalising by
    # ``norm`` pins the DC gain to 1 for every Q.  a1 carries the sign of
    # 2*(K**2 - 1) (negative below Nyquist/2), keeping both poles inside
    # the unit circle.
    return {
        "b0": k2 / norm,
        "b1": 2.0 * k2 / norm,
        "b2": k2 / norm,
        "a1": 2.0 * (k2 - 1.0) / norm,
        "a2": (1.0 - k / q + k2) / norm,
        "q": q,
        "freq": biquad_freq,
    }


def highpass(freq, q, sample_rate):
    """Design a second-order highpass biquad by the bilinear transform.

    Same topology and normalisation as :func:`lowpass`, with the response
    mirrored about ``freq``: the passband is above the corner and the DC
    gain is exactly 0.

    Parameters
    ----------
    freq : float
        Cutoff frequency in Hz, ``0 < freq < sample_rate / 2``.
    q : float
        Quality factor, ``q > 0``.  For ``q = 1 / sqrt(2)`` the magnitude
        at ``freq`` is ``-3.0103 dB``.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    dict
        Biquad dict as documented in the module docstring (keys ``"b0"``,
        ``"b1"``, ``"b2"``, ``"a1"``, ``"a2"``, plus ``"q"`` and ``"freq"``
        echoing the arguments).  Both poles lie strictly inside the unit
        circle.
    """
    k = prewarp(freq, sample_rate)
    k2 = k * k
    norm = 1.0 + k / q + k2

    # The highpass shares the lowpass denominator (same pole locations);
    # the numerator (1, -2, 1) is the lowpass numerator with K**2 swapped
    # for 1 and the middle sign flipped, which swaps DC and Nyquist
    # behaviour: gain 0 at z = 1, unity gain at z = -1.
    return {
        "b0": 1.0 / norm,
        "b1": -2.0 / norm,
        "b2": 1.0 / norm,
        "a1": 2.0 * (k2 - 1.0) / norm,
        "a2": (1.0 - k / q + k2) / norm,
        "q": q,
        "freq": freq,
    }


def _convolve(x, y):
    """Polynomial product of coefficient lists in ascending ``z**-1`` order."""
    out = [0.0] * (len(x) + len(y) - 1)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            out[i + j] += xi * yj
    return out


def cascade(sections):
    """Combine biquad sections into one polynomial (direct-form) filter.

    The overall transfer function is the product of the section transfer
    functions, so numerator and denominator polynomials are obtained by
    convolving the individual ``(b0, b1, b2)`` and ``(1, a1, a2)``
    coefficient sequences.  Sections are combined in the order given;
    use :func:`order_topology` first when a particular ordering is wanted.

    Parameters
    ----------
    sections : sequence of dict
        Biquad dicts as returned by :func:`lowpass` / :func:`highpass`.

    Returns
    -------
    dict
        ``{"b": [...], "a": [...], "sections": [...]}`` where ``b`` has
        length ``2 * len(sections) + 1`` and ``a`` likewise, both in
        ascending powers of ``z**-1`` with ``a[0] == 1.0``.  ``sections``
        echoes the input sections in the same order.
    """
    b = [1.0]
    a = [1.0]
    for section in sections:
        section_b = [section["b0"], section["b1"], section["b2"]]
        section_a = [1.0, section["a1"], section["a2"]]
        b = _convolve(b, section_b)
        a = _convolve(a, section_a)
    return {"b": b, "a": a, "sections": list(sections)}


def _coeff_lists(coeffs):
    """Return ``(b, a)`` coefficient lists for a biquad or cascade dict."""
    if "b" in coeffs and "a" in coeffs:
        return list(coeffs["b"]), list(coeffs["a"])
    return (
        [coeffs["b0"], coeffs["b1"], coeffs["b2"]],
        [1.0, coeffs["a1"], coeffs["a2"]],
    )


def freq_response(coeffs, freqs, sample_rate):
    """Evaluate the magnitude response in dB on the unit circle.

    For each frequency the transfer function is evaluated at
    ``z = exp(1j * 2 * pi * f / sample_rate)`` and converted with
    ``20 * log10(abs(H))``.

    Parameters
    ----------
    coeffs : dict
        Either a biquad dict (``"b0"`` ... ``"a2"``) or a cascade dict with
        ``"b"`` / ``"a"`` coefficient lists.
    freqs : sequence of float
        Frequencies in Hz at which to evaluate the response.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    list of float
        One magnitude in dB per requested frequency, in the same order.
        A zero magnitude is reported as ``float("-inf")``.
    """
    b, a = _coeff_lists(coeffs)
    result = []
    for freq in freqs:
        omega = 2.0 * math.pi * freq / sample_rate
        z = cmath.exp(1j * omega)
        # H(z) = sum(b_k z**-k) / sum(a_k z**-k), evaluated on z = e**jw.
        numerator = 0.0 + 0.0j
        denominator = 0.0 + 0.0j
        z_inv_power = 1.0 + 0.0j
        for bk in b:
            numerator += bk * z_inv_power
            z_inv_power /= z
        z_inv_power = 1.0 + 0.0j
        for ak in a:
            denominator += ak * z_inv_power
            z_inv_power /= z
        magnitude = abs(numerator / denominator)
        if magnitude == 0.0:
            result.append(float("-inf"))
        else:
            result.append(20.0 * math.log10(magnitude))
    return result


def _quadratic_pole_radius(a1, a2):
    """Largest root magnitude of ``z**2 + a1*z + a2``."""
    discriminant = a1 * a1 - 4.0 * a2
    root_1 = (-a1 + cmath.sqrt(discriminant)) / 2.0
    root_2 = (-a1 - cmath.sqrt(discriminant)) / 2.0
    return max(abs(root_1), abs(root_2))


def pole_radius(coeffs):
    """Return the largest pole magnitude of a filter.

    The filter is stable iff the result is strictly smaller than 1.

    Parameters
    ----------
    coeffs : dict
        Either a biquad dict (``"b0"`` ... ``"a2"``) or a cascade dict; for
        a cascade dict the maximum is taken over the poles of every entry
        of its ``"sections"`` list.

    Returns
    -------
    float
        ``max(abs(z))`` over all poles ``z`` of the denominator polynomial
        ``1 + a1 * z**-1 + a2 * z**-2 (+ ...)``, i.e. the roots of
        ``z**n + a1 * z**(n-1) + ... + an``.
    """
    if "sections" in coeffs:
        return max(
            _quadratic_pole_radius(section["a1"], section["a2"])
            for section in coeffs["sections"]
        )
    return _quadratic_pole_radius(coeffs["a1"], coeffs["a2"])


def order_topology(sections):
    """Order cascade sections for a low-overflow fixed-point topology.

    Sections are sorted by ascending ``q``.  Putting the low-Q sections
    first keeps the running peak gain from jumping at the input of the
    chain, which is what makes high-Q-first cascades overflow internally
    before the later sections can attenuate.

    Parameters
    ----------
    sections : sequence of dict
        Biquad dicts carrying a ``"q"`` entry.

    Returns
    -------
    list of dict
        A new list (the input sequence is not modified) holding the same
        sections sorted by ascending ``q``; the highest-Q section is last.
    """
    return sorted(sections, key=lambda section: section["q"])


def design_batch(specs, config):
    """Design several independent filters in one call.

    Each spec is designed from scratch: no coefficient, buffer or other
    state may be carried over between specs or between calls.  Designing a
    spec alone must therefore reproduce exactly the same result (to
    ``1e-9``) as designing it inside a batch.

    Parameters
    ----------
    specs : sequence of dict
        Each spec has ``"type"`` (``"lowpass"`` or ``"highpass"``),
        ``"freq"`` (Hz) and ``"q"``.
    config : dict
        ``{"sample_rate": float}`` with the sampling rate in Hz (defaults
        to ``48000.0`` when the key is absent).

    Returns
    -------
    dict
        ``{"sample_rate": float, "designs": [design, ...]}`` with one
        design per spec in the input order.  A design is::

            {"type": str, "freq": float, "q": float,
             "sample_rate": float,
             "sections": [biquad, ...],       # in order_topology order
             "coefficients": {"b": [...], "a": [...],
                              "sections": [...]}}
    """
    sample_rate = config.get("sample_rate", 48000.0)
    designs = []
    for spec in specs:
        if spec["type"] == "lowpass":
            section = lowpass(spec["freq"], spec["q"], sample_rate)
        else:
            section = highpass(spec["freq"], spec["q"], sample_rate)
        # A fresh section per spec keeps every design independent; ordering
        # a single-section list is a no-op but keeps the contract explicit.
        sections = order_topology([section])
        coefficients = cascade(sections)
        designs.append(
            {
                "type": spec["type"],
                "freq": spec["freq"],
                "q": spec["q"],
                "sample_rate": sample_rate,
                "sections": sections,
                "coefficients": coefficients,
            }
        )
    return {"sample_rate": sample_rate, "designs": designs}
