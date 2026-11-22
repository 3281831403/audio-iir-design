"""Behavioural tests for the audio IIR design toolkit.

The tests only use the public API.  Expected values are derived from the
bilinear transform, the RBJ biquad topology and the definition of the
frequency response, so they are independent of the implementation under
test.  Unless a tighter bound is stated, numeric agreement is expected to
within 1e-9 (absolute for O(1) quantities, with a relative cushion).
"""

import math

from iir.design import (
    cascade,
    design_batch,
    freq_response,
    highpass,
    lowpass,
    order_topology,
    pole_radius,
    prewarp,
)

FS = 48000.0
TOL = 1e-9
MINUS_3DB = 20.0 * math.log10(1.0 / math.sqrt(2.0))  # -3.010299956639812


def close(got, expected, tol=TOL):
    """Absolute comparison with a relative cushion for large magnitudes."""
    return abs(got - expected) <= tol * max(1.0, abs(expected))


def conv(x, y):
    """Polynomial product for coefficient lists in ascending z**-1 order."""
    out = [0.0] * (len(x) + len(y) - 1)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            out[i + j] += xi * yj
    return out


def coeff_lists(section):
    """Return (b, a) ascending coefficient lists of one biquad."""
    return (
        [float(section["b0"]), float(section["b1"]), float(section["b2"])],
        [1.0, float(section["a1"]), float(section["a2"])],
    )


def freq_grid(step=50.0, stop=23950.0):
    out = []
    f = step
    while f <= stop:
        out.append(f)
        f += step
    return out


def assert_same_coefficients(got, expected):
    for key in ("b", "a"):
        assert len(got[key]) == len(expected[key]), (key, len(got[key]), len(expected[key]))
        for g, e in zip(got[key], expected[key]):
            assert close(g, e), (key, g, e)


def assert_same_design(got, expected):
    assert got["type"] == expected["type"]
    assert close(got["freq"], expected["freq"])
    assert close(got["q"], expected["q"])
    assert close(got["sample_rate"], expected["sample_rate"])
    assert_same_coefficients(got["coefficients"], expected["coefficients"])
    assert len(got["sections"]) == len(expected["sections"])
    for got_sec, exp_sec in zip(got["sections"], expected["sections"]):
        assert close(got_sec["q"], exp_sec["q"])
        for key in ("b0", "b1", "b2", "a1", "a2"):
            assert close(got_sec[key], exp_sec[key]), (key, got_sec[key], exp_sec[key])


def test_prewarp_uses_tan_of_half_angle():
    # The bilinear transform prewarps with tan(pi * f / fs); tan(2 * pi * f / fs)
    # would move every corner and is off by a large factor here.
    assert close(prewarp(1000.0, FS), math.tan(math.pi * 1000.0 / FS), 1e-12)
    assert close(prewarp(6000.0, FS), math.tan(math.pi * 6000.0 / FS), 1e-12)
    assert close(prewarp(12000.0, FS), math.tan(math.pi * 12000.0 / FS), 1e-12)
    assert prewarp(1000.0, FS) > 0.0
    assert prewarp(6000.0, FS) > prewarp(1000.0, FS)
    wrong_scale = math.tan(2.0 * math.pi * 6000.0 / FS)
    assert abs(prewarp(6000.0, FS) - wrong_scale) > 0.1


def test_lowpass_a1_sign_and_pole_mapping():
    # Bilinear image of the analogue prototype poles:
    #   s = K * (-sigma +/- j * mu), K = tan(pi f / fs),
    #   sigma = 1 / (2Q), mu = sqrt(1 - sigma**2),
    #   z = (1 + s) / (1 - s).
    # The denominator must be z**2 + a1 * z + a2, so sum(poles) == -a1 and
    # product(poles) == a2.  A flipped a1 sign breaks the sum and ruins the
    # pole mapping.
    for q in (0.7071067811865476, 5.0, 20.0):
        section = lowpass(1000.0, q, FS)
        a1 = float(section["a1"])
        a2 = float(section["a2"])
        k = math.tan(math.pi * 1000.0 / FS)
        sigma = 1.0 / (2.0 * q)
        mu = math.sqrt(1.0 - sigma * sigma)
        poles = []
        for sign in (1.0, -1.0):
            s = k * complex(-sigma, sign * mu)
            poles.append((1.0 + s) / (1.0 - s))
        assert abs(poles[0] * poles[1] - a2) <= 1e-9
        assert abs(-(poles[0] + poles[1]) - a1) <= 1e-9
        for pole in poles:
            assert abs(pole) < 1.0 - 1e-9
        assert close(pole_radius(section), max(abs(p) for p in poles))
        assert a1 < 0.0


def test_lowpass_dc_gain_normalized_to_one():
    for q in (0.5, 5.0, 20.0):
        section = lowpass(1000.0, q, FS)
        b_sum = float(section["b0"]) + float(section["b1"]) + float(section["b2"])
        a_sum = 1.0 + float(section["a1"]) + float(section["a2"])
        assert close(b_sum / a_sum, 1.0)
        assert abs(freq_response(section, [0.0], FS)[0]) <= 1e-9


def test_highpass_flips_the_lowpass_response():
    hp = highpass(1000.0, 0.7071067811865476, FS)
    lp = lowpass(1000.0, 0.7071067811865476, FS)
    # highpass shares the lowpass denominator but not its numerator
    assert close(float(hp["a1"]), float(lp["a1"]))
    assert close(float(hp["a2"]), float(lp["a2"]))
    at_corner = freq_response(hp, [1000.0], FS)[0]
    at_50hz = freq_response(hp, [50.0], FS)[0]
    at_12khz = freq_response(hp, [12000.0], FS)[0]
    assert abs(at_corner - MINUS_3DB) <= 0.1
    assert at_50hz < -40.0
    assert at_12khz > -0.5
    # a lowpass formula reused here would read ~0 dB at 50 Hz and ~-40 dB
    # at 12 kHz
    assert freq_response(hp, [0.0], FS)[0] < -60.0


def test_cascade_convolves_section_polynomials():
    first = lowpass(800.0, 0.6, FS)
    second = highpass(3000.0, 4.0, FS)
    combined = cascade([first, second])
    b0, a0 = coeff_lists(first)
    b1, a1 = coeff_lists(second)
    expected_b = conv(b0, b1)
    expected_a = conv(a0, a1)
    assert len(combined["b"]) == 5
    assert len(combined["a"]) == 5
    assert close(combined["a"][0], 1.0)
    for got, exp in zip(combined["b"], expected_b):
        assert close(got, exp), (got, exp)
    for got, exp in zip(combined["a"], expected_a):
        assert close(got, exp), (got, exp)
    # element-wise multiplication of the sections would keep length 3
    assert len(combined["b"]) != 3


def test_freq_response_evaluates_the_unit_circle():
    section = lowpass(1000.0, 0.7071067811865476, FS)
    grid = [0.0, 100.0, 1000.0, 4000.0, 12000.0, 20000.0]
    magnitudes = freq_response(section, grid, FS)
    assert len(magnitudes) == len(grid)
    assert abs(magnitudes[0]) <= 1e-9
    assert abs(magnitudes[2] - MINUS_3DB) <= 0.1
    assert magnitudes[3] < -10.0
    assert magnitudes[5] < -40.0
    # evaluating at z = 1 instead of z = exp(jw) yields a flat curve
    assert max(magnitudes) - min(magnitudes) > 40.0


def test_freq_response_uses_20_log10():
    # H(z) = 0.25 + 0.5 z**-1 + 0.25 z**-2 at w = pi/2 is -0.5j, |H| = 0.5.
    # 20*log10(0.5) = -6.020599913279624; 10*log10(0.5) = -3.0102999566...
    coefficients = {"b0": 0.25, "b1": 0.5, "b2": 0.25, "a1": 0.0, "a2": 0.0}
    value = freq_response(coefficients, [FS / 4.0], FS)[0]
    expected = 20.0 * math.log10(0.5)
    assert close(value, expected)
    assert abs(value - 10.0 * math.log10(0.5)) > 1.0
    # a second anchor away from the axis: DC gain of this section is 1
    assert abs(freq_response(coefficients, [0.0], FS)[0]) <= 1e-9


def test_pole_radius_uses_a2():
    # z**2 + 0.81 has poles at +/- 0.9j: radius 0.9, while a1 alone is 0.
    imaginary = {"b0": 1.0, "b1": 0.0, "b2": 0.0, "a1": 0.0, "a2": 0.81}
    assert close(pole_radius(imaginary), 0.9)
    assert pole_radius(imaginary) > 0.5
    # z**2 - 1.6 z + 0.81 has a complex pair of radius 0.9 as well
    complex_pair = {"b0": 1.0, "b1": 0.0, "b2": 0.0, "a1": -1.6, "a2": 0.81}
    assert close(pole_radius(complex_pair), 0.9)
    # a second-order lowpass at 1 kHz / Q=20 is only just inside the circle
    section = lowpass(1000.0, 20.0, FS)
    assert pole_radius(section) > 0.99


def test_order_topology_sorts_q_ascending():
    sections = [
        lowpass(2000.0, 8.0, FS),
        lowpass(500.0, 0.6, FS),
        lowpass(1200.0, 3.0, FS),
    ]
    ordered = order_topology(sections)
    assert len(ordered) == 3
    qs = [float(section["q"]) for section in ordered]
    assert qs == sorted(qs)
    assert close(qs[0], 0.6, 1e-12)
    assert close(qs[-1], 8.0, 1e-12)
    # the input order must be left alone
    assert [float(section["q"]) for section in sections] == [8.0, 0.6, 3.0]
    # peak gain ramps up: low-Q sections first, the high-Q resonance last
    grid = freq_grid()
    peaks = [max(freq_response(section, grid, FS)) for section in ordered]
    for low, high in zip(peaks, peaks[1:]):
        assert low <= high + 1e-9
    assert peaks[0] < 3.0


def test_design_batch_has_no_reused_state():
    config = {"sample_rate": FS}
    spec_a = {"type": "lowpass", "freq": 700.0, "q": 0.7071067811865476}
    spec_b = {"type": "highpass", "freq": 5000.0, "q": 6.0}
    batch = design_batch([spec_a, spec_b], config)
    solo_a = design_batch([spec_a], config)
    solo_b = design_batch([spec_b], config)
    assert len(batch["designs"]) == 2
    assert_same_design(batch["designs"][0], solo_a["designs"][0])
    assert_same_design(batch["designs"][1], solo_b["designs"][0])
    # the two specs cannot share coefficients
    assert batch["designs"][0]["coefficients"]["b"] != batch["designs"][1]["coefficients"]["b"]
    # a second call after other designs must still start from scratch
    again = design_batch([spec_a], config)
    assert_same_design(again["designs"][0], solo_a["designs"][0])
    repeat = design_batch([spec_a, spec_b], config)
    assert_same_coefficients(repeat["designs"][0]["coefficients"],
                             batch["designs"][0]["coefficients"])
    assert_same_coefficients(repeat["designs"][1]["coefficients"],
                             batch["designs"][1]["coefficients"])


def test_lowpass_minus_3db_at_cutoff():
    section = lowpass(1000.0, 0.7071067811865476, FS)
    value = freq_response(section, [1000.0], FS)[0]
    assert abs(value - MINUS_3DB) <= 0.1


def test_all_poles_strictly_inside_unit_circle():
    for freq in (80.0, 1000.0, 6000.0, 15000.0):
        for q in (0.5, 0.7071067811865476, 4.0, 20.0):
            for section in (lowpass(freq, q, FS), highpass(freq, q, FS)):
                radius = pole_radius(section)
                assert radius < 1.0 - 1e-9, (freq, q, radius)
                assert radius > 0.0
    combined = cascade([lowpass(300.0, 12.0, FS), highpass(9000.0, 0.5, FS)])
    assert pole_radius(combined) < 1.0 - 1e-9


def test_cascade_response_equals_product_of_sections():
    sections = [
        lowpass(600.0, 0.7071067811865476, FS),
        highpass(2000.0, 2.0, FS),
        lowpass(8000.0, 5.0, FS),
    ]
    combined = cascade(sections)
    freqs = [10.0, 200.0, 600.0, 1500.0, 4000.0, 9000.0, 18000.0]
    total = freq_response(combined, freqs, FS)
    for index, freq in enumerate(freqs):
        expected = sum(freq_response(section, [freq], FS)[0] for section in sections)
        assert abs(total[index] - expected) <= 1e-9, (freq, total[index], expected)


def test_dc_gain_unity_for_lowpass_cascade():
    first = lowpass(400.0, 0.5411961001461971, FS)
    second = lowpass(2500.0, 1.3065629648763766, FS)
    combined = cascade([first, second])
    assert abs(freq_response(combined, [0.0], FS)[0]) <= 1e-9
    assert close(sum(combined["b"]) / sum(combined["a"]), 1.0)
    assert abs(freq_response(combined, [0.0], FS)[0]) <= 1e-9


def test_batch_matches_individual_designs():
    config = {"sample_rate": FS}
    spec_low = {"type": "lowpass", "freq": 1200.0, "q": 0.9}
    spec_high = {"type": "highpass", "freq": 3500.0, "q": 2.5}
    batch = design_batch([spec_low, spec_high], config)
    solo_low = design_batch([spec_low], config)
    solo_high = design_batch([spec_high], config)
    assert_same_design(batch["designs"][0], solo_low["designs"][0])
    assert_same_design(batch["designs"][1], solo_high["designs"][0])
    assert close(batch["sample_rate"], FS)
    for design, spec in zip(batch["designs"], (spec_low, spec_high)):
        assert design["type"] == spec["type"]
        # one spec designs a single section; low-Q first ordering is trivial
        assert len(design["sections"]) == 1
        assert len(design["coefficients"]["b"]) == 3
        assert len(design["coefficients"]["a"]) == 3
        # response sanity: lowpass is flat at DC, highpass is not
        dc = freq_response(design["coefficients"], [0.0], FS)[0]
        if spec["type"] == "lowpass":
            assert abs(dc) <= 1e-9
