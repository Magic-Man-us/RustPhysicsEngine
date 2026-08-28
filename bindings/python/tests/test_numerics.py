"""The numbers that come back are the right numbers.

The library tests its own mathematics thoroughly; these tests are not a
second opinion on that. They check the crossing: that arguments arrive in
the order and the units the Rust function expects, that a `Vec<f64>` comes
back as a list of the same length in the same order, and that nothing is
transposed, truncated or scaled on the way through. Each expected value is
a closed form or an exact identity, so a wrong answer is a wrong binding
rather than a stale golden file.
"""

import math

import pytest

import rust_physics_engine as rpe


def test_projectile_range_matches_the_closed_form():
    speed, angle, g = 20.0, math.radians(45.0), 9.80665
    got = rpe.classical.projectile_range(speed, angle, g)
    assert got == pytest.approx(speed**2 * math.sin(2 * angle) / g)
    # 45 degrees maximises the range, which is a fact about the physics
    # and so a check that the angle is in radians and in the right place.
    for other in (30.0, 60.0, 44.0, 46.0):
        assert rpe.classical.projectile_range(speed, math.radians(other), g) < got


def test_arguments_are_not_silently_reordered():
    # displacement(v0, a, t) = v0*t + a*t²/2. Asymmetric in its arguments,
    # so any permutation gives a different number.
    assert rpe.classical.displacement(3.0, 2.0, 4.0) == pytest.approx(3 * 4 + 0.5 * 2 * 16)


def test_orbital_and_escape_velocities_are_related_by_root_two():
    m, r = 5.972e24, 6.371e6
    v_orbit = rpe.gravitation.orbital_velocity(m, r)
    v_escape = rpe.gravitation.escape_velocity(m, r)
    assert v_escape == pytest.approx(v_orbit * math.sqrt(2.0))
    # And low Earth orbit really is about 7.9 km/s.
    assert 7_800 < v_orbit < 8_000


def test_a_linear_system_solves_to_the_known_answer():
    # [[2,1],[1,3]] x = [5,10]  ->  x = (1, 3)
    x = rpe.linalg.lu.solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert x == pytest.approx([1.0, 3.0])


def test_a_list_of_rows_and_a_Matrix_are_interchangeable():
    rows = [[4.0, 1.0], [1.0, 3.0]]
    m = rpe.linalg.Matrix.from_rows(rows)
    assert rpe.linalg.lu.solve(rows, [1.0, 2.0]) == pytest.approx(
        rpe.linalg.lu.solve(m, [1.0, 2.0])
    )


def test_the_fft_of_a_delta_is_flat_and_inverts():
    x = [1.0, 0.0, 0.0, 0.0]
    spectrum = rpe.transforms.fft.fft(x)
    assert len(spectrum) == 4
    assert all(isinstance(z, complex) for z in spectrum)
    assert all(z == pytest.approx(1 + 0j) for z in spectrum)
    back = rpe.transforms.fft.ifft(spectrum)
    assert [z.real for z in back] == pytest.approx(x, abs=1e-12)


def test_a_pure_tone_lands_in_one_bin():
    n = 64
    k = 5
    signal = [complex(math.cos(2 * math.pi * k * i / n), 0.0) for i in range(n)]
    mags = [abs(z) for z in rpe.transforms.fft.fft(signal)]
    peak = max(range(n), key=lambda i: mags[i])
    assert peak in (k, n - k)


def test_descriptive_statistics_on_a_known_sample():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert rpe.statistics.descriptive.mean(data) == pytest.approx(5.0)
    # Population variance of this textbook sample is exactly 4.
    assert rpe.statistics.descriptive.variance(data) == pytest.approx(4.0)
    assert rpe.statistics.descriptive.std_deviation(data) == pytest.approx(2.0)


def test_gamma_of_a_half_integer_is_root_pi():
    assert rpe.special.gamma.gamma(0.5) == pytest.approx(math.sqrt(math.pi))
    assert rpe.special.gamma.gamma(5.0) == pytest.approx(24.0)


def test_a_long_vector_survives_the_crossing_intact():
    """The GIL is released for array calls; the data must still be right."""
    n = 4096
    data = [math.sin(i / 10.0) for i in range(n)]
    out = rpe.statistics.descriptive.mean(data)
    assert out == pytest.approx(sum(data) / n)


def test_geodesy_reaches_an_associated_constant_and_a_tuple_return():
    """`Ellipsoid::WGS84` is a Rust associated constant, and the return is
    a three-tuple. Both have to survive."""
    wgs84 = rpe.geometry.geodesy.Ellipsoid.WGS84
    assert wgs84.a == pytest.approx(6_378_137.0)
    assert wgs84.f == pytest.approx(1.0 / 298.257223563)

    # London to New York is about 5,570 km along the geodesic.
    dist, az_fwd, az_rev = rpe.geometry.geodesy.vincenty_inverse(
        math.radians(51.5074), math.radians(-0.1278),
        math.radians(40.7128), math.radians(-74.0060),
        wgs84,
    )
    assert 5_500_000 < dist < 5_600_000
    assert -math.pi <= az_fwd <= math.pi or 0 <= az_fwd <= 2 * math.pi

    # A bare `(a, f)` pair stands in for the Ellipsoid.
    same, _, _ = rpe.geometry.geodesy.vincenty_inverse(
        math.radians(51.5074), math.radians(-0.1278),
        math.radians(40.7128), math.radians(-74.0060),
        (6_378_137.0, 1.0 / 298.257223563),
    )
    assert same == pytest.approx(dist)
