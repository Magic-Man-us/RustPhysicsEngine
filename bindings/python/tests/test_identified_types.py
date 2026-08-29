"""Three Rust types that are Python types.

`Complex`, `BigInt` and `Rational` have exact Python counterparts, so they
are translated rather than wrapped. The test a translation has to pass is
that the round trip loses nothing -- these check it at sizes and precisions
where a lossy conversion would show.
"""

from fractions import Fraction

import pytest

import numeria as nm


def test_complex_goes_both_ways_as_the_builtin():
    spectrum = nm.transforms.fft.fft([1.0, 2.0, 3.0, 4.0])
    assert all(isinstance(z, complex) for z in spectrum)
    # DC bin is the sum.
    assert spectrum[0] == pytest.approx(10 + 0j)
    # And a Python complex is accepted on the way in.
    back = nm.transforms.fft.ifft([complex(10, 0), -2 + 2j, -2 + 0j, -2 - 2j])
    assert [z.real for z in back] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_a_float_is_accepted_where_a_complex_is_expected():
    assert nm.transforms.fft.fft([1, 0, 0, 0])[0] == pytest.approx(1 + 0j)


def test_bigint_is_a_python_int_of_any_size():
    small = nm.exact.bigint.factorial(10)
    assert isinstance(small, int)
    assert small == 3_628_800

    # 100! has 158 digits: far past anything an f64 or an i64 can hold, so
    # a conversion that went through either would be visibly wrong.
    big = nm.exact.bigint.factorial(100)
    import math

    assert big == math.factorial(100)
    assert len(str(big)) == 158


def test_a_huge_python_int_survives_the_trip_in_and_out():
    n = (1 << 4000) - 1
    # gcd(n, n) == n exercises the conversion in both directions.
    assert nm.exact.bigint.gcd(n, n) == n
    assert nm.exact.bigint.gcd(-n, n) == n


def test_modular_exponentiation_agrees_with_python():
    base, exp, mod = 3, 10**40 + 7, 2**61 - 1
    assert nm.exact.bigint.mod_pow(base, exp, mod) == pow(base, exp, mod)


def test_rational_is_a_fraction_and_stays_exact():
    third = Fraction(1, 3)
    doubled = nm.exact.rational.add(third, third)
    assert isinstance(doubled, Fraction)
    assert doubled == Fraction(2, 3)

    # A tenth is exactly a tenth, which it would not be through a float.
    tenth = nm.exact.rational.add(Fraction(1, 20), Fraction(1, 20))
    assert tenth == Fraction(1, 10)
    assert tenth != 0.1


def test_a_plain_int_is_a_rational():
    assert nm.exact.rational.add(2, Fraction(1, 2)) == Fraction(5, 2)


def test_a_float_is_refused_where_a_rational_is_expected():
    """0.1 is not one tenth, and a module whose point is exactness should
    not pretend otherwise."""
    with pytest.raises(TypeError):
        nm.exact.rational.add(0.1, Fraction(1, 10))


def test_continued_fractions_round_trip():
    cf = nm.exact.rational.to_continued_fraction(Fraction(415, 93))
    assert cf == [4, 2, 6, 7]
    assert all(isinstance(x, int) for x in cf)
