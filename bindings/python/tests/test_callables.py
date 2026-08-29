"""Passing a Python function where the library wants `&dyn Fn`.

About two hundred routines take a function: an integrand, a residual, the
right-hand side of an ODE. Rust's signature has no room for failure --
`&dyn Fn(f64) -> f64` returns an `f64` and nothing else -- so the adapter
has to hold any exception and re-raise it after the routine returns. These
tests check both halves: that the results are right when nothing goes
wrong, and that the caller gets their own exception when something does.
"""

import math

import pytest

import numeria as nm


def test_integrating_a_python_function():
    got = nm.numerical.integrate.simpson(math.sin, 0.0, math.pi, 1000)
    assert got == pytest.approx(2.0, abs=1e-9)
    assert nm.numerical.integrate.trapezoid(lambda x: x * x, 0.0, 3.0, 10_000) == pytest.approx(
        9.0, rel=1e-6
    )


def test_a_closure_over_python_state_works():
    power = 3

    def f(x):
        return x**power

    # ∫₀¹ x³ dx = 1/4
    assert nm.numerical.integrate.simpson(f, 0.0, 1.0, 1000) == pytest.approx(0.25, abs=1e-9)


def test_root_finding_with_two_callbacks():
    root = nm.numerical.roots.newton_raphson(
        lambda x: x * x - 2.0, lambda x: 2.0 * x, 1.0, 1e-14, 100
    )
    assert root == pytest.approx(math.sqrt(2.0))


def test_a_method_that_cannot_converge_returns_None_not_a_wrong_answer():
    # No sign change on [2, 3] for x² − 2, so bisection has nothing to do.
    assert nm.numerical.roots.bisection(lambda x: x * x - 2.0, 2.0, 3.0, 1e-12, 100) is None


def test_a_two_argument_callback_integrates_an_ode():
    # y' = -y, y(0) = 1  ->  y(1) = 1/e. The result is a list of (t, y).
    trace = nm.numerical.ode.explicit.rk4_solve(lambda t, y: -y, 0.0, 1.0, 1.0, 1e-3)
    assert trace[0] == (0.0, 1.0)
    t_end, y_end = trace[-1]
    assert t_end == pytest.approx(1.0)
    assert y_end == pytest.approx(math.exp(-1.0), rel=1e-9)


def test_a_callback_taking_and_returning_a_sequence():
    """`&dyn Fn(f64, &[f64]) -> Vec<f64>` -- a vector-valued right-hand side."""
    # The harmonic oscillator: y'' = -y, as y' = v, v' = -y.
    step = nm.numerical.ode.explicit.rk4_step_vec(
        lambda t, y: [y[1], -y[0]], 0.0, [1.0, 0.0], 0.01
    )
    assert len(step) == 2
    assert step[0] == pytest.approx(math.cos(0.01), abs=1e-9)
    assert step[1] == pytest.approx(-math.sin(0.01), abs=1e-9)


def test_the_callback_sees_the_arguments_in_the_right_order():
    seen = []

    def rhs(t, y):
        seen.append((t, y))
        return 1.0

    nm.numerical.ode.explicit.rk4_solve(rhs, 0.0, 5.0, 1.0, 0.25)
    # First call is at the initial condition: t = 0, y = 5.
    assert seen[0] == (0.0, 5.0)


def test_many_callback_invocations_do_not_leak_or_slow_to_a_halt():
    calls = 0

    def f(x):
        nonlocal calls
        calls += 1
        return math.exp(-x * x)

    got = nm.numerical.integrate.simpson(f, -5.0, 5.0, 20_000)
    assert got == pytest.approx(math.sqrt(math.pi), abs=1e-6)
    assert calls > 20_000
