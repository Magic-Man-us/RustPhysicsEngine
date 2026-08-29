"""Failures arrive as exceptions, and carry what the caller needs.

Three things have to hold. A `Result::Err` becomes the matching exception
rather than a sentinel value. An `assert!` inside the library -- which is
how it validates arguments -- becomes an exception rather than an aborted
interpreter. And an exception raised inside a Python callable passed into
the library comes back out with its own traceback rather than turning into
a NaN.
"""

import math

import pytest

import numeria as nm


def test_the_hierarchy_is_what_the_readme_says():
    E = nm
    assert issubclass(E.InvalidArgumentError, E.PhysicsError)
    assert issubclass(E.SolverError, E.PhysicsError)
    assert issubclass(E.SingularMatrixError, E.SolverError)
    assert issubclass(E.NotPositiveDefiniteError, E.SolverError)
    assert issubclass(E.ConvergenceError, E.SolverError)
    assert issubclass(E.DimensionMismatchError, E.SolverError)
    assert issubclass(E.GeometryError, E.PhysicsError)
    assert issubclass(E.DegenerateGeometryError, E.GeometryError)
    assert issubclass(E.NotManifoldError, E.GeometryError)
    assert issubclass(E.EmptyInputError, E.GeometryError)
    assert issubclass(E.UnitsError, E.PhysicsError)
    assert issubclass(E.PhysicsError, Exception)


def test_a_singular_matrix_raises_rather_than_returning_nonsense():
    singular = [[1.0, 2.0], [2.0, 4.0]]
    with pytest.raises(nm.SingularMatrixError) as excinfo:
        nm.linalg.lu.solve(singular, [1.0, 2.0])
    assert "singular" in str(excinfo.value)
    # And it is catchable at every level above it.
    with pytest.raises(nm.SolverError):
        nm.linalg.lu.solve(singular, [1.0, 2.0])
    with pytest.raises(nm.PhysicsError):
        nm.linalg.lu.solve(singular, [1.0, 2.0])


def test_a_shape_mismatch_carries_the_two_shapes():
    with pytest.raises(nm.DimensionMismatchError) as excinfo:
        nm.linalg.lu.solve([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0])
    err = excinfo.value
    assert err.expected == 2
    assert err.got == 3


def test_an_assertion_in_the_library_becomes_an_exception_with_its_message():
    """`assert!(mass > 0.0, "mass must be positive")` is an argument check.

    In Rust it is the right one: a negative mass is a programming error.
    Crossing into Python it must not abort the interpreter, and the
    message the library author wrote is exactly the message a caller
    wants.
    """
    with pytest.raises(nm.InvalidArgumentError) as excinfo:
        nm.classical.acceleration(force=10.0, mass=-1.0)
    assert "mass must be positive" in str(excinfo.value)


def test_a_panic_does_not_leave_the_next_call_broken():
    with pytest.raises(nm.InvalidArgumentError):
        nm.classical.acceleration(force=10.0, mass=0.0)
    # The guard has to reset its state, or every later call inherits it.
    assert nm.classical.acceleration(force=10.0, mass=2.0) == 5.0


def test_an_exception_inside_a_callback_comes_back_out():
    def bad(x):
        raise ZeroDivisionError("from the integrand")

    with pytest.raises(ZeroDivisionError) as excinfo:
        nm.numerical.integrate.simpson(bad, 0.0, 1.0, 100)
    assert "from the integrand" in str(excinfo.value)


def test_a_callback_returning_the_wrong_type_is_a_TypeError():
    with pytest.raises(TypeError):
        nm.numerical.integrate.simpson(lambda x: "not a number", 0.0, 1.0, 10)


def test_a_callback_that_is_not_callable_is_a_TypeError():
    with pytest.raises(TypeError):
        nm.numerical.integrate.simpson(42, 0.0, 1.0, 10)


def test_the_integrand_error_wins_over_whatever_it_caused():
    """A failed callback returns NaN so the Rust routine can finish.

    That NaN may then trip an assertion further in. The caller needs the
    original exception, not the consequence, so the callback's error is
    checked first.
    """

    calls = []

    def bad(x):
        calls.append(x)
        raise ValueError("mine")

    with pytest.raises(ValueError, match="mine"):
        nm.numerical.integrate.simpson(bad, 0.0, 1.0, 1000)
    # And it short-circuits rather than calling a thousand times after the
    # first failure.
    assert len(calls) < 10


def test_a_geometry_failure_maps_to_the_geometry_branch():
    with pytest.raises(nm.EmptyInputError):
        # A triangulation needs at least three points.
        nm.fem.fem2d.FemMesh2.from_delaunay([(0.0, 0.0), (1.0, 0.0)])
    with pytest.raises(nm.DegenerateGeometryError) as excinfo:
        # Three collinear points enclose no area.
        nm.fem.fem2d.FemMesh2.from_delaunay([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    assert "collinear" in str(excinfo.value) or "degenerate" in str(excinfo.value)


def test_wrong_python_types_are_TypeError_not_a_crash():
    with pytest.raises(TypeError):
        nm.classical.displacement("fast", 1.0, 1.0)
    with pytest.raises(TypeError):
        nm.statistics.descriptive.mean("abc")
