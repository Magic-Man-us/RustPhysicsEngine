"""Signatures that need more than a value copied across.

Four shapes in the Rust API do not map to a Python argument by themselves,
and each gets its own treatment in the generator. If any of them regresses
the failure is quiet -- the call still runs, it just does nothing, or does
it to a copy -- so each has a test that would notice.
"""

import math

import pytest

import numeria as nm


def test_a_builder_returns_the_same_object_so_calls_chain():
    """`&mut self -> &mut Self` is Rust's chaining idiom.

    Handing back a copy would compile and read the same, and would throw
    away every gate after the first.
    """
    c = nm.quantum.circuit.Circuit(2)
    returned = c.h(0)
    assert returned is c

    c.cx(0, 1)
    state = c.run(nm.quantum.circuit.QState.zero(2))
    probabilities = [round(abs(z) ** 2, 6) for z in state.amps]
    # A Bell pair: |00> and |11> at one half each, nothing in between.
    assert probabilities == [0.5, 0.0, 0.0, 0.5]


def test_chaining_in_one_expression_keeps_every_gate():
    c = nm.quantum.circuit.Circuit(3)
    c.h(0).cx(0, 1).cx(1, 2)
    state = c.run(nm.quantum.circuit.QState.zero(3))
    probabilities = [round(abs(z) ** 2, 6) for z in state.amps]
    assert probabilities[0] == 0.5
    assert probabilities[7] == 0.5
    assert sum(probabilities[1:7]) == 0.0


def test_a_mutable_slice_of_objects_is_written_back():
    """`&mut [Vec2]` -- the list is the output, and the elements are
    wrapper objects rather than numbers."""
    points = [
        nm.math.Vec2(3.0, 3.0),
        nm.math.Vec2(0.0, 0.0),
        nm.math.Vec2(1.0, 1.0),
    ]
    nm.patterns.space_filling.sort_by_hilbert(points, 4)
    ordered = [p.tolist() for p in points]
    # Every point is still there, as a Vec2, and the order has changed:
    # the Hilbert index rises along this diagonal.
    assert sorted(ordered) == [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
    assert ordered[0] == [0.0, 0.0]
    assert ordered[-1] == [3.0, 3.0]
    assert all(isinstance(p, nm.math.Vec2) for p in points)


def test_a_borrowed_argument_whose_type_is_not_clone():
    """Some types have no `Clone`, so the wrapper is borrowed rather than
    copied. The call still has to work."""
    grid = nm.cfd.grid.MacGrid2(8, 8, 1.0)
    field = nm.cfd.grid.CellField2(8, 8, 1.0)
    out = nm.cfd.advection.advect_semi_lagrangian_2d(field, grid, 0.1)
    assert out is not None


def test_an_associated_constant_is_a_class_attribute():
    assert nm.math.Vec3.ZERO.tolist() == [0.0, 0.0, 0.0]
    assert nm.units.quantity.Dim.LENGTH.exponents() == [1, 0, 0, 0, 0, 0, 0]


def test_a_reexport_is_reachable_by_its_short_path():
    """`linalg` re-exports `Matrix` and `solve`; the crate's own docs use
    the short names."""
    assert nm.linalg.Matrix is nm.linalg.matrix.Matrix
    assert nm.linalg.solve is nm.linalg.lu.solve


def test_a_submodule_wins_a_name_it_shares_with_a_reexport():
    """`special::gamma` is both a module and a re-exported function.

    Python has one namespace, so the module keeps the name and the
    function stays reachable one level down. COVERAGE.md says so too.
    """
    import types

    assert isinstance(nm.special.gamma, types.ModuleType)
    assert nm.special.gamma.gamma(0.5) == pytest.approx(math.sqrt(math.pi))
    assert isinstance(nm.transforms.fft, types.ModuleType)
    assert nm.transforms.fft.fft([1.0, 0.0])


def test_an_optional_argument_defaults_to_none():
    sig = nm.mesh.analyze.self_intersections.__text_signature__
    assert "bvh=None" in sig
