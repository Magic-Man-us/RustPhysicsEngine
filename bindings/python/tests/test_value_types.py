"""Wrapper classes, and the literals that may stand in for them.

`Vec3(1, 2, 3)` and `(1, 2, 3)` should be interchangeable as arguments,
because requiring the constructor everywhere would make a Rust API read
like a Rust API. The classes still have to behave as Python objects: they
index, iterate, compare, copy and print.
"""

import copy
import math

import pytest

import rust_physics_engine as rpe


def test_a_tuple_stands_in_for_a_vector():
    a = rpe.classical.position_3d((0.0, 0.0, 100.0), (5.0, 0.0, 0.0), (0.0, 0.0, -9.81), 2.0)
    b = rpe.classical.position_3d(
        rpe.math.Vec3(0.0, 0.0, 100.0),
        rpe.math.Vec3(5.0, 0.0, 0.0),
        rpe.math.Vec3(0.0, 0.0, -9.81),
        2.0,
    )
    assert a.tolist() == pytest.approx(b.tolist())
    assert a.tolist() == pytest.approx([10.0, 0.0, 100.0 - 0.5 * 9.81 * 4])


def test_a_list_stands_in_too_and_the_wrong_length_is_rejected():
    assert rpe.math.Vec3(1, 0, 0).dot([0, 1, 0]) == 0.0
    with pytest.raises(TypeError) as excinfo:
        rpe.math.Vec3(1, 0, 0).dot([0, 1])
    assert "Vec3" in str(excinfo.value)


def test_vectors_behave_like_sequences():
    v = rpe.math.Vec3(1.0, 2.0, 2.0)
    assert len(v) == 3
    assert v[0] == 1.0 and v[-1] == 2.0
    assert list(v) == [1.0, 2.0, 2.0]
    assert v.tolist() == [1.0, 2.0, 2.0]
    with pytest.raises(IndexError):
        v[3]


def test_vector_algebra_through_operators():
    a = rpe.math.Vec3(1.0, 2.0, 3.0)
    b = rpe.math.Vec3(4.0, 5.0, 6.0)
    assert (a + b).tolist() == [5.0, 7.0, 9.0]
    assert (b - a).tolist() == [3.0, 3.0, 3.0]
    assert (a * 2.0).tolist() == [2.0, 4.0, 6.0]
    assert (-a).tolist() == [-1.0, -2.0, -3.0]
    assert a.dot(b) == 32.0
    assert a.cross(b).tolist() == [-3.0, 6.0, -3.0]
    assert rpe.math.Vec3(3.0, 4.0, 0.0).magnitude() == 5.0


def test_equality_repr_and_copying():
    v = rpe.math.Vec3(1.0, 2.0, 3.0)
    assert v == rpe.math.Vec3(1.0, 2.0, 3.0)
    assert v != rpe.math.Vec3(1.0, 2.0, 4.0)
    assert repr(v) == "Vec3(x=1.0, y=2.0, z=3.0)"
    assert copy.copy(v) == v
    assert copy.deepcopy(v) == v
    assert type(v).__module__ == "rust_physics_engine.math"


def test_fields_are_readable_and_writable():
    v = rpe.math.Vec3(1.0, 2.0, 3.0)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    v.x = 10.0
    assert v.x == 10.0
    assert v.tolist() == [10.0, 2.0, 3.0]


def test_matrix_indexing_and_shape():
    m = rpe.linalg.Matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert m.shape == (2, 3)
    assert len(m) == 2
    assert m[0, 2] == 3.0
    assert m[1] == [4.0, 5.0, 6.0]
    assert m[-1, -1] == 6.0
    m[0, 0] = 9.0
    assert m[0, 0] == 9.0
    assert m.tolist() == [[9.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(IndexError):
        m[5, 0]


def test_quaternion_rotation_is_a_rotation():
    q = rpe.quaternion.Quaternion.from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    turned = q.rotate_vec((1.0, 0.0, 0.0))
    assert turned.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert q.normalize().tolist() == pytest.approx(q.tolist())


def test_a_stateful_object_keeps_its_state_across_calls():
    """A wrapper holds the Rust value, so `&mut self` methods really mutate."""
    rng = rpe.monte_carlo.Rng(12345)
    first = [rng.next_f64() for _ in range(5)]
    assert all(0.0 <= x < 1.0 for x in first)
    assert len(set(first)) == 5

    # And the same seed replays exactly.
    again = rpe.monte_carlo.Rng(12345)
    assert [again.next_f64() for _ in range(5)] == first


def test_an_rng_passed_into_a_free_function_advances():
    """`&mut Rng` arguments are the wrapper itself, not a copy of it."""
    rng = rpe.monte_carlo.Rng(7)
    walk = rpe.monte_carlo.random_walk_1d(50, 1.0, rng)
    assert len(walk) == 51
    # The walk moved: if the RNG had been copied in and thrown away, every
    # step would repeat the same draw.
    assert len(set(walk)) > 5

    # A second walk from the advanced generator differs from the first.
    assert rpe.monte_carlo.random_walk_1d(50, 1.0, rng) != walk

    # And pi comes out near pi, which needs both the draws and the state.
    fresh = rpe.monte_carlo.Rng(99)
    assert rpe.monte_carlo.mc_estimate_pi(200_000, fresh) == pytest.approx(math.pi, abs=0.02)


def test_a_unit_variant_enum_crosses_as_a_class_with_members():
    """Fieldless Rust enums become Python enums: comparable, printable,
    and accepted wherever the Rust function wants the enum."""
    compounding = rpe.finance.rates.Compounding
    members = [n for n in dir(compounding) if not n.startswith("_")]
    assert members
    first = getattr(compounding, members[0])
    assert first == getattr(compounding, members[0])
    assert repr(first).startswith("Compounding.")
