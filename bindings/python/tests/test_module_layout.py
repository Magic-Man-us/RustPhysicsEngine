"""The shape of the package: what is importable, and by what name.

The Rust module tree is the map users navigate by. If `linalg.lu` is a
module in Rust it should be one here, reachable by attribute *and* by
`import`, and holding the same names. These tests fix that correspondence
so a change to the generator cannot quietly rearrange it.
"""

import importlib
import sys

import pytest

import rust_physics_engine as rpe


def test_version_is_reported():
    assert rpe.__version__
    assert rpe.__version__[0].isdigit()


def test_every_top_level_rust_module_is_an_attribute():
    # A representative spread rather than all 71: enough that a wholesale
    # regression shows up, few enough that adding a module to the library
    # does not fail this test.
    for name in (
        "classical", "linalg", "numerical", "statistics", "quantum",
        "transforms", "geometry", "spatial", "exact", "units", "math",
        "optimization", "graph", "audio", "fem", "manifold", "codes",
    ):
        assert hasattr(rpe, name), name


def test_submodules_are_importable_by_name():
    """`import rust_physics_engine.linalg.lu` has to work.

    PyO3 builds the tree as attributes of the extension module, which is
    enough for attribute access and not enough for `import`: that goes
    through `sys.modules`. `__init__.py` installs them; this is the check
    that it did.
    """
    mod = importlib.import_module("rust_physics_engine.linalg.lu")
    assert hasattr(mod, "solve")
    from rust_physics_engine.numerical import integrate

    assert hasattr(integrate, "simpson")


def test_attribute_and_import_give_the_same_object():
    from rust_physics_engine import linalg

    assert linalg.lu is sys.modules["rust_physics_engine.linalg.lu"]
    assert rpe.linalg.lu is linalg.lu


def test_module_docstrings_come_from_the_rust_headers():
    assert "Newtonian" in rpe.classical.__doc__
    assert rpe.linalg.__doc__


def test_function_docstrings_survive_the_crossing():
    doc = rpe.classical.projectile_range.__doc__
    assert "Range of a projectile" in doc
    # Every docstring names the Rust item it came from, so a reader can
    # follow it back to the source.
    assert "classical::projectile_range" in doc


def test_signatures_are_introspectable():
    import inspect

    sig = inspect.signature(rpe.classical.projectile_range)
    assert list(sig.parameters) == ["speed", "angle_rad", "g"]


def test_keyword_arguments_work():
    a = rpe.classical.projectile_range(20.0, 0.7853981633974483, 9.80665)
    b = rpe.classical.projectile_range(g=9.80665, speed=20.0, angle_rad=0.7853981633974483)
    assert a == b


def test_constants_are_exposed_both_ways():
    assert rpe.constants is sys.modules["rust_physics_engine.math.constants"]
    assert rpe.math.constants.C == 299_792_458.0
    # The 2019 SI redefinition made these exact, and the crate says so.
    assert rpe.constants.H == 6.626_070_15e-34
    assert rpe.constants.K_B == 1.380_649e-23


def test_the_package_docstring_example_actually_runs():
    """The example in `__init__.py` is a doctest, so it cannot go stale."""
    import doctest

    result = doctest.testmod(rpe, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0


def test_a_module_that_does_not_exist_raises():
    with pytest.raises(ImportError):
        importlib.import_module("rust_physics_engine.not_a_module")
