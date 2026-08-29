"""The shape of the package: what is importable, and by what name.

The Rust module tree is the map users navigate by. If `linalg.lu` is a
module in Rust it should be one here, reachable by attribute *and* by
`import`, and holding the same names. These tests fix that correspondence
so a change to the generator cannot quietly rearrange it.
"""

import importlib
import sys

import pytest

import numeria as nm


def test_version_is_reported():
    assert nm.__version__
    assert nm.__version__[0].isdigit()


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
    """`import numeria.linalg.lu` has to work.

    PyO3 builds the tree as attributes of the extension module, which is
    enough for attribute access and not enough for `import`: that goes
    through `sys.modules`. `__init__.py` installs them; this is the check
    that it did.
    """
    mod = importlib.import_module("numeria.linalg.lu")
    assert hasattr(mod, "solve")
    from numeria.numerical import integrate

    assert hasattr(integrate, "simpson")


def test_attribute_and_import_give_the_same_object():
    from numeria import linalg

    assert linalg.lu is sys.modules["numeria.linalg.lu"]
    assert nm.linalg.lu is linalg.lu


def test_module_docstrings_come_from_the_rust_headers():
    assert "Newtonian" in nm.classical.__doc__
    assert nm.linalg.__doc__


def test_function_docstrings_survive_the_crossing():
    doc = nm.classical.projectile_range.__doc__
    assert "Range of a projectile" in doc
    # Every docstring names the Rust item it came from, so a reader can
    # follow it back to the source.
    assert "classical::projectile_range" in doc


def test_signatures_are_introspectable():
    import inspect

    sig = inspect.signature(nm.classical.projectile_range)
    assert list(sig.parameters) == ["speed", "angle_rad", "g"]


def test_keyword_arguments_work():
    a = nm.classical.projectile_range(20.0, 0.7853981633974483, 9.80665)
    b = nm.classical.projectile_range(g=9.80665, speed=20.0, angle_rad=0.7853981633974483)
    assert a == b


def test_constants_are_exposed_both_ways():
    assert nm.constants is sys.modules["numeria.math.constants"]
    assert nm.math.constants.C == 299_792_458.0
    # The 2019 SI redefinition made these exact, and the crate says so.
    assert nm.constants.H == 6.626_070_15e-34
    assert nm.constants.K_B == 1.380_649e-23


def test_the_package_docstring_example_actually_runs():
    """The example in `__init__.py` is a doctest, so it cannot go stale."""
    import doctest

    result = doctest.testmod(nm, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0


def test_a_module_that_does_not_exist_raises():
    with pytest.raises(ImportError):
        importlib.import_module("numeria.not_a_module")
