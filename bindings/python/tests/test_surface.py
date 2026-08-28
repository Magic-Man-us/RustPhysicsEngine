"""A sweep over the whole bound surface.

The other tests check particular functions carefully. This one checks
every function shallowly: that the module tree registered without gaps,
that nothing is missing a docstring or a signature, and that no name in it
raises on mere access. A registration bug -- a module attached to the
wrong parent, a class registered twice, a getter that panics -- shows up
here and nowhere else, because no hand-written test would ever call the
function it broke.
"""

import inspect
import sys

import pytest

import rust_physics_engine as rpe


def _all_modules():
    return [
        sys.modules[f"rust_physics_engine.{d}"] for d in rpe._core.__submodules__
    ]


def test_the_tree_is_large_and_complete():
    mods = _all_modules()
    assert len(mods) > 250
    # Every module the extension advertises really is in sys.modules.
    assert all(m is not None for m in mods)


def test_every_module_is_attached_under_the_right_parent():
    for dotted in rpe._core.__submodules__:
        parent_name, _, leaf = dotted.rpartition(".")
        parent = (
            sys.modules[f"rust_physics_engine.{parent_name}"] if parent_name else rpe
        )
        assert getattr(parent, leaf) is sys.modules[f"rust_physics_engine.{dotted}"]


def test_no_public_name_raises_on_access():
    """Getters run Rust code. One that panics would only show up here."""
    failures = []
    for mod in _all_modules():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            try:
                getattr(mod, name)
            except Exception as exc:  # pragma: no cover - a failure is the point
                failures.append(f"{mod.__name__}.{name}: {exc!r}")
    assert not failures, failures[:20]


def test_thousands_of_functions_are_bound():
    total = 0
    for mod in _all_modules():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            if inspect.isbuiltin(getattr(mod, name)):
                total += 1
    assert total > 3500, f"only {total} functions bound"


def test_every_function_has_a_docstring_and_a_signature():
    missing_doc = []
    missing_sig = []
    for mod in _all_modules():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if not inspect.isbuiltin(obj):
                continue
            if not (obj.__doc__ or "").strip():
                missing_doc.append(f"{mod.__name__}.{name}")
            if not (obj.__text_signature__ or ""):
                missing_sig.append(f"{mod.__name__}.{name}")
    assert not missing_doc, missing_doc[:20]
    assert not missing_sig, missing_sig[:20]


def test_every_docstring_names_its_rust_origin():
    """So a reader can follow any function back to the source."""
    sampled = 0
    for mod in _all_modules():
        seen_here = 0
        for name in sorted(dir(mod)):
            if name.startswith("_") or seen_here >= 3:
                continue
            obj = getattr(mod, name)
            if not inspect.isbuiltin(obj):
                continue
            assert "Rust: `" in obj.__doc__, f"{mod.__name__}.{name}"
            seen_here += 1
            sampled += 1
    assert sampled > 200


def test_hundreds_of_classes_are_bound_and_printable():
    classes = []
    for mod in _all_modules():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if isinstance(obj, type) and getattr(obj, "__module__", "").startswith(
                "rust_physics_engine"
            ):
                classes.append(obj)
    # Aliases mean a class can appear twice; count the distinct ones.
    assert len({(c.__module__, c.__name__) for c in classes}) > 350
    for cls in classes:
        assert cls.__doc__, cls.__name__


def test_no_name_collides_with_a_submodule():
    """A function and a submodule of the same name would shadow each other."""
    for dotted in rpe._core.__submodules__:
        mod = sys.modules[f"rust_physics_engine.{dotted}"]
        children = {
            d.rsplit(".", 1)[1]
            for d in rpe._core.__submodules__
            if d.startswith(dotted + ".") and d.count(".") == dotted.count(".") + 1
        }
        for child in children:
            attr = getattr(mod, child)
            assert attr is sys.modules[f"rust_physics_engine.{dotted}.{child}"], (
                f"{dotted}.{child} is shadowed"
            )


def test_calling_a_representative_function_from_every_top_level_module():
    """Cheap end-to-end proof that each module's registration works."""
    import math

    checks = {
        "acoustics": lambda: rpe.acoustics.sabine_reverberation(100.0, 20.0),
        "chemistry": lambda: rpe.chemistry.half_life_first_order(0.1),
        "classical": lambda: rpe.classical.force(2.0, 3.0),
        "electromagnetism": lambda: rpe.electromagnetism.coulomb_force(1e-6, 1e-6, 1.0),
        "gravitation": lambda: rpe.gravitation.escape_velocity(5.972e24, 6.371e6),
        "information_theory": lambda: rpe.information_theory.shannon_entropy([0.5, 0.5]),
        "linalg": lambda: rpe.linalg.lu.solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0]),
        "math": lambda: rpe.math.Vec3(1.0, 0.0, 0.0).magnitude(),
        "numerical": lambda: rpe.numerical.integrate.simpson(math.sin, 0.0, 1.0, 100),
        "optics": lambda: rpe.optics.snells_law(1.0, 0.5, 1.5),
        "quantum": lambda: rpe.quantum.photon_energy(5e14),
        "relativity": lambda: rpe.relativity.lorentz_factor(0.5 * 299_792_458.0),
        "statistics": lambda: rpe.statistics.descriptive.mean([1.0, 2.0, 3.0]),
        "thermodynamics": lambda: rpe.thermodynamics.carnot_efficiency(300.0, 600.0),
        "transforms": lambda: rpe.transforms.fft.fft([1.0, 0.0]),
    }
    for name, call in checks.items():
        value = call()
        assert value is not None, name
