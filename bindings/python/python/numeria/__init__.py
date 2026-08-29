"""numeria -- the rust_physics_engine library, from Python.

Generated from the Rust source by ``bindings/python/generate.py``. Every
module here mirrors a Rust module of the same name, and every function
keeps the name, the argument order and the units it has in Rust.

    >>> import numeria as nm
    >>> import math
    >>> round(nm.classical.projectile_range(20.0, math.pi / 4, 9.80665), 4)
    40.7886

Units are SI and angles are radians unless a docstring says otherwise.
Everything this package raises derives from :class:`PhysicsError`.
"""

from __future__ import annotations

import sys as _sys

from . import _core
from ._core import (
    ConvergenceError,
    DegenerateGeometryError,
    DimensionMismatchError,
    EmptyInputError,
    GeometryError,
    InvalidArgumentError,
    NotManifoldError,
    NotPositiveDefiniteError,
    PhysicsError,
    SingularMatrixError,
    SolverError,
    UnitsError,
)

__version__ = _core.__version__


def _install() -> list[str]:
    """Make every submodule importable by name.

    PyO3 builds the module tree as attributes of the extension module,
    which is enough for ``rpe.linalg.lu``. It is not enough for ``import
    numeria.linalg.lu``, or for ``from numeria.linalg import lu``:
    both go through ``sys.modules``, and nothing has put the submodules
    there. This does, once, at import.
    """
    installed = []
    for dotted in _core.__submodules__:
        obj = _core
        for part in dotted.split("."):
            obj = getattr(obj, part)
        _sys.modules[f"{__name__}.{dotted}"] = obj
        if "." not in dotted:
            globals()[dotted] = obj
            installed.append(dotted)
    return installed


_MODULES = _install()

#: The physical and mathematical constants, as a module of their own.
constants = _sys.modules[f"{__name__}.math.constants"]

__all__ = [
    "PhysicsError",
    "InvalidArgumentError",
    "SolverError",
    "SingularMatrixError",
    "NotPositiveDefiniteError",
    "ConvergenceError",
    "DimensionMismatchError",
    "GeometryError",
    "DegenerateGeometryError",
    "NotManifoldError",
    "EmptyInputError",
    "UnitsError",
    "constants",
    *_MODULES,
]
