#!/usr/bin/env python3
"""Check that the type stubs describe the extension that was actually built.

A stub that disagrees with its module is worse than no stub at all: it
type-checks code that will fail at run time, and it hides code that would
have worked. Both halves come out of the same generator, so they agree
when the generator is right -- and this is what establishes that, by
importing the built extension and comparing what it exposes against what
the `.pyi` files declare, name by name.

Run it after `maturin develop`:

    python3 bindings/python/check_stubs.py
"""

from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUBS = os.path.join(HERE, "python", "rust_physics_engine")


def stub_names(path: str) -> set[str]:
    """The top-level names a stub file declares."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def stub_path(dotted: str) -> str | None:
    rel = dotted.replace(".", os.sep)
    for candidate in (
        os.path.join(STUBS, rel + ".pyi"),
        os.path.join(STUBS, rel, "__init__.pyi"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def module_names(mod) -> set[str]:
    out = set()
    for name in dir(mod):
        if name.startswith("_"):
            continue
        value = getattr(mod, name)
        if type(value).__name__ == "module":
            continue
        out.add(name)
    return out


def main() -> int:
    try:
        import rust_physics_engine as rpe
    except ImportError as exc:  # pragma: no cover - the message is the point
        print(f"cannot import the extension: {exc}")
        print("build it first:  maturin develop --release -m bindings/python/Cargo.toml")
        return 2

    problems: list[str] = []
    checked = 0
    for dotted in rpe._core.__submodules__:
        mod = sys.modules.get(f"rust_physics_engine.{dotted}")
        if mod is None:
            problems.append(f"{dotted}: not installed in sys.modules")
            continue
        path = stub_path(dotted)
        if path is None:
            problems.append(f"{dotted}: no stub file")
            continue
        checked += 1
        declared = stub_names(path)
        actual = module_names(mod)
        for missing in sorted(actual - declared):
            problems.append(f"{dotted}.{missing}: in the module, not in the stub")
        # A stub may legitimately name imported classes it only refers to,
        # so only flag a declared name the module does not have if the
        # stub defines it rather than importing it.
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        defined = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        } | {
            n.target.id
            for n in tree.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        for extra in sorted(defined - actual):
            problems.append(f"{dotted}.{extra}: in the stub, not in the module")

    if problems:
        print(f"{len(problems)} disagreement(s) between the stubs and the module:")
        for p in problems[:60]:
            print("   ", p)
        if len(problems) > 60:
            print(f"    ... and {len(problems) - 60} more")
        return 1
    print(f"stubs agree with the module across {checked} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
