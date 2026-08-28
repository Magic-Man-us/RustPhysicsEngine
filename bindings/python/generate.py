#!/usr/bin/env python3
"""Generate the PyO3 bindings for `rust_physics_engine`.

The library is about 266,000 lines across 71 top-level modules, and its
public surface is roughly 4,100 free functions, 2,200 inherent methods,
336 structs, 88 enums and 117 constants. Writing that by hand is not the
hard part -- keeping it in step with the library afterwards is. A binding
written once goes stale on the first commit that adds a function, and the
staleness is invisible: nothing fails, the function simply is not there.

So the bindings are derived from the source instead. `rustscan` reads the
crate's public API; this file decides how each item crosses into Python
and writes the wrapper. Re-running it after a change to the library
produces the binding for the changed library, and CI re-runs it and fails
if the committed output differs, which is the only thing that keeps
generated code honest.

What it cannot bind it says so about, in COVERAGE.md, with the reason --
a generic parameter it cannot monomorphise, a `&dyn Trait` argument with
no Python equivalent. Silence about a gap is worse than the gap.

Usage:
    python3 generate.py            # write the bindings
    python3 generate.py --check    # fail if what is committed is stale
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rustscan  # noqa: E402

CRATE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(CRATE_ROOT, "src")
OUT_RS = os.path.join(HERE, "src", "generated")
OUT_PY = os.path.join(HERE, "python", "rust_physics_engine")
PKG = "rust_physics_engine"

# Error types are exceptions, not classes.
ERROR_TYPES = {
    "error::SolveError",
    "error::GeomError",
    "units::quantity::DimError",
    "codes::reed_solomon::TooManyErrors",
    "graph::paths::NegativeCycle",
}

# Types with an exact Python counterpart, translated rather than wrapped.
IDENTIFIED = {
    "fractals::Complex": "complex",
    "exact::bigint::BigInt": "bigint",
    "exact::rational::Rational": "rational",
}

PRIMS = {
    "f64": "float",
    "f32": "float",
    "usize": "int",
    "isize": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "bool": "bool",
    "char": "str",
}

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    "match", "case", "type",
}

RUST_KEYWORDS = {
    "as", "break", "const", "continue", "crate", "dyn", "else", "enum",
    "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop",
    "match", "mod", "move", "mut", "pub", "ref", "return", "self", "static",
    "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "async", "await", "box", "final", "macro", "override", "priv",
    "try", "typeof", "unsized", "virtual", "yield", "abstract", "become",
    "do",
}


# ── The type language ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Ty:
    """A Rust type, reduced to the shapes that matter for binding.

    `by_ref` and `mutable` are kept because they change what the wrapper
    has to pass. `Vec3` and `&Vec3` need different call sites, and
    `&mut [f64]` is not an input at all -- it is an output written through
    an argument, and a binding that quietly dropped the writes would be
    worse than one that refused to bind it.
    """

    kind: str
    inner: tuple = ()
    name: str = ""
    n: int = 0
    by_ref: bool = False
    mutable: bool = False

    def amp(self, expr: str) -> str:
        """`expr`, borrowed if this type is a reference."""
        return f"&{expr}" if self.by_ref else expr

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pre = ("&mut " if self.mutable else "&") if self.by_ref else ""
        if self.kind in ("prim", "user"):
            return f"{pre}{self.kind}:{self.name}"
        return f"{pre}{self.kind}({', '.join(map(repr, self.inner))})"


BAD = Ty("bad")


class Resolver:
    """Turns a type as written into a fully-qualified crate path."""

    def __init__(self, crate: rustscan.Crate):
        self.crate = crate
        self.by_path: dict[str, object] = {}
        self.by_name: dict[str, list[str]] = {}
        for item in list(crate.structs) + list(crate.enums):
            self.by_path[item.path] = item
            self.by_name.setdefault(item.name, []).append(item.path)
        self.aliases = {}
        for a in crate.aliases:
            self.aliases[f"{a.module}::{a.name}" if a.module else a.name] = a.target
        # Every module that exists, so `use` targets can be checked.
        self.modules = set()
        for item in list(crate.structs) + list(crate.enums) + list(crate.funcs):
            parts = item.module.split("::")
            for i in range(len(parts)):
                self.modules.add("::".join(parts[: i + 1]))

    def resolve(self, name: str, file: str, module: str) -> str | None:
        """Full crate path for the type `name` as written inside `file`."""
        name = name.strip()
        if name.startswith("crate::"):
            name = name[len("crate::") :]
        if name.startswith("self::"):
            name = f"{module}::{name[6:]}"
        head = name.split("::")[0]
        tail = name.split("::")[1:]

        candidates: list[str] = []
        # Defined in the same module.
        candidates.append(f"{module}::{name}" if module else name)
        # Imported by name.
        mapped = self.crate.uses.get(file, {}).get(head)
        if mapped:
            candidates.append("::".join([mapped] + tail))
        # Brought in by a glob.
        for g in self.crate.glob_uses.get(file, []):
            candidates.append(f"{g}::{name}")
        # An ancestor module.
        parts = module.split("::") if module else []
        for i in range(len(parts) - 1, -1, -1):
            candidates.append("::".join(parts[:i] + [name]))
        # Written out in full already.
        candidates.append(name)
        # A unique match anywhere in the crate.
        if len(self.by_name.get(name, [])) == 1:
            candidates.append(self.by_name[name][0])

        for c in candidates:
            if c in self.by_path:
                return c
            if c in self.aliases:
                return None  # an alias; the caller re-parses the target
        return None

    def alias_target(self, name: str, file: str, module: str) -> str | None:
        name = name.strip().removeprefix("crate::")
        head = name.split("::")[0]
        for cand in (
            f"{module}::{name}" if module else name,
            self.crate.uses.get(file, {}).get(head, ""),
            name,
        ):
            if cand and cand in self.aliases:
                return self.aliases[cand]
        return None


def parse_type(text: str, res: Resolver, file: str, module: str, depth: int = 0) -> Ty:
    """Parse a Rust type into the reduced language above."""
    if depth > 8:
        return BAD
    s = " ".join(text.split()).strip()
    if not s:
        return Ty("unit")
    if s in ("()", "!"):
        return Ty("unit")

    # A leading reference belongs to the type it points at, not to a
    # wrapper around it: `&[f64]` is the slice type, borrowed.
    if s.startswith("&"):
        m = re.match(r"^&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s+)?(mut\s+)?", s)
        inner = parse_type(s[m.end() :], res, file, module, depth + 1)
        if inner.kind == "bad":
            return BAD
        return dataclasses.replace(inner, by_ref=True, mutable=bool(m.group(1)))
    if s.startswith("*"):
        return BAD

    # Slices and arrays.
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1]
        parts = rustscan._split_top(body)
        if len(parts) == 1 and ";" in body:
            base, count = body.rsplit(";", 1)
            count = count.strip()
            elem = parse_type(base, res, file, module, depth + 1)
            if elem.kind == "bad" or not count.isdigit():
                return BAD
            return Ty("array", (elem,), n=int(count))
        elem = parse_type(body, res, file, module, depth + 1)
        return BAD if elem.kind == "bad" else Ty("vec", (elem,))

    # Tuples.
    if s.startswith("(") and s.endswith(")") and _balanced(s):
        parts = rustscan._split_top(s[1:-1])
        if len(parts) == 1:
            return parse_type(parts[0], res, file, module, depth + 1)
        elems = tuple(parse_type(p, res, file, module, depth + 1) for p in parts)
        if any(e.kind == "bad" for e in elems):
            return BAD
        return Ty("tuple", elems)

    # Callables. A bare `fn(..)` pointer is deliberately not one: a Python
    # callable has no address to hand over.
    m = re.match(r"^(?:dyn\s+|impl\s+)(?:Fn|FnMut|FnOnce)\s*\((.*?)\)\s*(?:->\s*(.+?))?(?:\s*\+\s*.*)?$", s)
    if m:
        argtypes = tuple(
            parse_type(p, res, file, module, depth + 1) for p in rustscan._split_top(m.group(1))
        )
        ret = parse_type(m.group(2) or "()", res, file, module, depth + 1)
        if any(a.kind == "bad" for a in argtypes) or ret.kind == "bad":
            return BAD
        return Ty("callable", argtypes + (ret,))

    # `impl Iterator<Item = T>`: a Python caller gets the list. The
    # iterators here are all finite by construction -- subsets, dyck
    # paths, partitions -- so collecting is a change of laziness, not of
    # termination.
    m = re.match(r"^impl\s+Iterator\s*<\s*Item\s*=\s*(.+?)\s*>(?:\s*\+.*)?$", s)
    if m:
        elem = parse_type(m.group(1), res, file, module, depth + 1)
        return BAD if elem.kind == "bad" else Ty("iter", (elem,))
    if s.startswith("dyn ") or s.startswith("impl "):
        return BAD

    if s in ("String", "str", "std::string::String"):
        return Ty("str")
    if s in PRIMS:
        return Ty("prim", name=s)

    # Generic containers.
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_:]*)\s*<(.+)>$", s)
    if m:
        head = m.group(1).split("::")[-1]
        args = [parse_type(p, res, file, module, depth + 1) for p in rustscan._split_top(m.group(2))]
        if head in ("Vec", "VecDeque"):
            return BAD if args[0].kind == "bad" else Ty("vec", (args[0],))
        if head == "Box":
            return args[0] if args else BAD
        if head == "Option":
            return BAD if args[0].kind == "bad" else Ty("opt", (args[0],))
        if head == "Result":
            ok = args[0]
            parts = rustscan._split_top(m.group(2))
            errname = parts[1].strip() if len(parts) > 1 else ""
            if ok.kind == "bad":
                return BAD
            return Ty("result", (ok,), name=errname.split("::")[-1])
        return BAD

    # A plain path: a struct, an enum, or an alias for one.
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:]*$", s):
        if s.split("::")[-1] == "Self":
            return Ty("selfty")
        target = res.alias_target(s, file, module)
        if target is not None:
            return parse_type(target, res, file, module, depth + 1)
        full = res.resolve(s, file, module)
        if full:
            return Ty("user", name=full)
    return BAD


def _balanced(s: str) -> bool:
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


# ── Wrappers ────────────────────────────────────────────────────────────


@dataclass
class Wrapper:
    """A Python class standing in for one Rust struct or enum."""

    item: object
    ident: str  # the Rust identifier of the wrapper struct
    py_name: str
    py_module: str
    rust_path: str
    is_enum: bool
    simple_enum: bool
    clone: bool
    debug: bool
    partial_eq: bool
    copy: bool
    coerce_n: int = 0  # >0: accepts a sequence of this many floats
    coerce_rows: bool = False  # accepts a sequence of rows
    coerce_seq: bool = False  # accepts a sequence of floats of any length
    unsendable: bool = False
    fields: list = field(default_factory=list)  # bindable (name, Ty, rust_name)
    consts: list = field(default_factory=list)

    @property
    def arg_ident(self) -> str:
        return f"{self.ident}Arg"


def camel(s: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_:]+", s) if p)


class Generator:
    def __init__(self) -> None:
        self.crate = rustscan.scan_crate(SRC)
        _add_macro_items(self.crate)
        self.res = Resolver(self.crate)
        self.wrappers: dict[str, Wrapper] = {}
        self.skipped: list[tuple[str, str, str]] = []  # (module, item, reason)
        self.counts: dict[str, dict[str, int]] = {}
        self.modules: list[str] = []
        self._plan_wrappers()

    # ── planning ────────────────────────────────────────────────────

    def _plan_wrappers(self) -> None:
        items = [(s, False) for s in self.crate.structs] + [(e, True) for e in self.crate.enums]
        by_name: dict[str, list] = {}
        keep = []
        for item, is_enum in items:
            if item.path in ERROR_TYPES or item.path in IDENTIFIED:
                continue
            if item.generics.strip():
                self.skipped.append((item.module, item.name, "generic type"))
                continue
            if item.name.startswith("_"):
                continue
            keep.append((item, is_enum))
            by_name.setdefault(item.name, []).append(item)

        for item, is_enum in keep:
            derives = item.derives()
            if len(by_name[item.name]) == 1:
                ident = "Py" + item.name
            else:
                ident = "Py" + camel(item.module.split("::")[-1]) + item.name
                if sum(1 for o in by_name[item.name] if o.module.split("::")[-1] == item.module.split("::")[-1]) > 1:
                    ident = "Py" + camel(item.module) + item.name
            simple_enum = is_enum and all(not v.payload for v in item.variants)
            unsendable = any(
                re.search(r"\bdyn\b|\bRc<|\bCell<|\bRefCell<", f.ty) for f in getattr(item, "fields", [])
            )
            self.wrappers[item.path] = Wrapper(
                item=item,
                ident=ident,
                py_name=item.name,
                py_module=f"{PKG}.{item.module.replace('::', '.')}",
                rust_path=f"rust_physics_engine::{item.path}",
                is_enum=is_enum,
                simple_enum=simple_enum,
                clone="Clone" in derives,
                debug="Debug" in derives,
                partial_eq="PartialEq" in derives,
                copy="Copy" in derives,
                unsendable=unsendable,
            )

        # Which structs can be built from a bare sequence?
        for path, w in self.wrappers.items():
            if w.is_enum or not w.clone:
                continue
            fields = w.item.fields
            if w.item.kind == "named" and fields and all(f.public and f.ty == "f64" for f in fields):
                if len(fields) <= 6:
                    w.coerce_n = len(fields)
            if w.item.kind == "named" and len(fields) == 1 and fields[0].ty == "Vec<f64>" and fields[0].public:
                w.coerce_seq = True
        matrix = self.wrappers.get("linalg::matrix::Matrix")
        if matrix:
            matrix.coerce_rows = True

        # A type is unsendable if it holds something unsendable, however
        # deep. PyO3 needs that to be right: a `#[pyclass]` that is not
        # `Send` and not declared `unsendable` will not compile, and one
        # that is declared `unsendable` must not have the GIL released
        # around it. Iterate to a fixed point rather than looking only one
        # field deep.
        changed = True
        while changed:
            changed = False
            for w in self.wrappers.values():
                if w.unsendable or w.is_enum:
                    continue
                for f in w.item.fields:
                    ty = parse_type(f.ty, self.res, w.item.file, w.item.module)
                    if self._holds_unsendable(ty):
                        w.unsendable = True
                        changed = True
                        break

        # Bindable fields, for getters.
        for path, w in self.wrappers.items():
            if w.is_enum or w.item.kind != "named":
                continue
            for f in w.item.fields:
                if not f.public:
                    continue
                ty = parse_type(f.ty, self.res, w.item.file, w.item.module)
                if ty.kind != "bad" and self.ret_plan(ty, "x") is not None:
                    w.fields.append((f.name, ty))

        # Associated constants.
        for c in self.crate.consts:
            if not c.owner:
                continue
            for path, w in self.wrappers.items():
                if w.item.name == c.owner and w.item.module == c.module:
                    ty = parse_type(c.ty, self.res, c.file, c.module)
                    if self.ret_plan(ty, "x") is not None:
                        w.consts.append((c.name, ty, c.doc))
                    break

    def _holds_unsendable(self, ty: Ty, depth: int = 0) -> bool:
        if depth > 6:
            return False
        if ty.kind == "user":
            w = self.wrappers.get(ty.name)
            return bool(w and w.unsendable)
        return any(self._holds_unsendable(t, depth + 1) for t in ty.inner)

    def is_clone(self, ty: Ty, depth: int = 0) -> bool:
        """Whether an owned value of `ty` can be cloned out of a field."""
        if depth > 6:
            return False
        k = ty.kind
        if k in ("prim", "str", "unit"):
            return True
        if k in ("vec", "array", "opt"):
            return self.is_clone(ty.inner[0], depth + 1)
        if k == "tuple":
            return all(self.is_clone(t, depth + 1) for t in ty.inner)
        if k == "user":
            if ty.name in IDENTIFIED:
                return True
            w = self.wrappers.get(ty.name)
            return bool(w and (w.clone or w.simple_enum))
        return False

    # ── argument plans ──────────────────────────────────────────────
    #
    # Every argument is described by two things: the type the wrapper
    # declares, and a Rust expression turning a value of that type into
    # what the library wants. Keeping the conversion an *expression*
    # rather than a statement is what lets it nest -- the conversion for
    # `&[(Vec2, f64)]` is the conversion for `(Vec2, f64)` inside a
    # `.map()`, and that in turn is the conversion for `Vec2` and for
    # `f64`. Two shapes cannot be expressions and get their own paths:
    # callables, which need an object that outlives the call, and `&mut`
    # slices, which are outputs and have to be written back afterwards.

    def param_type(self, ty: Ty) -> str | None:
        """The type the generated wrapper declares for `ty`."""
        k = ty.kind
        if k == "prim":
            return ty.name
        if k == "str":
            return "String"
        if k == "user":
            return self._user_param(ty.name)
        if k == "vec":
            inner = self.param_type(ty.inner[0])
            return None if inner is None else f"Vec<{inner}>"
        if k == "array":
            inner = self.param_type(ty.inner[0])
            return None if inner is None else f"Vec<{inner}>"
        if k == "tuple":
            parts = [self.param_type(t) for t in ty.inner]
            if any(p is None for p in parts):
                return None
            return "(" + ", ".join(parts) + ")"
        if k == "opt":
            inner = self.param_type(ty.inner[0])
            return None if inner is None else f"Option<{inner}>"
        return None

    def _user_param(self, path: str) -> str | None:
        if path in IDENTIFIED:
            return {
                "complex": "crate::runtime::coerce::ComplexArg",
                "bigint": "crate::runtime::coerce::BigIntArg",
                "rational": "crate::runtime::coerce::RationalArg",
            }[IDENTIFIED[path]]
        w = self.wrappers.get(path)
        if w is None or not (w.clone or w.simple_enum):
            return None
        if w.coerce_n or w.coerce_rows or w.coerce_seq:
            return f"crate::generated::types::{w.arg_ident}"
        return f"crate::generated::types::{w.ident}"

    def conv_expr(self, ty: Ty, var: str) -> tuple[str, bool] | None:
        """Rust expression turning `var` into an owned value of `ty`.

        The flag says whether the expression uses `?`, which decides
        whether a surrounding `.map()` has to collect into a `PyResult`.
        """
        k = ty.kind
        if k in ("prim", "str"):
            return var, False
        if k == "user":
            path = ty.name
            if path in IDENTIFIED:
                return f"{var}.0", False
            w = self.wrappers.get(path)
            if w is None:
                return None
            if w.simple_enum:
                return f"{var}.to_rust()", False
            if w.coerce_n or w.coerce_rows or w.coerce_seq:
                return f"{var}.0", False
            if not w.clone:
                return None
            return f"{var}.inner", False
        if k == "vec":
            inner = self.conv_expr(ty.inner[0], "__e")
            if inner is None:
                return None
            expr, fallible = inner
            if expr == "__e":
                return var, False
            if fallible:
                elem = self.rust_type(ty.inner[0])
                return (
                    f"{var}.into_iter().map(|__e| -> PyResult<{elem}> {{ Ok({expr}) }})"
                    f".collect::<PyResult<Vec<_>>>()?",
                    True,
                )
            return f"{var}.into_iter().map(|__e| {expr}).collect::<Vec<_>>()", False
        if k == "array":
            elem = self.rust_type(ty.inner[0])
            if elem is None:
                return None
            inner = self.conv_expr(ty.inner[0], "__e")
            if inner is None:
                return None
            expr, fallible = inner
            body = var if expr == "__e" else (
                f"{var}.into_iter().map(|__e| {expr}).collect::<Vec<_>>()"
            )
            if fallible:
                return None
            return (
                f"<[{elem}; {ty.n}]>::try_from({body}).map_err(|__v: Vec<{elem}>| "
                f'pyo3::exceptions::PyValueError::new_err(format!("expected {ty.n} values, got {{}}", __v.len())))?',
                True,
            )
        if k == "tuple":
            parts = [self.conv_expr(t, f"{var}.{i}") for i, t in enumerate(ty.inner)]
            if any(p is None for p in parts):
                return None
            fallible = any(f for _e, f in parts)
            return "(" + ", ".join(e for e, _f in parts) + ")", fallible
        if k == "opt":
            inner = self.conv_expr(ty.inner[0], "__o")
            if inner is None:
                return None
            expr, fallible = inner
            if expr == "__o":
                return var, False
            if fallible:
                elem = self.rust_type(ty.inner[0])
                return (
                    f"match {var} {{ Some(__o) => Some({expr}), None => None }}",
                    True,
                )
            return f"{var}.map(|__o| {expr})", False
        return None

    def rust_type(self, ty: Ty) -> str | None:
        """The library-side Rust type an owned conversion produces."""
        k = ty.kind
        if k == "prim":
            return ty.name
        if k == "str":
            return "String"
        if k == "user":
            if ty.name in IDENTIFIED:
                return {
                    "complex": "rust_physics_engine::fractals::Complex",
                    "bigint": "rust_physics_engine::exact::bigint::BigInt",
                    "rational": "rust_physics_engine::exact::rational::Rational",
                }[IDENTIFIED[ty.name]]
            w = self.wrappers.get(ty.name)
            return w.rust_path if w else None
        if k == "vec":
            inner = self.rust_type(ty.inner[0])
            return None if inner is None else f"Vec<{inner}>"
        if k == "array":
            inner = self.rust_type(ty.inner[0])
            return None if inner is None else f"[{inner}; {ty.n}]"
        if k == "tuple":
            parts = [self.rust_type(t) for t in ty.inner]
            if any(p is None for p in parts):
                return None
            return "(" + ", ".join(parts) + ")"
        if k == "opt":
            inner = self.rust_type(ty.inner[0])
            return None if inner is None else f"Option<{inner}>"
        return None

    def arg_plan(self, ty: Ty, name: str) -> dict | None:
        """How to accept `ty` from Python.

        Returns `param` (the wrapper's parameter type), `pre` (statements
        run before the call), `expr` (what is passed to the Rust routine),
        `post` (statements run after it), `py` (the stub annotation) and
        `owns` (whether the value is free of Python state, and so safe to
        hold with the GIL released).
        """
        if ty.kind == "callable":
            return self._callable_arg(ty, name)
        if ty.mutable:
            return self._mut_arg(ty, name)
        if ty.kind in ("unit", "bad", "selfty", "result", "iter"):
            return None
        if ty.kind == "user" and ty.by_ref and ty.name in self.wrappers:
            w = self.wrappers[ty.name]
            if not (w.clone or w.simple_enum):
                # No `Clone`, so there is no owned value to make. The
                # wrapper is borrowed for the duration of the call
                # instead, which is what `&T` means anyway.
                return dict(
                    param=f"pyo3::PyRef<'_, crate::generated::types::{w.ident}>",
                    pre=[],
                    expr=f"&{name}.inner",
                    post=[],
                    py=w.py_name,
                    owns=False,
                )
        param = self.param_type(ty)
        conv = self.conv_expr(ty, name)
        if param is None or conv is None:
            return None
        expr, _fallible = conv
        pre = [] if expr == name else [f"let {name} = {expr};"]
        # An `&[T]` parameter takes a borrow of the owned vector; a `Vec<T>`
        # parameter takes it whole. Slices of references need one more
        # step, because `&Vec<Vec<f64>>` is not `&[&[f64]]`.
        call = self._borrow(ty, name, pre)
        if call is None:
            return None
        return dict(
            param=param,
            pre=pre,
            expr=call,
            post=[],
            py=self._py_of(ty),
            owns=True,
        )

    def borrow_rust_type(self, ty: Ty) -> str | None:
        """The Rust type as the callee writes it, references and all."""
        base = self.rust_type(dataclasses.replace(ty, by_ref=False, mutable=False))
        if ty.kind == "vec":
            elem = self.borrow_rust_type(ty.inner[0])
            if elem is None:
                return None
            base = f"[{elem}]" if ty.by_ref else f"Vec<{elem}>"
        elif ty.kind == "tuple":
            parts = [self.borrow_rust_type(t) for t in ty.inner]
            if any(p is None for p in parts):
                return None
            base = "(" + ", ".join(parts) + ")"
        elif ty.kind == "opt":
            inner = self.borrow_rust_type(ty.inner[0])
            if inner is None:
                return None
            base = f"Option<{inner}>"
        elif ty.kind == "str" and ty.by_ref:
            return "&str"
        if base is None:
            return None
        return f"&{base}" if ty.by_ref else base

    def _needs_deep_borrow(self, ty: Ty, top: bool = True) -> bool:
        """Whether anything below the outermost level is a reference.

        `&[f64]` does not need one: a `&Vec<f64>` coerces. `&[&str]` and
        `&[(&str, Dim)]` do, because no coercion turns an owned collection
        into a collection of borrows.
        """
        if not top and ty.by_ref:
            return True
        return any(self._needs_deep_borrow(t, False) for t in ty.inner)

    def _place_expr(self, ty: Ty, place: str) -> str | None:
        """The borrowed form of `ty`, given `place` names the owned value."""
        if ty.by_ref:
            if ty.kind == "str":
                return f"{place}.as_str()"
            if ty.kind == "vec":
                return f"{place}.as_slice()"
            return f"&{place}"
        if ty.kind == "prim":
            return place
        if ty.kind == "tuple":
            parts = [self._place_expr(t, f"{place}.{i}") for i, t in enumerate(ty.inner)]
            return None if any(p is None for p in parts) else "(" + ", ".join(parts) + ")"
        if self.is_clone(ty):
            return f"{place}.clone()"
        return None

    def _borrow(self, ty: Ty, name: str, pre: list[str]) -> str | None:
        """What to write at the call site, given `name` holds the owned value."""
        if ty.kind == "vec" and self._needs_deep_borrow(ty.inner[0], top=False):
            elem_ty = self.borrow_rust_type(ty.inner[0])
            elem_expr = self._place_expr(ty.inner[0], "(*__b)")
            if elem_ty is None or elem_expr is None:
                return None
            pre.append(
                f"let {name}__b: Vec<{elem_ty}> = {name}.iter().map(|__b| {elem_expr}).collect();"
            )
            return f"&{name}__b" if ty.by_ref else f"{name}__b"
        if ty.kind == "tuple" and any(
            self._needs_deep_borrow(t, top=False) or t.by_ref for t in ty.inner
        ):
            parts = [self._place_expr(t, f"{name}.{i}") for i, t in enumerate(ty.inner)]
            if any(p is None for p in parts):
                return None
            inner = "(" + ", ".join(parts) + ")"
            return f"&{inner}" if ty.by_ref else inner
        if ty.kind == "opt" and ty.inner[0].by_ref:
            inner = ty.inner[0]
            if inner.kind == "vec":
                take = "__o.as_slice()"
            elif inner.kind == "str":
                take = "__o.as_str()"
            else:
                take = "__o"
            return f"{name}.as_ref().map(|__o| {take})"
        if ty.kind == "array" and (
            ty.inner[0].by_ref or self._needs_deep_borrow(ty.inner[0], top=False)
        ):
            return None
        return ty.amp(name)

    MUT_WRITEBACK = {"f64", "f32", "usize", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "bool"}

    def _mut_arg(self, ty: Ty, name: str) -> dict | None:
        """`&mut` arguments: an output written through the argument."""
        if ty.kind == "user":
            w = self.wrappers.get(ty.name)
            if w is None or w.simple_enum:
                return None
            return dict(
                param=f"pyo3::PyRefMut<'_, crate::generated::types::{w.ident}>",
                pre=[f"let mut {name} = {name};"],
                expr=f"&mut {name}.inner",
                post=[],
                py=w.py_name,
                owns=False,
            )
        if ty.kind == "vec" and ty.inner[0].kind == "user":
            elem = ty.inner[0]
            param = self.param_type(elem)
            conv = self.conv_expr(elem, "__e")
            back = self._writeback_object(elem, "__e")
            if param is None or conv is None or back is None or conv[1]:
                return None
            rust_elem = self.rust_type(elem)
            return dict(
                param="pyo3::Bound<'py, pyo3::PyAny>",
                pre=[
                    f"let mut {name}__v: Vec<{rust_elem}> = {name}.extract::<Vec<{param}>>()?"
                    f".into_iter().map(|__e| {conv[0]}).collect();",
                ],
                expr=f"&mut {name}__v",
                post=[
                    f"crate::runtime::coerce::write_back_objects(&{name}, "
                    f"{name}__v.into_iter().map(|__e| {back}).collect::<Vec<_>>())?;"
                ],
                py=f"MutableSequence[{self._py_of(elem)}]",
                owns=False,
                lifetime=True,
            )
        if ty.kind == "vec" and ty.inner[0].kind == "prim" and ty.inner[0].name in self.MUT_WRITEBACK:
            p = ty.inner[0].name
            return dict(
                param="pyo3::Bound<'py, pyo3::PyAny>",
                pre=[
                    f"let mut {name}__v: Vec<{p}> = {name}.extract()?;",
                ],
                expr=f"&mut {name}__v",
                post=[f"crate::runtime::coerce::write_back(&{name}, &{name}__v)?;"],
                py="MutableSequence[" + PRIMS[p] + "]",
                owns=False,
                lifetime=True,
            )
        return None

    def _writeback_object(self, ty: Ty, var: str) -> str | None:
        """An expression turning an owned Rust value back into a Python one."""
        if ty.kind != "user":
            return None
        if ty.name in IDENTIFIED:
            return f"crate::runtime::coerce::Cx({var})" if IDENTIFIED[ty.name] == "complex" else None
        w = self.wrappers.get(ty.name)
        if w is None or w.simple_enum:
            return None
        return f"crate::generated::types::{w.ident} {{ inner: {var} }}"

    CALLABLE_FALLBACK = {
        "f64": "f64::NAN",
        "f32": "f32::NAN",
        "bool": "false",
        "usize": "0",
        "u8": "0",
        "u16": "0",
        "u32": "0",
        "u64": "0",
        "i8": "0",
        "i16": "0",
        "i32": "0",
        "i64": "0",
    }

    def _callable_ret(self, ty: Ty) -> tuple[str, str, str] | None:
        """(extracted type, fallback expression, expression turning it into the Rust type)."""
        if ty.kind == "prim":
            fb = self.CALLABLE_FALLBACK.get(ty.name)
            return (ty.name, fb, "__r") if fb else None
        if ty.kind == "unit":
            return "()", "()", "__r"
        if ty.kind == "str":
            return "String", "String::new()", "__r"
        if ty.kind == "vec" and ty.inner[0].kind == "prim":
            return f"Vec<{ty.inner[0].name}>", "Vec::new()", "__r"
        if ty.kind == "tuple" and all(t.kind == "prim" for t in ty.inner):
            rust = "(" + ", ".join(t.name for t in ty.inner) + ")"
            fb = "(" + ", ".join(self.CALLABLE_FALLBACK.get(t.name, "0") for t in ty.inner) + ")"
            return rust, fb, "__r"
        if ty.kind == "opt":
            inner = self._callable_ret(ty.inner[0])
            if inner is None or inner[2] != "__r":
                return None
            return f"Option<{inner[0]}>", "None", "__r"
        if ty.kind == "user":
            if ty.name in IDENTIFIED and IDENTIFIED[ty.name] == "complex":
                return (
                    "crate::runtime::coerce::ComplexArg",
                    "crate::runtime::coerce::ComplexArg(rust_physics_engine::fractals::Complex::new(f64::NAN, f64::NAN))",
                    "__r.0",
                )
            w = self.wrappers.get(ty.name)
            if w is None or not w.clone:
                return None
            if w.coerce_n:
                nan = ", ".join(f"{f.name}: f64::NAN" for f in w.item.fields)
                return (
                    f"crate::generated::types::{w.arg_ident}",
                    f"crate::generated::types::{w.arg_ident}({w.rust_path} {{ {nan} }})",
                    "__r.0",
                )
            if w.coerce_seq:
                fname = w.item.fields[0].name
                return (
                    f"crate::generated::types::{w.arg_ident}",
                    f"crate::generated::types::{w.arg_ident}({w.rust_path} {{ {fname}: Vec::new() }})",
                    "__r.0",
                )
        return None

    def _callable_arg(self, ty: Ty, name: str) -> dict | None:
        *argtys, retty = ty.inner
        params, passed = [], []
        for i, a in enumerate(argtys):
            v = f"__a{i}"
            if a.kind == "prim":
                params.append(f"{v}: {a.name}")
                passed.append(v)
            elif a.kind == "str":
                params.append(f"{v}: &str")
                passed.append(f"{v}.to_string()")
            elif a.kind == "vec" and a.inner[0].kind == "prim":
                params.append(f"{v}: &[{a.inner[0].name}]")
                passed.append(f"{v}.to_vec()")
            elif a.kind == "user" and a.name in IDENTIFIED and IDENTIFIED[a.name] == "complex":
                params.append(f"{v}: rust_physics_engine::fractals::Complex")
                passed.append(f"crate::runtime::coerce::Cx({v})")
            elif a.kind == "user" and a.name in self.wrappers and self.wrappers[a.name].clone:
                w = self.wrappers[a.name]
                ref = "&" if a.by_ref else ""
                params.append(f"{v}: {ref}{w.rust_path}")
                passed.append(
                    f"crate::generated::types::{w.ident} {{ inner: {v}{'.clone()' if a.by_ref else ''} }}"
                )
            else:
                return None
        ret = self._callable_ret(retty)
        if ret is None:
            return None
        extracted, fallback, unwrap = ret
        rust_ret = self.rust_type(retty) if retty.kind != "unit" else "()"
        if rust_ret is None:
            return None
        args_tuple = f"({', '.join(passed)},)" if len(passed) == 1 else f"({', '.join(passed)})"
        body = f"__cb_{name}.call::<_, {extracted}>({args_tuple}, {fallback})"
        if unwrap != "__r":
            body = f"{{ let __r = {body}; {unwrap} }}"
        # The closure owns a handle rather than borrowing one, because
        # some of these arguments carry a `+ 'static` bound and a borrow
        # of a local cannot satisfy it. The wrapper keeps its own handle
        # so that it can still ask, after the call, whether the callable
        # raised.
        body = body.replace(f"__cb_{name}", "__cb")
        pre = [
            f"let __cb_{name} = std::rc::Rc::new(crate::runtime::Callback::new({name}));",
            f"let {name} = {{ let __cb = __cb_{name}.clone(); "
            f"move |{', '.join(params)}| -> {rust_ret} {{ {body} }} }};",
        ]
        py_sig = (
            "Callable[[" + ", ".join(self._py_of(a) for a in argtys) + "], " + self._py_of(retty) + "]"
        )
        return dict(
            param="pyo3::Py<pyo3::PyAny>",
            pre=pre,
            expr=ty.amp(name),
            post=[],
            py=py_sig,
            owns=False,
            callback=f"__cb_{name}",
        )

    def _coerce_py(self, w: Wrapper) -> str:
        if w.coerce_rows:
            return f"{w.py_name} | Sequence[Sequence[float]]"
        return f"{w.py_name} | Sequence[float]"

    def _py_of(self, ty: Ty) -> str:
        if ty.kind == "prim":
            return PRIMS[ty.name]
        if ty.kind == "str":
            return "str"
        if ty.kind == "unit":
            return "None"
        if ty.kind in ("vec", "array", "iter"):
            return f"list[{self._py_of(ty.inner[0])}]"
        if ty.kind == "tuple":
            return "tuple[" + ", ".join(self._py_of(t) for t in ty.inner) + "]"
        if ty.kind == "opt":
            return f"Optional[{self._py_of(ty.inner[0])}]"
        if ty.kind == "result":
            return self._py_of(ty.inner[0])
        if ty.kind == "callable":
            *a, r = ty.inner
            return "Callable[[" + ", ".join(self._py_of(x) for x in a) + "], " + self._py_of(r) + "]"
        if ty.kind == "user":
            if ty.name in IDENTIFIED:
                return {"complex": "complex", "bigint": "int", "rational": "Fraction"}[
                    IDENTIFIED[ty.name]
                ]
            w = self.wrappers.get(ty.name)
            if w is None:
                return "Any"
            if w.coerce_n or w.coerce_rows or w.coerce_seq:
                return self._coerce_py(w)
            return w.py_name
        return "Any"

    # ── return plans ────────────────────────────────────────────────

    def ret_plan(self, ty: Ty, expr: str) -> dict | None:
        """How to hand `ty` back to Python. `expr` names the Rust value."""
        k = ty.kind
        if ty.by_ref and k not in ("str", "unit"):
            # A returned reference cannot outlive the call on the Python
            # side, so it is copied out. `&mut` is different: it exists to
            # be written through, and a copy would look like it worked.
            if ty.mutable:
                return None
            base = dataclasses.replace(ty, by_ref=False, mutable=False)
            if k == "prim":
                return self.ret_plan(base, f"(*{expr})")
            if not self.is_clone(base):
                return None
            if k == "vec":
                return self.ret_plan(base, f"{expr}.to_vec()")
            return self.ret_plan(base, f"{expr}.clone()")
        if k == "iter":
            return self.ret_plan(Ty("vec", ty.inner), f"{expr}.collect::<Vec<_>>()")
        if k == "unit":
            return dict(rust="()", conv="()", py="None", fallible=False)
        if k == "prim":
            return dict(rust=ty.name, conv=expr, py=PRIMS[ty.name], fallible=False)
        if k == "str":
            return dict(rust="String", conv=f"{expr}.to_string()", py="str", fallible=False)
        if k == "user":
            return self._user_ret(ty.name, expr)
        if k == "vec":
            inner = self.ret_plan(ty.inner[0], "__x")
            if inner is None:
                return None
            if inner["conv"] == "__x":
                return dict(
                    rust=f"Vec<{inner['rust']}>",
                    conv=expr,
                    py=f"list[{inner['py']}]",
                    fallible=False,
                )
            if inner["fallible"]:
                return dict(
                    rust=f"Vec<{inner['rust']}>",
                    conv=(
                        f"{expr}.into_iter()"
                        f".map(|__x| -> PyResult<{inner['rust']}> {{ Ok({inner['conv']}) }})"
                        ".collect::<PyResult<Vec<_>>>()?"
                    ),
                    py=f"list[{inner['py']}]",
                    fallible=True,
                )
            return dict(
                rust=f"Vec<{inner['rust']}>",
                conv=f"{expr}.into_iter().map(|__x| {inner['conv']}).collect::<Vec<_>>()",
                py=f"list[{inner['py']}]",
                fallible=False,
            )
        if k == "array":
            inner = self.ret_plan(ty.inner[0], "__x")
            if inner is None:
                return None
            if inner["conv"] == "__x":
                return dict(
                    rust=f"Vec<{inner['rust']}>",
                    conv=f"{expr}.to_vec()",
                    py=f"list[{inner['py']}]",
                    fallible=False,
                )
            if inner["fallible"]:
                return None
            return dict(
                rust=f"Vec<{inner['rust']}>",
                conv=f"{expr}.into_iter().map(|__x| {inner['conv']}).collect::<Vec<_>>()",
                py=f"list[{inner['py']}]",
                fallible=False,
            )
        if k == "tuple":
            plans = [self.ret_plan(t, f"{expr}.{i}") for i, t in enumerate(ty.inner)]
            if any(p is None for p in plans):
                return None
            return dict(
                rust="(" + ", ".join(p["rust"] for p in plans) + ")",
                conv="(" + ", ".join(p["conv"] for p in plans) + ")",
                py="tuple[" + ", ".join(p["py"] for p in plans) + "]",
                fallible=any(p["fallible"] for p in plans),
            )
        if k == "opt":
            elem = ty.inner[0]
            if elem.by_ref and not elem.mutable and elem.kind == "vec":
                base = dataclasses.replace(elem, by_ref=False, mutable=False)
                inner = self.ret_plan(base, "__x.to_vec()") if self.is_clone(base) else None
            else:
                inner = self.ret_plan(elem, "__x")
            if inner is None:
                return None
            if inner["fallible"]:
                return dict(
                    rust=f"Option<{inner['rust']}>",
                    conv=f"match {expr} {{ Some(__x) => Some({inner['conv']}), None => None }}",
                    py=f"Optional[{inner['py']}]",
                    fallible=True,
                )
            return dict(
                rust=f"Option<{inner['rust']}>",
                conv=f"{expr}.map(|__x| {inner['conv']})",
                py=f"Optional[{inner['py']}]",
                fallible=False,
            )
        return None

    ERR_MAP = {
        "GeomError": "crate::runtime::map_geom",
        "SolveError": "crate::runtime::map_solve",
        "DimError": "crate::runtime::errors::map_dim",
        # These carry a message, so `Display` is the right rendering.
        "TooManyErrors": "crate::runtime::errors::map_display",
        "NegativeCycle": "crate::runtime::errors::map_display",
        "Error": "crate::runtime::errors::map_display",
        "String": "crate::runtime::errors::map_display",
        # An unnamed error type: `io::Result<T>` and friends.
        "": "crate::runtime::errors::map_display",
    }

    def _user_ret(self, path: str, expr: str) -> dict | None:
        if path in IDENTIFIED:
            kind = IDENTIFIED[path]
            if kind == "complex":
                return dict(
                    rust="pyo3::Bound<'py, pyo3::types::PyComplex>",
                    conv=f"crate::runtime::coerce::complex_out(py, {expr})",
                    py="complex",
                    fallible=False,
                )
            if kind == "bigint":
                return dict(
                    rust="pyo3::Bound<'py, pyo3::PyAny>",
                    conv=f"crate::runtime::coerce::bigint_out(py, &{expr})?",
                    py="int",
                    fallible=True,
                )
            if kind == "rational":
                return dict(
                    rust="pyo3::Bound<'py, pyo3::PyAny>",
                    conv=f"crate::runtime::coerce::rational_out(py, &{expr})?",
                    py="Fraction",
                    fallible=True,
                )
        w = self.wrappers.get(path)
        if w is None:
            return None
        if w.simple_enum:
            return dict(
                rust=f"crate::generated::types::{w.ident}",
                conv=f"crate::generated::types::{w.ident}::from_rust(&{expr})",
                py=w.py_name,
                fallible=False,
            )
        return dict(
            rust=f"crate::generated::types::{w.ident}",
            conv=f"crate::generated::types::{w.ident} {{ inner: {expr} }}",
            py=w.py_name,
            fallible=False,
        )



# ── Hand-written supplements ────────────────────────────────────────────

# Methods that no generator would produce, injected verbatim into a
# class's `#[pymethods]` block. PyO3 allows one such block per class, so
# extras cannot live in a file of their own; they are spliced in here.
# Everything below is either a Python protocol the Rust type has no
# equivalent for (`__len__`, `__getitem__`, `__iter__`) or a conversion
# out of the wrapper and into a plain Python value.
EXTRAS: dict[str, list[str]] = {
    "math::Vec2": [
        'fn __len__(&self) -> usize { 2 }',
        'fn __iter__(slf: pyo3::PyRef<\'_, Self>) -> PyResult<pyo3::Py<pyo3::PyAny>> {'
        ' let v = vec![slf.inner.x, slf.inner.y];'
        ' Ok(v.into_pyobject(slf.py())?.try_iter()?.unbind().into_any()) }',
        'fn __getitem__(&self, i: isize) -> PyResult<f64> {'
        ' match i { 0 | -2 => Ok(self.inner.x), 1 | -1 => Ok(self.inner.y),'
        ' _ => Err(pyo3::exceptions::PyIndexError::new_err("Vec2 index out of range")) } }',
        '/// The components as a plain list.\n    fn tolist(&self) -> Vec<f64> { vec![self.inner.x, self.inner.y] }',
    ],
    "math::Vec3": [
        'fn __len__(&self) -> usize { 3 }',
        'fn __iter__(slf: pyo3::PyRef<\'_, Self>) -> PyResult<pyo3::Py<pyo3::PyAny>> {'
        ' let v = vec![slf.inner.x, slf.inner.y, slf.inner.z];'
        ' Ok(v.into_pyobject(slf.py())?.try_iter()?.unbind().into_any()) }',
        'fn __getitem__(&self, i: isize) -> PyResult<f64> {'
        ' match i { 0 | -3 => Ok(self.inner.x), 1 | -2 => Ok(self.inner.y), 2 | -1 => Ok(self.inner.z),'
        ' _ => Err(pyo3::exceptions::PyIndexError::new_err("Vec3 index out of range")) } }',
        '/// The components as a plain list.\n    fn tolist(&self) -> Vec<f64> {'
        ' vec![self.inner.x, self.inner.y, self.inner.z] }',
    ],
    "manifold::vecn::VecN": [
        'fn __len__(&self) -> usize { self.inner.data.len() }',
        'fn __getitem__(&self, i: isize) -> PyResult<f64> {'
        ' let n = self.inner.data.len() as isize; let j = if i < 0 { i + n } else { i };'
        ' if j < 0 || j >= n { return Err(pyo3::exceptions::PyIndexError::new_err("VecN index out of range")); }'
        ' Ok(self.inner.data[j as usize]) }',
        'fn __iter__(slf: pyo3::PyRef<\'_, Self>) -> PyResult<pyo3::Py<pyo3::PyAny>> {'
        ' let v = slf.inner.data.clone();'
        ' Ok(v.into_pyobject(slf.py())?.try_iter()?.unbind().into_any()) }',
        '/// The components as a plain list.\n    fn tolist(&self) -> Vec<f64> { self.inner.data.clone() }',
    ],
    "linalg::matrix::Matrix": [
        'fn __len__(&self) -> usize { self.inner.rows }',
        '/// `m[i, j]`, or `m[i]` for a whole row.\n    '
        'fn __getitem__(&self, py: Python<\'_>, key: pyo3::Py<pyo3::PyAny>) -> PyResult<pyo3::Py<pyo3::PyAny>> {'
        ' let k = key.bind(py);'
        ' if let Ok((i, j)) = k.extract::<(isize, isize)>() {'
        '   let (r, c) = (self.wrap_row(i)?, self.wrap_col(j)?);'
        '   return Ok(self.inner.data[r * self.inner.cols + c].into_pyobject(py)?.unbind().into_any()); }'
        ' let i = k.extract::<isize>().map_err(|_| pyo3::exceptions::PyTypeError::new_err('
        '   "index a Matrix with m[i, j] or m[i]"))?;'
        ' let r = self.wrap_row(i)?;'
        ' let row: Vec<f64> = self.inner.data[r * self.inner.cols..(r + 1) * self.inner.cols].to_vec();'
        ' Ok(row.into_pyobject(py)?.unbind().into_any()) }',
        '/// `m[i, j] = v`.\n    '
        'fn __setitem__(&mut self, key: (isize, isize), v: f64) -> PyResult<()> {'
        ' let (r, c) = (self.wrap_row(key.0)?, self.wrap_col(key.1)?);'
        ' let cols = self.inner.cols; self.inner.data[r * cols + c] = v; Ok(()) }',
        '/// The rows as a list of lists.\n    fn tolist(&self) -> Vec<Vec<f64>> {'
        ' self.inner.data.chunks(self.inner.cols).map(<[f64]>::to_vec).collect() }',
        '/// `(rows, cols)`.\n    #[getter]\n    fn shape(&self) -> (usize, usize) {'
        ' (self.inner.rows, self.inner.cols) }',
    ],
    "quaternion::Quaternion": [
        'fn __len__(&self) -> usize { 4 }',
        '/// `(w, x, y, z)` as a plain list.\n    fn tolist(&self) -> Vec<f64> {'
        ' vec![self.inner.w, self.inner.x, self.inner.y, self.inner.z] }',
    ],
}

# Private helpers on the wrapper struct (outside `#[pymethods]`).
PRIVATE_IMPLS: dict[str, str] = {
    "linalg::matrix::Matrix": """
impl PyMatrix {
    fn wrap_row(&self, i: isize) -> PyResult<usize> {
        let n = self.inner.rows as isize;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err("row index out of range"));
        }
        Ok(j as usize)
    }
    fn wrap_col(&self, i: isize) -> PyResult<usize> {
        let n = self.inner.cols as isize;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err("column index out of range"));
        }
        Ok(j as usize)
    }
}
""",
}

# Operator traits worth exposing as Python dunders.
DUNDERS = {
    ("Add", "add"): "__add__",
    ("Sub", "sub"): "__sub__",
    ("Mul", "mul"): "__mul__",
    ("Div", "div"): "__truediv__",
    ("Neg", "neg"): "__neg__",
}


def _without_detach(plan: dict) -> dict:
    """Turn off GIL release, and stop asking for a `py` nobody now uses."""
    needs_py = bool(
        re.search(r"\bpy\b", plan["ret"]["conv"])
        or re.search(r"\bpy\b", " ".join(plan["pre"]))
    )
    return dict(plan, detach=False, needs_py=needs_py)


def sanitize_py(name: str) -> str:
    name = name.lstrip("_") or "arg"
    if name in PY_KEYWORDS:
        return name + "_"
    return name


def sanitize_rust(name: str) -> str:
    if name in RUST_KEYWORDS:
        return "r#" + name if name not in ("self", "super", "crate", "Self") else name + "_"
    return name


def clean_markup(text: str) -> str:
    """Strip the Rust-specific markup from a doc comment."""
    # rustdoc disambiguators: `[`mod@cholesky`]` names a module, not a
    # function; the prefix means nothing outside rustdoc.
    text = re.sub(
        r"\b(?:mod|fn|struct|enum|trait|type|macro|value|derive|prim|const|static|union)@",
        "",
        text,
    )
    text = re.sub(r"\[`([^`]+)`\]\([^)]*\)", r"`\1`", text)
    text = re.sub(r"\[`([^`]+)`\]", r"`\1`", text)
    text = re.sub(r"\[([^\]\[]+)\]\([^)]*\)", r"\1", text)
    return text.replace("crate::", "")


def pydoc(doc: str, rust_path: str) -> list[str]:
    """Rust doc comment -> Python docstring lines."""
    out: list[str] = []
    fenced = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if stripped.startswith("# "):
            out.append(stripped[2:].rstrip(".") + ":")
            continue
        out.append(line)
    text = clean_markup("\n".join(out))
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append("")
    lines.append(f"Rust: `{rust_path}`")
    return lines


def rust_doc_lines(lines: list[str]) -> str:
    out = []
    for ln in lines:
        ln = ln.replace("\r", "")
        out.append("/// " + ln if ln else "///")
    return "\n".join(out)


def py_doc_literal(lines: list[str], indent: str) -> str:
    body = "\n".join(lines).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        body += " "
    return f'{indent}"""' + ("\n" + body + "\n" + indent if body else "") + '"""'


# ── Emission ────────────────────────────────────────────────────────────


@dataclass
class Emitted:
    py_name: str
    code: str
    stub: str
    kind: str = "function"


LIFETIME_ONLY = re.compile(r"^<\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*,?\s*)*>$")


class Emitter(Generator):
    """Turns the plans in `Generator` into Rust and `.pyi` text."""

    def __init__(self) -> None:
        super().__init__()
        self.by_module: dict[str, list[Emitted]] = {}
        self.class_of_module: dict[str, list[Wrapper]] = {}
        self.consts_of_module: dict[str, list] = {}
        self.class_by_name: dict[str, list[Wrapper]] = {}
        for w in self.wrappers.values():
            self.class_by_name.setdefault(w.py_name, []).append(w)
            self.class_of_module.setdefault(w.item.module, []).append(w)

    # ── shared call planning ────────────────────────────────────────

    def _self_subst(self, text: str, owner: object | None) -> str:
        if owner is None:
            return text
        return re.sub(r"\bSelf\b", f"crate::{owner.path}", text)

    def _plan(self, fn, owner=None):
        """Plan one call. Returns a dict, or a string naming why it cannot be bound."""
        if fn.is_unsafe:
            return "unsafe fn"
        if fn.generics.strip() and not LIFETIME_ONLY.match(fn.generics.strip()):
            return f"generic ({fn.generics.strip()})"
        if fn.where_clause.strip() and not re.match(
            r"^(?:'[A-Za-z_][A-Za-z0-9_]*\s*:[^,]*,?\s*)*$", fn.where_clause.strip()
        ):
            return "where clause"

        params, pre, post, callargs, annots, callbacks = [], [], [], [], [], []
        owns = True
        seen: set[str] = set()
        heavy = False
        for pat, ty_text in fn.args:
            name = pat.replace("mut ", "").strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                return f"argument pattern `{pat}`"
            py_name = sanitize_py(name)
            if py_name in seen:
                return f"duplicate argument `{py_name}`"
            seen.add(py_name)
            ty = parse_type(self._self_subst(ty_text, owner), self.res, fn.file, fn.module)
            if ty.kind == "bad":
                return f"argument `{name}: {ty_text}`"
            plan = self.arg_plan(ty, py_name)
            if plan is None:
                return f"argument `{name}: {ty_text}`"
            params.append(f"{py_name}: {plan['param']}")
            pre.extend(plan["pre"])
            post.extend(plan.get("post", []))
            callargs.append(plan["expr"])
            annots.append((py_name, plan["py"], ty))
            owns = owns and plan["owns"]
            if "callback" in plan:
                callbacks.append(plan["callback"])
            if ty.kind in ("vec", "array") or (ty.kind == "user" and ty.name in self.wrappers
                                               and self.wrappers[ty.name].coerce_rows):
                heavy = True

        ret_text = self._self_subst(fn.ret, owner)
        ret_ty = parse_type(ret_text, self.res, fn.file, fn.module) if ret_text else Ty("unit")
        if ret_ty.kind == "bad":
            return f"return type `{fn.ret}`"
        err_map = None
        if ret_ty.kind == "result":
            err_map = self.ERR_MAP.get(ret_ty.name, "crate::runtime::errors::map_debug")
            ret_ty = ret_ty.inner[0]
        rp = self.ret_plan(ret_ty, "__v")
        if rp is None:
            return f"return type `{fn.ret}`"
        if ret_ty.kind in ("vec", "array"):
            heavy = True

        needs_py = bool(re.search(r"\bpy\b", rp["conv"]) or re.search(r"\bpy\b", " ".join(pre)))
        plain_ret = self._is_plain(ret_ty)
        detach = owns and not callbacks and plain_ret and heavy
        return dict(
            params=params,
            pre=pre,
            post=post,
            callargs=callargs,
            annots=annots,
            callbacks=callbacks,
            ret=rp,
            err_map=err_map,
            detach=detach,
            needs_py=needs_py or detach,
            lifetime="'py" in rp["rust"] or any("'py" in p for p in params),
        )

    def _is_plain(self, ty: Ty) -> bool:
        if ty.kind in ("prim", "str", "unit"):
            return True
        if ty.kind in ("vec", "array", "opt"):
            return self._is_plain(ty.inner[0])
        if ty.kind == "tuple":
            return all(self._is_plain(t) for t in ty.inner)
        return False

    def _body(self, plan, callee: str) -> str:
        lines: list[str] = []
        lines.extend(plan["pre"])
        call = f"{callee}({', '.join(plan['callargs'])})"
        if plan.get("discard") == "unit":
            # A builder returns `&mut Self` for chaining. Keeping that
            # borrow alive past the call would stop the wrapper being
            # handed back, and the mutation is the whole point anyway.
            call = f"{{ {call}; }}"
        elif plan.get("discard") == "result":
            # The fallible form: the error still matters, the borrow does
            # not, and dropping it inside the closure ends it there.
            call = f"{call}.map(|_| ())"
        mv = "move " if plan["detach"] else ""
        guarded = f"crate::runtime::guard({mv}|| {call})"
        if plan["detach"]:
            guarded = f"py.detach(move || {guarded})"
        lines.append(f"let __r = {guarded};")
        if plan["callbacks"]:
            refs = ", ".join(f"&{c}" for c in plan["callbacks"])
            lines.append(f"crate::runtime::callback::check(&[{refs}], ())?;")
        lines.append(
            "let __v = __r.map_err(crate::runtime::errors::InvalidArgumentError::new_err)?;"
        )
        if plan["err_map"]:
            lines.append(f"let __v = __v.map_err({plan['err_map']})?;")
        lines.extend(plan.get("post", []))
        lines.append(f"Ok({plan['ret']['conv']})")
        return "\n".join("    " + ln for ln in lines)

    def _signature_attr(self, annots) -> str:
        parts = []
        for i, (name, _py, ty) in enumerate(annots):
            trailing = all(a[2].kind == "opt" for a in annots[i:])
            if ty.kind == "opt" and trailing:
                parts.append(f"{name}=None")
            else:
                parts.append(name)
        return "(" + ", ".join(parts) + ")"

    def _stub_sig(self, annots, ret_py: str) -> str:
        parts = []
        for i, (name, py, ty) in enumerate(annots):
            trailing = all(a[2].kind == "opt" for a in annots[i:])
            parts.append(f"{name}: {py} = None" if (ty.kind == "opt" and trailing) else f"{name}: {py}")
        return "(" + ", ".join(parts) + ") -> " + ret_py

    # ── free functions ──────────────────────────────────────────────

    def emit_free(self, fn) -> Emitted | None:
        plan = self._plan(fn)
        if isinstance(plan, str):
            self.skipped.append((fn.module, fn.name + "()", plan))
            return None
        py_name = sanitize_py(fn.name)
        ident = "pyfn_" + fn.name
        rust_path = f"rust_physics_engine::{fn.module}::{fn.name}" if fn.module else f"rust_physics_engine::{fn.name}"
        doc = pydoc(fn.doc, f"{fn.module}::{fn.name}")
        params = plan["params"]
        head_params = []
        if plan["needs_py"]:
            head_params.append("py: Python<'py>")
        head_params.extend(params)
        lt = "<'py>" if (plan["needs_py"] or plan["lifetime"]) else ""
        code = "\n".join(
            [
                rust_doc_lines(doc),
                "#[pyfunction]",
                f'#[pyo3(name = "{py_name}", signature = {self._signature_attr(plan["annots"])})]',
                f"pub fn {ident}{lt}({', '.join(head_params)}) -> PyResult<{plan['ret']['rust']}> {{",
                self._body(plan, rust_path),
                "}",
            ]
        )
        stub = "\n".join(
            [
                f"def {py_name}{self._stub_sig(plan['annots'], plan['ret']['py'])}:",
                py_doc_literal(doc, "    "),
                "    ...",
            ]
        )
        return Emitted(py_name=py_name, code=code, stub=stub)

    def emit_identified_method(self, fn, path: str) -> Emitted | None:
        """A method of `BigInt`, `Rational` or `Complex`, as a free function.

        These three types cross over as `int`, `Fraction` and `complex`,
        so there is no class to hang their methods on -- but the methods
        are not all redundant. Python has `math.factorial` and three-
        argument `pow`; it has no integer `nth_root`, no
        `is_perfect_square`, and no `to_continued_fraction`. Each becomes
        a function in the module the type is defined in, with the receiver
        as its first argument.
        """
        owner = self.res.by_path.get(path)
        if owner is None:
            return None
        if fn.self_kind == "&mut self":
            # The receiver is a Python `int`, `Fraction` or `complex`, all
            # immutable. A method that works by mutating the value in place
            # has nowhere to put the result.
            self.skipped.append(
                (fn.module, f"{fn.impl_type}.{fn.name}()", "mutates an immutable Python type")
            )
            return None
        plan = self._plan(fn, owner=owner)
        if isinstance(plan, str):
            self.skipped.append((fn.module, f"{fn.impl_type}.{fn.name}()", plan))
            return None
        recv_ty = Ty("user", name=path, by_ref=fn.self_kind != "self")
        recv_name = {"complex": "z", "bigint": "n", "rational": "q"}[IDENTIFIED[path]]
        # `BigInt::nth_root(&self, n: u32)` already uses the short name, so
        # the receiver falls back to the type's own.
        if any(p.split(":")[0].strip() == recv_name for p in plan["params"]):
            recv_name = {"complex": "z_value", "bigint": "bigint", "rational": "rational"}[
                IDENTIFIED[path]
            ]
        if fn.self_kind:
            recv = self.arg_plan(recv_ty, recv_name)
            if recv is None:
                return None
            plan = dict(
                plan,
                params=[f"{recv_name}: {recv['param']}"] + plan["params"],
                pre=recv["pre"] + plan["pre"],
                callargs=plan["callargs"],
                annots=[(recv_name, recv["py"], recv_ty)] + plan["annots"],
            )
            callee = f"{recv['expr'].lstrip('&')}.{fn.name}"
        else:
            callee = f"{self.rust_type(Ty('user', name=path))}::{fn.name}"
        py_name = sanitize_py(fn.name)
        ident = f"pyfn_{IDENTIFIED[path]}_{fn.name}"
        doc = pydoc(fn.doc, f"{path}::{fn.name}")
        head = []
        if plan["needs_py"]:
            head.append("py: Python<'py>")
        head.extend(plan["params"])
        lt = "<'py>" if (plan["needs_py"] or plan["lifetime"]) else ""
        code = "\n".join(
            [
                rust_doc_lines(doc),
                "#[pyfunction]",
                f'#[pyo3(name = "{py_name}", signature = {self._signature_attr(plan["annots"])})]',
                f"pub fn {ident}{lt}({', '.join(head)}) -> PyResult<{plan['ret']['rust']}> {{",
                self._body(plan, callee),
                "}",
            ]
        )
        stub = "\n".join(
            [
                f"def {py_name}{self._stub_sig(plan['annots'], plan['ret']['py'])}:",
                py_doc_literal(doc, "    "),
                "    ...",
            ]
        )
        return Emitted(py_name=py_name, code=code, stub=stub)

    # ── classes ─────────────────────────────────────────────────────

    def emit_class(self, w: Wrapper) -> tuple[str, str]:
        """Returns (rust code, stub text) for one wrapper class."""
        item = w.item
        doc = pydoc(item.doc, item.path)
        flags = [f'name = "{w.py_name}"', f'module = "{w.py_module}"']
        if w.clone:
            flags.append("from_py_object")
        if w.partial_eq or w.simple_enum:
            flags.append("eq")
        if w.simple_enum:
            flags.append("eq_int")
        if w.unsendable:
            flags.append("unsendable")
        derives = ["Clone"] if (w.clone or w.simple_enum) else []
        if w.simple_enum:
            derives += ["Copy", "PartialEq"]
        elif w.partial_eq:
            derives.append("PartialEq")

        rust: list[str] = [rust_doc_lines(doc)]
        rust.append(f"#[pyclass({', '.join(flags)})]")
        if derives:
            rust.append(f"#[derive({', '.join(derives)})]")
        if w.simple_enum:
            rust.append(f"pub enum {w.ident} {{")
            for v in item.variants:
                rust.append(f"    {v.name},")
            rust.append("}")
            rust.append(f"impl {w.ident} {{")
            rust.append(f"    pub fn to_rust(&self) -> {w.rust_path} {{ match self {{")
            for v in item.variants:
                rust.append(f"        Self::{v.name} => {w.rust_path}::{v.name},")
            rust.append("    } }")
            rust.append(f"    pub fn from_rust(v: &{w.rust_path}) -> Self {{ match v {{")
            for v in item.variants:
                rust.append(f"        {w.rust_path}::{v.name} => Self::{v.name},")
            rust.append("    } }")
            rust.append("}")
        else:
            rust.append(f"pub struct {w.ident} {{ pub inner: {w.rust_path} }}")

        rust.append(PRIVATE_IMPLS.get(item.path, "").strip())
        methods, stub_methods = self.emit_methods(w)
        rust.append(f"#[pymethods]\nimpl {w.ident} {{\n" + "\n\n".join(methods) + "\n}")

        if w.coerce_n or w.coerce_rows or w.coerce_seq:
            rust.append(self._arg_adapter(w))

        stub = [f"class {w.py_name}:", py_doc_literal(doc, "    ")]
        stub.extend(stub_methods or ["    ..."])
        return "\n".join(x for x in rust if x), "\n".join(stub)

    def _arg_adapter(self, w: Wrapper) -> str:
        item = w.item
        if w.coerce_n:
            names = [f.name for f in item.fields]
            build = ", ".join(f"{n}: __v[{i}]" for i, n in enumerate(names))
            body = (
                f"        let __v = crate::runtime::coerce::floats_exact(obj, {w.coerce_n}, "
                f'"{w.py_name}")?;\n'
                f"        Ok({w.arg_ident}({w.rust_path} {{ {build} }}))"
            )
        elif w.coerce_rows:
            body = (
                f'        let __rows = crate::runtime::coerce::rows(obj, "{w.py_name}")?;\n'
                "        let __refs: Vec<&[f64]> = __rows.iter().map(Vec::as_slice).collect();\n"
                f"        {w.rust_path}::from_rows(&__refs)\n"
                f"            .map({w.arg_ident})\n"
                "            .map_err(crate::runtime::map_solve)"
            )
        else:
            field_name = item.fields[0].name
            body = (
                "        let __v: Vec<f64> = obj.extract().map_err(|_| "
                f'pyo3::exceptions::PyTypeError::new_err("{w.py_name} expects a sequence of floats"))?;\n'
                f"        Ok({w.arg_ident}({w.rust_path} {{ {field_name}: __v }}))"
            )
        return f"""
/// A `{w.py_name}` argument, or anything that can stand in for one.
pub struct {w.arg_ident}(pub {w.rust_path});

impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for {w.arg_ident} {{
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> Result<Self, pyo3::PyErr> {{
        if let Ok(__w) = obj.extract::<{w.ident}>() {{
            return Ok({w.arg_ident}(__w.inner));
        }}
{body}
    }}
}}
"""

    def emit_methods(self, w: Wrapper) -> tuple[list[str], list[str]]:
        item = w.item
        out: list[str] = []
        stubs: list[str] = []
        taken: set[str] = set()

        methods = [
            f
            for f in self.crate.funcs
            if f.impl_type == item.name and f.module == item.module and not f.impl_trait
        ]
        # `new` is the constructor whether it returns `Self` or
        # `Result<Self, E>`; the fallible form is common enough here that
        # missing it would leave a third of the classes unconstructible.
        ctor_ret = re.compile(
            rf"^(?:Self|{re.escape(item.name)}"
            rf"|Result\s*<\s*(?:Self|{re.escape(item.name)})\s*,.*>)$"
        )
        ctor = next(
            (
                f
                for f in methods
                if f.name == "new" and not f.self_kind and ctor_ret.match(f.ret.strip())
            ),
            None,
        )

        # A constructor from the fields, when the type has no `new` and its
        # fields are all public and all bindable.
        if ctor is None and not w.is_enum and item.kind == "named" and item.fields:
            plans = []
            ok = all(f.public for f in item.fields)
            for f in item.fields:
                ty = parse_type(f.ty, self.res, item.file, item.module)
                if ty.kind in ("callable", "bad") or ty.mutable or ty.by_ref:
                    ok = False
                    break
                p = self.arg_plan(ty, sanitize_py(f.name))
                if p is None:
                    ok = False
                    break
                plans.append((sanitize_py(f.name), f.name, p, ty))
            if ok and plans:
                params = ", ".join(f"{n}: {p['param']}" for n, _r, p, _t in plans)
                pre = "\n".join("        " + s for _n, _r, p, _t in plans for s in p["pre"])
                build = ", ".join(
                    f"{r}: {p['expr'].lstrip('&')}" if not t.by_ref else f"{r}: {p['expr']}"
                    for _n, r, p, t in plans
                )
                sig = ", ".join(n for n, _r, _p, _t in plans)
                fallible = any("?" in st for _n, _r, p, _t in plans for st in p["pre"])
                ret = "PyResult<Self>" if fallible else "Self"
                value = f"Self {{ inner: {w.rust_path} {{ {build} }} }}"
                value = f"Ok({value})" if fallible else value
                out.append(
                    f"    /// Builds a `{w.py_name}` from its fields.\n"
                    f"    #[new]\n    #[pyo3(signature = ({sig}))]\n"
                    f"    fn __new__({params}) -> {ret} {{\n{pre}\n"
                    f"        {value}\n    }}"
                )
                stubs.append(
                    "    def __init__(self, "
                    + ", ".join(f"{n}: {p['py']}" for n, _r, p, _t in plans)
                    + ") -> None: ..."
                )
                taken.add("__new__")

        for fn in methods:
            name = sanitize_py(fn.name)
            if name in taken:
                continue
            is_ctor = fn is ctor
            # A builder -- `&mut self` in, `&mut Self` out, for chaining.
            # Python gets the same object back, so `c.h(0).cx(0, 1)` reads
            # as it does in Rust.
            builder = False
            if fn.self_kind == "&mut self":
                own = re.escape(item.name)
                plain = re.match(rf"^&\s*mut\s+(?:Self|{own})$", fn.ret.strip())
                wrapped = re.match(
                    rf"^Result\s*<\s*&\s*mut\s+(?:Self|{own})\s*,\s*(.+)>$", fn.ret.strip()
                )
                if plain:
                    fn = dataclasses.replace(fn, ret="")
                    builder = True
                elif wrapped:
                    fn = dataclasses.replace(fn, ret=f"Result<(), {wrapped.group(1)}>")
                    builder = True
            plan = self._plan(fn, owner=item)
            if isinstance(plan, str):
                self.skipped.append((item.module, f"{item.name}.{fn.name}()", plan))
                continue
            if w.unsendable and fn.self_kind:
                plan = _without_detach(plan)
            taken.add(name)
            doc = pydoc(fn.doc, f"{item.path}::{fn.name}")
            head: list[str] = []
            recv = ""
            if builder:
                plan = dict(
                    _without_detach(plan),
                    discard="result" if plan["err_map"] else "unit",
                )
                head.append("mut slf: pyo3::PyRefMut<'py, Self>")
                recv = "slf.inner."
            elif fn.self_kind == "&self":
                head.append("&self")
                recv = "self.to_rust()." if w.simple_enum else "self.inner."
            elif fn.self_kind == "&mut self":
                if w.simple_enum:
                    self.skipped.append(
                        (item.module, f"{item.name}.{fn.name}()", "mutates a unit-variant enum")
                    )
                    continue
                head.append("&mut self")
                recv = "self.inner."
            elif fn.self_kind == "self":
                if w.simple_enum:
                    head.append("&self")
                    recv = "self.to_rust()."
                elif not w.clone:
                    self.skipped.append(
                        (item.module, f"{item.name}.{fn.name}()", "takes self by value, not Clone")
                    )
                    continue
                else:
                    head.append("&self")
                    recv = "self.inner.clone()."
            if plan["needs_py"] and not builder:
                head.append("py: Python<'py>")
            head.extend(plan["params"])
            lt = "<'py>" if (plan["needs_py"] or plan["lifetime"] or builder) else ""
            attrs = []
            if is_ctor:
                attrs.append("    #[new]")
            elif not fn.self_kind:
                attrs.append("    #[staticmethod]")
            attrs.append(f'    #[pyo3(signature = {self._signature_attr(plan["annots"])})]')
            if is_ctor:
                callee = f"{w.rust_path}::{fn.name}"
                body = self._body(plan, callee)
                ident = "__new__"
            else:
                callee = f"{recv}{fn.name}" if recv else f"{w.rust_path}::{fn.name}"
                body = self._body(plan, callee)
                ident = sanitize_rust(fn.name)
            ret_rust = plan["ret"]["rust"]
            ret_py = plan["ret"]["py"]
            if builder:
                ret_rust = "pyo3::PyRefMut<'py, Self>"
                ret_py = w.py_name
                body = body[: body.rindex("Ok(())")] + "Ok(slf)"
            rust_name_attr = f'    #[pyo3(name = "{name}")]' if not is_ctor else ""
            block = [rust_doc_lines(doc).replace("///", "    ///")]
            if rust_name_attr:
                block.append(rust_name_attr)
            block.extend(attrs)
            block.append(f"    fn {ident}{lt}({', '.join(head)}) -> PyResult<{ret_rust}> {{")
            block.append("    " + body.replace("\n", "\n    "))
            block.append("    }")
            out.append("\n".join(block))
            if is_ctor:
                stubs.append(
                    "    def __init__(self, "
                    + ", ".join(f"{n}: {p}" for n, p, _t in plan["annots"])
                    + ") -> None: ..."
                )
            elif fn.self_kind:
                stubs.append(
                    f"    def {name}(self"
                    + ("".join(f", {n}: {p}" for n, p, _t in plan["annots"]))
                    + f") -> {ret_py}: ..."
                )
            else:
                stubs.append("    @staticmethod")
                stubs.append(
                    f"    def {name}" + self._stub_sig(plan["annots"], ret_py) + ": ..."
                )

        out.extend(self._operator_methods(w, taken, stubs))
        out.extend(self._field_accessors(w, taken, stubs))
        out.extend(self._const_attrs(w))
        out.extend(f"    {e}" for e in EXTRAS.get(item.path, []))
        out.append(self._repr(w))
        if w.clone:
            out.append(
                "    fn __copy__(&self) -> Self { self.clone() }\n\n"
                "    #[pyo3(signature = (_memo=None))]\n"
                "    fn __deepcopy__(&self, _memo: Option<pyo3::Py<pyo3::PyAny>>) -> Self "
                "{ self.clone() }"
            )
        return out, stubs

    def _repr(self, w: Wrapper) -> str:
        item = w.item
        if w.simple_enum:
            arms = "\n".join(
                f'            Self::{v.name} => "{w.py_name}.{v.name}",' for v in item.variants
            )
            return (
                "    fn __repr__(&self) -> &'static str {\n"
                f"        match self {{\n{arms}\n        }}\n    }}"
            )
        simple = [
            (f.name, ty)
            for f in getattr(item, "fields", [])
            if f.public
            for ty in [parse_type(f.ty, self.res, item.file, item.module)]
            if ty.kind == "prim"
        ]
        if getattr(item, "kind", "") == "named" and 0 < len(simple) <= 6 and len(simple) == len(
            [f for f in item.fields if f.public]
        ):
            fmt = ", ".join(f"{n}={{:?}}" for n, _ in simple)
            args = ", ".join(f"self.inner.{n}" for n, _ in simple)
            return (
                f'    fn __repr__(&self) -> String {{ format!("{w.py_name}({fmt})", {args}) }}'
            )
        if w.debug:
            return (
                '    fn __repr__(&self) -> String { format!("{:?}", self.inner)'
                f'.replacen("{item.name}", "{w.py_name}", 1) }}'
            )
        return f'    fn __repr__(&self) -> String {{ "<{w.py_name}>".to_string() }}'

    def _operator_methods(self, w: Wrapper, taken: set[str], stubs: list[str]) -> list[str]:
        """`__add__` and friends, from the crate's `std::ops` impls.

        The call is written out in full -- `<Vec3 as std::ops::Add>::add(a,
        b)` -- rather than as `a.add(b)`. Method syntax needs the trait in
        scope, and bringing all of `std::ops` into scope would let an
        inherent `add` on some unrelated type silently win the lookup. The
        qualified form names exactly the impl the crate wrote.
        """
        item = w.item
        out = []
        for fn in self.crate.funcs:
            if fn.impl_type != item.name or fn.module != item.module or not fn.impl_trait:
                continue
            trait = fn.impl_trait.split("::")[-1].strip()
            base = trait.split("<")[0].strip()
            dunder = DUNDERS.get((base, fn.name))
            if not dunder or dunder in taken or not w.clone:
                continue
            if fn.self_kind != "self" or len(fn.args) > 1:
                continue
            # `impl Mul<f64> for Vec3` is worth naming; `impl Mul<Vec3> for
            # f64` is the same operator from the other side and Python
            # reaches it through `__rmul__`, which is not generated.
            generic = trait[len(base) :].strip()
            if generic and not re.fullmatch(r"<\s*(f64|f32|i64|i32|u64|u32|usize|isize)\s*>", generic):
                continue
            plan = self._plan(fn, owner=item)
            if isinstance(plan, str) or plan["needs_py"] or plan["lifetime"]:
                continue
            taken.add(dunder)
            plan = dict(plan, detach=False, callargs=["self.inner.clone()"] + plan["callargs"])
            callee = f"<{w.rust_path} as std::ops::{base}{generic}>::{fn.name}"
            body = self._body(plan, callee)
            head = ["&self"] + plan["params"]
            out.append(
                f"    fn {dunder}({', '.join(head)}) -> PyResult<{plan['ret']['rust']}> {{\n"
                + "    "
                + body.replace("\n", "\n    ")
                + "\n    }"
            )
            args = "".join(f", {n}: {p}" for n, p, _t in plan["annots"])
            stubs.append(f"    def {dunder}(self{args}) -> {plan['ret']['py']}: ...")
        return out

    def _field_accessors(self, w: Wrapper, taken: set[str], stubs: list[str]) -> list[str]:
        """Read (and, for numbers, write) access to a struct's public fields.

        A getter hands out a copy, so a field whose type cannot be cloned
        has no getter -- returning a borrow of the wrapper's interior is
        not something the Python object model can express. The Rust
        identifiers are prefixed because a struct is free to have both a
        field `phase` and a method `set_phase`, and the two would collide.
        """
        out = []
        for name, ty in w.fields:
            py_name = sanitize_py(name)
            if py_name in taken:
                continue
            needs_clone = ty.kind not in ("prim", "str")
            if needs_clone and not self.is_clone(ty):
                continue
            access = f"self.inner.{name}.clone()" if needs_clone else f"self.inner.{name}"
            rp = self.ret_plan(ty, access)
            if rp is None:
                continue
            taken.add(py_name)
            needs_py = bool(re.search(r"\bpy\b", rp["conv"]))
            lt = "<'py>" if needs_py else ""
            args = "&self, py: Python<'py>" if needs_py else "&self"
            out.append(
                f"    #[getter]\n"
                f'    #[pyo3(name = "{py_name}")]\n'
                f"    fn py_get_{sanitize_rust(name)}{lt}({args}) -> PyResult<{rp['rust']}> "
                f"{{ Ok({rp['conv']}) }}"
            )
            stubs.append(f"    @property\n    def {py_name}(self) -> {rp['py']}: ...")
            if ty.kind == "prim":
                out.append(
                    f"    #[setter]\n"
                    f'    #[pyo3(name = "{py_name}")]\n'
                    f"    fn py_set_{sanitize_rust(name)}(&mut self, v: {ty.name}) "
                    f"{{ self.inner.{name} = v; }}"
                )
        return out

    def _const_attrs(self, w: Wrapper) -> list[str]:
        out = []
        for name, ty, doc in w.consts:
            rp = self.ret_plan(ty, f"{w.rust_path}::{name}")
            if rp is None or re.search(r"\bpy\b", rp["conv"]):
                continue
            out.append(
                f"    #[classattr]\n"
                f'    #[pyo3(name = "{name}")]\n'
                f"    fn const_{name.lower()}() -> {rp['rust']} {{ {rp['conv']} }}"
            )
        return out


# ── Items the scanner cannot see: macro expansions ──────────────────────


def _add_macro_items(crate: rustscan.Crate) -> None:
    """Add the items that `macro_rules!` produces.

    `rustscan` reads source, not expanded source, so anything a macro
    defines is invisible to it. Two macros in this crate define public
    API: `unit_ctor!` in `units::quantity`, which defines 31 constructors
    on `Quantity`, and `kd_impl!` in `spatial::kdtree`, which defines the
    two k-d tree types outright. Both are regular enough to reconstruct
    from their invocations, which is better than leaving 40-odd public
    items unbound and better than pretending they do not exist.
    """
    qfile = os.path.join(SRC, "units", "quantity.rs")
    if os.path.exists(qfile):
        text = open(qfile, encoding="utf-8").read()
        for m in re.finditer(r'unit_ctor!\(\s*(\w+)\s*,[^,]+,\s*(.+?)\s*,\s*"(.*?)"\s*\)', text):
            crate.funcs.append(
                rustscan.Func(
                    name=m.group(1),
                    module="units::quantity",
                    file=qfile,
                    doc=m.group(3),
                    attrs=[],
                    args=[("v", "f64")],
                    ret="Quantity",
                    generics="",
                    where_clause="",
                    self_kind="",
                    impl_type="Quantity",
                    impl_trait="",
                    is_const=False,
                    is_unsafe=False,
                )
            )

    kfile = os.path.join(SRC, "spatial", "kdtree.rs")
    if os.path.exists(kfile):
        text = open(kfile, encoding="utf-8").read()
        methods = [
            ("build", [("points", "&[{vec}]")], "Self", "", "Builds by recursive median split."),
            ("nearest", [("p", "{vec}")], "Option<(usize, f64)>", "&self",
             "Index of and distance to the nearest stored point."),
            ("k_nearest", [("p", "{vec}"), ("k", "usize")], "Vec<(usize, f64)>", "&self",
             "The `k` nearest stored points, nearest first."),
            ("within_radius", [("p", "{vec}"), ("r", "f64")], "Vec<(usize, f64)>", "&self",
             "Every stored point within `r` of `p`."),
            ("all_pairs_within", [("r", "f64")], "Vec<(usize, usize)>", "&self",
             "Every pair of stored points closer than `r`."),
        ]
        for m in re.finditer(r"kd_impl!\(\s*(\w+)\s*,\s*(\w+)\s*,", text):
            name, vec = m.group(1), m.group(2)
            crate.structs.append(
                rustscan.Struct(
                    name=name,
                    module="spatial::kdtree",
                    file=kfile,
                    doc=f"Median-split k-d tree over `{vec}` points.",
                    attrs=["derive(Debug, Clone)"],
                    fields=[],
                    kind="named",
                    generics="",
                )
            )
            for mname, margs, mret, mself, mdoc in methods:
                crate.funcs.append(
                    rustscan.Func(
                        name=mname,
                        module="spatial::kdtree",
                        file=kfile,
                        doc=mdoc,
                        attrs=[],
                        args=[(a, t.format(vec=vec)) for a, t in margs],
                        ret=mret,
                        generics="",
                        where_clause="",
                        self_kind=mself,
                        impl_type=name,
                        impl_trait="",
                        is_const=False,
                        is_unsafe=False,
                    )
                )


# ── Writing it all out ──────────────────────────────────────────────────

HEADER = """// @generated by bindings/python/generate.py -- do not edit.
//
// Regenerate with:  python3 bindings/python/generate.py
"""

ALLOWS = """#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(deprecated)]
#![allow(rustdoc::all)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
"""


class Build(Emitter):
    def run(self, out_rs: str, out_py: str) -> None:
        self.identified_fns = 0
        modules = self._module_list()
        free_by_module: dict[str, list[Emitted]] = {m: [] for m in modules}
        for fn in self.crate.funcs:
            if fn.impl_type or fn.impl_trait:
                continue
            if fn.module not in free_by_module:
                continue
            e = self.emit_free(fn)
            if e is not None:
                free_by_module[fn.module].append(e)

        # The three identified types have no class, so their methods are
        # emitted as functions in the module that defines them.
        for path in IDENTIFIED:
            owner = self.res.by_path.get(path)
            if owner is None or owner.module not in free_by_module:
                continue
            taken = {e.py_name for e in free_by_module[owner.module]}
            for fn in self.crate.funcs:
                if fn.impl_type != owner.name or fn.module != owner.module or fn.impl_trait:
                    continue
                e = self.emit_identified_method(fn, path)
                if e is None:
                    continue
                self.identified_fns += 1
                if e.py_name in taken:
                    # A name the module already uses: qualify rather than
                    # shadow. `fractals.norm` stays the free function;
                    # the method becomes `fractals.complex_norm`.
                    prefixed = f"{IDENTIFIED[path]}_{e.py_name}"
                    if prefixed in taken:
                        continue
                    e = Emitted(
                        py_name=prefixed,
                        code=e.code.replace(f'name = "{e.py_name}"', f'name = "{prefixed}"'),
                        stub=e.stub.replace(f"def {e.py_name}(", f"def {prefixed}(", 1),
                    )
                taken.add(e.py_name)
                free_by_module[owner.module].append(e)

        consts_by_module: dict[str, list] = {m: [] for m in modules}
        for c in self.crate.consts:
            if c.owner or c.module not in consts_by_module:
                continue
            ty = parse_type(c.ty, self.res, c.file, c.module)
            rp = self.ret_plan(ty, f"rust_physics_engine::{c.module}::{c.name}")
            if rp is None or rp["fallible"] or re.search(r"\bpy\b", rp["conv"]):
                self.skipped.append((c.module, c.name, f"constant of type `{c.ty}`"))
                continue
            consts_by_module[c.module].append((c, rp))

        # Rust keeps types, values and modules in separate namespaces;
        # Python does not. `special::gamma` is both a module and, through a
        # re-export, a function, and only one of them can be
        # `special.gamma`. The module wins, because dropping it would take
        # `special.gamma.gamma_p` and everything beside it with it.
        self._resolve_shadowing(modules, free_by_module, consts_by_module)
        self.aliases = self._compute_aliases(modules, free_by_module)

        os.makedirs(out_rs, exist_ok=True)
        self._write_types(out_rs)
        for mod in modules:
            self._write_module(out_rs, mod, free_by_module[mod], consts_by_module[mod])
        self._write_mod_rs(out_rs, modules, consts_by_module)
        self._write_python(out_py, modules, free_by_module, consts_by_module)
        self._write_coverage(modules, free_by_module, consts_by_module)

    def _child_modules(self, modules) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for m in modules:
            parent, _, leaf = m.rpartition("::")
            if parent:
                out.setdefault(parent, set()).add(leaf)
        return out

    def _resolve_shadowing(self, modules, free_by_module, consts_by_module) -> None:
        """Drop anything a submodule of the same name would shadow."""
        children = self._child_modules(modules)
        for mod, names in children.items():
            kept = [e for e in free_by_module.get(mod, []) if e.py_name not in names]
            for e in free_by_module.get(mod, []):
                if e.py_name in names:
                    self.skipped.append(
                        (mod, f"{e.py_name}()", f"shadowed by the submodule `{mod}::{e.py_name}`")
                    )
            free_by_module[mod] = kept
            for w in list(self.class_of_module.get(mod, [])):
                if w.py_name in names:
                    self.class_of_module[mod].remove(w)
                    self.skipped.append(
                        (mod, w.py_name, f"shadowed by the submodule `{mod}::{w.py_name}`")
                    )
            consts_by_module[mod] = [
                (c, rp) for c, rp in consts_by_module.get(mod, []) if c.name not in names
            ]

        # A class and a function of the same name in one module would also
        # collide. Rust's naming conventions make it unlikely rather than
        # impossible, and "unlikely" is not a thing to leave unchecked.
        for mod in modules:
            class_names = {w.py_name for w in self.class_of_module.get(mod, [])}
            clash = [e for e in free_by_module.get(mod, []) if e.py_name in class_names]
            for e in clash:
                free_by_module[mod].remove(e)
                self.skipped.append(
                    (mod, f"{e.py_name}()", f"a class in `{mod}` already has that name")
                )

    def _compute_aliases(self, modules, free_by_module) -> dict[str, list[tuple[str, str]]]:
        """Where a `pub use` re-export puts a name, put the Python name too.

        `linalg` re-exports `Matrix` from `linalg::matrix` and `solve` from
        `linalg::lu`, and the crate's own documentation refers to them by
        the short path. A binding that only offered the long one would send
        readers of those docs to an `AttributeError`.
        """
        modset = set(modules)
        children = self._child_modules(modules)
        fn_names = {m: {e.py_name for e in v} for m, v in free_by_module.items()}
        class_names = {
            m: {w.py_name for w in ws} for m, ws in self.class_of_module.items()
        }
        out: dict[str, list[tuple[str, str]]] = {}
        for mod, entries in sorted(self.crate.pub_uses.items()):
            if mod not in modset:
                continue
            # A submodule of the same name is already there and must stay.
            here = fn_names.get(mod, set()) | class_names.get(mod, set()) | children.get(
                mod, set()
            )
            for name, target in sorted(entries.items()):
                # `use matrix::Matrix` inside `linalg` is a relative path.
                for full in (target, f"{mod}::{target}"):
                    src_mod = "::".join(full.split("::")[:-1])
                    tail = full.split("::")[-1]
                    if src_mod == mod or src_mod not in modset or tail != name:
                        continue
                    if name in here:
                        if name in children.get(mod, set()):
                            self.skipped.append(
                                (
                                    mod,
                                    name,
                                    f"re-export shadowed by the submodule `{mod}::{name}`;"
                                    f" reach it at `{src_mod.replace('::', '.')}.{name}`",
                                )
                            )
                        break
                    if name in fn_names.get(src_mod, set()) or name in class_names.get(
                        src_mod, set()
                    ):
                        out.setdefault(mod, []).append((name, src_mod))
                        here.add(name)
                        break
        return out

    def _module_list(self) -> list[str]:
        mods: set[str] = set()
        for fn in self.crate.funcs:
            if not fn.impl_type:
                mods.add(fn.module)
        for w in self.wrappers.values():
            mods.add(w.item.module)
        for c in self.crate.consts:
            if not c.owner:
                mods.add(c.module)
        # The three identified types have no class, and their module may
        # contain nothing else -- `exact::bigint` is only `BigInt`. Their
        # methods still become functions there, so the module has to exist.
        for path in IDENTIFIED:
            item = self.res.by_path.get(path)
            if item is not None:
                mods.add(item.module)
        # Every ancestor has to exist so the tree can be attached.
        full = set()
        for m in mods:
            parts = m.split("::")
            for i in range(len(parts)):
                full.add("::".join(parts[: i + 1]))
        full.discard("")
        return sorted(full, key=lambda m: (m.count("::"), m))

    @staticmethod
    def _rs_name(mod: str) -> str:
        return "m_" + mod.replace("::", "__")

    # ── Rust ────────────────────────────────────────────────────────

    def _write_types(self, out_rs: str) -> None:
        tdir = os.path.join(out_rs, "types")
        os.makedirs(tdir, exist_ok=True)
        groups: dict[str, list[Wrapper]] = {}
        for w in sorted(self.wrappers.values(), key=lambda w: w.item.path):
            groups.setdefault(w.item.module.split("::")[0], []).append(w)
        self._class_stubs: dict[str, list[str]] = {}
        names = []
        for top, ws in sorted(groups.items()):
            body = [HEADER, ALLOWS, "use pyo3::prelude::*;\n"]
            for w in ws:
                code, stub = self.emit_class(w)
                body.append(code)
                self._class_stubs[w.item.path] = stub
            with open(os.path.join(tdir, f"{top}.rs"), "w") as fh:
                fh.write("\n\n".join(body) + "\n")
            names.append(top)
        mod = [HEADER, ALLOWS]
        for n in names:
            mod.append(f"mod {n};\npub use {n}::*;")
        with open(os.path.join(tdir, "mod.rs"), "w") as fh:
            fh.write("\n".join(mod) + "\n")

    def _write_module(self, out_rs: str, mod: str, fns: list[Emitted], consts: list) -> None:
        path = os.path.join(out_rs, self._rs_name(mod) + ".rs")
        body = [HEADER, ALLOWS, "use pyo3::prelude::*;\nuse pyo3::types::PyModule;\n"]
        body.extend(e.code for e in fns)
        reg = [
            "/// Registers this module's contents.",
            "pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {",
            "    let _ = (py, m);",
        ]
        for e in fns:
            reg.append(f"    m.add_function(wrap_pyfunction!({self._ident_of(e)}, m)?)?;")
        for c, rp in consts:
            reg.append(f'    m.add("{c.name}", {rp["conv"]})?;')
        for w in sorted(self.class_of_module.get(mod, []), key=lambda w: w.py_name):
            reg.append(f"    m.add_class::<crate::generated::types::{w.ident}>()?;")
        reg.append("    Ok(())")
        reg.append("}")
        body.append("\n".join(reg))
        with open(path, "w") as fh:
            fh.write("\n\n".join(body) + "\n")

    @staticmethod
    def _ident_of(e: Emitted) -> str:
        return e.code.split("pub fn ", 1)[1].split("<", 1)[0].split("(", 1)[0]

    def _write_mod_rs(self, out_rs: str, modules: list[str], consts_by_module) -> None:
        lines = [HEADER, ALLOWS, "use pyo3::prelude::*;", "use pyo3::types::PyModule;",
                 "use std::collections::HashMap;", "", "pub mod types;"]
        for m in modules:
            lines.append(f"pub mod {self._rs_name(m)};")
        lines.append("")
        lines.append("/// Builds the module tree under the extension module.")
        lines.append(
            "pub fn register<'py>(py: Python<'py>, root: &Bound<'py, PyModule>) -> PyResult<()> {"
        )
        lines.append("    let mut mods: HashMap<&'static str, Bound<'py, PyModule>> = HashMap::new();")
        for m in modules:
            short = m.split("::")[-1]
            parent = "::".join(m.split("::")[:-1])
            doc = clean_markup(self.crate.module_docs.get(m, "")).replace(
                "\\", "\\\\"
            ).replace('"', '\\"')
            lines.append("    {")
            lines.append(f'        let sub = PyModule::new(py, "{PKG}.{m.replace("::", ".")}")?;')
            if doc:
                lines.append(f'        sub.setattr("__doc__", "{doc}")?;')
            lines.append(f"        {self._rs_name(m)}::register(py, &sub)?;")
            if parent:
                lines.append(f'        mods["{parent}"].add("{short}", &sub)?;')
            else:
                lines.append(f'        root.add("{short}", &sub)?;')
            lines.append(f'        mods.insert("{m}", sub);')
            lines.append("    }")
        for mod, entries in self.aliases.items():
            for name, src_mod in entries:
                lines.append(
                    f'    {{ let v = mods["{src_mod}"].getattr("{name}")?; '
                    f'mods["{mod}"].add("{name}", v)?; }}'
                )
        lines.append("    let names: Vec<&str> = vec![")
        lines.append(
            "        " + ", ".join(f'"{m.replace("::", ".")}"' for m in modules)
        )
        lines.append("    ];")
        lines.append('    root.add("__submodules__", names)?;')
        lines.append("    Ok(())")
        lines.append("}")
        with open(os.path.join(out_rs, "mod.rs"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # ── Python ──────────────────────────────────────────────────────

    STUB_PRELUDE = (
        "# @generated by bindings/python/generate.py -- do not edit.\n"
        "from __future__ import annotations\n"
        "from collections.abc import Callable, Sequence\n"
        "from fractions import Fraction\n"
        "from typing import Any, Optional\n"
    )

    def _write_python(self, out_py: str, modules, free_by_module, consts_by_module) -> None:
        # Wipe the generated subtree, keeping the hand-written files.
        for entry in sorted(os.listdir(out_py)) if os.path.isdir(out_py) else []:
            full = os.path.join(out_py, entry)
            if os.path.isdir(full):
                shutil.rmtree(full)
            elif entry.endswith(".pyi") and entry != "py.typed":
                os.remove(full)
        os.makedirs(out_py, exist_ok=True)

        unique = {n: ws[0] for n, ws in self.class_by_name.items() if len(ws) == 1}
        for mod in modules:
            parts = mod.split(".") if "." in mod else mod.split("::")
            rel = os.path.join(*parts)
            children = [m for m in modules if m.startswith(mod + "::") and m.count("::") == mod.count("::") + 1]
            if children:
                target = os.path.join(out_py, rel, "__init__.pyi")
                os.makedirs(os.path.join(out_py, rel), exist_ok=True)
            else:
                target = os.path.join(out_py, rel + ".pyi")
                os.makedirs(os.path.dirname(target), exist_ok=True)

            classes = sorted(self.class_of_module.get(mod, []), key=lambda w: w.py_name)
            fns = free_by_module.get(mod, [])
            chunks = [self._class_stubs[w.item.path] for w in classes]
            chunks += [e.stub for e in fns]
            chunks += [f"{c.name}: {rp['py']}" for c, rp in consts_by_module.get(mod, [])]
            aliased = self.aliases.get(mod, [])
            local = {w.py_name for w in classes} | {n for n, _ in aliased}
            imports = self._stub_imports("\n".join(chunks), mod, local, unique)
            imports += [
                f"from {PKG}.{src.replace('::', '.')} import {name} as {name}"
                for name, src in aliased
            ]
            # The docstring has to be the first statement in the file, or
            # it is not the module's docstring at all.
            doc = clean_markup(self.crate.module_docs.get(mod, ""))
            head = [py_doc_literal([doc], "")] if doc else []
            head.append(self.STUB_PRELUDE)
            if children:
                head.append("from . import " + ", ".join(sorted(c.split("::")[-1] for c in children)))
            head.extend(imports)
            with open(target, "w") as fh:
                fh.write("\n".join(head) + "\n\n" + "\n\n".join(chunks) + "\n")

        with open(os.path.join(out_py, "py.typed"), "w") as fh:
            fh.write("")
        self._write_init(out_py, modules)

    def _stub_imports(self, text: str, mod: str, local: set[str], unique) -> list[str]:
        used = set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text))
        out = []
        for name in sorted(used):
            if name in local or name in ("Any", "Optional", "Sequence", "Callable", "Fraction", "None"):
                continue
            w = unique.get(name)
            if w is None:
                continue
            out.append(f"from {PKG}.{w.item.module.replace('::', '.')} import {name}")
        return out

    def _write_init(self, out_py: str, modules) -> None:
        top = [m for m in modules if "::" not in m]
        text = f'''"""{PKG}: Python bindings for the rust_physics_engine library.

Generated from the Rust source by ``bindings/python/generate.py``. Every
module here mirrors a Rust module of the same name, and every function
keeps the name, the argument order and the units it has in Rust.

    >>> import {PKG} as rpe
    >>> import math
    >>> round(rpe.classical.projectile_range(20.0, math.pi / 4, 9.80665), 4)
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
    {PKG}.linalg.lu``, or for ``from {PKG}.linalg import lu``:
    both go through ``sys.modules``, and nothing has put the submodules
    there. This does, once, at import.
    """
    installed = []
    for dotted in _core.__submodules__:
        obj = _core
        for part in dotted.split("."):
            obj = getattr(obj, part)
        _sys.modules[f"{{__name__}}.{{dotted}}"] = obj
        if "." not in dotted:
            globals()[dotted] = obj
            installed.append(dotted)
    return installed


_MODULES = _install()

#: The physical and mathematical constants, as a module of their own.
constants = _sys.modules[f"{{__name__}}.math.constants"]

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
'''
        with open(os.path.join(out_py, "__init__.py"), "w") as fh:
            fh.write(text)

        stub = [
            self.STUB_PRELUDE,
            "from . import " + ", ".join(sorted(top)),
            "",
            "__version__: str",
            "constants: Any",
            "",
            "class PhysicsError(Exception): ...",
            "class InvalidArgumentError(PhysicsError): ...",
            "class SolverError(PhysicsError): ...",
            "class SingularMatrixError(SolverError): ...",
            "class NotPositiveDefiniteError(SolverError): ...",
            "class ConvergenceError(SolverError):",
            "    iterations: int",
            "    residual: float",
            "class DimensionMismatchError(SolverError):",
            "    expected: int",
            "    got: int",
            "class GeometryError(PhysicsError): ...",
            "class DegenerateGeometryError(GeometryError): ...",
            "class NotManifoldError(GeometryError): ...",
            "class EmptyInputError(GeometryError): ...",
            "class UnitsError(PhysicsError): ...",
        ]
        with open(os.path.join(out_py, "__init__.pyi"), "w") as fh:
            fh.write("\n".join(stub) + "\n")

    # ── the coverage report ─────────────────────────────────────────

    def _write_coverage(self, modules, free_by_module, consts_by_module) -> None:
        total_fns = sum(1 for f in self.crate.funcs if not f.impl_type and not f.impl_trait)
        bound_fns = sum(len(v) for v in free_by_module.values()) - self.identified_fns
        total_methods = sum(1 for f in self.crate.funcs if f.impl_type and not f.impl_trait)
        bound_classes = len(self.wrappers)
        skipped_methods = sum(1 for _m, item, _r in self.skipped if "." in item)
        by_reason: dict[str, int] = {}
        for _m, _i, reason in self.skipped:
            key = re.sub(r"`[^`]*`", "`...`", reason)
            by_reason[key] = by_reason.get(key, 0) + 1

        lines = [
            "<!-- @generated by bindings/python/generate.py -- do not edit. -->",
            "",
            "# What is bound, and what is not",
            "",
            "Generated alongside the bindings themselves, so it cannot drift",
            "from them. Every item the generator could not bind is listed at",
            "the bottom with the reason.",
            "",
            "## Totals",
            "",
            "| | in Rust | bound |",
            "|---|---:|---:|",
            f"| Free functions | {total_fns} | {bound_fns} |",
            f"| Methods | {total_methods} | {total_methods - skipped_methods} |",
            f"| Classes | {len(self.crate.structs) + len(self.crate.enums)} | {bound_classes} |",
            f"| Constants | {sum(1 for c in self.crate.consts if not c.owner)} | "
            f"{sum(len(v) for v in consts_by_module.values())} |",
            "",
            f"Of those methods, {self.identified_fns} belong to `Complex`, `BigInt` and",
            "`Rational`. Those three have no wrapper class -- they cross over as",
            "Python's own `complex`, `int` and `Fraction` -- so their methods appear",
            "as functions in the module that defines the type:",
            "`exact.bigint.mod_pow(base, exponent, modulus)` rather than",
            "`BigInt.mod_pow`.",
            "",
            f"The tree comes to {len(modules)} Python modules.",
            "",
            "## Why the rest is not bound",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {n} |")
        lines += ["", "## Per module", "", "| module | functions bound | classes |", "|---|---:|---:|"]
        for m in modules:
            nf = len(free_by_module.get(m, []))
            nc = len(self.class_of_module.get(m, []))
            if nf or nc:
                lines.append(f"| `{m}` | {nf} | {nc} |")
        lines += ["", "## Every unbound item", "", "| module | item | reason |", "|---|---|---|"]
        for m, item, reason in sorted(self.skipped):
            lines.append(f"| `{m}` | `{item}` | {reason} |")
        with open(os.path.join(HERE, "COVERAGE.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the committed output is stale")
    args = ap.parse_args()

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            rs = os.path.join(tmp, "generated")
            py = os.path.join(tmp, PKG)
            os.makedirs(py)
            Build().run(rs, py)
            stale = []
            for old, new in ((OUT_RS, rs), (OUT_PY, py)):
                stale += _diff_trees(old, new)
            if stale:
                print("The committed bindings are stale. Re-run:")
                print("    python3 bindings/python/generate.py")
                for s in stale[:40]:
                    print("   ", s)
                return 1
        print("bindings are up to date")
        return 0

    Build().run(OUT_RS, OUT_PY)
    print(f"wrote {OUT_RS} and {OUT_PY}")
    return 0


def _diff_trees(a: str, b: str) -> list[str]:
    def listing(root: str) -> dict[str, str]:
        out = {}
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                full = os.path.join(dirpath, f)
                out[os.path.relpath(full, root)] = open(full, encoding="utf-8").read()
        return out

    old, new = listing(a), listing(b)
    diffs = []
    for k in sorted(set(old) | set(new)):
        if k not in old:
            diffs.append(f"missing: {k}")
        elif k not in new:
            diffs.append(f"stale:   {k}")
        elif old[k] != new[k]:
            diffs.append(f"differs: {k}")
    return diffs


if __name__ == "__main__":
    raise SystemExit(main())
