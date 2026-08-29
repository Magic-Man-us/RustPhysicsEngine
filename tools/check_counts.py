#!/usr/bin/env python3
"""Check counts quoted in prose against the files that derive them.

MODULE_MAP.md and COVERAGE.md disagree by design -- the map counts a
macro-generated item once, the generator counts what it emitted -- so each
claim names which of them owns it.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODULE_MAP = "docs/MODULE_MAP.md"
COVERAGE = "bindings/python/COVERAGE.md"


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def plain(n: str) -> int:
    return int(n.replace(",", ""))


def module_map_facts() -> dict[str, int]:
    text = read(MODULE_MAP)
    pairs = {
        "modules": r"\*\*([\d,]+) modules\*\*",
        "top_level": r"\*\*([\d,]+) public top-level modules\*\*",
        "files": r"\*\*([\d,]+) files\*\*",
        "map_functions": r"\*\*([\d,]+) public functions\*\*",
        "map_methods": r"\*\*([\d,]+) public methods\*\*",
        "map_types": r"\*\*([\d,]+) public types\*\*",
    }
    out = {}
    for key, pat in pairs.items():
        m = re.search(pat, text)
        if not m:
            sys.exit(f"{MODULE_MAP}: could not find the {key} figure -- regenerate it")
        out[key] = plain(m.group(1))
    return out


def coverage_facts() -> dict[str, int]:
    text = read(COVERAGE)
    rows = {
        "functions": "Free functions",
        "methods": "Methods",
        "classes": "Classes",
        "constants": "Constants",
    }
    out = {}
    for key, label in rows.items():
        m = re.search(rf"^\|\s*{label}\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text, re.M)
        if not m:
            sys.exit(f"{COVERAGE}: could not find the {label!r} row -- regenerate it")
        out[f"rust_{key}"] = int(m.group(1))
        out[f"bound_{key}"] = int(m.group(2))
    return out


# A literal space in a pattern means any whitespace: these sentences are
# hard wrapped and a reflow must not read as a missing claim.
CLAIMS: list[tuple[str, str, str]] = [
    ("Cargo.toml", r"([\d,]+) public functions", "map_functions"),
    ("Cargo.toml", r"public functions across ([\d,]+) modules", "modules"),
    ("README.md", r"the ([\d,]+) modules", "modules"),
    ("docs/GUIDE.md", r"([\d,]+) modules", "modules"),
    ("bindings/python/pyproject.toml", r"([\d,]+) functions across", "bound_functions"),
    ("bindings/python/pyproject.toml", r"functions across ([\d,]+) domains", "top_level"),
    ("bindings/python/README.md", r"^([\d,]+) functions, ", "bound_functions"),
    ("bindings/python/README.md", r"functions, ([\d,]+) methods", "bound_methods"),
    ("bindings/python/README.md", r"methods, ([\d,]+) classes", "bound_classes"),
    ("bindings/python/README.md", r"classes and ([\d,]+) constants", "bound_constants"),
    ("bindings/python/README.md", r"constants across ([\d,]+) domains", "top_level"),
    ("bindings/python/README.md", r"^([\d,]+) of the library's", "bound_functions"),
    ("bindings/python/README.md", r"library's ([\d,]+) free functions", "rust_functions"),
    ("bindings/python/README.md", r"free functions, ([\d,]+) of its", "bound_methods"),
    ("bindings/python/README.md", r"of its ([\d,]+) methods", "rust_methods"),
    ("bindings/python/README.md", r"methods, ([\d,]+) of its", "bound_classes"),
    ("bindings/python/README.md", r"of its ([\d,]+) types", "rust_classes"),
    ("bindings/python/README.md", r"all ([\d,]+) of its constants", "bound_constants"),
    ("bindings/python/README.md", r"constants, across ([\d,]+) modules", "modules"),
]


def compile_claim(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern.replace(" ", r"\s+"), re.M)


def main() -> int:
    facts = module_map_facts() | coverage_facts()
    problems: list[str] = []
    for rel, pattern, key in CLAIMS:
        text = read(rel)
        m = compile_claim(pattern).search(text)
        if not m:
            problems.append(f"{rel}: no text matching {pattern!r} -- the sentence moved")
            continue
        found, want = plain(m.group(1)), facts[key]
        if found != want:
            line = text[: m.start(1)].count("\n") + 1
            problems.append(
                f"{rel}:{line}: says {found:,} but {key} is {want:,}"
            )
    if problems:
        print("count check failed:", file=sys.stderr)
        for p in problems:
            print(f"    {p}", file=sys.stderr)
        print(
            f"\nthe figures come from {MODULE_MAP} and {COVERAGE}, both generated;"
            "\nfix the prose, not those files",
            file=sys.stderr,
        )
        return 1
    print(f"{len(CLAIMS)} quoted counts agree with {MODULE_MAP} and {COVERAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
