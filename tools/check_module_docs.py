#!/usr/bin/env python3
"""Fail if any source file lacks a `//!` module summary.

Rustdoc does not warn about a missing module doc -- `missing_docs` covers
items, not modules -- so an undocumented module shows up as a blank page
in the generated documentation and as an empty row in
`docs/MODULE_MAP.md`. Nothing catches that but this.

A file passes when its first line of substance (ignoring blank lines and
inner attributes such as `#![allow(...)]`) is a `//!` comment.

    python3 tools/check_module_docs.py
"""

from __future__ import annotations

import os
import sys

SRC = "src"


def has_module_doc(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#!["):
                continue
            return stripped.startswith("//!")
    return False  # an empty file documents nothing


def main() -> int:
    if not os.path.isdir(SRC):
        print("run from the repository root", file=sys.stderr)
        return 2

    missing, total = [], 0
    for root, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if not name.endswith(".rs"):
                continue
            total += 1
            path = os.path.join(root, name)
            if not has_module_doc(path):
                missing.append(path)

    if missing:
        print(f"{len(missing)} of {total} source files have no `//!` module doc:",
              file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"all {total} source files carry a module doc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
