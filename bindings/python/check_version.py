#!/usr/bin/env python3
"""Check that the library, the bindings and the tag all claim one version.

A publish is the one operation here that cannot be taken back: PyPI does
not allow a version to be re-uploaded, even after a delete. So the thing
worth checking before it happens is the thing that is easy to get wrong --
tagging `v0.2.0` while `Cargo.toml` still says `0.1.0`, and publishing
0.1.0 under a name that can never be used again.

The Python package is the same library seen from Python, so its version
tracks the crate's rather than moving on its own.

    python3 bindings/python/check_version.py            # the two agree
    python3 bindings/python/check_version.py v0.2.0     # ...and match the tag
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def package_version(cargo_toml: str) -> str | None:
    """The `version` of the `[package]` table, ignoring every other table."""
    with open(cargo_toml, encoding="utf-8") as fh:
        text = fh.read()
    in_package = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_package = stripped == "[package]"
            continue
        if not in_package:
            continue
        m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
        if m:
            return m.group(1)
    return None


def main(argv: list[str]) -> int:
    crate = package_version(os.path.join(ROOT, "Cargo.toml"))
    bindings = package_version(os.path.join(HERE, "Cargo.toml"))

    problems = []
    if crate is None:
        problems.append("the crate's Cargo.toml has no [package] version")
    if bindings is None:
        problems.append("the bindings' Cargo.toml has no [package] version")
    if crate and bindings and crate != bindings:
        problems.append(
            f"the crate is {crate} but the bindings are {bindings}; "
            "the Python package is the same library, so the two move together"
        )

    if len(argv) > 1:
        tag = argv[1].removeprefix("refs/tags/")
        wanted = tag.removeprefix("v")
        if not re.fullmatch(r"\d+\.\d+\.\d+([-.+][0-9A-Za-z.-]+)?", wanted):
            problems.append(f"the tag {tag!r} is not a version tag of the form vX.Y.Z")
        elif bindings and wanted != bindings:
            problems.append(
                f"the tag says {wanted} but Cargo.toml says {bindings}. "
                "PyPI will not let a version be re-uploaded, so publishing the "
                "wrong one is not recoverable -- bump Cargo.toml, or retag"
            )

    if problems:
        print("version check failed:")
        for p in problems:
            print(f"    {p}")
        return 1

    described = f"{crate}" + (f", matching tag {argv[1]}" if len(argv) > 1 else "")
    print(f"crate and bindings agree on {described}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
