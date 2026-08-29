"""The generator itself, and the promises the package makes about itself.

The bindings are derived from the library's source, so the parts worth
testing directly are the ones a bad derivation would get wrong quietly:
the Rust reader's view of the source, and whether the shipped stubs and
the shipped module still agree.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BINDINGS = os.path.dirname(HERE)
sys.path.insert(0, BINDINGS)

import rustscan  # noqa: E402


def scan(source: str):
    """Run the scanner over a snippet, as if it were a file."""
    import tempfile

    crate = rustscan.Crate()
    with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as fh:
        fh.write(source)
        path = fh.name
    try:
        rustscan.parse_file(path, "demo", crate)
    finally:
        os.unlink(path)
    return crate


def test_the_scanner_finds_a_function_and_its_doc():
    crate = scan(
        """
        /// Adds two numbers.
        pub fn add(a: f64, b: f64) -> f64 { a + b }
        """
    )
    assert len(crate.funcs) == 1
    fn = crate.funcs[0]
    assert fn.name == "add"
    assert fn.args == [("a", "f64"), ("b", "f64")]
    assert fn.ret == "f64"
    assert fn.doc == "Adds two numbers."


def test_a_brace_inside_a_doctest_does_not_unbalance_the_item():
    """Doc comments are masked before brace matching.

    A `{` inside a doctest would otherwise close the wrong block and lose
    every item after it -- silently, because the file still parses.
    """
    crate = scan(
        """
        /// ```
        /// let m = Foo { x: 1 };
        /// ```
        pub fn first() -> f64 { 1.0 }

        pub fn second() -> f64 { 2.0 }
        """
    )
    assert [f.name for f in crate.funcs] == ["first", "second"]


def test_a_semicolon_in_a_return_type_is_not_the_end_of_a_declaration():
    crate = scan("pub fn corners() -> [f64; 6] { [0.0; 6] }\npub fn after() -> f64 { 1.0 }")
    assert [f.name for f in crate.funcs] == ["corners", "after"]
    assert crate.funcs[0].ret == "[f64; 6]"


def test_test_modules_are_not_part_of_the_public_api():
    crate = scan(
        """
        pub fn real() -> f64 { 1.0 }

        #[cfg(test)]
        mod tests {
            pub fn helper() -> f64 { 2.0 }
        }
        """
    )
    assert [f.name for f in crate.funcs] == ["real"]


def test_a_string_containing_a_brace_does_not_confuse_the_scanner():
    crate = scan(
        """
        pub fn label() -> String { "}{".to_string() }
        pub fn after() -> f64 { 1.0 }
        """
    )
    assert [f.name for f in crate.funcs] == ["label", "after"]


def test_methods_are_attributed_to_their_impl_type():
    crate = scan(
        """
        pub struct Thing { pub x: f64 }

        impl Thing {
            /// Doubles it.
            pub fn double(&self) -> f64 { self.x * 2.0 }
        }
        """
    )
    assert crate.structs[0].name == "Thing"
    assert crate.structs[0].fields[0].name == "x"
    method = crate.funcs[0]
    assert method.impl_type == "Thing"
    assert method.self_kind == "&self"


def test_trait_impls_are_recorded_so_operators_can_be_found():
    crate = scan(
        """
        pub struct V { pub x: f64 }

        impl std::ops::Add for V {
            type Output = V;
            fn add(self, rhs: V) -> V { V { x: self.x + rhs.x } }
        }
        """
    )
    add = [f for f in crate.funcs if f.name == "add"][0]
    assert add.impl_trait.endswith("Add")
    assert add.impl_type == "V"


def test_use_statements_resolve_short_names():
    crate = scan("use crate::math::{Vec2, Vec3 as V3};\npub fn f(a: Vec3) -> f64 { 0.0 }")
    uses = list(crate.uses.values())[0]
    assert uses["Vec2"] == "math::Vec2"
    assert uses["V3"] == "math::Vec3"


# ── the shipped package ─────────────────────────────────────────────────


def test_the_package_declares_itself_typed():
    assert os.path.exists(os.path.join(BINDINGS, "python", "numeria", "py.typed"))


def test_every_module_has_a_stub_and_the_stub_agrees():
    """`check_stubs.py` is the real check; this runs it."""
    result = subprocess.run(
        [sys.executable, os.path.join(BINDINGS, "check_stubs.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_committed_bindings_match_the_library_source():
    """Regenerating must be a no-op, or what is committed is stale."""
    result = subprocess.run(
        [sys.executable, os.path.join(BINDINGS, "generate.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_coverage_report_exists_and_reports_most_of_the_api():
    path = os.path.join(BINDINGS, "COVERAGE.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "Free functions" in text
    # Pull the bound/total counts out of the totals table.
    import re

    row = re.search(r"\| Free functions \| (\d+) \| (\d+) \|", text)
    assert row, "the totals table changed shape"
    total, bound = int(row.group(1)), int(row.group(2))
    assert bound / total > 0.95, f"only {bound} of {total} free functions are bound"
