#!/usr/bin/env python3
"""Generate docs/MODULE_MAP.md from the source tree.

The map is generated rather than written by hand so that it cannot drift
out of date: `--check` re-derives it and fails if the committed file
differs, which is what CI runs.

Every figure in the output comes from parsing the sources -- module
summary from the first `//!` line, counts of public items, lines of
code. Nothing is transcribed.

    python3 tools/gen_module_map.py            # rewrite docs/MODULE_MAP.md
    python3 tools/gen_module_map.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import os
import re
import sys

SRC = "src"
OUT = "docs/MODULE_MAP.md"

# Groupings for the top-level listing. Anything not named here is
# collected under "Other" so a new module shows up rather than vanishing.
AREAS: list[tuple[str, list[str]]] = [
    ("Numeric foundations", ["core", "math", "linalg", "numerical", "special", "error"]),
    ("Exact and symbolic", ["exact", "discrete", "graph", "codes"]),
    ("Classical mechanics", ["classical", "gravitation", "solid_mechanics",
                             "continuum_mechanics", "resonance", "geophysics"]),
    ("Thermal and statistical", ["thermodynamics", "statistical_mechanics", "radiation"]),
    ("Electromagnetism", ["electromagnetism", "electronics", "rf", "photonics",
                          "plasma", "magnetohydrodynamics"]),
    ("Waves and signals", ["waves", "optics", "acoustics", "transforms", "dsp",
                           "signal_processing", "audio"]),
    ("Fluids", ["fluids", "cfd", "fluid_instabilities", "propulsion"]),
    ("Modern physics", ["relativity", "general_relativity", "quantum",
                        "particle_physics", "nuclear", "neutronics"]),
    ("Space", ["astrophysics"]),
    ("PDE solvers", ["fem", "sim", "fields", "vector_calculus"]),
    ("Chemistry and life", ["chemistry", "biophysics"]),
    ("Probability and data", ["statistics", "stochastic", "monte_carlo",
                              "information_theory", "learn"]),
    ("Decisions", ["optimization", "finance"]),
    ("Geometry", ["geometry", "curves", "trigonometry", "quaternion", "manifold",
                  "spatial", "mesh"]),
    ("Patterns and chaos", ["fractals", "patterns", "nonlinear"]),
    ("Reference and utility", ["units", "materials", "color_science",
                               "control_systems", "atmosphere", "verification"]),
]

PUB_FN = re.compile(r"pub (?:const |async )?fn ")
PUB_TYPE = re.compile(r"^pub (?:struct|enum|trait) ")


def scan(path: str) -> dict:
    """Public item counts, line count and summary for one source file.

    Items inside `#[cfg(test)]` are skipped by tracking brace depth, so a
    test helper never inflates the public surface.
    """
    free = meth = types = 0
    depth = 0
    test_depth: int | None = None
    summary_parts: list[str] = []
    in_summary = True
    total = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            stripped = line.strip()

            if in_summary:
                if stripped.startswith("//!"):
                    text = stripped[3:].strip()
                    if text:
                        summary_parts.append(text)
                    elif summary_parts:
                        in_summary = False
                elif stripped and not stripped.startswith("#!["):
                    in_summary = False

            if test_depth is None and stripped.startswith("#[cfg(test)]"):
                test_depth = depth
            if test_depth is None:
                if PUB_FN.match(stripped) and stripped.startswith("pub "):
                    if line.startswith("pub "):
                        free += 1
                    else:
                        meth += 1
                if PUB_TYPE.match(line):
                    types += 1
            depth += line.count("{") - line.count("}")
            if test_depth is not None and depth <= test_depth:
                test_depth = None

    summary = " ".join(summary_parts)
    # First sentence only, and never longer than a table cell wants.
    match = re.match(r"(.+?[.!?])(?:\s|$)", summary)
    if match:
        summary = match.group(1)
    summary = summary.replace("|", r"\|")
    if len(summary) > 150:
        summary = summary[:147].rsplit(" ", 1)[0] + "…"
    return {"free": free, "meth": meth, "types": types,
            "lines": total, "summary": summary or "—"}


def public_top_levels() -> set[str]:
    """Top-level modules declared `pub mod` in lib.rs.

    `verification` is declared as a bare `mod`, so it is compiled and
    tested but not part of the public API. Counting it as a public module
    would make the totals here disagree with the README.
    """
    out = set()
    with open(os.path.join(SRC, "lib.rs"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("pub mod "):
                out.add(line.strip()[len("pub mod "):].rstrip(";"))
    return out


def collect() -> dict[str, dict]:
    """Every source file, keyed by its Rust module path."""
    out = {}
    for root, _dirs, files in os.walk(SRC):
        for name in sorted(files):
            if not name.endswith(".rs"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, SRC)
            mod = rel[:-3].replace(os.sep, "::")
            if mod.endswith("::mod"):
                mod = mod[:-5]
            out[mod] = scan(path) | {"path": path, "rel": rel}
    return out


def render(mods: dict[str, dict]) -> str:
    tops = sorted({m.split("::")[0] for m in mods if m != "lib"})
    public = public_top_levels()
    private = [t for t in tops if t not in public]
    named = {n for _, names in AREAS for n in names}
    areas = AREAS + [("Other", sorted(set(tops) - named))]

    total_lines = sum(m["lines"] for m in mods.values())
    total_free = sum(m["free"] for m in mods.values())
    total_meth = sum(m["meth"] for m in mods.values())
    total_types = sum(m["types"] for m in mods.values())

    L: list[str] = []
    L.append("# Module map")
    L.append("")
    L.append("**Generated file — do not edit.** Produced by")
    L.append("[`tools/gen_module_map.py`](../tools/gen_module_map.py) from the source")
    L.append("tree; CI fails if it is out of date. Regenerate with:")
    L.append("")
    L.append("```bash")
    L.append("python3 tools/gen_module_map.py")
    L.append("```")
    L.append("")
    L.append("Every figure below is parsed from the sources. Summaries are the first")
    L.append("sentence of each module's `//!` documentation. Public-item counts exclude")
    L.append("anything inside `#[cfg(test)]`.")
    L.append("")
    L.append(f"**{len(mods) - 1} modules** across **{len(public)} public top-level "
             f"modules**, **{total_lines:,} lines** in "
             f"**{len(mods)} files** (the modules plus the crate root `src/lib.rs`), "
             f"**{total_free:,} public functions** and **{total_meth:,} public methods**, "
             f"**{total_types:,} public types**.")
    if private:
        names = ", ".join(f"`{p}`" for p in private)
        L.append("")
        L.append(f"{names} is compiled and tested but declared `mod` rather than "
                 "`pub mod`, so it is not part of the public API and is excluded from "
                 "the module count above.")
    L.append("")

    # ---- tree -------------------------------------------------------
    L.append("## Tree")
    L.append("")
    L.append("```")
    L.append(f"src/{' ' * 28}{'lines':>8}")
    L.append(f"├── {'lib.rs':<28}{mods['lib']['lines']:>8,}   (crate root)")
    for i, top in enumerate(tops):
        children = sorted(m for m in mods if m.startswith(top + "::"))
        last_top = i == len(tops) - 1
        stem = "└── " if last_top else "├── "
        tag = "" if top in public else "   (private)"
        if children:
            own = mods.get(top)
            n = sum(mods[c]["lines"] for c in children) + (own["lines"] if own else 0)
            L.append(f"{stem}{(top + '/'):<28}{n:>8,}{tag}")
            pad = "    " if last_top else "│   "
            for j, child in enumerate(children):
                leaf = child.split("::")[-1]
                cstem = "└── " if j == len(children) - 1 else "├── "
                L.append(f"{pad}{cstem}{(leaf + '.rs'):<24}{mods[child]['lines']:>8,}")
        else:
            L.append(f"{stem}{(top + '.rs'):<28}{mods[top]['lines']:>8,}{tag}")
    L.append("```")
    L.append("")

    # ---- by area ----------------------------------------------------
    L.append("## By area")
    L.append("")
    for area, names in areas:
        present = [n for n in names if n in tops]
        if not present:
            continue
        L.append(f"### {area}")
        L.append("")
        L.append("| Module | Lines | Public fns | Types | What it is |")
        L.append("|---|--:|--:|--:|---|")
        for name in present:
            subs = sorted(m for m in mods
                          if m != "lib" and (m == name or m.startswith(name + "::")))
            lines = sum(mods[s]["lines"] for s in subs)
            fns = sum(mods[s]["free"] + mods[s]["meth"] for s in subs)
            types = sum(mods[s]["types"] for s in subs)
            summary = mods[name]["summary"] if name in mods else "—"
            L.append(f"| **`{name}`** | {lines:,} | {fns:,} | {types:,} | {summary} |")
        L.append("")

    # ---- every module ----------------------------------------------
    L.append("## Every module")
    L.append("")
    L.append("| Path | Module | Lines | Fns | Methods | Types | Summary |")
    L.append("|---|---|--:|--:|--:|--:|---|")
    for mod in sorted(mods):
        if mod == "lib":
            continue
        m = mods[mod]
        L.append(f"| `{m['rel']}` | `{mod}` | {m['lines']:,} | {m['free']} "
                 f"| {m['meth']} | {m['types']} | {m['summary']} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    if not os.path.isdir(SRC):
        print("run from the repository root", file=sys.stderr)
        return 2
    text = render(collect())
    check = "--check" in sys.argv
    existing = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else None
    if check:
        if existing == text:
            print(f"{OUT} is up to date")
            return 0
        print(f"{OUT} is STALE — run `python3 {sys.argv[0]}` and commit the result",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
