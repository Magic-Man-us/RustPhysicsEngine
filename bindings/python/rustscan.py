"""A small Rust source reader: enough of the grammar to list a crate's public API.

This is not a Rust parser. It is a scanner that knows where the code is --
that is, which byte offsets are inside a string, a character literal or a
comment -- and can therefore match braces reliably, and on top of that a
walker that recognises the handful of item forms this crate actually uses:
modules, `use`, structs, enums, functions, `impl` blocks, constants and
type aliases.

Being a scanner rather than a parser is a deliberate limit. It cannot
resolve a macro, and it does not try to; a generic bound spanning several
lines it records verbatim rather than interpreting. Everything it cannot
place it reports as unhandled instead of guessing, so the generator that
consumes it can decide what to skip and say why.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterator


# ── Masking: which offsets are code ─────────────────────────────────────


def code_mask(src: str) -> bytearray:
    """Return a byte-per-character mask: 1 where `src` is code, 0 elsewhere.

    Doc comments (`///`, `//!`, `/** */`, `/*! */`) are masked out like any
    other comment; they are recovered separately by `doc_before`, which
    reads the raw text. Masking them keeps a `{` inside a doctest from
    unbalancing an item's braces.
    """
    n = len(src)
    mask = bytearray(b"\x01" * n)
    i = 0
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                mask[k] = 0
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            # Rust block comments nest.
            depth = 1
            j = i + 2
            while j < n and depth:
                if src[j] == "/" and j + 1 < n and src[j + 1] == "*":
                    depth += 1
                    j += 2
                elif src[j] == "*" and j + 1 < n and src[j + 1] == "/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            for k in range(i, min(j, n)):
                mask[k] = 0
            i = j
        elif c == "r" and _raw_string_start(src, i):
            hashes = 0
            j = i + 1
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            close = '"' + "#" * hashes
            end = src.find(close, j + 1)
            end = n if end < 0 else end + len(close)
            for k in range(i, end):
                mask[k] = 0
            i = end
        elif c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                mask[k] = 0
            i = j
        elif c == "'":
            # A quote is either a char literal or a lifetime. Only the
            # literal forms hide code.
            m = re.match(r"'(?:\\.|[^\\'])'", src[i : i + 8])
            if m:
                for k in range(i, i + m.end()):
                    mask[k] = 0
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return mask


def _raw_string_start(src: str, i: int) -> bool:
    j = i + 1
    while j < len(src) and src[j] == "#":
        j += 1
    return j < len(src) and src[j] == '"'


def match_brace(src: str, mask: bytearray, open_at: int) -> int:
    """Index just past the `}` closing the `{` at `open_at`."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    closer = pairs[src[open_at]]
    opener = src[open_at]
    depth = 0
    i = open_at
    n = len(src)
    while i < n:
        if mask[i]:
            if src[i] == opener:
                depth += 1
            elif src[i] == closer:
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return n


def find_code(src: str, mask: bytearray, ch: str, start: int, end: int) -> int:
    for i in range(start, min(end, len(src))):
        if mask[i] and src[i] == ch:
            return i
    return -1


# ── Items ───────────────────────────────────────────────────────────────


@dataclass
class Field:
    name: str
    ty: str
    public: bool


@dataclass
class Variant:
    name: str
    payload: str  # "" for a unit variant, else the tuple/struct body


@dataclass
class Struct:
    name: str
    module: str
    file: str
    doc: str
    attrs: list[str]
    fields: list[Field]
    kind: str  # "named" | "tuple" | "unit"
    generics: str

    @property
    def path(self) -> str:
        return f"{self.module}::{self.name}" if self.module else self.name

    def derives(self) -> set[str]:
        out: set[str] = set()
        for a in self.attrs:
            m = re.match(r"derive\((.*)\)$", a, re.S)
            if m:
                out |= {x.strip() for x in m.group(1).split(",") if x.strip()}
        return out


@dataclass
class Enum:
    name: str
    module: str
    file: str
    doc: str
    attrs: list[str]
    variants: list[Variant]
    generics: str

    @property
    def path(self) -> str:
        return f"{self.module}::{self.name}" if self.module else self.name

    def derives(self) -> set[str]:
        out: set[str] = set()
        for a in self.attrs:
            m = re.match(r"derive\((.*)\)$", a, re.S)
            if m:
                out |= {x.strip() for x in m.group(1).split(",") if x.strip()}
        return out


@dataclass
class Func:
    name: str
    module: str
    file: str
    doc: str
    attrs: list[str]
    args: list[tuple[str, str]]  # (pattern, type); `self` appears as ("self", "")
    ret: str
    generics: str
    where_clause: str
    self_kind: str  # "" | "self" | "&self" | "&mut self"
    impl_type: str  # "" for a free function, else the receiver type as written
    impl_trait: str  # non-empty when the impl block implements a trait
    is_const: bool
    is_unsafe: bool

    @property
    def is_method(self) -> bool:
        return bool(self.impl_type)


@dataclass
class Const:
    name: str
    module: str
    file: str
    doc: str
    ty: str
    value: str
    owner: str = ""  # the impl type, for an associated constant

    @property
    def path(self) -> str:
        return f"{self.module}::{self.name}" if self.module else self.name


@dataclass
class Alias:
    name: str
    module: str
    target: str


@dataclass
class Crate:
    structs: list[Struct] = field(default_factory=list)
    enums: list[Enum] = field(default_factory=list)
    funcs: list[Func] = field(default_factory=list)
    consts: list[Const] = field(default_factory=list)
    aliases: list[Alias] = field(default_factory=list)
    traits: set[str] = field(default_factory=set)
    # file -> {short name: full path}
    uses: dict[str, dict[str, str]] = field(default_factory=dict)
    # file -> [module paths brought in by a glob import]
    glob_uses: dict[str, list[str]] = field(default_factory=dict)
    # module path -> {name exposed there: the path it re-exports}
    pub_uses: dict[str, dict[str, str]] = field(default_factory=dict)
    # module path -> the `//!` summary
    module_docs: dict[str, str] = field(default_factory=dict)
    unhandled: list[str] = field(default_factory=list)


ATTR_RE = re.compile(r"#!?\[")
DOC_LINE_RE = re.compile(r"^\s*//[/!] ?(.*)$")


def _doc_and_attrs(src: str, mask: bytearray, start: int, stop: int) -> tuple[str, list[str], int]:
    """Read doc comments and attributes ending at `stop`, scanning from `start`.

    Returns the doc text, the attribute bodies, and the offset where the
    item's own tokens begin.
    """
    docs: list[str] = []
    attrs: list[str] = []
    i = start
    while i < stop:
        # Whitespace.
        if src[i].isspace():
            i += 1
            continue
        # Comment (doc or not).
        if src[i] == "/" and i + 1 < stop and src[i + 1] == "/":
            j = src.find("\n", i)
            j = stop if j < 0 else j
            line = src[i:j]
            m = re.match(r"//[/!] ?(.*)$", line)
            if m:
                docs.append(m.group(1))
            elif not line.startswith("////"):
                docs.clear() if False else None
            i = j + 1
            continue
        if src[i] == "/" and i + 1 < stop and src[i + 1] == "*":
            depth, j = 1, i + 2
            while j < stop and depth:
                if src[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif src[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            i = j
            continue
        if src[i] == "#":
            m = ATTR_RE.match(src, i)
            if m:
                open_at = m.end() - 1
                close = match_brace(src, mask, open_at)
                attrs.append(src[open_at + 1 : close - 1].strip())
                i = close
                continue
        break
    return "\n".join(docs).strip(), attrs, i


def _split_top(text: str) -> list[str]:
    """Split on commas that are not nested inside <>, (), [] or {}."""
    out, depth, cur = [], 0, ""
    angle = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "<":
            angle += 1
        elif c == ">":
            # `->` is not a closing angle bracket.
            if i and text[i - 1] == "-":
                pass
            else:
                angle -= 1
        if c == "," and depth == 0 and angle <= 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def _split_generics(text: str, start: int) -> tuple[str, int]:
    """Read a `<...>` generic list starting at `start`; return it and the end."""
    if start >= len(text) or text[start] != "<":
        return "", start
    depth, i = 0, start
    while i < len(text):
        if text[i] == "<":
            depth += 1
        elif text[i] == ">":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        elif text[i] == "-" and i + 1 < len(text) and text[i + 1] == ">":
            i += 1
        i += 1
    return text[start:], len(text)


def parse_file(path: str, module: str, crate: Crate) -> None:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    mask = code_mask(src)
    crate.uses.setdefault(path, {})
    crate.glob_uses.setdefault(path, [])
    # The `//!` header, taken before anything masks it away.
    header: list[str] = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("//!"):
            header.append(s[3:].strip())
        elif s and not s.startswith("//"):
            break
    if header:
        crate.module_docs[module] = " ".join(header).strip()
    _walk(src, mask, 0, len(src), module, path, crate, in_impl=None)


def _walk(
    src: str,
    mask: bytearray,
    start: int,
    end: int,
    module: str,
    path: str,
    crate: Crate,
    in_impl,
) -> None:
    i = start
    while i < end:
        doc, attrs, i = _doc_and_attrs(src, mask, i, end)
        if i >= end:
            return
        # Where does this item's head end? At `{`, `(`, `;` or `=`.
        head_start = i
        rest = src[i:end]
        if not rest.strip():
            return

        cfg_test = any(a.startswith("cfg(test") or a == "test" for a in attrs)
        cfg_kani = any("kani" in a for a in attrs)
        hidden = any(a.startswith("doc(hidden") for a in attrs)

        tok = re.match(r"[A-Za-z_][A-Za-z0-9_]*|.", rest)
        word = tok.group(0) if tok else ""

        # Visibility prefix.
        vis = ""
        vm = re.match(r"pub(\s*\([^)]*\))?\s+", rest)
        if vm:
            vis = "pub" if not vm.group(1) else f"pub{vm.group(1).strip()}"
            after_vis = i + vm.end()
        else:
            after_vis = i
        body = src[after_vis:end]
        kw = re.match(
            r"(unsafe\s+|const\s+|async\s+|extern\s+\"[^\"]*\"\s+|default\s+)*"
            r"(mod|use|struct|enum|fn|impl|const|static|type|trait|union|macro_rules)\b",
            body,
        )
        if not kw:
            # Something we do not recognise: skip to the next top-level
            # boundary so one odd item cannot derail the rest of the file.
            nxt = _skip_item(src, mask, i, end)
            if nxt <= i:
                return
            snippet = src[i:nxt].strip().splitlines()
            if snippet:
                crate.unhandled.append(f"{path}: {snippet[0][:70]}")
            i = nxt
            continue

        prefixes = kw.group(0)[: kw.start(2)]
        kind = kw.group(2)
        after_kw = after_vis + kw.end()

        if kind == "use":
            semi = find_code(src, mask, ";", after_kw, end)
            if semi < 0:
                return
            _record_use(src[after_kw:semi], module, path, crate, vis == "pub")
            i = semi + 1
            continue

        if kind == "mod":
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", src[after_kw:end])
            if not m:
                i = _skip_item(src, mask, i, end)
                continue
            name = m.group(1)
            j = after_kw + m.end()
            if j < end and src[j] == ";":
                i = j + 1
                continue
            if j < end and src[j] == "{":
                close = match_brace(src, mask, j)
                if not (cfg_test or cfg_kani or name == "tests"):
                    sub = f"{module}::{name}" if module else name
                    inner_doc: list[str] = []
                    for line in src[j + 1 : close].splitlines():
                        s = line.strip()
                        if s.startswith("//!"):
                            inner_doc.append(s[3:].strip())
                        elif s and not s.startswith("//"):
                            break
                    if inner_doc:
                        crate.module_docs[sub] = " ".join(inner_doc)
                    elif doc:
                        crate.module_docs[sub] = doc.splitlines()[0]
                    if vis == "pub":
                        _walk(src, mask, j + 1, close - 1, sub, path, crate, in_impl)
                i = close
                continue
            i = _skip_item(src, mask, i, end)
            continue

        if kind in ("struct", "union"):
            i = _parse_struct(
                src, mask, after_kw, end, module, path, crate, doc, attrs, vis, cfg_test or hidden
            )
            continue

        if kind == "enum":
            i = _parse_enum(
                src, mask, after_kw, end, module, path, crate, doc, attrs, vis, cfg_test or hidden
            )
            continue

        if kind == "trait":
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", src[after_kw:end])
            if m:
                crate.traits.add(m.group(1))
            i = _skip_item(src, mask, i, end)
            continue

        if kind == "fn":
            i = _parse_fn(
                src,
                mask,
                after_kw,
                end,
                module,
                path,
                crate,
                doc,
                attrs,
                vis,
                prefixes,
                in_impl,
                cfg_test or hidden,
            )
            continue

        if kind == "impl":
            i = _parse_impl(src, mask, after_kw, end, module, path, crate, cfg_test or cfg_kani)
            continue

        if kind in ("const", "static"):
            m = re.match(r"\s*(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*", src[after_kw:end])
            eq = find_code(src, mask, "=", after_kw, end)
            semi = _stmt_end(src, mask, after_kw, end)
            if m and vis == "pub" and not cfg_test and eq > 0:
                ty = src[after_kw + m.end() : eq].strip()
                crate.consts.append(
                    Const(
                        name=m.group(1),
                        module=module,
                        file=path,
                        doc=doc,
                        ty=ty,
                        value=src[eq + 1 : semi].strip(),
                        owner=in_impl[0] if in_impl else "",
                    )
                )
            i = semi + 1
            continue

        if kind == "type":
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", src[after_kw:end])
            eq = find_code(src, mask, "=", after_kw, end)
            semi = _stmt_end(src, mask, after_kw, end)
            if m and eq > 0 and eq < semi:
                crate.aliases.append(
                    Alias(name=m.group(1), module=module, target=src[eq + 1 : semi].strip())
                )
            i = semi + 1
            continue

        i = _skip_item(src, mask, i, end)


def _stmt_end(src: str, mask: bytearray, start: int, end: int) -> int:
    """Offset of the `;` ending a statement, skipping balanced brackets."""
    i = start
    while i < end:
        if mask[i]:
            if src[i] in "{([":
                i = match_brace(src, mask, i)
                continue
            if src[i] == ";":
                return i
        i += 1
    return end


def _skip_item(src: str, mask: bytearray, start: int, end: int) -> int:
    """Skip one item: to the end of its block, or to its terminating `;`."""
    i = start
    while i < end:
        if mask[i]:
            if src[i] == "{":
                return match_brace(src, mask, i)
            if src[i] == ";":
                return i + 1
            if src[i] in "([":
                i = match_brace(src, mask, i)
                continue
        i += 1
    return end


def _record_use(text: str, module: str, path: str, crate: Crate, is_pub: bool = False) -> None:
    text = " ".join(text.split())
    if is_pub:
        crate.pub_uses.setdefault(module, {})

    def expand(prefix: str, body: str) -> None:
        body = body.strip()
        if body.startswith("{") and body.endswith("}"):
            for part in _split_top(body[1:-1]):
                expand(prefix, part)
            return
        if "::{" in body:
            head, tail = body.split("::{", 1)
            expand(f"{prefix}::{head}".strip(":"), "{" + tail)
            return
        if body == "*":
            crate.glob_uses[path].append(_normalise(prefix, module))
            return
        if " as " in body:
            target, alias = body.split(" as ", 1)
            full = _normalise(f"{prefix}::{target}".strip(":"), module)
            crate.uses[path][alias.strip()] = full
            if is_pub:
                crate.pub_uses[module][alias.strip()] = full
            return
        if not body:
            return
        full = _normalise(f"{prefix}::{body}".strip(":"), module)
        crate.uses[path][body.split("::")[-1]] = full
        if is_pub:
            crate.pub_uses[module][body.split("::")[-1]] = full

    expand("", text)


def _normalise(p: str, module: str) -> str:
    p = p.strip().strip(":")
    if p.startswith("crate::"):
        return p[len("crate::") :]
    if p.startswith("self::"):
        return f"{module}::{p[len('self::'):]}" if module else p[len("self::") :]
    if p.startswith("super::"):
        parent = "::".join(module.split("::")[:-1])
        rest = p[len("super::") :]
        while rest.startswith("super::"):
            parent = "::".join(parent.split("::")[:-1])
            rest = rest[len("super::") :]
        return f"{parent}::{rest}" if parent else rest
    return p


def _parse_struct(src, mask, i, end, module, path, crate, doc, attrs, vis, skip) -> int:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", src[i:end])
    if not m:
        return _skip_item(src, mask, i, end)
    name = m.group(1)
    j = i + m.end()
    generics, j = _split_generics(src[:end], j)
    while j < end and src[j].isspace():
        j += 1
    fields: list[Field] = []
    if j < end and src[j] == "{":
        close = match_brace(src, mask, j)
        inner = src[j + 1 : close - 1]
        imask = mask[j + 1 : close - 1]
        for part in _split_fields(inner, imask):
            fm = re.match(r"\s*((?:pub(?:\s*\([^)]*\))?\s+)?)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", part, re.S)
            if fm:
                fields.append(
                    Field(
                        name=fm.group(2),
                        ty=" ".join(fm.group(3).split()),
                        public=fm.group(1).strip().startswith("pub"),
                    )
                )
        kind = "named"
        nxt = close
    elif j < end and src[j] == "(":
        close = match_brace(src, mask, j)
        inner = src[j + 1 : close - 1]
        for k, part in enumerate(_split_top(inner)):
            pm = re.match(r"\s*((?:pub(?:\s*\([^)]*\))?\s+)?)(.+)$", part, re.S)
            if pm:
                fields.append(
                    Field(
                        name=str(k),
                        ty=" ".join(pm.group(2).split()),
                        public=pm.group(1).strip().startswith("pub"),
                    )
                )
        kind = "tuple"
        nxt = _stmt_end(src, mask, close, end) + 1
    else:
        kind = "unit"
        nxt = _stmt_end(src, mask, j, end) + 1
    if vis == "pub" and not skip:
        crate.structs.append(
            Struct(
                name=name,
                module=module,
                file=path,
                doc=doc,
                attrs=attrs,
                fields=fields,
                kind=kind,
                generics=generics,
            )
        )
    return nxt


def _split_fields(inner: str, imask: bytearray) -> list[str]:
    """Split a struct body on commas that are outside comments and nesting."""
    out, cur, depth = [], "", 0
    angle = 0
    for idx, c in enumerate(inner):
        live = imask[idx] if idx < len(imask) else 1
        if live:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "<":
                angle += 1
            elif c == ">" and not (idx and inner[idx - 1] == "-"):
                angle -= 1
            if c == "," and depth == 0 and angle <= 0:
                out.append(cur)
                cur = ""
                continue
        cur += c if live else " "
    if cur.strip():
        out.append(cur)
    return out


def _parse_enum(src, mask, i, end, module, path, crate, doc, attrs, vis, skip) -> int:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", src[i:end])
    if not m:
        return _skip_item(src, mask, i, end)
    name = m.group(1)
    j = i + m.end()
    generics, j = _split_generics(src[:end], j)
    while j < end and src[j].isspace():
        j += 1
    variants: list[Variant] = []
    if j < end and src[j] == "{":
        close = match_brace(src, mask, j)
        inner = src[j + 1 : close - 1]
        imask = mask[j + 1 : close - 1]
        for part in _split_fields(inner, imask):
            vm = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$", part, re.S)
            if vm:
                variants.append(Variant(name=vm.group(1), payload=vm.group(2).strip()))
        nxt = close
    else:
        nxt = _stmt_end(src, mask, j, end) + 1
    if vis == "pub" and not skip:
        crate.enums.append(
            Enum(
                name=name,
                module=module,
                file=path,
                doc=doc,
                attrs=attrs,
                variants=variants,
                generics=generics,
            )
        )
    return nxt


def _parse_impl(src, mask, i, end, module, path, crate, skip) -> int:
    generics, j = _split_generics(src[:end], _skip_ws(src, i, end))
    open_at = find_code(src, mask, "{", j, end)
    if open_at < 0:
        return _skip_item(src, mask, i, end)
    head = src[j:open_at].strip()
    close = match_brace(src, mask, open_at)
    if skip:
        return close
    # `impl Trait for Type` vs `impl Type`.
    trait_name = ""
    target = head
    fm = re.search(r"\bfor\b", head)
    if fm:
        trait_name = head[: fm.start()].strip()
        target = head[fm.end() :].strip()
    target = re.sub(r"\bwhere\b.*$", "", target, flags=re.S).strip()
    base = re.match(r"([A-Za-z_][A-Za-z0-9_:]*)", target)
    target_name = base.group(1).split("::")[-1] if base else target
    _walk(
        src,
        mask,
        open_at + 1,
        close - 1,
        module,
        path,
        crate,
        in_impl=(target_name, trait_name, generics),
    )
    return close


def _fn_tail(src: str, mask: bytearray, start: int, end: int) -> tuple[int, int]:
    """Offsets of the body's `{` and of a declaration's `;` after an arg list.

    Either may be -1. Balanced `[...]` and `(...)` are stepped over, so the
    `;` in a `-> [f64; 6]` return type is not mistaken for the end of a
    declaration.
    """
    i = start
    while i < end:
        if mask[i]:
            c = src[i]
            if c in "([":
                i = match_brace(src, mask, i)
                continue
            if c == "{":
                return i, -1
            if c == ";":
                return -1, i
        i += 1
    return -1, -1


def _skip_ws(src: str, i: int, end: int) -> int:
    while i < end and src[i].isspace():
        i += 1
    return i


def _parse_fn(
    src, mask, i, end, module, path, crate, doc, attrs, vis, prefixes, in_impl, skip
) -> int:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", src[i:end])
    if not m:
        return _skip_item(src, mask, i, end)
    name = m.group(1)
    j = i + m.end()
    generics, j = _split_generics(src[:end], _skip_ws(src, j, end))
    j = _skip_ws(src, j, end)
    if j >= end or src[j] != "(":
        return _skip_item(src, mask, i, end)
    close = match_brace(src, mask, j)
    arglist = src[j + 1 : close - 1]
    amask = mask[j + 1 : close - 1]
    clean = "".join(c if amask[k] else " " for k, c in enumerate(arglist))
    # Return type and where clause run to the body (or the `;` of a decl).
    # A return type may itself contain brackets and semicolons -- `[f64; 6]`
    # is the common case here -- so step over balanced brackets rather than
    # taking the first `;` or `{` literally.
    body_open, semi = _fn_tail(src, mask, close, end)
    if body_open < 0 or (0 <= semi < body_open):
        tail_end = semi if semi >= 0 else end
        nxt = tail_end + 1
    else:
        tail_end = body_open
        nxt = match_brace(src, mask, body_open)
    tail = src[close:tail_end]
    tail = "".join(c if mask[close + k] else " " for k, c in enumerate(tail))
    where_clause = ""
    wm = re.search(r"\bwhere\b", tail)
    if wm:
        where_clause = tail[wm.end() :].strip()
        tail = tail[: wm.start()]
    ret = ""
    rm = re.search(r"->", tail)
    if rm:
        ret = " ".join(tail[rm.end() :].split()).strip()

    in_trait_impl = bool(in_impl and in_impl[1])
    if skip or (vis != "pub" and not in_trait_impl):
        return nxt

    args: list[tuple[str, str]] = []
    self_kind = ""
    for part in _split_top(clean):
        part = part.strip()
        if not part or part.startswith("#["):
            continue
        sm = re.match(r"^(&\s*(?:'[a-z_]+\s*)?(mut\s+)?)?(mut\s+)?self$", part)
        if sm:
            if not sm.group(1):
                self_kind = "self"
            elif sm.group(2):
                self_kind = "&mut self"
            else:
                self_kind = "&self"
            continue
        if ":" not in part:
            continue
        pat, ty = part.split(":", 1)
        args.append((pat.strip(), " ".join(ty.split()).strip()))

    impl_type, impl_trait = ("", "")
    if in_impl:
        impl_type, impl_trait = in_impl[0], in_impl[1]
    crate.funcs.append(
        Func(
            name=name,
            module=module,
            file=path,
            doc=doc,
            attrs=attrs,
            args=args,
            ret=ret,
            generics=generics,
            where_clause=where_clause,
            self_kind=self_kind,
            impl_type=impl_type,
            impl_trait=impl_trait,
            is_const="const" in prefixes,
            is_unsafe="unsafe" in prefixes,
        )
    )
    return nxt


# ── Crate walk ──────────────────────────────────────────────────────────


def module_for(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.replace(os.sep, "/").split("/")
    if parts[-1] in ("lib.rs", "main.rs"):
        parts = parts[:-1]
    elif parts[-1] == "mod.rs":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return "::".join(parts)


def scan_crate(src_root: str) -> Crate:
    crate = Crate()
    files = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for fn in filenames:
            if fn.endswith(".rs"):
                files.append(os.path.join(dirpath, fn))
    for path in sorted(files):
        module = module_for(src_root, path)
        if module.split("::")[0] == "verification":
            continue
        parse_file(path, module, crate)
    return crate


def iter_public_modules(crate: Crate) -> Iterator[str]:
    seen = set()
    for coll in (crate.structs, crate.enums, crate.funcs, crate.consts):
        for item in coll:
            mod = getattr(item, "module", "")
            if mod and mod not in seen:
                seen.add(mod)
                yield mod
