"""
sibling_resolver.py — resolve and symlink dd4hep compact XML sibling files.
Only <include>, <gdmlFile>, <file> tags are treated as file refs; chemical
element/material tables are ignored. Refs resolve against the SOURCE compact's
directory; symlinks are created in the RUN directory. Recurses only into
<include>d detector fragments, never into GDML element/material tables.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set

from bilevel_opt.util import log

_FILE_REF_TAGS = {"include", "gdmlfile", "file"}
_REF_ATTRS = ("ref", "url", "file")
_ENVVAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
_XML_SUFFIXES = {".xml"}


def _local_file_refs(xml_path: Path):
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError) as exc:
        log(f"sibling_resolver: cannot parse {xml_path} ({exc}); skipping")
        return
    for elem in tree.iter():
        tag = elem.tag.lower()
        if tag not in _FILE_REF_TAGS:
            continue
        for attr in _REF_ATTRS:
            ref = elem.attrib.get(attr)
            if not ref:
                continue
            if _ENVVAR_RE.search(ref):
                continue
            if ref.startswith(("/", "http://", "https://", "file://")):
                continue
            yield tag, ref


def _link(source: Path, target: Path) -> str:
    if not source.exists():
        log(f"sibling_resolver: source missing, skipping: {source}")
        return "missing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        try:
            existing = target.resolve(strict=False)
        except OSError:
            existing = None
        if existing == source.resolve(strict=False):
            return "matched"
        link_target = os.readlink(target) if target.is_symlink() else "(regular file)"
        log(f"sibling_resolver: WARNING — {target} exists and points at "
            f"{link_target}, expected {source}; leaving as-is")
        return "conflict"
    os.symlink(source, target)
    return "created"


def link_siblings(generated_compact: Path, source_compact: Path, run_dir: Path) -> Dict[str, str]:
    source_dir = source_compact.parent.resolve()
    run_dir = run_dir.resolve()
    manifest: Dict[str, str] = {}
    seen: Set[Path] = set()

    def walk(xml_to_scan: Path, resolve_base: Path) -> None:
        real = xml_to_scan.resolve()
        if real in seen:
            return
        seen.add(real)
        for tag, ref in _local_file_refs(xml_to_scan):
            if ref in manifest:
                continue
            source_path = (resolve_base / ref).resolve()
            target_path = (run_dir / ref).resolve()
            if source_path.is_dir():
                continue
            status = _link(source_path, target_path)
            manifest[ref] = status
            if (tag == "include" and status in ("created", "matched")
                    and source_path.suffix.lower() in _XML_SUFFIXES):
                walk(source_path, source_path.parent)

    walk(generated_compact, source_dir)
    return manifest


def summarize(manifest: Dict[str, str]) -> str:
    counts: Dict[str, int] = {}
    for status in manifest.values():
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{n} {status}" for status, n in sorted(counts.items())]
    return f"sibling_resolver: {len(manifest)} refs " + ", ".join(parts)
