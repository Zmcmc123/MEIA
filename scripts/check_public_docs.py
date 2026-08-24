#!/usr/bin/env python3
"""Validate local targets referenced by tracked public Markdown files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable
from urllib.parse import unquote, urlsplit


MARKDOWN_LINK_RE = re.compile(r"(?P<image>!)?\[[^\]]*\]\((?P<target>[^)\n]+)\)")
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*\bsrc=[\"'](?P<target>[^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinkReference:
    line_number: int
    target: str
    is_image: bool


def _link_destination(raw_target: str) -> str:
    """Return the URL/path portion before an optional Markdown title."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def _references(markdown: str) -> Iterable[LinkReference]:
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        for match in MARKDOWN_LINK_RE.finditer(line):
            yield LinkReference(
                line_number=line_number,
                target=_link_destination(match.group("target")),
                is_image=bool(match.group("image")),
            )
        for match in HTML_IMAGE_RE.finditer(line):
            yield LinkReference(
                line_number=line_number,
                target=match.group("target").strip(),
                is_image=True,
            )


def _target_kind(root: Path, relative_path: Path) -> str:
    """Classify a repository-relative target without hiding case mismatches."""

    current = root
    for part in relative_path.parts:
        if not current.is_dir():
            return "missing"
        names = {child.name for child in current.iterdir()}
        if part not in names:
            if any(part.casefold() == name.casefold() for name in names):
                return "case"
            return "missing"
        current = current / part
    return "ok" if current.exists() else "missing"


def _valid_image_bytes(path: Path) -> bool:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix in {".jpg", ".jpeg"}:
        return len(data) >= 4 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    return True


def check_markdown_files(root: Path, markdown_files: Iterable[Path]) -> list[str]:
    """Return human-readable errors for broken repository-local references."""

    root = root.resolve()
    errors: list[str] = []
    for supplied_path in markdown_files:
        markdown_path = supplied_path
        if not markdown_path.is_absolute():
            markdown_path = root / markdown_path
        try:
            relative_markdown = markdown_path.resolve().relative_to(root)
        except ValueError:
            errors.append(f"{supplied_path}: outside repository root")
            continue
        if not markdown_path.is_file():
            errors.append(f"{relative_markdown}: missing Markdown file")
            continue
        markdown = markdown_path.read_text(encoding="utf-8")
        for reference in _references(markdown):
            parsed = urlsplit(reference.target)
            if (
                parsed.scheme
                or parsed.netloc
                or reference.target.startswith("//")
                or not parsed.path
            ):
                continue
            decoded_path = unquote(parsed.path)
            lexical_target = markdown_path.parent / decoded_path
            try:
                relative_target = lexical_target.resolve().relative_to(root)
            except ValueError:
                errors.append(
                    f"{relative_markdown}:{reference.line_number}: "
                    f"target leaves repository root: {reference.target}"
                )
                continue
            target_kind = _target_kind(root, relative_target)
            if target_kind == "case":
                errors.append(
                    f"{relative_markdown}:{reference.line_number}: "
                    f"path case mismatch: {reference.target}"
                )
                continue
            if target_kind == "missing":
                errors.append(
                    f"{relative_markdown}:{reference.line_number}: "
                    f"missing local target: {reference.target}"
                )
                continue
            target_path = root / relative_target
            if reference.is_image and target_path.is_file() and not _valid_image_bytes(target_path):
                label = "JPEG" if target_path.suffix.lower() in {".jpg", ".jpeg"} else "image"
                errors.append(
                    f"{relative_markdown}:{reference.line_number}: "
                    f"invalid {label} data: {reference.target}"
                )
    return errors


def _discover_markdown(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*.md",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check repository-local links and image files in public Markdown."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    markdown_files = args.files if args.files is not None else _discover_markdown(root)
    errors = check_markdown_files(root, markdown_files)
    for error in errors:
        print(error)
    count = len(markdown_files)
    noun = "file" if count == 1 else "files"
    print(f"checked {count} Markdown {noun}; {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
