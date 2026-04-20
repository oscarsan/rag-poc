from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import frontmatter

from app.domain import Document, Language

log = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^\s*##\s+(English|Suomi)\s*$", re.IGNORECASE | re.MULTILINE)
_HEADER_TO_LANG: dict[str, Language] = {"english": "en", "suomi": "fi"}


def split_language_sections(body: str) -> dict[Language, str]:
    """Split a markdown body into English/Finnish sections using ## headers.

    Content before the first language header is ignored. Missing sections are
    simply absent from the returned mapping. Section text is stripped.
    """
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        return {}

    sections: dict[Language, str] = {}
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower()
        lang = _HEADER_TO_LANG.get(header)
        if lang is None:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections[lang] = text
    return sections


def parse_markdown_file(path: Path) -> Document:
    """Parse one markdown file with YAML frontmatter into a Document.

    The frontmatter must contain an `id` field; otherwise the file stem is
    used as a fallback.
    """
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    fm: dict[str, Any] = dict(post.metadata)
    doc_id = str(fm.get("id") or path.stem)
    sections = split_language_sections(post.content)

    if not sections:
        log.warning("No language sections found in %s", path)

    return Document(
        doc_id=doc_id,
        source_file=str(path),
        frontmatter=fm,
        sections=sections,
    )


def iter_markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Content root does not exist: {root}")
    return sorted(p for p in root.rglob("*.md") if p.is_file())
