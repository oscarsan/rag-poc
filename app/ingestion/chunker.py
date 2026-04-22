from __future__ import annotations

import re
from collections.abc import Iterable

from app.domain import Chunk, Document

# Paragraph = one or more blank lines.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_section(text: str, *, max_words: int = 400) -> list[str]:
    """Split a section into chunks of at most `max_words` words, at paragraph
    boundaries where possible.

    If a single paragraph exceeds `max_words`, it is greedily split on
    sentence boundaries, then on word boundaries as a last resort.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_words = 0

    for para in paragraphs:
        pw = _word_count(para)
        if pw > max_words:
            flush()
            chunks.extend(_split_long_paragraph(para, max_words))
            continue
        if buf_words + pw > max_words:
            flush()
        buf.append(para)
        buf_words += pw

    flush()
    return [c for c in chunks if c]


def _split_long_paragraph(para: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for sent in sentences:
        sw = _word_count(sent)
        if sw > max_words:
            if buf:
                out.append(" ".join(buf))
                buf = []
                buf_words = 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i : i + max_words]))
            continue
        if buf_words + sw > max_words:
            out.append(" ".join(buf))
            buf = []
            buf_words = 0
        buf.append(sent)
        buf_words += sw
    if buf:
        out.append(" ".join(buf))
    return out


def document_to_chunks(doc: Document, *, max_words: int = 400) -> list[Chunk]:
    """Expand a Document into chunks, one Chunk per language-section chunk.

    Chunk IDs are deterministic: `{doc_id}__{language}__{idx}`.
    Metadata is the frontmatter plus `source_file`.
    """
    chunks: list[Chunk] = []
    for lang, text in doc.sections.items():
        pieces = chunk_section(text, max_words=max_words)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}__{lang}__{idx}",
                    doc_id=doc.doc_id,
                    language=lang,
                    text=piece,
                    metadata={**doc.frontmatter, "source_file": doc.source_file},
                )
            )
    return chunks


def documents_to_chunks(
    docs: Iterable[Document], *, max_words: int = 400
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(document_to_chunks(doc, max_words=max_words))
    return out
