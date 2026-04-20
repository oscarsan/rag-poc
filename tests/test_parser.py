from pathlib import Path

from app.ingestion.parser import (
    iter_markdown_files,
    parse_markdown_file,
    split_language_sections,
)

SAMPLE = """\
---
id: test-doc
type: activity
price_eur_adult: 42
---

## English

English paragraph one.

English paragraph two.

## Suomi

Suomenkielinen kappale.
"""


def test_split_language_sections_basic():
    body = "## English\n\nHello world.\n\n## Suomi\n\nHei maailma."
    sections = split_language_sections(body)
    assert sections == {"en": "Hello world.", "fi": "Hei maailma."}


def test_split_language_sections_case_insensitive_and_order_independent():
    body = "## suomi\n\nHei.\n\n## ENGLISH\n\nHi."
    sections = split_language_sections(body)
    assert sections == {"fi": "Hei.", "en": "Hi."}


def test_split_language_sections_missing_section():
    body = "## English\n\nOnly english here."
    sections = split_language_sections(body)
    assert sections == {"en": "Only english here."}


def test_split_language_sections_no_headers_returns_empty():
    assert split_language_sections("just plain text") == {}


def test_parse_markdown_file(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text(SAMPLE, encoding="utf-8")
    doc = parse_markdown_file(f)
    assert doc.doc_id == "test-doc"
    assert doc.frontmatter["price_eur_adult"] == 42
    assert doc.sections["en"].startswith("English paragraph one.")
    assert "Suomenkielinen" in doc.sections["fi"]
    assert doc.source_file == str(f)


def test_parse_markdown_file_uses_stem_when_no_id(tmp_path: Path):
    f = tmp_path / "fallback.md"
    f.write_text("---\ntype: x\n---\n\n## English\n\nHi.\n", encoding="utf-8")
    doc = parse_markdown_file(f)
    assert doc.doc_id == "fallback"


def test_iter_markdown_files_sorted(tmp_path: Path):
    (tmp_path / "b.md").write_text("", encoding="utf-8")
    (tmp_path / "a.md").write_text("", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("", encoding="utf-8")
    files = [p.name for p in iter_markdown_files(tmp_path)]
    assert files == sorted(files)
    assert "c.md" in files
