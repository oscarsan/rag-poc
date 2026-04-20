from app.domain import Document
from app.ingestion.chunker import chunk_section, document_to_chunks


def test_chunk_section_keeps_small_text_as_one_chunk():
    text = "One paragraph.\n\nTwo paragraph."
    chunks = chunk_section(text, max_words=400)
    assert len(chunks) == 1
    assert "One paragraph." in chunks[0]
    assert "Two paragraph." in chunks[0]


def test_chunk_section_splits_on_paragraph_boundary():
    para_a = " ".join(["word"] * 50)
    para_b = " ".join(["token"] * 50)
    chunks = chunk_section(f"{para_a}\n\n{para_b}", max_words=60)
    assert len(chunks) == 2
    assert chunks[0].startswith("word")
    assert chunks[1].startswith("token")


def test_chunk_section_splits_long_paragraph():
    huge = " ".join(["w"] * 250)
    chunks = chunk_section(huge, max_words=100)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 100


def test_document_to_chunks_deterministic_ids():
    doc = Document(
        doc_id="abc",
        source_file="/x/abc.md",
        frontmatter={"type": "activity"},
        sections={"en": "Hello.", "fi": "Hei."},
    )
    chunks = document_to_chunks(doc)
    ids = {c.chunk_id for c in chunks}
    assert ids == {"abc__en__0", "abc__fi__0"}
    # Metadata carries frontmatter and source_file.
    for c in chunks:
        assert c.metadata["type"] == "activity"
        assert c.metadata["source_file"] == "/x/abc.md"


def test_document_to_chunks_multi_chunk_indexing():
    long_en = "\n\n".join([" ".join(["w"] * 200)] * 3)
    doc = Document(
        doc_id="long",
        source_file="long.md",
        frontmatter={},
        sections={"en": long_en},
    )
    chunks = document_to_chunks(doc, max_words=200)
    assert [c.chunk_id for c in chunks] == [f"long__en__{i}" for i in range(len(chunks))]
    assert len(chunks) >= 2
