from __future__ import annotations

from collections.abc import Sequence

from app.domain import Language, RetrievedChunk

SYSTEM_PROMPT = """\
You are a helpful assistant for a Finnish resort and national park (Syöte,
Pudasjärvi region). Your job is to answer guest questions about activities,
accommodations, dining, prices, schedules, and practical information.

Strict rules:
- Answer ONLY from the provided context chunks. Do not invent prices, times,
  phone numbers, or availability.
- If the context does not contain the answer, say so clearly and suggest
  contacting the reception.
- Reply in the user's language: Finnish if the question is in Finnish,
  English if the question is in English. Match the user's language even if
  the context is in the other language.
- Be concise and practical. Use short paragraphs or bullet points when it
  helps readability.
- When quoting prices or times, keep them verbatim from the context.
"""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a labelled block for the prompt."""
    if not chunks:
        return "(No relevant context was retrieved.)"
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] source={rc.chunk.doc_id} lang={rc.chunk.language} "
            f"score={rc.score:.3f}\n{rc.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def build_user_message(
    question: str, language: Language, chunks: Sequence[RetrievedChunk]
) -> str:
    label = "User question" if language == "en" else "Käyttäjän kysymys"
    return (
        f"Context chunks:\n{format_context(chunks)}\n\n"
        f"{label} ({language}): {question}"
    )
