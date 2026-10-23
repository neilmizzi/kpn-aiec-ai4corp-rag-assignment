"""Answer generation grounded in retrieved chunks, with doc-ID citations and abstention."""

import re

from rag.llm import chat

ABSTAIN = "The knowledge source does not contain enough information to answer this question."
MAX_CHUNK_CHARS = 4000

SYSTEM_PROMPT = f"""You answer technical questions about LangChain using ONLY the numbered \
sources provided. Do not use outside knowledge.

Rules:
- Cite every claim with the source ID in square brackets, e.g. [langchain/docs/x.mdx_0_996]. \
Copy IDs exactly. Cite only sources that support the claim.
- Many questions are debugging questions. If the sources show the relevant API, setup or \
configuration, explain what they indicate (likely cause, correct usage, steps to check) even if \
they do not mention the user's exact error.
- If the sources answer only part of the question, answer that part and say what is missing.
- Only if the sources are unrelated to the question, reply with exactly:
  {ABSTAIN}
  Then, in one sentence, say what is missing. Do not guess.
- Include short code snippets when the sources contain them and they help.
- Be concise."""

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_ABSTAIN_PATTERN = re.compile(r"do(?:es)? not contain enough information", re.IGNORECASE)


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources labelled with their doc IDs.

    Args:
        chunks: Retrieved chunks with ``id`` and ``text``.

    Returns:
        Context string for the prompt; each chunk is truncated to ``MAX_CHUNK_CHARS``.
    """
    return "\n\n".join(
        f"### Source {i}: [{c['id']}]\n{c['text'][:MAX_CHUNK_CHARS]}"
        for i, c in enumerate(chunks, 1)
    )


def extract_citations(text: str, allowed_ids: list[str]) -> list[str]:
    """Find cited doc IDs in an answer, keeping only IDs that were actually retrieved.

    Args:
        text: Answer text with ``[doc_id]`` citations; one bracket may hold several IDs
            separated by commas or semicolons.
        allowed_ids: IDs of the retrieved chunks.

    Returns:
        Unique valid cited IDs, in order of first appearance.
    """
    allowed = set(allowed_ids)
    cited = []
    for group in _CITATION.findall(text):
        for doc_id in re.split(r"[;,]\s*", group):
            doc_id = doc_id.strip()
            if doc_id in allowed and doc_id not in cited:
                cited.append(doc_id)
    return cited


def answer(question: str, chunks: list[dict]) -> dict:
    """Answer a question from retrieved chunks, or state that the corpus lacks the information.

    Args:
        question: The user question.
        chunks: Retrieved corpus chunks to ground the answer in.

    Returns:
        Dict with ``answer`` (text with ``[doc_id]`` citations), ``citations`` (valid cited IDs)
        and ``abstained`` (True if the answer opens by saying the sources lack the information
        and cites nothing).
    """
    if not chunks:
        return {"answer": ABSTAIN, "citations": [], "abstained": True}
    text = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n\n{format_context(chunks)}\n\nQuestion:\n{question}",
            },
        ]
    )
    citations = extract_citations(text, [c["id"] for c in chunks])
    first_line = text.strip().split("\n", 1)[0]
    return {
        "answer": text,
        "citations": citations,
        "abstained": not citations and bool(_ABSTAIN_PATTERN.search(first_line)),
    }
