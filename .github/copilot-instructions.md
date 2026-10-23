# Copilot instructions

## Keep the user in the loop

Before changing code or running experiments, explain the next bounded action, why it matters,
and its expected duration. Get agreement before moving to a new stage or making an architecture,
model, dependency or scope decision. Do not treat approval of one step as approval of later steps.
Explain results briefly, distinguish measured facts from assumptions, and pause for the user
to review them before continuing. Commit only when explicitly requested.

## Keep it simple

Build only what the assignment or a measured problem requires. Prefer the standard library
and existing dependencies. Avoid speculative abstractions, configuration and new services.
Keep changes small, preserve error handling, and verify behavior with focused checks.

## Docstrings

Every function and method you add or change gets a Google-style docstring with:

1. A one-line description (plus detail paragraph if needed).
2. `Args:` for every parameter (no types; they live in the signature, which must be fully
   type-hinted).
3. `Returns:` describing the return value, unless it returns `None`.
4. `Raises:` for every exception raised in the body.

```python
def retrieve(question: str, k: int) -> list[dict]:
    """Retrieve the top-k corpus chunks for a question.

    Args:
        question: The user question.
        k: Number of chunks to return.

    Returns:
        Chunks ordered by fused relevance score, best first.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
```

Enforced by `pydocstyle` and `pydoclint` in pre-commit (`uv run pre-commit run --all-files`).
Tests are exempt.
