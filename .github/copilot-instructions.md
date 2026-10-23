# Copilot instructions

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
