"""Live connectivity check against the configured LLM provider.

Skipped unless ``RUN_LIVE_TESTS=1``, since it needs ``az login`` (or an OpenAI key) and network.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="live test")


def test_chat():
    from rag.llm import chat

    assert chat([{"role": "user", "content": "Reply with just: pong"}]).strip().lower() == "pong"
