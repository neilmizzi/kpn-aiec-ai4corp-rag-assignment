"""OpenAI-compatible client for chat and embeddings.

Two providers are supported:

* ``kpn_gateway`` (default): KPN GenAI Gateway, authenticated with an EntraID token from the local
  ``az login`` session. No API key is needed; KPN's private CA is trusted via the OS certificate
  store (``truststore``).
* ``openai``: public OpenAI API using ``OPENAI_API_KEY``, for running outside the KPN network.
"""

import json
import shutil
import subprocess
import time
from functools import lru_cache

from openai import OpenAI

from rag.config import EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER

GATEWAY_ENDPOINT = "https://api.acc.genai.kpn.org/openai/v1"
GATEWAY_SCOPE = "api://api.acc.genai.kpn.org/.default"

_token = {"value": "", "expires_on": 0}


def _gateway_token() -> str:
    """Get an EntraID token for the gateway from the Azure CLI, cached until 5 min before expiry.

    Returns:
        A bearer access token.

    Raises:
        RuntimeError: If the Azure CLI cannot provide a token, e.g. when not logged in.
    """
    if time.time() > _token["expires_on"] - 300:
        az = shutil.which("az") or "az"
        result = subprocess.run(
            [az, "account", "get-access-token", "--scope", GATEWAY_SCOPE, "-o", "json"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Azure CLI token request failed; run `az login --allow-no-subscriptions`.\n"
                f"{result.stderr.strip()}"
            )
        data = json.loads(result.stdout)
        _token.update(value=data["accessToken"], expires_on=int(data["expires_on"]))
    return _token["value"]


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Build the OpenAI client for the configured provider.

    Returns:
        A client pointed at the KPN GenAI Gateway or the public OpenAI API.

    Raises:
        ValueError: If ``LLM_PROVIDER`` is not ``kpn_gateway`` or ``openai``.
    """
    if LLM_PROVIDER == "openai":
        return OpenAI()
    if LLM_PROVIDER != "kpn_gateway":
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")

    import truststore

    truststore.inject_into_ssl()
    return OpenAI(base_url=GATEWAY_ENDPOINT, api_key=_gateway_token)


def chat(messages: list[dict], model: str | None = None) -> str:
    """Run a chat completion.

    Args:
        messages: OpenAI-style chat messages.
        model: Model/deployment name; defaults to ``LLM_MODEL``.

    Returns:
        The assistant message content.
    """
    response = get_client().chat.completions.create(model=model or LLM_MODEL, messages=messages)
    return response.choices[0].message.content


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with ``EMBEDDING_MODEL``.

    Args:
        texts: Texts to embed.

    Returns:
        One embedding vector per input text, in input order.
    """
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
