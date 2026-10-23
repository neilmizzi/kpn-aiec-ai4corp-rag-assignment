"""Runtime settings read from environment variables (see README)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "kpn_gateway")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-54-mini-2026-03-17")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-54-2026-03-05")
TOP_K = int(os.getenv("TOP_K", "8"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
