"""Minimal secrets loader — reads a gitignored .env into the environment.

No python-dotenv dependency. Secrets are NEVER hardcoded or committed
(CLAUDE.md §5); they live in .env (gitignored) or real environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Populate os.environ from a .env file (idempotent; existing vars win)."""
    global _loaded
    if _loaded:
        return
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    _loaded = True


def get_secret(name: str, default: str | None = None) -> str:
    """Return a secret from .env / environment, or raise a clear error."""
    load_dotenv()
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(
            f"Missing secret {name!r}. Set it in .env or export it as an "
            f"environment variable."
        )
    return value


def has_secret(name: str) -> bool:
    load_dotenv()
    return name in os.environ
