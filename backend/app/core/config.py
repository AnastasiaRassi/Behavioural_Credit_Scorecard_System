"""Runtime settings, read from the environment once.

The database and the session signing key need configuring; both are secrets and
both live in the gitignored .env. Model artifacts are resolved from the module's
own location (see feature_engineering.project_root), never from the working
directory or a setting.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationMissing(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def database_url() -> str:
    load_dotenv(project_root() / ".env")
    url = os.environ.get("DATABASE_URL")
    # No default: credentials live only in the gitignored .env.
    if not url:
        raise ConfigurationMissing(
            "DATABASE_URL is not set. Add it to .env, e.g.\n"
            "  DATABASE_URL=postgresql+psycopg://user:password@127.0.0.1:5432/dbname"
        )
    if not url.startswith("postgresql"):
        raise ConfigurationMissing(
            f"DATABASE_URL must be a postgresql:// URL, got {url!r}. "
            "The schema uses Postgres uuid and jsonb columns."
        )
    return url


@lru_cache(maxsize=1)
def session_secret() -> str:
    """The key session tokens are signed with."""
    load_dotenv(project_root() / ".env")
    secret = os.environ.get("SESSION_SECRET")
    # No default, and no generated-per-process fallback. A hardcoded default
    # would let anyone mint a valid token; a per-process one would sign people
    # out on every restart and hide the missing setting until deployment.
    if not secret:
        raise ConfigurationMissing(
            "SESSION_SECRET is not set, and sign-in tokens are signed with it. "
            "Add a random value to .env:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return secret


# Ollama serves the free local model behind the cardholder assistant. Neither
# setting is a secret, so both have working defaults for a local install.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"


@lru_cache(maxsize=1)
def ollama_url() -> str:
    load_dotenv(project_root() / ".env")
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


@lru_cache(maxsize=1)
def ollama_model() -> str:
    load_dotenv(project_root() / ".env")
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


# OpenAI is the assistant's first-choice provider when a key is set; without one the
# assistant falls back to Ollama, then to templates (ai/llm_client.py).
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=1)
def openai_api_key() -> str | None:
    """The key, or None when unset. Never raises: no key just means no OpenAI."""
    load_dotenv(project_root() / ".env")
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


@lru_cache(maxsize=1)
def openai_model() -> str:
    load_dotenv(project_root() / ".env")
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
