"""Runtime settings, read from the environment once.

Only the database needs configuring. Model artifacts are resolved from the
module's own location (see feature_engineering.project_root), never from the
working directory or a setting.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationMissing(RuntimeError):
    pass


# Local Postgres. Set DATABASE_URL in .env to supply the password, which is why
# no credentials are hardcoded here. psycopg (v3) driver, hence the +psycopg
# suffix; a plain postgresql:// URL would reach for psycopg2 instead.
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres@127.0.0.1:5432/postgres"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def database_url() -> str:
    # .env holds the password and is gitignored, so credentials never reach the repo.
    load_dotenv(project_root() / ".env")
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if not url.startswith("postgresql"):
        raise ConfigurationMissing(
            f"DATABASE_URL must be a postgresql:// URL, got {url!r}. "
            "The schema uses Postgres uuid and jsonb columns."
        )
    return url
