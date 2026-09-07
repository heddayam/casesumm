"""Data paths and API keys from environment variables or the project .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Read the project .env even when running from another directory.
# Values already set in the environment take precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def configured_path(name: str, default: str | Path) -> Path:
    """Resolve a directory from an environment variable or its default."""
    value = os.environ.get(name)
    if value is not None and not value.strip():
        raise ValueError(f"{name} must be a nonempty directory path")
    return Path(value if value is not None else default).expanduser().resolve()


DATA_DIR = configured_path("CASESUMM_DATA_DIR", PROJECT_ROOT / "data")


def require_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"Set {name} in the project .env or your environment before using this provider"
        )
    return value
