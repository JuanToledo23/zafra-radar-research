"""Configuration loaded from the project .env file.

Credential values are never printed or logged. Use `require()` to get a key at the
point it is needed; it raises a clear error naming the missing variable.
"""

import os

from dotenv import load_dotenv

from zafra.paths import (  # noqa: F401  (re-exported for convenience)
    DATA_DIR,
    FIELDS_DIR,
    FIRMS_DIR,
    NISAR_DIR,
    OUTPUTS_DIR,
    ROOT,
    S1_DIR,
)

load_dotenv(ROOT / ".env")


class MissingConfigError(RuntimeError):
    """Raised when a required environment variable is absent or empty."""


def get(name: str, default: str | None = None) -> str | None:
    """Return an optional setting, or `default` if unset/empty."""
    value = os.getenv(name, "").strip()
    return value or default


def require(name: str) -> str:
    """Return a required setting or raise MissingConfigError naming it."""
    value = get(name)
    if value is None:
        raise MissingConfigError(
            f"Missing required setting {name}. Add it to {ROOT / '.env'} (see .env.example)."
        )
    return value


def earthdata_credentials() -> tuple[str, str]:
    return require("EARTHDATA_USERNAME"), require("EARTHDATA_PASSWORD")


def firms_map_key() -> str:
    return require("FIRMS_MAP_KEY")


def ee_project() -> str:
    return require("EE_PROJECT")
