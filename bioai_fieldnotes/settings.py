from __future__ import annotations

import os
from pathlib import Path


_LOADED_ENV_FILES: set[Path] = set()


def load_dotenv(path: str | Path | None = None) -> None:
    """Load simple KEY=value pairs from .env without overwriting real env vars."""
    paths = [Path(path)] if path else _default_env_paths()
    for candidate in paths:
        _load_one_env_file(candidate)


def _load_one_env_file(path: Path) -> None:
    env_path = path.expanduser().resolve()
    if env_path in _LOADED_ENV_FILES or not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, value = _parse_env_line(line)
        if key and key not in os.environ:
            os.environ[key] = value
    _LOADED_ENV_FILES.add(env_path)


def env(name: str, default: str | None = None) -> str | None:
    load_dotenv()
    return os.environ.get(name, default)


def has_env(name: str) -> bool:
    value = env(name)
    return bool(value and value.strip())


def runtime_status() -> dict[str, bool]:
    return {
        "x_bearer_token": has_env("X_BEARER_TOKEN"),
        "openai_api_key": has_env("OPENAI_API_KEY"),
        "modelprices": has_env("BIOAI_MODELPRICES_PATH")
        or env("BIOAI_ALLOW_FALLBACK_PRICING") == "1",
    }


def _default_env_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("BIOAI_ENV_PATH")
    if explicit:
        paths.append(Path(explicit))
    paths.append(Path.cwd() / ".env")
    return paths


def _parse_env_line(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, ""
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]
    return key, value
