from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from app.core.config import Settings


def build_settings_from_env_file(env_file: str | None) -> Settings:
    if not env_file:
        return Settings()

    env_values: dict[str, Any] = {
        _to_settings_field_name(key): value
        for key, value in dotenv_values(Path(env_file)).items()
        if value is not None
    }
    return Settings(_env_file=env_file, **env_values)


def _to_settings_field_name(env_key: str) -> str:
    field_name = env_key.lower()
    if field_name in Settings.model_fields:
        return field_name
    return env_key
