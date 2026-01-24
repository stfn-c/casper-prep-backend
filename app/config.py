from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # R2 Configuration
    r2_endpoint: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = ""

    # Database
    database_url: str = ""

    # OpenRouter API
    openrouter_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# For convenience - lazy loaded
_settings: Optional[Settings] = None


def get_settings_sync() -> Settings:
    """Get settings (lazy loaded)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
