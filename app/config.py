"""Application configuration from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"

    github_token: str = ""

    host: str = "0.0.0.0"
    port: int = 8080

    max_tool_rounds: int = 15
    shell_timeout_seconds: int = 60

    workspace_dir: str = "/app/workspace"

    # Phase 2 — interactive desktop
    desktop_api_url: str = "http://desktop:7090"
    novnc_public_url: str = "http://localhost:6080"

    @property
    def workspace_path(self) -> Path:
        path = Path(self.workspace_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
