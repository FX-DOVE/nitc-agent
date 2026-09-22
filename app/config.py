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
    openai_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "meta-llama/llama-3.3-70b-instruct:free"

    github_token: str = ""

    host: str = "0.0.0.0"
    port: int = 8080

    max_tool_rounds: int = 15
    shell_timeout_seconds: int = 60

    # Parallel agent job workers (FIFO queue; never cancel in-flight)
    nitc_job_concurrency: int = 2

    workspace_dir: str = "/app/workspace"

    # Phase 2 — interactive desktop
    desktop_api_url: str = "http://desktop:7090"
    # Direct :6080 URL (optional fullscreen / legacy). Embed uses same-origin /novnc/.
    novnc_public_url: str = "http://localhost:6080"
    # Upstream for the agent’s /novnc/ reverse proxy (Docker DNS).
    novnc_upstream: str = "http://desktop:6080"

    @property
    def workspace_path(self) -> Path:
        path = Path(self.workspace_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
