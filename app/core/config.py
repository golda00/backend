"""
Central application configuration.
All settings are loaded from environment variables or a .env file.
"""
import logging
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


# Base directory of this file -> project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App Meta ---
    APP_NAME: str = "Telecom Vision API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    BASE_DIR: Path = BASE_DIR

    # --- Model Weights ---
    MODELS_DIR: Path = BASE_DIR / "model_weights"

    def _resolve_model(self, filename: str) -> Path:
        """
        Safely resolves a model path. 
        If the specific model file (e.g. 'fiber_node_model.pt') is missing, 
        it falls back to 'best.pt' to prevent FileNotFoundError crashes.
        """
        path = self.MODELS_DIR / filename
        if not path.exists():
            fallback = self.MODELS_DIR / "best.pt"
            if fallback.exists():
                logger.info(f"⚠️ Model {filename} not found, falling back to {fallback.name}")
                return fallback
        return path

    @property
    def MAIN_MODEL_PATH(self) -> Path:
        return self._resolve_model("best.pt")

    @property
    def PS_MODEL_PATH(self) -> Path:
        return self._resolve_model("power_supply_best.pt")

    @property
    def NODE_MODEL_PATH(self) -> Path:
        return self._resolve_model("3x3_4x4_new_model.pt")

    @property
    def INTERNAL_MODEL_PATH(self) -> Path:
        return self._resolve_model("Internal_best.pt")

    @property
    def FIBER_NODE_MODEL_PATH(self) -> Path:
        return self._resolve_model("fiber_node_model.pt")

    # --- Storage ---
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = BASE_DIR / "storage" / "uploads"
    OUTPUTS_DIR: Path = BASE_DIR / "storage" / "outputs"

    # --- Processing ---
    PDF_DPI: int = 300
    TILE_SIZE: int = 640
    TILE_OVERLAP: float = 0.2
    USE_GPU: bool = True

    # --- API ---
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]

    # --- Safety & Performance ---
    # Max upload size per file (default 200 MB)
    MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024
    # Max seconds a pipeline job may run before being killed (default 1 hour)
    JOB_TIMEOUT_SECONDS: float = 3600.0
    # Number of dedicated pipeline worker threads (separate from FastAPI's I/O pool)
    PIPELINE_WORKERS: int = 4
    # Hours after which completed/failed jobs and their files are auto-deleted
    JOB_RETENTION_HOURS: float = 24.0

    # --- Redis & Celery (Batch 3) ---
    REDIS_URL: str = "redis://localhost:6379/0"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.REDIS_URL

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.REDIS_URL


@lru_cache
def get_settings() -> Settings:
    """Returns a cached singleton of the Settings object."""
    return Settings()
