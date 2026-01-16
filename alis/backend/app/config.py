"""
Configuration management for ALIS application.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "ALIS - Aadhaar Lifecycle Intelligence System"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]
    
    # Paths (defined early so DATABASE_URL can use them)
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Database - use absolute path based on BASE_DIR
    @property
    def DATABASE_URL(self) -> str:
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
        return f"sqlite:///{self.DATA_DIR / 'alis.db'}"
    
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    # ML Model Settings
    FORECAST_HORIZON_DAYS: int = 30
    ANOMALY_CONFIDENCE_THRESHOLD: float = 0.7
    RISK_SCORE_MAX: int = 100
    
    # Data Settings
    MIN_K_ANONYMITY: int = 5
    DATA_FRESHNESS_HOURS: int = 24
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600
    
    # Aadhaar API Settings (data.gov.in)
    AADHAAR_API_KEY: str = Field(
        default="579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        env="AADHAAR_API_KEY"
    )
    AADHAAR_API_URL: str = Field(
        default="https://api.data.gov.in/resource",
        env="AADHAAR_API_URL"
    )
    AADHAAR_RESOURCE_ID: str = Field(
        default="ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        env="AADHAAR_RESOURCE_ID"
    )
    API_FETCH_LIMIT: int = Field(default=1000, env="API_FETCH_LIMIT")
    API_MAX_RECORDS: int = Field(default=0, env="API_MAX_RECORDS")  # 0 = no limit
    
    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    
    # Create directories if they don't exist
    for directory in [settings.DATA_DIR, settings.RAW_DATA_DIR, 
                      settings.PROCESSED_DATA_DIR, settings.MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return settings


settings = get_settings()