"""
Configuration management with Pydantic settings
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    ENV: str = Field(default="development", env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")

    # Model Configuration
    MODEL_TYPE: str = Field(default="isolation_forest", env="MODEL_TYPE")
    MODEL_PATH: str = Field(default="models/trained/", env="MODEL_PATH")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")

    # Performance
    BATCH_SIZE: int = Field(default=1000, env="BATCH_SIZE")
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    USE_GPU: bool = Field(default=False, env="USE_GPU")

    # Security
    API_KEY: str = Field(default="", env="API_KEY")
    ENABLE_CORS: bool = Field(default=True, env="ENABLE_CORS")
    ALLOWED_ORIGINS: str = Field(default="*", env="ALLOWED_ORIGINS")

    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    CONFIG_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "config"
        / "config.yaml"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_data = self._load_yaml_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if self.CONFIG_PATH.exists():
            with open(self.CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        return {}

    def get_config(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value"""
        data = self._config_data
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
                if data is None:
                    return default
            else:
                return default
        return data

    @property
    def model_cfg(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get_config("model", default={})

    @property
    def feature_cfg(self) -> Dict[str, Any]:
        """Get feature configuration"""
        return self.get_config("features", default={})

    @property
    def parser_cfg(self) -> Dict[str, Any]:
        """Get parser configuration"""
        return self.get_config("parser", default={})

    @property
    def threshold_cfg(self) -> Dict[str, Any]:
        """Get threshold configuration"""
        return self.get_config("threshold", default={})

    @property
    def training_cfg(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get_config("training", default={})


# Global settings instance
settings = Settings()
