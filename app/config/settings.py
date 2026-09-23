from typing import Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    MODEL_PATH: str = "ZhengPeng7/BiRefNet"
    DEVICE: str = "auto"

    IMAGE_SIZE: int = 1024
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    OUTPUT_DIR: str = "output"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
