import torch
from transformers import AutoModelForImageSegmentation

from app.config import settings
from app.core import (
    logger,
    ImageProcessingError,
)


def get_device() -> str:
    """
    Returns the best available device.
    """

    if settings.DEVICE.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return settings.DEVICE.lower()


def load_model():
    """
    Load BiRefNet model.
    """

    device = get_device()

    logger.info(f"Loading model on {device.upper()}...")

    try:

        model = AutoModelForImageSegmentation.from_pretrained(
            settings.MODEL_PATH,
            trust_remote_code=True,
        )

        model = model.float()

        model.to(device)

        model.eval()

        logger.info("Model loaded successfully.")

        return model, device

    except Exception as e:

        logger.exception("Failed to load model.")

        raise ImageProcessingError(
            f"Unable to load model: {e}"
        )