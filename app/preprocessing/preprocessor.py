from PIL import Image
from torchvision import transforms

from app.config import settings
from app.core import (
    logger,
    InvalidImageError,
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        settings.NORMALIZE_MEAN,
        settings.NORMALIZE_STD
    )
])


def preprocess_image(image_path: str, device: str):
    """
    Load and preprocess an image for BiRefNet.
    """

    logger.info(f"Loading image: {image_path}")

    try:

        image = Image.open(image_path).convert("RGB")

    except Exception as e:

        logger.exception("Failed to load image.")

        raise InvalidImageError(
            f"Unable to load image: {e}"
        )

    input_tensor = (
        IMAGE_TRANSFORM(image)
        .unsqueeze(0)
        .to(device)
    )

    logger.info("Image preprocessed successfully.")

    return image, input_tensor