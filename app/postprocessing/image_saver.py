from pathlib import Path
from PIL import Image

from app.config import settings
from app.core import (
    logger,
    ImageProcessingError,
)


def save_image(
    image: Image.Image,
    filename: str = "output.png",
):
    """
    Save RGBA image to the output directory.

    Args:
        image: RGBA image
        filename: Output filename

    Returns:
        Path to saved image
    """

    logger.info("Saving output image...")

    try:

        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_dir / filename

        image.save(output_path)

        logger.info(
            f"Image saved successfully: {output_path}"
        )

        return output_path

    except Exception as e:

        logger.exception("Failed to save image.")

        raise ImageProcessingError(
            f"Unable to save image: {e}"
        )