from pathlib import Path

from app.core import (
    InvalidImageError,
)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".jfif",
}

def validate_image_path(
    image_path: str,
) -> Path:
    """
    Validate the input image path.

    Args:
        image_path: Path to input image.

    Returns:
        Validated Path object.

    Raises:
        InvalidImageError
    """

    path = Path(image_path)

    if not path.exists():
        raise InvalidImageError(
            f"Image not found: {image_path}"
        )

    if not path.is_file():
        raise InvalidImageError(
            f"Not a file: {image_path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported image format: {path.suffix}"
        )

    return path

def generate_output_filename(
    image_path: str,
) -> str:
    """
    Generate output filename.

    Example:
        person.jpg
            ↓
        person_removed.png
    """

    path = Path(image_path)

    return f"{path.stem}_removed.png"