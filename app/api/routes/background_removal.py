from pathlib import Path
import tempfile
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.services.background_removal_service import (
    BackgroundRemovalService,
)
from app.core.logger import logger
from app.config.constants import MAX_IMAGE_SIZE_MB


router = APIRouter(
    prefix="/api/v1",
    tags=["Background Removal"],
)


# Supported input formats
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".jfif",
}


# Create the service once when the API process starts.
# This prevents BiRefNet from being loaded for every request.
_service = None


def get_service() -> BackgroundRemovalService:
    global _service

    if _service is None:
        logger.info(
            "Creating BackgroundRemovalService for FastAPI..."
        )
        _service = BackgroundRemovalService()

    return _service


def _delete_file(path: Path) -> None:
    """Best-effort cleanup after a streamed API response is sent."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to remove generated output file.")


@router.post(
    "/remove-background",
    summary="Remove image background",
    response_class=FileResponse,
)
async def remove_background(
    file: UploadFile = File(...),
):
    """
    Remove the background from an uploaded image.

    The existing BackgroundRemovalService performs:

        Upload
          ↓
        Validation
          ↓
        Preprocessing
          ↓
        BiRefNet inference
          ↓
        Postprocessing
          ↓
        Transparent PNG

    Returns:
        Transparent PNG image.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                f"Supported formats: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    temp_input_path = None

    try:
        # Create a unique temporary input filename.
        temp_filename = (
            f"{uuid.uuid4().hex}{extension}"
        )

        temp_dir = Path(
            tempfile.gettempdir()
        ) / "bg_remover_inputs"

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_input_path = (
            temp_dir / temp_filename
        )

        # Save uploaded file with a bounded request size.
        max_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
        received_bytes = 0

        with temp_input_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                received_bytes += len(chunk)
                if received_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.",
                    )

                buffer.write(chunk)

        logger.info("API received an image upload.")

        # Get the already-existing production service.
        service = get_service()

        # Unique output filename.
        output_filename = (
            f"{uuid.uuid4().hex}.png"
        )

        # Run existing background-removal pipeline.
        saved_path = service.remove_background(
            image_path=str(
                temp_input_path
            ),
            output_filename=output_filename,
        )

        saved_path = Path(saved_path)

        if not saved_path.exists():
            raise RuntimeError(
                "Background removal completed, "
                "but output file was not found."
            )

        logger.info(
            f"API background removal completed: "
            f"{saved_path}"
        )

        return FileResponse(
            path=str(saved_path),
            media_type="image/png",
            filename="background_removed.png",
            background=BackgroundTask(_delete_file, saved_path),
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "API background removal failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Background removal failed."
            ),
        ) from e

    finally:

        # Delete temporary uploaded input.
        if (
            temp_input_path is not None
            and temp_input_path.exists()
        ):
            try:
                temp_input_path.unlink()
            except Exception:
                logger.warning(
                    "Failed to remove temporary "
                    f"input file: {temp_input_path}"
                )