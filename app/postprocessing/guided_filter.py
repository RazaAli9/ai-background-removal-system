"""
Guided Filter Module

This module performs edge-aware refinement of segmentation masks
using OpenCV's Guided Filter algorithm.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from app.core.logger import logger
from app.core.exceptions import PostprocessingError


def prepare_guidance_image(image: Image.Image) -> np.ndarray:
    """
    Prepare the original RGB image for guided filtering.

    Args:
        image: Original input image.

    Returns:
        RGB image as a normalized float32 NumPy array.

    Raises:
        PostprocessingError: If image preparation fails.
    """
    try:
        logger.info("Preparing guidance image...")

        # Ensure RGB format
        image = image.convert("RGB")

        # Convert to NumPy array
        guidance = np.asarray(image, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        guidance /= 255.0

        logger.info("Guidance image prepared successfully.")

        return guidance

    except Exception as e:
        logger.exception("Failed to prepare guidance image.")
        raise PostprocessingError(
            "Failed to prepare guidance image."
        ) from e


def prepare_mask(mask: Image.Image) -> np.ndarray:
    """
    Prepare the segmentation mask for guided filtering.

    Args:
        mask: Segmentation mask.

    Returns:
        Grayscale mask as a normalized float32 NumPy array.

    Raises:
        PostprocessingError: If mask preparation fails.
    """
    try:
        logger.info("Preparing segmentation mask...")

        # Ensure grayscale format
        mask = mask.convert("L")

        # Convert to NumPy array
        mask_array = np.asarray(mask, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        mask_array /= 255.0

        logger.info("Segmentation mask prepared successfully.")

        return mask_array

    except Exception as e:
        logger.exception("Failed to prepare segmentation mask.")
        raise PostprocessingError(
            "Failed to prepare segmentation mask."
        ) from e


def guided_filter(
    guidance: np.ndarray,
    mask: np.ndarray,
    radius: int,
    epsilon: float,
) -> np.ndarray:
    """
    Apply OpenCV's guided filter to refine mask edges.

    Args:
        guidance: Normalized RGB guidance image.
        mask: Normalized grayscale mask.
        radius: Radius of the local filtering window.
        epsilon: Regularization parameter.

    Returns:
        Refined mask as a float32 NumPy array.

    Raises:
        PostprocessingError: If guided filtering fails.
    """
    try:
        logger.info("Applying guided filter...")

        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guidance,
            src=mask,
            radius=radius,
            eps=epsilon,
        )

        logger.info("Guided filter applied successfully.")

        return refined_mask

    except Exception as e:
        logger.exception("Failed to apply guided filter.")
        raise PostprocessingError(
            "Failed to apply guided filter."
        ) from e


def refine_edges(
    image: Image.Image,
    mask: Image.Image,
    radius: int = 8,
    epsilon: float = 1e-4,
) -> Image.Image:
    """
    Refine segmentation mask edges using Guided Filtering.

    Args:
        image: Original RGB image.
        mask: Refined segmentation mask.
        radius: Radius of the local filtering window.
        epsilon: Regularization parameter.

    Returns:
        Edge-refined segmentation mask as a PIL Image.

    Raises:
        PostprocessingError:
            If edge refinement fails.
    """
    try:
        logger.info("Starting edge refinement...")

        # Prepare inputs
        guidance = prepare_guidance_image(image)
        prepared_mask = prepare_mask(mask)

        # Apply guided filter
        refined_mask = guided_filter(
            guidance=guidance,
            mask=prepared_mask,
            radius=radius,
            epsilon=epsilon,
        )

        # Convert back to 8-bit image
        refined_mask = np.clip(refined_mask * 255.0, 0, 255).astype(np.uint8)

        # Convert NumPy array to PIL Image
        refined_mask = Image.fromarray(refined_mask)

        logger.info("Edge refinement completed successfully.")

        return refined_mask

    except Exception as e:
        logger.exception("Failed to refine mask edges.")
        raise PostprocessingError(
            "Failed to refine mask edges."
        ) from e
