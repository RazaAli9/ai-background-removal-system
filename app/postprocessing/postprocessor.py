from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.postprocessing.mask_refiner import refine_mask
from app.postprocessing.alpha_refiner import refine_alpha_mask
from app.postprocessing.hair_refiner import refine_hair
from app.postprocessing.foreground_refiner import refine_foreground
from app.core import (
    logger,
    ImageProcessingError,
)


def extract_prediction(
    preds: Union[torch.Tensor, list, tuple],
) -> torch.Tensor:
    """
    Extract the final prediction tensor from BiRefNet output.
    """
    if isinstance(preds, (list, tuple)):
        pred = preds[-1]
    else:
        pred = preds

    return pred


def normalize_prediction(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Apply sigmoid and return the 2D soft BiRefNet prediction.

    Values remain in float32 range [0, 1]. This soft prediction is
    intentionally preserved for image-aware mask refinement.
    """
    prediction = prediction.sigmoid()

    prediction = prediction.cpu()[0][0]

    return prediction


def resize_mask(
    mask: Image.Image,
    original_size: tuple[int, int],
) -> Image.Image:
    """
    Resize a display/processing mask back to original image size.
    """
    return mask.resize(
        original_size,
        Image.Resampling.LANCZOS,
    )


def resize_soft_prediction(
    prediction: torch.Tensor,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize the original BiRefNet sigmoid prediction while preserving
    float32 confidence values in [0, 1].

    This is deliberately separate from resize_mask(), because converting
    the prediction to an 8-bit PIL image too early loses confidence
    information useful to the intelligent hole classifier.
    """
    soft = prediction.detach().cpu().numpy().astype(np.float32)

    if soft.ndim != 2:
        soft = np.squeeze(soft)

    if soft.ndim != 2:
        raise ValueError(
            f"Expected 2D prediction, got shape {soft.shape}"
        )

    soft = cv2.resize(
        soft,
        original_size,
        interpolation=cv2.INTER_CUBIC,
    )

    return np.clip(
        soft,
        0.0,
        1.0,
    )


def apply_threshold(
    mask: Image.Image,
    threshold: int | None = None,
) -> Image.Image:
    """
    Apply optional binary threshold.

    The threshold affects the working mask only. The original soft
    BiRefNet prediction remains available separately.
    """
    if threshold is None:
        return mask

    return mask.point(
        lambda p: 255 if p >= threshold else 0
    )


def create_rgba_image(
    original_image: Image.Image,
    alpha_mask: Image.Image,
) -> Image.Image:
    """
    Apply alpha mask to original image.
    """
    rgba = original_image.convert("RGBA")
    rgba.putalpha(alpha_mask)
    return rgba


def postprocess_prediction(
    preds,
    original_image: Image.Image,
    threshold: int | None = None,
) -> Image.Image:
    """
    Convert BiRefNet prediction into the final transparent RGBA image.

    Pipeline:
        BiRefNet prediction
            ↓
        preserve soft prediction
            ↓
        resize mask
            ↓
        optional threshold
            ↓
        image-aware mask refinement
            ↓
        foreground structural refinement
            ↓
        alpha refinement
            ↓
        hair/fur refinement
            ↓
        final RGBA
    """
    logger.info("Starting postprocessing...")

    try:
        # ========================================================
        # 1. EXTRACT + NORMALIZE
        # ========================================================

        prediction = extract_prediction(preds)

        # IMPORTANT:
        # Keep this tensor as the original soft BiRefNet prediction.
        # Do not threshold it before image-aware refinement.
        prediction = normalize_prediction(
            prediction
        )

        # ========================================================
        # 2. PRESERVE SOFT PREDICTION
        # ========================================================

        soft_prediction = resize_soft_prediction(
            prediction=prediction,
            original_size=original_image.size,
        )

        # ========================================================
        # 3. CREATE WORKING MASK
        # ========================================================

        mask = transforms.ToPILImage()(
            prediction
        )

        mask = resize_mask(
            mask,
            original_image.size,
        )

        # ========================================================
        # 4. OPTIONAL THRESHOLD
        # ========================================================

        mask = apply_threshold(
            mask,
            threshold,
        )

        # ========================================================
        # 5. IMAGE-AWARE MASK REFINEMENT
        # ========================================================

        # The refiner receives:
        #   - working mask
        #   - original image
        #   - original soft BiRefNet confidence
        #
        # Therefore hole filling is no longer based only on
        # fixed hole size.
        mask = refine_mask(
            mask=mask,
            image=original_image,
            soft_mask=soft_prediction,
        )

        # ========================================================
        # 6. FOREGROUND STRUCTURAL REFINEMENT
        # ========================================================

        refined_foreground = refine_foreground(
            mask,
        )

        mask = Image.fromarray(
            np.clip(
                refined_foreground * 255.0,
                0,
                255,
            ).astype(np.uint8),
            mode="L",
        )

        # ========================================================
        # 7. ALPHA REFINEMENT
        # ========================================================

        alpha = refine_alpha_mask(
            image=original_image,
            mask=mask,
        )

        # ========================================================
        # 8. HAIR / FUR REFINEMENT
        # ========================================================

        alpha = refine_hair(
            image=original_image,
            alpha=alpha,
        )

        # ========================================================
        # 9. FINAL RGBA
        # ========================================================

        output = create_rgba_image(
            original_image,
            alpha,
        )

        logger.info(
            "Postprocessing completed successfully."
        )

        return output

    except Exception as e:
        logger.exception(
            "Postprocessing failed."
        )

        raise ImageProcessingError(
            f"Unable to process prediction: {e}"
        ) from e
