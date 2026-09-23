from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from app.core.exceptions import PostprocessingError
from app.core.logger import logger


EDGE_CONFIDENCE_WEIGHT = 0.75
CONTRAST_WEIGHT = 0.25

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0



def prepare_image(
    image: Image.Image,
) -> np.ndarray:
    """
    Prepare the original RGB image for hair refinement.

    Args:
        image:
            Original input image.

    Returns:
        Normalized RGB image as a float32 NumPy array.

    Raises:
        PostprocessingError:
            If image preparation fails.
    """

    try:

        logger.info(
            "Preparing image for hair refinement..."
        )

        # Ensure RGB format
        image = image.convert("RGB")

        # Convert to NumPy array
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        # Normalize pixel values to [0, 1]
        image_array /= 255.0

        logger.info(
            "Image prepared successfully."
        )

        return image_array

    except Exception as e:

        logger.exception(
            "Failed to prepare image."
        )

        raise PostprocessingError(
            "Failed to prepare image."
        ) from e

def prepare_alpha(
    alpha: Image.Image,
) -> np.ndarray:
    """
    Prepare the alpha mask for hair refinement.

    Args:
        alpha:
            Refined alpha mask.

    Returns:
        Normalized alpha mask as a float32 NumPy array.

    Raises:
        PostprocessingError:
            If alpha preparation fails.
    """

    try:

        logger.info(
            "Preparing alpha mask for hair refinement..."
        )

        # Ensure grayscale format
        alpha = alpha.convert("L")

        # Convert to NumPy array
        alpha_array = np.asarray(
            alpha,
            dtype=np.float32,
        )

        # Normalize to [0, 1]
        alpha_array /= 255.0

        logger.info(
            "Alpha mask prepared successfully."
        )

        return alpha_array

    except Exception as e:

        logger.exception(
            "Failed to prepare alpha mask."
        )

        raise PostprocessingError(
            "Failed to prepare alpha mask."
        ) from e


def compute_multiscale_gradients(
    image: np.ndarray,
) -> np.ndarray:
    """
    Compute multi-scale gradient magnitude of the input image.

    The image is analyzed at multiple Gaussian scales to detect
    fine, medium, and coarse edge structures. The resulting
    normalized gradient maps are averaged into a single response.

    Args:
        image:
            Normalized RGB image as a float32 NumPy array
            with shape (H, W, 3).

    Returns:
        Multi-scale gradient magnitude as a float32 array
        with shape (H, W) and values in the range [0, 1].

    Raises:
        PostprocessingError:
            If gradient computation fails.
    """

    try:

        logger.info(
            "Computing multi-scale gradients..."
        )

        # Convert RGB image to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # Gaussian kernel sizes
        scales = (3, 5, 9)

        gradient_maps = []

        for kernel_size in scales:

            # Blur image at current scale
            blurred = cv2.GaussianBlur(
                gray,
                (kernel_size, kernel_size),
                sigmaX=0,
            )

            # Compute Sobel gradients
            grad_x = cv2.Sobel(
                blurred,
                cv2.CV_32F,
                1,
                0,
                ksize=3,
            )

            grad_y = cv2.Sobel(
                blurred,
                cv2.CV_32F,
                0,
                1,
                ksize=3,
            )

            # Gradient magnitude
            magnitude = cv2.magnitude(
                grad_x,
                grad_y,
            )
            gradient_maps.append(magnitude)

        # Average gradients across all scales
        gradient_response = np.mean(
            gradient_maps,
            axis=0,
        ).astype(np.float32)

        # Normalize final response
        gradient_response = cv2.normalize(
            gradient_response,
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        logger.info(
            "Multi-scale gradients computed successfully."
        )

        return gradient_response

    except Exception as e:

        logger.exception(
            "Failed to compute multi-scale gradients."
        )

        raise PostprocessingError(
            "Failed to compute multi-scale gradients."
        ) from e


def compute_local_contrast(
    image: np.ndarray,
    window_size: int = 9,
) -> np.ndarray:
    """
    Compute the local contrast map of the input image.

    Local contrast is estimated using the local standard deviation
    within a sliding window. Hair, beard, fur, and other textured
    regions exhibit higher local contrast than smooth surfaces.

    Args:
        image:
            Normalized RGB image as a float32 NumPy array
            with shape (H, W, 3).

        window_size:
            Size of the local neighborhood used for
            contrast estimation.

    Returns:
        Local contrast response as a float32 array
        with shape (H, W) and values in the range [0, 1].

    Raises:
        PostprocessingError:
            If local contrast computation fails.
    """

    try:

        logger.info(
            "Computing local contrast..."
        )

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # Compute local mean
        local_mean = cv2.boxFilter(
            gray,
            ddepth=cv2.CV_32F,
            ksize=(window_size, window_size),
            normalize=True,
        )

        # Compute mean of squared intensities
        local_mean_sq = cv2.boxFilter(
            gray * gray,
            ddepth=cv2.CV_32F,
            ksize=(window_size, window_size),
            normalize=True,
        )

        # Variance
        variance = local_mean_sq - (local_mean * local_mean)

        # Prevent tiny negative values caused by floating-point precision
        variance = np.maximum(
            variance,
            0.0,
        )

        # Standard deviation
        contrast = np.sqrt(
            variance,
        )

        # Normalize once
        contrast = cv2.normalize(
            contrast,
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        logger.info(
            "Local contrast computed successfully."
        )

        return contrast.astype(np.float32)

    except Exception as e:

        logger.exception(
            "Failed to compute local contrast."
        )

        raise PostprocessingError(
            "Failed to compute local contrast."
        ) from e


def compute_texture_response(
    image: np.ndarray,
) -> np.ndarray:
    """
    Compute the texture response of the input image.

    Texture response is estimated using the Laplacian operator,
    which highlights fine, high-frequency structures such as
    hair strands, beard, fur, and feathers.

    Args:
        image:
            Normalized RGB image as a float32 NumPy array
            with shape (H, W, 3).

    Returns:
        Texture response as a float32 array
        with shape (H, W) and values in the range [0, 1].

    Raises:
        PostprocessingError:
            If texture computation fails.
    """

    try:

        logger.info(
            "Computing texture response..."
        )

        # Convert RGB to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # Light denoising while preserving texture
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            sigmaX=0,
        )

        # Compute Laplacian response
        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_32F,
            ksize=3,
        )

        # Magnitude only
        texture = np.abs(
            laplacian,
        )

        # Normalize once
        texture = cv2.normalize(
            texture,
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        logger.info(
            "Texture response computed successfully."
        )

        return texture.astype(
            np.float32,
        )

    except Exception as e:

        logger.exception(
            "Failed to compute texture response."
        )

        raise PostprocessingError(
            "Failed to compute texture response."
        ) from e


def detect_hair_candidates(
    gradient: np.ndarray,
    contrast: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    """
    Generate a hair probability map by combining multiple feature
    responses.

    Args:
        gradient:
            Multi-scale gradient response.

        contrast:
            Local contrast response.

        texture:
            Texture response.

    Returns:
        Hair probability map as a float32 array
        with values in the range [0, 1].

    Raises:
        PostprocessingError:
            If hair candidate detection fails.
    """

    try:

        logger.info(
            "Detecting hair candidate regions..."
        )

        # --------------------------------------------------
        # Normalize feature maps
        # --------------------------------------------------

        gradient = cv2.normalize(
            gradient,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        contrast = cv2.normalize(
            contrast,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        texture = cv2.normalize(
            texture,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Hair-like structures should have BOTH
        # strong gradients and strong texture.

        edge_confidence = gradient * texture

        edge_confidence = cv2.normalize(
            edge_confidence,
            None,
            alpha=0.0,
            beta=1.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # --------------------------------------------------
        # Stage 2: Hair probability
        # --------------------------------------------------

        # Local contrast strengthens already detected
        # hair structures instead of dominating them.

        # hair_probability = (
        #     0.75 * edge_confidence
        #    + 0.25 * contrast
        # )
        hair_probability = (
        EDGE_CONFIDENCE_WEIGHT * edge_confidence
        + CONTRAST_WEIGHT * contrast
    )

        # --------------------------------------------------
        # Remove isolated noise
        # --------------------------------------------------

        hair_probability = cv2.GaussianBlur(
            hair_probability,
            GAUSSIAN_KERNEL,
            GAUSSIAN_SIGMA,
        )

        # --------------------------------------------------
        # Normalize final probability map
        # --------------------------------------------------

        hair_probability = cv2.normalize(
            hair_probability,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        logger.info(
            "Hair candidate detection completed successfully."
        )

        return hair_probability.astype(
            np.float32,
        )

    except Exception as e:

        logger.exception(
            "Failed to detect hair candidates."
        )

        raise PostprocessingError(
            "Failed to detect hair candidates."
        ) from e


def enhance_hair(
    alpha: np.ndarray,
    hair_probability: np.ndarray,
    enhancement_strength: float = 0.30,
) -> np.ndarray:
    """
    Generate a hair enhancement map using the detected
    hair probability map.

    Enhancement is applied only around uncertain alpha
    boundary pixels. This function does NOT modify the
    alpha matte. It only computes the enhancement map.

    Args:
        alpha:
            Normalized alpha matte with values in [0, 1].

        hair_probability:
            Hair probability map with values in [0, 1].

        enhancement_strength:
            Strength of hair enhancement.

    Returns:
        Hair enhancement map with values in [0, 1].

    Raises:
        PostprocessingError:
            If hair enhancement fails.
    """

    try:

        logger.info(
            "Enhancing hair regions..."
        )

        # -----------------------------------------
        # Boundary region
        # -----------------------------------------

        boundary_mask = (
            (alpha > 0.10)
            &
            (alpha < 0.90)
        ).astype(np.float32)

        # -----------------------------------------
        # Hair enhancement map
        # -----------------------------------------

        enhancement = (
            boundary_mask
            * hair_probability
            * enhancement_strength
        )

        enhancement = cv2.GaussianBlur(
            enhancement,
            (3, 3),
            0,
        )

        # Keep enhancement only on boundary pixels
        enhancement *= boundary_mask

        enhancement = np.clip(
            enhancement,
            0.0,
            1.0,
        )

        logger.info(
            "Hair enhancement completed successfully."
        )

        return enhancement.astype(
            np.float32,
        )

    except Exception as e:

        logger.exception(
            "Failed to enhance hair."
        )

        raise PostprocessingError(
            "Failed to enhance hair."
        ) from e


def merge_alpha(
    alpha: np.ndarray,
    enhancement: np.ndarray,
    enhancement_strength: float = 1.0,
) -> np.ndarray:
    """
    Merge the original alpha matte with the hair enhancement map.
    The enhancement map contains only boundary-region values.
    These values are added to the original alpha matte to
    strengthen fine hair structures while preserving confident
    foreground and background regions.

    Args:
        alpha:
            Original normalized alpha matte
            with values in the range [0, 1].
        enhancement:
            Hair enhancement map produced by
            ``enhance_hair()``.
        enhancement_strength:
            Global multiplier controlling how much
            enhancement is applied.
    Returns:
        Refined alpha matte as a float32 NumPy array
        with values in the range [0, 1].
    Raises:
        PostprocessingError:
            If alpha merging fails.
    """

    try:

        logger.info(
            "Merging hair enhancement into alpha matte..."
        )

        # Validate shapes
        if alpha.shape != enhancement.shape:

            raise ValueError(
                "Alpha matte and enhancement map "
                "must have identical shapes."
            )

        # Merge enhancement
        merged_alpha = (
            alpha
            + enhancement_strength * enhancement
        )

        # Keep values valid
        merged_alpha = np.clip(
            merged_alpha,
            0.0,
            1.0,
        )

        merged_alpha = merged_alpha.astype(
            np.float32,
        )

        logger.info(
            "Hair enhancement merged successfully."
        )

        return merged_alpha

    except Exception as e:

        logger.exception(
            "Failed to merge alpha matte."
        )

        raise PostprocessingError(
            "Failed to merge alpha matte."
        ) from e


def refine_hair(
    image: Image.Image,
    alpha: Image.Image,
) -> Image.Image:
    """
    Perform complete hair refinement.

    Pipeline

        Original Image
              │
              ▼
        prepare_image()
              │
              ▼
        Prepared Image
              │
              ├─────────────► compute_multiscale_gradients()
              │
              ├─────────────► compute_local_contrast()
              │
              └─────────────► compute_texture_response()
                              │
                              ▼
                    detect_hair_candidates()
                              │
                              ▼
                       prepare_alpha()
                              │
                              ▼
                      enhance_hair()
                              │
                              ▼
                       merge_alpha()
                              │
                              ▼
                     Refined Alpha Matte
                              │
                              ▼
                         PIL Image

    Args:
        image:
            Original RGB image.

        alpha:
            Alpha matte produced by the previous module.

    Returns:
        Hair-refined alpha matte as a PIL grayscale image.

    Raises:
        PostprocessingError:
            If hair refinement fails.
    """

    try:

        logger.info(
            "Starting hair refinement..."
        )

        # ---------------------------------
        # Prepare inputs
        # ---------------------------------

        prepared_image = prepare_image(
            image,
        )

        prepared_alpha = prepare_alpha(
            alpha,
        )

        # ---------------------------------
        # Hair detection
        # ---------------------------------

        gradient = compute_multiscale_gradients(
            prepared_image,
        )

        contrast = compute_local_contrast(
            prepared_image,
        )

        texture = compute_texture_response(
            prepared_image,
        )

        hair_probability = detect_hair_candidates(
            gradient=gradient,
            contrast=contrast,
            texture=texture,
        )

        # ---------------------------------
        # Hair enhancement
        # ---------------------------------

        enhancement = enhance_hair(
            alpha=prepared_alpha,
            hair_probability=hair_probability,
        )

        # ---------------------------------
        # Merge
        # ---------------------------------

        refined_alpha = merge_alpha(
            alpha=prepared_alpha,
            enhancement=enhancement,
        )

        # ---------------------------------
        # Convert to PIL Image
        # ---------------------------------

        refined_alpha = (
            refined_alpha * 255.0
        ).astype(
            np.uint8,
        )

        refined_alpha = Image.fromarray(
            refined_alpha,
            mode="L",
        )

        logger.info(
            "Hair refinement completed successfully."
        )

        return refined_alpha

    except Exception as e:

        logger.exception(
            "Hair refinement failed."
        )

        raise PostprocessingError(
            "Hair refinement failed."
        ) from e


