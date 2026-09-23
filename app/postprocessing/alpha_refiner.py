"""
Adaptive Alpha Refiner
----------------------

Production-oriented, image-aware alpha refinement for the background
removal pipeline.

Design goals:
- Keep BiRefNet's original decision as the primary authority.
- Refine only where the RGB image provides boundary evidence.
- Avoid globally blurring the foreground.
- Avoid fixed "replace unknown pixels" behavior.
- Adapt refinement scale to image resolution and boundary geometry.
- Use the existing guided_filter.py as the low-level edge-aware filter.
- Apply confidence-weighted blending instead of replacing the whole
  trimap unknown region.
- Protect confident foreground/background and reject unsafe changes.

This module intentionally uses deterministic computer-vision analysis
instead of training another model. Constants are safety bounds, not
per-image decisions.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.postprocessing.guided_filter import refine_edges
from app.core.logger import logger
from app.core.exceptions import PostprocessingError


# ---------------------------------------------------------------------
# Safety bounds
# ---------------------------------------------------------------------
# These are guardrails only. Actual values are estimated from the
# image/mask statistics at runtime.

_MIN_RADIUS = 2
_MAX_RADIUS = 12

_MIN_TRIMAP_RADIUS = 1
_MAX_TRIMAP_RADIUS = 8

_MIN_EPSILON = 1e-5
_MAX_EPSILON = 5e-3

# Maximum fraction of the complete image that may be changed by alpha
# refinement before the change is considered suspicious.
_MAX_CHANGED_FRACTION = 0.12

# Maximum average absolute alpha change over the complete image.
_MAX_GLOBAL_MEAN_CHANGE = 0.025

# Strong RGB edge evidence threshold.
_EDGE_PERCENTILE = 70.0


# ---------------------------------------------------------------------
# Basic validation / conversion
# ---------------------------------------------------------------------

def prepare_mask(mask: Image.Image) -> np.ndarray:
    """
    Convert a PIL mask to normalized float32 alpha in [0, 1].
    """
    try:
        if not isinstance(mask, Image.Image):
            raise PostprocessingError("Mask must be a PIL Image.")

        if mask.width <= 0 or mask.height <= 0:
            raise PostprocessingError("Mask cannot have zero dimensions.")

        alpha = np.asarray(
            mask.convert("L"),
            dtype=np.float32,
        ) / 255.0

        return np.clip(alpha, 0.0, 1.0)

    except PostprocessingError:
        raise
    except Exception as e:
        logger.exception("Failed to prepare alpha mask.")
        raise PostprocessingError(
            "Failed to prepare alpha mask."
        ) from e


def create_alpha_mask(alpha: np.ndarray) -> Image.Image:
    """
    Convert normalized alpha [0, 1] to a grayscale PIL image.
    """
    try:
        if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
            raise PostprocessingError(
                "Alpha must be a 2D NumPy array."
            )

        alpha = np.clip(
            np.asarray(alpha, dtype=np.float32),
            0.0,
            1.0,
        )

        return Image.fromarray(
            np.round(alpha * 255.0).astype(np.uint8),
            mode="L",
        )

    except PostprocessingError:
        raise
    except Exception as e:
        logger.exception("Failed to create alpha mask.")
        raise PostprocessingError(
            "Failed to create alpha mask."
        ) from e


# ---------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------

def _prepare_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """
    Convert and align the RGB guidance image to mask dimensions.
    """
    if not isinstance(image, Image.Image):
        raise PostprocessingError("Image must be a PIL Image.")

    if image.size != size:
        logger.warning(
            "Alpha refinement image/mask dimensions differ. "
            f"Image={image.size}, Mask={size}. "
            "Aligning image to mask dimensions."
        )
        image = image.resize(
            size,
            Image.Resampling.BILINEAR,
        )

    return np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0


def _normalized_gradient(gray: np.ndarray) -> np.ndarray:
    """
    Compute a normalized Sobel gradient magnitude.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)

    p = float(np.percentile(magnitude, 99.0))
    if p <= 1e-8:
        return np.zeros_like(magnitude, dtype=np.float32)

    return np.clip(magnitude / p, 0.0, 1.0).astype(np.float32)


def _alpha_boundary_strength(alpha: np.ndarray) -> np.ndarray:
    """
    Measure how strongly alpha changes locally.

    Unlike a binary trimap, this works with both soft BiRefNet masks and
    masks that have already become mostly binary.
    """
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    p = float(np.percentile(magnitude, 99.0))

    if p <= 1e-8:
        return np.zeros_like(alpha, dtype=np.float32)

    return np.clip(magnitude / p, 0.0, 1.0).astype(np.float32)


def _uncertainty_map(alpha: np.ndarray) -> np.ndarray:
    """
    Confidence/uncertainty from alpha.

    Values near 0 or 1 are confident.
    Values near 0.5 are uncertain.

    This is an evidence signal, not a final decision.
    """
    uncertainty = 1.0 - np.abs(2.0 * alpha - 1.0)
    return np.clip(uncertainty, 0.0, 1.0).astype(np.float32)


def _edge_alignment(
    alpha: np.ndarray,
    image_edges: np.ndarray,
) -> np.ndarray:
    """
    Estimate whether image edges occur near alpha boundaries.

    A small local maximum filter is used so a true RGB edge does not need
    to fall on exactly the same pixel as the mask edge.
    """
    boundary = _alpha_boundary_strength(alpha)

    # Convert the boundary into a soft neighborhood.
    boundary_neighborhood = cv2.GaussianBlur(
        boundary,
        (0, 0),
        sigmaX=1.5,
        sigmaY=1.5,
    )

    alignment = boundary_neighborhood * image_edges

    max_value = float(np.max(alignment))
    if max_value > 1e-8:
        alignment /= max_value

    return np.clip(alignment, 0.0, 1.0).astype(np.float32)


def _boundary_band(
    alpha: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Build a narrow candidate band around the actual foreground boundary.

    This is intentionally based on the existing mask boundary, not a
    global morphology operation over the entire image.
    """
    foreground = (alpha >= 0.5).astype(np.uint8)

    eroded = cv2.erode(
        foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
        iterations=1,
    )

    dilated = cv2.dilate(
        foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
        iterations=1,
    )

    band = ((dilated > 0) & (eroded == 0)).astype(np.uint8)

    return band


def _estimate_boundary_scale(
    alpha: np.ndarray,
    image_edges: np.ndarray,
) -> tuple[int, int]:
    """
    Estimate safe trimap/radius scale from image resolution and the
    observed boundary structure.

    The calculation is adaptive; the constants only bound the result.
    """
    h, w = alpha.shape
    min_dimension = min(h, w)

    # Resolution-derived baseline. Logically proportional rather than
    # one fixed number for every input resolution.
    resolution_scale = max(
        1.0,
        np.sqrt(float(min_dimension)) / 32.0,
    )

    boundary = _alpha_boundary_strength(alpha)
    strong_boundary = boundary >= np.percentile(
        boundary,
        85.0,
    )

    edge_density = float(
        np.mean(
            image_edges[strong_boundary]
        )
    ) if np.any(strong_boundary) else 0.0

    boundary_density = float(np.mean(strong_boundary))

    # More complex boundaries need a slightly wider analysis band, while
    # clean sparse boundaries remain conservative.
    complexity = np.clip(
        0.55 * edge_density
        + 0.45 * min(boundary_density * 20.0, 1.0),
        0.0,
        1.0,
    )

    radius = int(
        round(
            resolution_scale
            * (0.75 + 0.85 * complexity)
        )
    )

    radius = int(
        np.clip(
            radius,
            _MIN_RADIUS,
            _MAX_RADIUS,
        )
    )

    trimap_radius = int(
        np.clip(
            round(radius * 0.55),
            _MIN_TRIMAP_RADIUS,
            _MAX_TRIMAP_RADIUS,
        )
    )

    return radius, trimap_radius


def _estimate_epsilon(
    image_rgb: np.ndarray,
    image_edges: np.ndarray,
) -> float:
    """
    Estimate guided-filter regularization from image variance and
    boundary complexity.

    The guidance image is normalized to [0, 1], so epsilon is also
    expressed on that scale.
    """
    gray = cv2.cvtColor(
        np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    variance = float(np.var(gray))
    edge_mean = float(np.mean(image_edges))

    complexity = np.clip(
        0.6 * np.sqrt(max(variance, 0.0)) * 2.0
        + 0.4 * edge_mean,
        0.0,
        1.0,
    )

    # Stronger/complex images get a little more regularization to avoid
    # following texture. Clean edges remain more responsive.
    epsilon = (
        _MIN_EPSILON
        + complexity * (
            _MAX_EPSILON - _MIN_EPSILON
        )
    )

    return float(
        np.clip(
            epsilon,
            _MIN_EPSILON,
            _MAX_EPSILON,
        )
    )


# ---------------------------------------------------------------------
# Adaptive trimap
# ---------------------------------------------------------------------

def generate_trimap(
    alpha: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2,
) -> np.ndarray:
    """
    Backwards-compatible trimap generator.

    This function remains available for existing callers. New refinement
    uses the adaptive version internally.
    """
    try:
        if alpha.ndim != 2:
            raise PostprocessingError("Alpha must be a 2D array.")

        kernel_size = int(max(1, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        iterations = int(max(1, iterations))

        mask = np.round(
            np.clip(alpha, 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        foreground = cv2.erode(
            mask,
            kernel,
            iterations=iterations,
        )

        background = cv2.dilate(
            mask,
            kernel,
            iterations=iterations,
        )

        trimap = np.full_like(
            mask,
            128,
            dtype=np.uint8,
        )

        trimap[background == 0] = 0
        trimap[foreground == 255] = 255

        return trimap

    except Exception as e:
        logger.exception("Failed to generate trimap.")
        raise PostprocessingError(
            "Failed to generate trimap."
        ) from e


def generate_adaptive_trimap(
    alpha: np.ndarray,
    image_edges: np.ndarray,
    trimap_radius: int,
) -> np.ndarray:
    """
    Generate a conservative, image-aware trimap.

    The unknown area is restricted to the actual alpha boundary and is
    not allowed to spread across the complete image.
    """
    try:
        boundary = _boundary_band(
            alpha,
            radius=trimap_radius,
        )

        # A soft-alpha uncertainty region is useful when BiRefNet has
        # already produced a genuine matte.
        uncertainty = _uncertainty_map(alpha)

        uncertainty_threshold = float(
            np.clip(
                np.percentile(
                    uncertainty[boundary > 0],
                    35.0,
                )
                if np.any(boundary)
                else 0.35,
                0.20,
                0.75,
            )
        )

        unknown = (
            (boundary > 0)
            &
            (
                (uncertainty >= uncertainty_threshold)
                | (image_edges >= 0.20)
            )
        )

        trimap = np.where(
            alpha <= 0.01,
            0,
            255,
        ).astype(np.uint8)

        trimap[unknown] = 128

        # Ensure a narrow transition exists around strong alpha
        # boundaries, even when the incoming mask is almost binary.
        transition = (
            (boundary > 0)
            & (
                (alpha > 0.02)
                & (alpha < 0.98)
            )
        )
        trimap[transition] = 128

        return trimap

    except Exception as e:
        logger.exception(
            "Failed to generate adaptive trimap."
        )
        raise PostprocessingError(
            "Failed to generate adaptive trimap."
        ) from e


# ---------------------------------------------------------------------
# Confidence / blending
# ---------------------------------------------------------------------

def _compute_refinement_confidence(
    alpha: np.ndarray,
    image_edges: np.ndarray,
    alignment: np.ndarray,
    trimap: np.ndarray,
) -> np.ndarray:
    """
    Compute a pixel-wise confidence map for guided-filter influence.

    High confidence requires multiple independent signals:
    - pixel is actually near the mask boundary
    - mask is uncertain or transitional
    - RGB contains an edge
    - RGB edge aligns with the mask boundary

    The guided result therefore cannot globally replace the original
    alpha.
    """
    boundary = (trimap == 128).astype(np.float32)
    uncertainty = _uncertainty_map(alpha)

    # Edge evidence is deliberately softened. Texture alone should not
    # be enough to authorize refinement.
    edge_evidence = np.sqrt(
        np.clip(image_edges, 0.0, 1.0)
    )

    confidence = (
        0.35 * uncertainty
        + 0.35 * alignment
        + 0.20 * edge_evidence
        + 0.10 * boundary
    )

    # A pixel outside the actual candidate boundary must never be
    # modified by the adaptive alpha stage.
    confidence *= boundary

    # Remove isolated confidence noise.
    confidence = cv2.GaussianBlur(
        confidence.astype(np.float32),
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
    )

    return np.clip(
        confidence,
        0.0,
        1.0,
    ).astype(np.float32)


def _apply_soft_refinement(
    alpha: np.ndarray,
    guided_alpha: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """
    Blend original and guided alpha using pixel-wise confidence.
    """
    refined = (
        alpha * (1.0 - confidence)
        + guided_alpha * confidence
    )

    return np.clip(
        refined,
        0.0,
        1.0,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------

# def _validate_refinement(
#     original: np.ndarray,
#     refined: np.ndarray,
#     confidence: np.ndarray,
# ) -> tuple[np.ndarray, dict]:
#     """
#     Validate the refinement and progressively reduce its strength if the
#     result changes too much.

#     This is a safety mechanism, not an image-specific rule.
#     """
#     original = np.asarray(original, dtype=np.float32)
#     refined = np.asarray(refined, dtype=np.float32)

#     delta = np.abs(refined - original)

#     changed = delta > 0.01
#     changed_fraction = float(np.mean(changed))
#     mean_change = float(np.mean(delta))
#     max_change = float(np.max(delta))

#     result = refined.copy()
#     fallback = False
#     scale = 1.0

#     if (
#         changed_fraction > _MAX_CHANGED_FRACTION
#         or mean_change > _MAX_GLOBAL_MEAN_CHANGE
#     ):
#         # First reduce refinement rather than immediately discarding it.
#         if changed_fraction > 0:
#             scale = min(
#                 1.0,
#                 _MAX_CHANGED_FRACTION / changed_fraction,
#             )

#         if mean_change > 0:
#             scale = min(
#                 scale,
#                 _MAX_GLOBAL_MEAN_CHANGE / mean_change,
#             )

#         scale = float(
#             np.clip(
#                 scale,
#                 0.0,
#                 1.0,
#             )
#         )

#         result = original + (
#             refined - original
#         ) * scale

#         # Re-evaluate after damping.
#         new_delta = np.abs(result - original)
#         new_changed_fraction = float(
#             np.mean(new_delta > 0.01)
#         )
#         new_mean_change = float(
#             np.mean(new_delta)
#         )

#         if (
#             new_changed_fraction > _MAX_CHANGED_FRACTION
#             or new_mean_change > _MAX_GLOBAL_MEAN_CHANGE
#         ):
#             result = original.copy()
#             fallback = True
#             scale = 0.0

#     stats = {
#         "changed_fraction": changed_fraction,
#         "mean_change": mean_change,
#         "max_change": max_change,
#         "applied_scale": scale,
#         "fallback": fallback,
#         "confidence_mean": float(np.mean(confidence)),
#     }

#     return np.clip(result, 0.0, 1.0), stats

def _validate_refinement(
    original: np.ndarray,
    refined: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Validate adaptive alpha refinement using distribution-based metrics.

    Important:
        A large single-pixel change is not automatically considered
        unsafe. Boundary refinement can legitimately contain strong
        local changes.

    Safety decisions are therefore based primarily on:
        - changed-pixel fraction
        - global mean change
        - high-percentile change
        - concentration of changes inside the candidate region

    All alpha values are normalized to [0, 1].
    """

    original = np.asarray(
        original,
        dtype=np.float32,
    )

    refined = np.asarray(
        refined,
        dtype=np.float32,
    )

    confidence = np.asarray(
        confidence,
        dtype=np.float32,
    )

    if original.shape != refined.shape:
        raise PostprocessingError(
            "Original and refined alpha dimensions must match."
        )

    if original.shape != confidence.shape:
        raise PostprocessingError(
            "Confidence and alpha dimensions must match."
        )

    original = np.clip(
        original,
        0.0,
        1.0,
    )

    refined = np.clip(
        refined,
        0.0,
        1.0,
    )

    confidence = np.clip(
        confidence,
        0.0,
        1.0,
    )

    # --------------------------------------------------------------
    # Absolute alpha difference
    # --------------------------------------------------------------

    delta = np.abs(
        refined - original
    )

    # A change greater than ~2.55/255 is considered measurable.
    change_threshold = 0.01

    changed = (
        delta > change_threshold
    )

    changed_fraction = float(
        np.mean(changed)
    )

    mean_change = float(
        np.mean(delta)
    )

    max_change = float(
        np.max(delta)
    )

    # --------------------------------------------------------------
    # Distribution statistics
    #
    # These are much more useful than max_change alone.
    # --------------------------------------------------------------

    p50_change = float(
        np.percentile(
            delta,
            50.0,
        )
    )

    p95_change = float(
        np.percentile(
            delta,
            95.0,
        )
    )

    p99_change = float(
        np.percentile(
            delta,
            99.0,
        )
    )

    p999_change = float(
        np.percentile(
            delta,
            99.9,
        )
    )

    # --------------------------------------------------------------
    # Significant-change statistics
    # Converted from 8-bit alpha levels to normalized alpha.
    # --------------------------------------------------------------

    threshold_10 = 10.0 / 255.0
    threshold_25 = 25.0 / 255.0
    threshold_50 = 50.0 / 255.0

    changes_ge_10 = int(
        np.sum(
            delta >= threshold_10
        )
    )

    changes_ge_25 = int(
        np.sum(
            delta >= threshold_25
        )
    )

    changes_ge_50 = int(
        np.sum(
            delta >= threshold_50
        )
    )

    # --------------------------------------------------------------
    # Candidate-region analysis
    # --------------------------------------------------------------

    candidate = confidence > 0.01

    candidate_fraction = float(
        np.mean(candidate)
    )

    changed_in_candidate = (
        changed & candidate
    )

    changed_outside_candidate = (
        changed & ~candidate
    )

    changed_candidate_fraction = (
        float(
            np.mean(
                changed_in_candidate
            )
        )
        if np.any(candidate)
        else 0.0
    )

    changed_outside_fraction = float(
        np.mean(
            changed_outside_candidate
        )
    )

    # The adaptive refiner should never modify pixels outside its
    # candidate region.
    outside_violation = bool(
        np.any(
            changed_outside_candidate
        )
    )

    # --------------------------------------------------------------
    # Safety decision
    #
    # Do NOT reject a result merely because max_change is large.
    # --------------------------------------------------------------

    result = refined.copy()

    fallback = False
    scale = 1.0

    unsafe_global_change = (
        changed_fraction
        > _MAX_CHANGED_FRACTION
    )

    unsafe_mean_change = (
        mean_change
        > _MAX_GLOBAL_MEAN_CHANGE
    )

    if (
        unsafe_global_change
        or unsafe_mean_change
    ):

        # Gradually reduce refinement instead of immediately
        # discarding it.

        if changed_fraction > 0:
            scale = min(
                scale,
                _MAX_CHANGED_FRACTION
                / changed_fraction,
            )

        if mean_change > 0:
            scale = min(
                scale,
                _MAX_GLOBAL_MEAN_CHANGE
                / mean_change,
            )

        scale = float(
            np.clip(
                scale,
                0.0,
                1.0,
            )
        )

        result = (
            original
            + (refined - original) * scale
        )

        result = np.clip(
            result,
            0.0,
            1.0,
        )

        # ----------------------------------------------------------
        # Recalculate after damping
        # ----------------------------------------------------------

        new_delta = np.abs(
            result - original
        )

        new_changed_fraction = float(
            np.mean(
                new_delta > change_threshold
            )
        )

        new_mean_change = float(
            np.mean(new_delta)
        )

        if (
            new_changed_fraction
            > _MAX_CHANGED_FRACTION
            or new_mean_change
            > _MAX_GLOBAL_MEAN_CHANGE
        ):
            result = original.copy()
            fallback = True
            scale = 0.0

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    stats = {
        "changed_fraction": changed_fraction,
        "candidate_fraction": candidate_fraction,
        "changed_candidate_fraction": (
            changed_candidate_fraction
        ),
        "changed_outside_fraction": (
            changed_outside_fraction
        ),
        "outside_candidate_violation": (
            outside_violation
        ),

        "mean_change": mean_change,

        "p50_change": p50_change,
        "p95_change": p95_change,
        "p99_change": p99_change,
        "p999_change": p999_change,

        "max_change": max_change,

        "changes_ge_10": changes_ge_10,
        "changes_ge_25": changes_ge_25,
        "changes_ge_50": changes_ge_50,

        "applied_scale": scale,
        "fallback": fallback,

        "confidence_mean": float(
            np.mean(confidence)
        ),
    }

    return (
        np.clip(
            result,
            0.0,
            1.0,
        ),
        stats,
    )

# ---------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------

def refine_alpha(
    image: Image.Image,
    alpha: np.ndarray,
    trimap: np.ndarray | None = None,
    radius: int | None = None,
    epsilon: float | None = None,
) -> np.ndarray:
    """
    Perform adaptive alpha refinement.

    Existing callers may still provide trimap/radius/epsilon. When they
    are omitted, all parameters are inferred from the image and mask.

    The guided filter is only used as an evidence-driven proposal.
    It never automatically replaces the complete unknown region.
    """
    try:
        if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
            raise PostprocessingError(
                "Alpha must be a 2D NumPy array."
            )

        alpha = np.clip(
            np.asarray(alpha, dtype=np.float32),
            0.0,
            1.0,
        )

        rgb = _prepare_rgb(
            image,
            (alpha.shape[1], alpha.shape[0]),
        )

        gray = cv2.cvtColor(
            np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32) / 255.0

        image_edges = _normalized_gradient(gray)

        adaptive_radius, adaptive_trimap_radius = (
            _estimate_boundary_scale(
                alpha,
                image_edges,
            )
        )

        if radius is None:
            radius = adaptive_radius
        else:
            radius = int(
                np.clip(
                    int(radius),
                    _MIN_RADIUS,
                    _MAX_RADIUS,
                )
            )

        if epsilon is None:
            epsilon = _estimate_epsilon(
                rgb,
                image_edges,
            )
        else:
            epsilon = float(
                np.clip(
                    float(epsilon),
                    _MIN_EPSILON,
                    _MAX_EPSILON,
                )
            )

        if trimap is None:
            trimap = generate_adaptive_trimap(
                alpha=alpha,
                image_edges=image_edges,
                trimap_radius=adaptive_trimap_radius,
            )
        else:
            trimap = np.asarray(
                trimap,
                dtype=np.uint8,
            )

            if trimap.shape != alpha.shape:
                raise PostprocessingError(
                    "Trimap dimensions must match alpha dimensions."
                )

        alignment = _edge_alignment(
            alpha,
            image_edges,
        )

        confidence = _compute_refinement_confidence(
            alpha=alpha,
            image_edges=image_edges,
            alignment=alignment,
            trimap=trimap,
        )

        candidate_pixels = confidence > 0.01

        if not np.any(candidate_pixels):
            logger.info(
                "No reliable boundary evidence found. "
                "Original alpha preserved."
            )
            return alpha.copy()

        logger.info(
            "Adaptive alpha analysis: "
            f"radius={radius}, "
            f"trimap_radius={adaptive_trimap_radius}, "
            f"epsilon={epsilon:.6g}, "
            f"candidate_fraction={float(np.mean(candidate_pixels)):.4f}, "
            f"confidence_mean={float(np.mean(confidence)):.4f}."
        )
        

        # The existing guided_filter module remains the low-level
        # filtering engine.
        guided_alpha_mask = refine_edges(
            image=Image.fromarray(
                np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
                mode="RGB",
            ),
            mask=create_alpha_mask(alpha),
            radius=radius,
            epsilon=epsilon,
        )

        guided_alpha = prepare_mask(
            guided_alpha_mask
        )

        # Never let the guided filter change confident interiors.
        refined = _apply_soft_refinement(
            alpha=alpha,
            guided_alpha=guided_alpha,
            confidence=confidence,
        )

        # Restore the original values outside the adaptive candidate
        # region exactly.
        refined[confidence <= 0.01] = alpha[
            confidence <= 0.01
        ]

        refined, stats = _validate_refinement(
            original=alpha,
            refined=refined,
            confidence=confidence,
        )

        logger.info(
            "Adaptive alpha refinement validation: "
            f"changed_fraction={stats['changed_fraction']:.4f}, "
            f"candidate_fraction={stats['candidate_fraction']:.4f}, "
            f"mean_change={stats['mean_change']:.6f}, "
            f"p95={stats['p95_change']:.6f}, "
            f"p99={stats['p99_change']:.6f}, "
            f"p99.9={stats['p999_change']:.6f}, "
            f"max={stats['max_change']:.6f}, "
            f">10={stats['changes_ge_10']}, "
            f">25={stats['changes_ge_25']}, "
            f">50={stats['changes_ge_50']}, "
            f"outside_candidate={stats['outside_candidate_violation']}, "
            f"applied_scale={stats['applied_scale']:.4f}, "
            f"fallback={stats['fallback']}."
        )

        return refined

    except PostprocessingError:
        raise
    except Exception as e:
        logger.exception(
            "Failed to refine alpha mask."
        )
        raise PostprocessingError(
            "Failed to refine alpha mask."
        ) from e


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def refine_alpha_mask(
    image: Image.Image,
    mask: Image.Image,
    kernel_size: int = 5,
    iterations: int = 2,
    radius: int = 8,
    epsilon: float = 1e-4,
) -> Image.Image:
    """
    Public compatibility API.

    The legacy arguments are retained so existing project imports and
    calls do not break. Refinement is adaptive internally.

    `radius` and `epsilon` are treated as optional hints bounded by
    safe ranges. Passing the legacy defaults does not force those
    values to be used blindly: the adaptive analysis remains in control
    of the refinement region and blending strength.
    """
    try:
        logger.info(
            "Starting adaptive alpha refinement..."
        )

        alpha = prepare_mask(mask)

        # Build image evidence first so the trimap can be adaptive.
        rgb = _prepare_rgb(
            image,
            (alpha.shape[1], alpha.shape[0]),
        )

        gray = cv2.cvtColor(
            np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32) / 255.0

        image_edges = _normalized_gradient(gray)

        adaptive_radius, adaptive_trimap_radius = (
            _estimate_boundary_scale(
                alpha,
                image_edges,
            )
        )

        # Legacy parameters are used only as bounded hints. If callers
        # pass the old defaults, adaptive scale wins when it differs
        # materially from the image-derived scale.
        supplied_radius = int(
            np.clip(
                int(radius),
                _MIN_RADIUS,
                _MAX_RADIUS,
            )
        )

        if supplied_radius == 8:
            selected_radius = adaptive_radius
        else:
            selected_radius = int(
                np.clip(
                    round(
                        0.35 * supplied_radius
                        + 0.65 * adaptive_radius
                    ),
                    _MIN_RADIUS,
                    _MAX_RADIUS,
                )
            )

        supplied_epsilon = float(
            np.clip(
                float(epsilon),
                _MIN_EPSILON,
                _MAX_EPSILON,
            )
        )

        adaptive_epsilon = _estimate_epsilon(
            rgb,
            image_edges,
        )

        if abs(supplied_epsilon - 1e-4) < 1e-12:
            selected_epsilon = adaptive_epsilon
        else:
            selected_epsilon = float(
                np.clip(
                    0.35 * supplied_epsilon
                    + 0.65 * adaptive_epsilon,
                    _MIN_EPSILON,
                    _MAX_EPSILON,
                )
            )

        trimap = generate_adaptive_trimap(
            alpha=alpha,
            image_edges=image_edges,
            trimap_radius=adaptive_trimap_radius,
        )

        refined_alpha = refine_alpha(
            image=image,
            alpha=alpha,
            trimap=trimap,
            radius=selected_radius,
            epsilon=selected_epsilon,
        )

        result = create_alpha_mask(
            refined_alpha
        )

        logger.info(
            "Adaptive alpha refinement completed successfully. "
            f"Radius={selected_radius}, "
            f"Epsilon={selected_epsilon:.6g}, "
            f"TrimapRadius={adaptive_trimap_radius}."
        )

        return result

    except PostprocessingError:
        raise
    except Exception as e:
        logger.exception(
            "Alpha refinement failed."
        )
        raise PostprocessingError(
            "Alpha refinement failed."
        ) from e
