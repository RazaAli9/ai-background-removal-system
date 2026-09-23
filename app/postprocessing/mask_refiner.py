import cv2
import numpy as np
from PIL import Image

from app.core.logger import logger


class MaskRefiner:
    """
    Image-aware adaptive mask refinement for BiRefNet masks.

    The refiner intentionally does NOT decide "fill vs preserve"
    from a fixed hole-area threshold.

    Hole decisions use:
        - BiRefNet soft confidence (when available)
        - local foreground color statistics
        - local background/context color statistics
        - foreground-ring confidence
        - edge/texture evidence

    A hole is filled only when the visual evidence indicates that
    the enclosed region belongs to the surrounding foreground.
    Ambiguous regions are preserved because a false fill is usually
    worse than leaving a small segmentation gap.
    """

    def __init__(
        self,
        threshold: int = 127,
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 3,
        artifact_ratio: float = 0.00005,
        hole_ratio: float = 0.002,
        enable_hole_filling: bool = False,
    ):
        self.threshold = int(threshold)
        self.opening_kernel_size = int(opening_kernel_size)
        self.closing_kernel_size = int(closing_kernel_size)

        # Kept for backwards compatibility with the existing config.
        # These are no longer used as the primary hole decision.
        self.artifact_ratio = float(artifact_ratio)
        self.hole_ratio = float(hole_ratio)

        # IMPORTANT:
        # A contour "hole" is not necessarily a segmentation error.
        # In multi-object images, spaces between touching people/animals
        # are enclosed by the union of foreground and therefore appear as
        # holes to RETR_CCOMP. Automatically filling them damages the mask.
        #
        # Safe production default: preserve all enclosed background regions.
        # Hole filling remains available as an explicit opt-in feature.
        self.enable_hole_filling = bool(enable_hole_filling)

    # ============================================================
    # BASIC CONVERSION
    # ============================================================

    @staticmethod
    def _to_numpy(mask: Image.Image) -> np.ndarray:
        if not isinstance(mask, Image.Image):
            raise TypeError("mask must be a PIL.Image.Image")

        return np.asarray(mask.convert("L"), dtype=np.uint8)

    @staticmethod
    def _to_pil(mask: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.clip(mask, 0, 255).astype(np.uint8),
            mode="L",
        )

    @staticmethod
    def _to_soft_numpy(
        soft_mask,
        expected_size: tuple[int, int],
    ) -> np.ndarray | None:
        """
        Convert an optional soft BiRefNet prediction to float32 [0, 1].

        Accepted inputs:
            - PIL.Image
            - NumPy array
            - torch Tensor (via detach/cpu/numpy)
        """
        if soft_mask is None:
            return None

        if isinstance(soft_mask, Image.Image):
            arr = np.asarray(
                soft_mask.convert("L"),
                dtype=np.float32,
            ) / 255.0

        elif isinstance(soft_mask, np.ndarray):
            arr = soft_mask.astype(np.float32)
            if arr.ndim == 3:
                arr = arr.squeeze()
            if arr.max(initial=0.0) > 1.0:
                arr /= 255.0

        elif hasattr(soft_mask, "detach"):
            arr = (
                soft_mask.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            arr = np.squeeze(arr)
            if arr.max(initial=0.0) > 1.0:
                arr /= 255.0

        else:
            raise TypeError(
                "soft_mask must be a PIL image, NumPy array, "
                "or torch Tensor."
            )

        if arr.ndim != 2:
            raise ValueError(
                f"soft_mask must be 2D after conversion; got shape {arr.shape}"
            )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        arr = np.clip(arr, 0.0, 1.0)

        width, height = expected_size
        if (arr.shape[1], arr.shape[0]) != (width, height):
            arr = cv2.resize(
                arr,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            arr = np.clip(arr, 0.0, 1.0)

        return arr

    # ============================================================
    # MORPHOLOGY
    # ============================================================

    @staticmethod
    def _odd_kernel_size(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 else value + 1

    def opening(
        self,
        mask: Image.Image,
        kernel_size: int | None = None,
    ) -> Image.Image:
        if kernel_size is None:
            kernel_size = self.opening_kernel_size

        kernel_size = self._odd_kernel_size(kernel_size)
        array = self._to_numpy(mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        result = cv2.morphologyEx(
            array,
            cv2.MORPH_OPEN,
            kernel,
        )

        return self._to_pil(result)

    def closing(
        self,
        mask: Image.Image,
        kernel_size: int | None = None,
    ) -> Image.Image:
        if kernel_size is None:
            kernel_size = self.closing_kernel_size

        kernel_size = self._odd_kernel_size(kernel_size)
        array = self._to_numpy(mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        result = cv2.morphologyEx(
            array,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return self._to_pil(result)

    # ============================================================
    # ADAPTIVE ARTIFACT REMOVAL
    # ============================================================

    def remove_small_artifacts(
        self,
        mask: Image.Image,
    ) -> Image.Image:
        logger.info("Removing small artifacts...")

        array = self._to_numpy(mask)

        binary = (
            array > self.threshold
        ).astype(np.uint8) * 255

        foreground_area = cv2.countNonZero(binary)

        if foreground_area == 0:
            logger.warning("Mask contains no foreground.")
            return self._to_pil(array)

        dynamic_min_area = max(
            1,
            int(foreground_area * self.artifact_ratio),
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(binary)

        for label in range(1, num_labels):
            component_area = stats[
                label,
                cv2.CC_STAT_AREA,
            ]

            if component_area >= dynamic_min_area:
                cleaned[labels == label] = 255

        # Preserve soft values of retained foreground pixels.
        result = np.where(
            cleaned > 0,
            array,
            0,
        ).astype(np.uint8)

        logger.info(
            "Small artifact removal completed successfully. "
            f"Dynamic minimum area: {dynamic_min_area} pixels."
        )

        return self._to_pil(result)

    # ============================================================
    # HOLE DETECTION
    # ============================================================

    @staticmethod
    def _detect_holes(
        binary: np.ndarray,
    ) -> list:
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        holes = []

        if hierarchy is None:
            return holes

        hierarchy = hierarchy[0]

        for index, contour in enumerate(contours):
            parent = hierarchy[index][3]

            if parent == -1:
                continue

            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if (
                x <= 0
                or y <= 0
                or x + w >= binary.shape[1] - 1
                or y + h >= binary.shape[0] - 1
            ):
                continue

            holes.append(
                {
                    "index": index,
                    "contour": contour,
                    "area": float(area),
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "parent": parent,
                }
            )

        return holes

    @staticmethod
    def _get_hole_mask(
        contour: np.ndarray,
        shape: tuple,
    ) -> np.ndarray:
        hole_mask = np.zeros(
            shape,
            dtype=np.uint8,
        )

        cv2.drawContours(
            hole_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        return hole_mask

    # ============================================================
    # LOCAL CONTEXT
    # ============================================================

    @staticmethod
    def _get_foreground_ring(
        hole_mask: np.ndarray,
        binary: np.ndarray,
        kernel_size: int = 11,
    ) -> np.ndarray:
        kernel_size = max(3, int(kernel_size))
        kernel_size = (
            kernel_size
            if kernel_size % 2
            else kernel_size + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        dilated = cv2.dilate(
            hole_mask,
            kernel,
            iterations=1,
        )

        ring = cv2.subtract(
            dilated,
            hole_mask,
        )

        return cv2.bitwise_and(
            ring,
            binary,
        )

    @staticmethod
    def _get_background_context(
        hole_mask: np.ndarray,
        binary: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Find nearby pixels that are outside the foreground.

        The context radius grows with the candidate's dimensions.
        This avoids using one fixed pixel radius for every image.
        """
        radius = int(
            np.clip(
                max(width, height) * 0.20,
                9,
                81,
            )
        )

        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        expanded = cv2.dilate(
            hole_mask,
            kernel,
            iterations=1,
        )

        # Outside the hole AND outside current foreground.
        background = cv2.bitwise_and(
            expanded,
            cv2.bitwise_not(binary),
        )

        background[
            hole_mask > 0
        ] = 0

        return background

    # ============================================================
    # IMAGE FEATURES
    # ============================================================

    @staticmethod
    def _lab_pixels(
        image_lab: np.ndarray,
        region: np.ndarray,
    ) -> np.ndarray:
        pixels = image_lab[region > 0]

        if pixels.size == 0:
            return np.empty(
                (0, 3),
                dtype=np.float32,
            )

        return pixels.astype(np.float32)

    @staticmethod
    def _median_color_distance(
        pixels: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        if pixels.size == 0:
            return float("inf")

        distances = np.linalg.norm(
            pixels - reference,
            axis=1,
        )

        return float(np.median(distances))

    @staticmethod
    def _region_edge_density(
        gray: np.ndarray,
        region: np.ndarray,
    ) -> float:
        """
        Edge density normalized to [0, 1].

        This is used as contextual evidence, not as a standalone
        fill rule.
        """
        if cv2.countNonZero(region) < 10:
            return 0.0

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        region_edges = cv2.bitwise_and(
            edges,
            edges,
            mask=region,
        )

        region_area = max(
            cv2.countNonZero(region),
            1,
        )

        return float(
            cv2.countNonZero(region_edges)
            / region_area
        )

    @staticmethod
    @staticmethod
    def _soft_statistics(
        soft_mask: np.ndarray | None,
        hole_mask: np.ndarray,
        foreground_ring: np.ndarray,
        background_context: np.ndarray | None = None,
    ) -> dict:
        """Return robust soft-mask statistics for candidate/context regions."""
        empty = {
            "hole_mean": None,
            "hole_p10": None,
            "hole_p90": None,
            "ring_mean": None,
            "background_mean": None,
            "background_p90": None,
            "separation": None,
        }

        if soft_mask is None:
            return empty

        hole_values = soft_mask[hole_mask > 0]
        ring_values = soft_mask[foreground_ring > 0]

        if background_context is not None:
            background_values = soft_mask[background_context > 0]
        else:
            background_values = np.empty(0, dtype=np.float32)

        if hole_values.size == 0:
            return empty

        hole_mean = float(np.mean(hole_values))
        hole_p10 = float(np.percentile(hole_values, 10))
        hole_p90 = float(np.percentile(hole_values, 90))

        ring_mean = float(np.mean(ring_values)) if ring_values.size else None
        background_mean = (
            float(np.mean(background_values))
            if background_values.size else None
        )
        background_p90 = (
            float(np.percentile(background_values, 90))
            if background_values.size else None
        )

        separation = (
            ring_mean - hole_mean
            if ring_mean is not None else None
        )

        return {
            "hole_mean": hole_mean,
            "hole_p10": hole_p10,
            "hole_p90": hole_p90,
            "ring_mean": ring_mean,
            "background_mean": background_mean,
            "background_p90": background_p90,
            "separation": separation,
        }

    @staticmethod
    def _border_background(binary: np.ndarray) -> np.ndarray:
        """Return the binary background component connected to the image border."""
        background = (binary == 0).astype(np.uint8)
        h, w = background.shape
        flood = background.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        for x in range(w):
            if flood[0, x]:
                cv2.floodFill(flood, flood_mask, (x, 0), 2, flags=8)
            if flood[h - 1, x]:
                cv2.floodFill(flood, flood_mask, (x, h - 1), 2, flags=8)

        for y in range(h):
            if flood[y, 0]:
                cv2.floodFill(flood, flood_mask, (0, y), 2, flags=8)
            if flood[y, w - 1]:
                cv2.floodFill(flood, flood_mask, (w - 1, y), 2, flags=8)

        return np.where(flood == 2, 255, 0).astype(np.uint8)

    @staticmethod
    def _background_connectivity(
        hole_mask: np.ndarray,
        background_context: np.ndarray,
        soft_mask: np.ndarray | None,
        border_background: np.ndarray | None = None,
    ) -> dict:
        """
        Detect whether a candidate behaves like negative space.

        The test is intentionally conservative. It learns the allowed
        soft-confidence range from the candidate's local background context,
        then checks whether low-confidence pixels inside the candidate touch
        that context. A strong connection is a veto against filling.
        """
        if soft_mask is None:
            return {
                "connected": False,
                "score": 0.0,
                "cutoff": None,
                "reason": "soft_mask_unavailable",
            }

        hole_values = soft_mask[hole_mask > 0]
        context_values = soft_mask[background_context > 0]

        if hole_values.size == 0:
            return {
                "connected": False,
                "score": 0.0,
                "cutoff": None,
                "reason": "empty_candidate",
            }

        if context_values.size >= 20:
            median = float(np.median(context_values))
            mad = float(np.median(np.abs(context_values - median)))
            robust_cutoff = median + 3.0 * max(mad, 0.01)
            percentile_cutoff = float(np.percentile(context_values, 95.0))
            cutoff = float(np.clip(max(robust_cutoff, percentile_cutoff), 0.0, 1.0))
        else:
            # Without reliable local context, do not manufacture a strong
            # background path from the candidate alone.
            return {
                "connected": False,
                "score": 0.0,
                "cutoff": None,
                "reason": "insufficient_context",
            }

        low_confidence = (soft_mask <= cutoff).astype(np.uint8)
        candidate_low = (hole_mask > 0) & (low_confidence > 0)
        candidate_ratio = float(
            np.count_nonzero(candidate_low)
            / max(np.count_nonzero(hole_mask), 1)
        )

        if candidate_ratio < 0.35:
            return {
                "connected": False,
                "score": candidate_ratio,
                "cutoff": cutoff,
                "reason": "candidate_not_background_like",
            }

        num_labels, labels = cv2.connectedComponents(
            low_confidence,
            connectivity=8,
        )

        candidate_labels = np.unique(labels[candidate_low])
        candidate_labels = candidate_labels[candidate_labels != 0]
        context_labels = np.unique(labels[background_context > 0])
        context_labels = context_labels[context_labels != 0]

        connected_labels = np.intersect1d(
            candidate_labels,
            context_labels,
        )

        if connected_labels.size:
            component = np.isin(labels, connected_labels)
            covered = np.count_nonzero(component & (hole_mask > 0))
            candidate_seed_count = np.count_nonzero(candidate_low)
            score = float(
                covered / max(candidate_seed_count, 1)
            )

            # If the low-confidence component also reaches the actual image
            # background, make the veto even stronger.
            border_touch = False
            if border_background is not None:
                border_touch = bool(
                    np.any(component & (border_background > 0))
                )

            if border_touch:
                score = max(score, 1.0)

            return {
                "connected": score >= 0.50,
                "score": float(np.clip(score, 0.0, 1.0)),
                "cutoff": cutoff,
                "reason": (
                    "low_confidence_path_to_background"
                    if score >= 0.50
                    else "weak_background_path"
                ),
            }

        # No component connection. A high proportion of low-confidence
        # pixels in both candidate and local background is still strong
        # evidence that this region is ordinary negative space.
        context_low_ratio = float(
            np.mean(context_values <= cutoff)
        )

        if candidate_ratio >= 0.75 and context_low_ratio >= 0.60:
            return {
                "connected": True,
                "score": float(min(candidate_ratio, context_low_ratio)),
                "cutoff": cutoff,
                "reason": "strong_local_background_evidence",
            }

        return {
            "connected": False,
            "score": 0.0,
            "cutoff": cutoff,
            "reason": "foreground_barrier",
        }

    # ============================================================
    # INTELLIGENT HOLE CLASSIFICATION
    # ============================================================


    def _analyze_hole(
        self,
        image_lab: np.ndarray,
        gray: np.ndarray,
        hole_mask: np.ndarray,
        foreground_ring: np.ndarray,
        background_context: np.ndarray,
        soft_mask: np.ndarray | None,
        connectivity: dict | None = None,
    ) -> dict:
        """
        Conservative foreground/background classifier.

        PRESERVE is the default. A candidate is filled only when:
            - it looks sufficiently like the surrounding foreground,
            - BiRefNet's original soft prediction supports foreground,
            - and there is no strong background evidence.
        """
        if connectivity is None:
            connectivity = self._background_connectivity(
                hole_mask=hole_mask,
                background_context=background_context,
                soft_mask=soft_mask,
            )

        hole_pixels = self._lab_pixels(image_lab, hole_mask)
        foreground_pixels = self._lab_pixels(image_lab, foreground_ring)
        background_pixels = self._lab_pixels(image_lab, background_context)

        if hole_pixels.size == 0:
            return {
                "decision": "preserve",
                "confidence": 1.0,
                "score": 0.0,
                "reason": "empty_candidate",
                "appearance_evidence": 0.0,
                "soft_evidence": 0.0,
                "texture_evidence": 0.0,
                "foreground_distance": float("inf"),
                "background_distance": float("inf"),
                "background_pixels": 0,
                "background_connected": False,
                "background_connectivity_score": 0.0,
            }

        if foreground_pixels.shape[0] >= 20:
            foreground_median = np.median(foreground_pixels, axis=0)
            foreground_distance = self._median_color_distance(
                hole_pixels, foreground_median
            )
        else:
            foreground_median = np.median(hole_pixels, axis=0)
            foreground_distance = float("inf")

        if background_pixels.shape[0] >= 20:
            background_median = np.median(background_pixels, axis=0)
            background_distance = self._median_color_distance(
                hole_pixels, background_median
            )

            # Positive => visually closer to foreground.
            appearance_evidence = float(np.clip(
                (background_distance - foreground_distance)
                / max(background_distance + foreground_distance, 1e-6),
                -1.0,
                1.0,
            ))
        else:
            background_distance = float("inf")
            appearance_evidence = 0.0

        soft = self._soft_statistics(
            soft_mask=soft_mask,
            hole_mask=hole_mask,
            foreground_ring=foreground_ring,
            background_context=background_context,
        )

        if soft["hole_mean"] is not None:
            # Map 0.50 -> 0 and 0.85 -> +1. A low soft value is direct
            # evidence that the candidate should remain transparent.
            soft_evidence = float(np.clip(
                (soft["hole_mean"] - 0.50) / 0.35,
                -1.0,
                1.0,
            ))
        else:
            soft_evidence = 0.0

        hole_edges = self._region_edge_density(gray, hole_mask)

        if background_pixels.shape[0] >= 20:
            background_edges = self._region_edge_density(
                gray, background_context
            )
            texture_background_evidence = float(np.clip(
                1.0 - abs(hole_edges - background_edges) * 6.0,
                0.0,
                1.0,
            ))
        else:
            texture_background_evidence = 0.0

        background_connected = bool(
            connectivity.get("connected", False)
        )
        connectivity_score = float(
            connectivity.get("score", 0.0)
        )

        # --------------------------------------------------------
        # HARD BACKGROUND VETO #1: topology
        # --------------------------------------------------------
        if background_connected and connectivity_score >= 0.50:
            return {
                "decision": "preserve",
                "confidence": float(np.clip(
                    0.75 + 0.25 * connectivity_score,
                    0.0,
                    1.0,
                )),
                "score": -1.0,
                "reason": connectivity.get(
                    "reason", "background_connectivity_veto"
                ),
                "appearance_evidence": appearance_evidence,
                "soft_evidence": soft_evidence,
                "texture_evidence": texture_background_evidence,
                "foreground_distance": float(foreground_distance),
                "background_distance": float(background_distance),
                "background_pixels": int(background_pixels.shape[0]),
                "soft_hole_mean": soft["hole_mean"],
                "soft_hole_p10": soft["hole_p10"],
                "soft_hole_p90": soft["hole_p90"],
                "soft_ring_mean": soft["ring_mean"],
                "soft_background_mean": soft["background_mean"],
                "soft_background_p90": soft["background_p90"],
                "soft_separation": soft["separation"],
                "background_connected": True,
                "background_connectivity_score": connectivity_score,
                "background_connectivity_cutoff": connectivity.get("cutoff"),
                "background_connectivity_reason": connectivity.get("reason"),
            }

        # --------------------------------------------------------
        # HARD BACKGROUND VETO #2: the candidate itself looks like
        # the local background. This is the key protection for gaps
        # between people/legs/arms and other negative space.
        # --------------------------------------------------------
        candidate_is_background_like = (
            appearance_evidence <= -0.10
            or (
                appearance_evidence < 0.05
                and soft["hole_mean"] is not None
                and soft["hole_mean"] < 0.42
            )
        )

        if candidate_is_background_like:
            return {
                "decision": "preserve",
                "confidence": 0.90,
                "score": float(appearance_evidence),
                "reason": "candidate_matches_background",
                "appearance_evidence": appearance_evidence,
                "soft_evidence": soft_evidence,
                "texture_evidence": texture_background_evidence,
                "foreground_distance": float(foreground_distance),
                "background_distance": float(background_distance),
                "background_pixels": int(background_pixels.shape[0]),
                "soft_hole_mean": soft["hole_mean"],
                "soft_hole_p10": soft["hole_p10"],
                "soft_hole_p90": soft["hole_p90"],
                "soft_ring_mean": soft["ring_mean"],
                "soft_background_mean": soft["background_mean"],
                "soft_background_p90": soft["background_p90"],
                "soft_separation": soft["separation"],
                "background_connected": background_connected,
                "background_connectivity_score": connectivity_score,
                "background_connectivity_cutoff": connectivity.get("cutoff"),
                "background_connectivity_reason": connectivity.get("reason"),
            }

        # --------------------------------------------------------
        # Foreground score. Positive evidence only.
        # --------------------------------------------------------
        appearance_positive = max(appearance_evidence, 0.0)
        soft_positive = max(soft_evidence, 0.0)
        texture_positive = max(0.0, 1.0 - texture_background_evidence)

        score = (
            0.60 * appearance_positive
            + 0.30 * soft_positive
            + 0.10 * texture_positive
        )

        # --------------------------------------------------------
        # HARD FOREGROUND REQUIREMENT
        # --------------------------------------------------------
        # No area threshold. No coordinates. No object-specific rule.
        # A fill needs both visual and model support.
        # --------------------------------------------------------
        fill = (
            appearance_evidence >= 0.25
            and soft["hole_mean"] is not None
            and soft["hole_mean"] >= 0.55
            and score >= 0.45
        )

        # If the candidate is almost as likely to be background as
        # foreground, preserve it. False positives are worse here.
        if (
            soft["hole_mean"] is not None
            and soft["background_mean"] is not None
            and soft["hole_mean"]
            <= soft["background_mean"] + 0.05
        ):
            fill = False

        if fill:
            decision = "fill"
            reason = "strong_foreground_evidence"
            confidence = float(np.clip(0.50 + score * 0.50, 0.0, 1.0))
        else:
            decision = "preserve"
            if soft["hole_mean"] is not None and soft["hole_mean"] < 0.50:
                reason = "soft_prediction_background"
            elif appearance_evidence < 0.25:
                reason = "weak_foreground_appearance"
            else:
                reason = "ambiguous_candidate"
            confidence = float(np.clip(0.55 + (1.0 - score) * 0.45, 0.0, 1.0))

        return {
            "decision": decision,
            "confidence": confidence,
            "score": float(score),
            "reason": reason,
            "appearance_evidence": float(appearance_evidence),
            "soft_evidence": float(soft_evidence),
            "texture_evidence": float(texture_background_evidence),
            "foreground_distance": float(foreground_distance),
            "background_distance": float(background_distance),
            "hole_edge_density": float(hole_edges),
            "background_pixels": int(background_pixels.shape[0]),
            "soft_hole_mean": soft["hole_mean"],
            "soft_hole_p10": soft["hole_p10"],
            "soft_hole_p90": soft["hole_p90"],
            "soft_ring_mean": soft["ring_mean"],
            "soft_background_mean": soft["background_mean"],
            "soft_background_p90": soft["background_p90"],
            "soft_separation": soft["separation"],
            "background_connected": background_connected,
            "background_connectivity_score": connectivity_score,
            "background_connectivity_cutoff": connectivity.get("cutoff"),
            "background_connectivity_reason": connectivity.get("reason"),
        }

    # ============================================================
    # HOLE FILLING
    # ============================================================

    def fill_holes(
        self,
        mask: Image.Image,
        image: Image.Image,
        soft_mask=None,
        force: bool = False,
    ) -> Image.Image:
        """
        Image-aware enclosed-hole refinement.

        There is intentionally no fixed maximum hole-area rule.
        Every detected candidate is analyzed against its local
        foreground and local background appearance.

        Ambiguous candidates are preserved.
        """
        array = self._to_numpy(mask)

        # ------------------------------------------------------------
        # PRODUCTION-SAFE DEFAULT
        # ------------------------------------------------------------
        # Do NOT automatically fill enclosed regions.
        #
        # Why:
        # OpenCV's contour hierarchy can tell us that a background region
        # is enclosed by the current foreground mask, but it cannot tell us
        # whether that region is:
        #
        #   1. a real object interior/transparent opening, OR
        #   2. negative space between two touching foreground objects.
        #
        # The user's diagnostic image is case (2). Filling such regions
        # merges separate objects and destroys valid background.
        #
        # Therefore hole filling is an explicit opt-in operation.
        if not self.enable_hole_filling and not force:
            logger.info(
                "Hole filling disabled (production-safe default). "
                "Preserving all enclosed background regions."
            )
            return self._to_pil(array)

        logger.info(
            "Starting image-aware hole analysis..."
        )

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        if image.size != mask.size:
            logger.warning(
                "Mask/image dimensions differ. "
                f"Mask: {mask.size}, "
                f"Image: {image.size}. "
                "Resizing image to mask dimensions."
            )

            image = image.resize(
                mask.size,
                Image.Resampling.BILINEAR,
            )

        soft_array = self._to_soft_numpy(
            soft_mask,
            expected_size=mask.size,
        )

        binary = (
            array > self.threshold
        ).astype(np.uint8) * 255

        foreground_area = cv2.countNonZero(
            binary
        )

        if foreground_area == 0:
            logger.warning(
                "No foreground detected. Skipping hole filling."
            )
            return self._to_pil(array)

        holes = self._detect_holes(binary)

        if not holes:
            logger.info(
                "No enclosed holes detected."
            )
            return self._to_pil(array)

        image_rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

        image_lab = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2LAB,
        )

        gray = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        border_background = self._border_background(binary)

        result = array.copy()

        filled_count = 0
        filled_area = 0.0
        preserved_count = 0

        logger.info(
            f"Detected {len(holes)} enclosed candidate region(s). "
            "Using image-aware classification instead of a fixed "
            "hole-area cutoff."
        )

        for hole_number, hole in enumerate(
            holes,
            start=1,
        ):
            hole_mask = self._get_hole_mask(
                contour=hole["contour"],
                shape=binary.shape,
            )

            # Dynamic local ring based on candidate dimensions.
            ring_size = int(
                np.clip(
                    max(
                        hole["width"],
                        hole["height"],
                    ) * 0.12,
                    7,
                    31,
                )
            )

            foreground_ring = (
                self._get_foreground_ring(
                    hole_mask=hole_mask,
                    binary=binary,
                    kernel_size=(
                        ring_size
                        if ring_size % 2
                        else ring_size + 1
                    ),
                )
            )

            background_context = (
                self._get_background_context(
                    hole_mask=hole_mask,
                    binary=binary,
                    width=hole["width"],
                    height=hole["height"],
                )
            )

            connectivity = self._background_connectivity(
                hole_mask=hole_mask,
                background_context=background_context,
                soft_mask=soft_array,
                border_background=border_background,
            )

            analysis = self._analyze_hole(
                image_lab=image_lab,
                gray=gray,
                hole_mask=hole_mask,
                foreground_ring=foreground_ring,
                background_context=background_context,
                soft_mask=soft_array,
                connectivity=connectivity,
            )

            logger.info(
                f"Hole {hole_number}: "
                f"area={hole['area']:.2f}, "
                f"decision={analysis['decision']}, "
                f"score={analysis['score']:.3f}, "
                f"reason={analysis['reason']}, "
                f"fg_dist={analysis['foreground_distance']:.2f}, "
                f"bg_dist={analysis['background_distance']:.2f}, "
                f"bg_pixels={analysis['background_pixels']}, "
                f"soft_hole={analysis['soft_hole_mean']}, "
                f"soft_ring={analysis['soft_ring_mean']}, "
                f"soft_bg={analysis.get('soft_background_mean')}, "
                f"bg_connected={analysis['background_connected']}, "
                f"connectivity={analysis['background_connectivity_score']:.3f}, "
                f"connectivity_reason={analysis['background_connectivity_reason']}"
            )

            if analysis["decision"] == "fill":
                cv2.drawContours(
                    result,
                    [hole["contour"]],
                    contourIdx=-1,
                    color=255,
                    thickness=cv2.FILLED,
                )

                filled_count += 1
                filled_area += hole["area"]

            else:
                preserved_count += 1

        logger.info(
            "Image-aware hole analysis completed. "
            f"Detected: {len(holes)}, "
            f"Filled: {filled_count}, "
            f"Preserved: {preserved_count}, "
            f"Filled area: {filled_area:.0f} pixels."
        )

        return self._to_pil(result)

    # ============================================================
    # FULL REFINEMENT
    # ============================================================

    def refine_mask(
        self,
        mask: Image.Image,
        image: Image.Image,
        soft_mask=None,
    ) -> Image.Image:
        """
        Full adaptive refinement.

        soft_mask is optional for backwards compatibility. When
        supplied, it should be the original BiRefNet sigmoid
        prediction before binary thresholding.
        """
        logger.info(
            "Starting adaptive mask refinement..."
        )

        if not isinstance(mask, Image.Image):
            raise TypeError(
                "mask must be a PIL.Image.Image"
            )

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        if mask.size != image.size:
            logger.warning(
                "Mask/image dimensions differ before refinement. "
                f"Mask: {mask.size}, "
                f"Image: {image.size}. "
                "Image will be aligned to mask dimensions."
            )

            image = image.resize(
                mask.size,
                Image.Resampling.BILINEAR,
            )

        refined = self.opening(mask)
        refined = self.closing(refined)
        refined = self.remove_small_artifacts(refined)

        refined = self.fill_holes(
            mask=refined,
            image=image,
            soft_mask=soft_mask,
        )

        logger.info(
            "Adaptive mask refinement completed successfully."
        )

        return refined


# ================================================================
# MODULE-LEVEL API
# ================================================================

_default_mask_refiner = MaskRefiner()


def refine_mask(
    mask: Image.Image,
    image: Image.Image,
    soft_mask=None,
) -> Image.Image:
    """
    Public API used by postprocessor.py.

    soft_mask is optional so existing diagnostic tests remain
    compatible.
    """
    return _default_mask_refiner.refine_mask(
        mask=mask,
        image=image,
        soft_mask=soft_mask,
    )
