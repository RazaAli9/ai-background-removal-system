from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from app.core.exceptions import PostprocessingError
from app.core.logger import logger
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import (
    skeletonize,
    convex_hull_image,
    reconstruction,
)



def prepare_alpha(
    alpha: Image.Image,
) -> np.ndarray:
    """
    Prepare the alpha matte for foreground cleaning.

    Args:
        alpha:
            Refined alpha mask from the previous
            postprocessing stage.

    Returns:
        Normalized float32 NumPy array with values
        in the range [0, 1].

    Raises:
        PostprocessingError:
            If alpha preparation fails.
    """

    try:

        logger.info(
            "Preparing alpha matte for foreground cleaning..."
        )

        # Convert to grayscale
        alpha = alpha.convert(
            "L"
        )

        # Convert to float32 NumPy array
        alpha_array = np.asarray(
            alpha,
            dtype=np.float32,
        )

        # Normalize to [0, 1]
        alpha_array /= 255.0

        logger.info(
            "Alpha matte prepared successfully."
        )

        return alpha_array

    except Exception as e:

        logger.exception(
            "Failed to prepare alpha matte."
        )

        raise PostprocessingError(
            "Failed to prepare alpha matte."
        ) from e



def analyze_foreground(
    alpha: np.ndarray,
) -> list[dict]:
    """
    Analyze connected foreground regions in the alpha matte.

    Each detected foreground component is converted into a
    structured region dictionary. No filtering or classification
    is performed at this stage.

    Args:
        alpha:
            Normalized alpha matte with values in the
            range [0, 1].

    Returns:
        List of foreground region dictionaries.
        Each dictionary contains:

        - label
        - mask
        - stats
        - centroid

    Raises:
        PostprocessingError:
            If foreground analysis fails.
    """

    try:

        logger.info(
            "Analyzing foreground regions..."
        )

        # Convert alpha to binary mask
        binary = (
            alpha > 0.5
        ).astype(
            np.uint8
        )

        # Connected component analysis
        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        regions = []

        # Skip label 0 (background)
        for label in range(
            1,
            num_labels,
        ):

            region = {
                "label": label,
                "mask": labels == label,
                "stats": stats[label],
                "centroid": centroids[label],
            }

            regions.append(
                region
            )

        logger.info(
            f"{len(regions)} foreground region(s) detected."
        )

        return regions

    except Exception as e:

        logger.exception(
            "Failed to analyze foreground."
        )

        raise PostprocessingError(
            "Failed to analyze foreground."
        ) from e



def compute_region_features(
    regions: list[dict],
    image_shape: tuple[int, int],
) -> list[dict]:
    """
    Compute geometric and statistical features for each
    connected foreground region using scikit-image.

    OpenCV performs connected-component extraction while
    scikit-image computes rich region descriptors.

    Args:
        regions:
            Foreground regions returned by analyze_regions().

        image_shape:
            Shape of the alpha matte.

    Returns:
        Enriched foreground regions.

    Raises:
        PostprocessingError:
            If feature computation fails.
    """

    try:

        logger.info(
            "Computing region features..."
        )

        image_height, image_width = image_shape

        enriched_regions = []

        for region in regions:

            mask = region["mask"].astype(bool)

            props = regionprops(
                mask.astype(np.uint8)
            )

            if not props:
                continue

            prop = props[0]

            bbox = {
                "x": int(prop.bbox[1]),
                "y": int(prop.bbox[0]),
                "width": int(
                    prop.bbox[3] - prop.bbox[1]
                ),
                "height": int(
                    prop.bbox[2] - prop.bbox[0]
                ),
            }

            touches_border = (

                bbox["x"] == 0

                or

                bbox["y"] == 0

                or

                bbox["x"] + bbox["width"] >= image_width

                or

                bbox["y"] + bbox["height"] >= image_height

            )

            enriched_regions.append({

                "label": region["label"],

                "mask": region["mask"],

                "area": int(prop.area),

                "relative_area":
                    float(
                        prop.area /
                        (image_height * image_width)
                    ),

                "bbox": bbox,

                "centroid": (
                    float(prop.centroid[1]),
                    float(prop.centroid[0]),
                ),

                "aspect_ratio":
                    float(
                        bbox["width"] /
                        bbox["height"]
                    )
                    if bbox["height"] > 0
                    else 0.0,

                "touches_border":
                    bool(touches_border),

                # -------- New Features --------

                "perimeter":
                    float(prop.perimeter),

                "eccentricity":
                    float(prop.eccentricity),

                "solidity":
                    float(prop.solidity),

                "extent":
                    float(prop.extent),

                "orientation":
                    float(prop.orientation),

                "major_axis_length":
                    float(prop.axis_major_length),

                "minor_axis_length":
                    float(prop.axis_minor_length),

                "equivalent_diameter":
                    float(
                        prop.equivalent_diameter_area
                    ),

                "convex_area":
                    int(prop.area_convex),

                "filled_area":
                    int(prop.area_filled),

                "euler_number":
                    int(prop.euler_number),

            })

        logger.info(
            "Region feature computation completed successfully."
        )

        return enriched_regions

    except Exception as e:

        logger.exception(
            "Failed to compute region features."
        )

        raise PostprocessingError(
            "Failed to compute region features."
        ) from e



def compute_skeleton(
    region: dict,
) -> np.ndarray:
    """
    Compute the morphological skeleton of a foreground region.

    The foreground region is first cropped to its bounding box
    to reduce computation. Skeletonization is then applied to
    the cropped region and restored to its original position.

    The resulting skeleton preserves topology while reducing
    the foreground to a one-pixel-wide representation.

    The skeleton is later used for:

    - Branch point detection
    - End point detection
    - Thin structure analysis
    - Attached structure detection

    Args:
        region:
            Foreground region returned by
            compute_region_features().

    Returns:
        Full-size binary skeleton image.

    Raises:
        PostprocessingError:
            If skeleton computation fails.
    """

    try:

        logger.info(
            f"Computing foreground skeleton for "
            f"region {region['label']}..."
        )

        # ----------------------------------
        # Read region information
        # ----------------------------------

        mask = region["mask"]

        bbox = region["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        # ----------------------------------
        # Crop region
        # ----------------------------------

        cropped_mask = mask[
            y:y + height,
            x:x + width,
        ]

        # ----------------------------------
        # Validate crop
        # ----------------------------------

        if cropped_mask.size == 0:

            raise PostprocessingError(
                "Foreground region is empty."
            )

        if not np.any(cropped_mask):

            raise PostprocessingError(
                "Foreground region contains no foreground pixels."
            )

        # ----------------------------------
        # Skeletonization
        # ----------------------------------

        cropped_skeleton = skeletonize(
            cropped_mask.astype(bool)
        )

        # ----------------------------------
        # Restore to original image size
        # ----------------------------------

        skeleton = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        skeleton[
            y:y + height,
            x:x + width,
        ] = cropped_skeleton.astype(
            np.uint8
        )

        logger.info(
            "Skeleton computed successfully."
        )

        return skeleton
        # region["shape_analysis"] = {
        #     "skeleton": skeleton,
        # }

        # return region

    except Exception as e:

        logger.exception(
            "Failed to compute skeleton."
        )

        raise PostprocessingError(
            "Failed to compute skeleton."
        ) from e

def compute_convex_hull(
    region: dict,
) -> np.ndarray:
    """
    Compute the convex hull of a foreground region.

    The foreground region is cropped to its bounding box before
    convex hull computation to reduce unnecessary processing.
    The resulting hull is then restored to the original image
    coordinates.

    The convex hull represents the smallest convex shape that
    completely contains the foreground region.

    It is later used for:

    - Protrusion analysis
    - Shape irregularity analysis
    - Attached structure detection
    - False foreground detection

    Args:
        region:
            Foreground region returned by
            compute_region_features().

    Returns:
        Full-size binary convex hull image.

    Raises:
        PostprocessingError:
            If convex hull computation fails.
    """

    try:

        logger.info(
            f"Computing convex hull for "
            f"region {region['label']}..."
        )

        # ----------------------------------
        # Read region information
        # ----------------------------------

        mask = region["mask"]

        bbox = region["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        # ----------------------------------
        # Crop region
        # ----------------------------------

        cropped_mask = mask[
            y:y + height,
            x:x + width,
        ]

        # ----------------------------------
        # Validate crop
        # ----------------------------------

        if cropped_mask.size == 0:

            raise PostprocessingError(
                "Foreground region is empty."
            )

        if not np.any(cropped_mask):

            raise PostprocessingError(
                "Foreground region contains no foreground pixels."
            )

        # ----------------------------------
        # Compute convex hull
        # ----------------------------------

        cropped_hull = convex_hull_image(
            cropped_mask.astype(bool)
        )

        # ----------------------------------
        # Restore to original image size
        # ----------------------------------

        convex_hull = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        convex_hull[
            y:y + height,
            x:x + width,
        ] = cropped_hull.astype(
            np.uint8
        )

        logger.info(
            "Convex hull computed successfully."
        )

        return convex_hull

    except Exception as e:

        logger.exception(
            "Failed to compute convex hull."
        )

        raise PostprocessingError(
            "Failed to compute convex hull."
        ) from e

def compute_distance_transform(
    region: dict,
) -> np.ndarray:
    """
    Compute the Euclidean distance transform of a foreground region.

    Each foreground pixel receives a value representing its Euclidean
    distance from the nearest background pixel.

    Larger values indicate thicker/interior regions, while smaller
    values indicate thin structures and boundaries.

    The computation is performed only inside the region bounding box
    and then restored to the original image coordinates.

    A one-pixel background padding is added before distance-transform
    computation so that regions touching the image boundary still have
    a valid background reference.

    The distance map is later used for:

    - Foreground thickness analysis
    - Thin structure detection
    - Hair/branch/wire analysis
    - Attached structure detection
    - Boundary analysis

    Args:
        region:
            Foreground region returned by
            compute_region_features().

    Returns:
        Full-size float32 distance-transform image.

    Raises:
        PostprocessingError:
            If distance-transform computation fails.
    """

    try:

        logger.info(
            f"Computing distance transform for "
            f"region {region['label']}..."
        )

        # ----------------------------------
        # Read region information
        # ----------------------------------

        mask = region["mask"]
        bbox = region["bbox"]

        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])

        # ----------------------------------
        # Crop region
        # ----------------------------------

        cropped_mask = mask[
            y:y + height,
            x:x + width,
        ]

        # ----------------------------------
        # Validate crop
        # ----------------------------------

        if cropped_mask.size == 0:

            raise PostprocessingError(
                "Foreground region is empty."
            )

        if not np.any(cropped_mask):

            raise PostprocessingError(
                "Foreground region contains no foreground pixels."
            )

        # ----------------------------------
        # Prepare binary mask
        # ----------------------------------

        binary_mask = (
            cropped_mask > 0
        )

        # ----------------------------------
        # Add background padding
        #
        # This guarantees that the cropped
        # region has an explicit background
        # boundary, including regions touching
        # the original image boundary.
        # ----------------------------------

        padding = 1

        padded_mask = np.pad(
            binary_mask,
            pad_width=padding,
            mode="constant",
            constant_values=False,
        )

        # ----------------------------------
        # Euclidean Distance Transform
        #
        # SciPy:
        # scipy.ndimage.distance_transform_edt
        #
        # Non-zero pixels are treated as
        # foreground and receive their
        # distance to the nearest zero pixel.
        # ----------------------------------

        padded_distance = distance_transform_edt(
            padded_mask,
            sampling=(1.0, 1.0),
        )

        # ----------------------------------
        # Remove padding
        # ----------------------------------

        cropped_distance = padded_distance[
            padding:-padding,
            padding:-padding,
        ]

        # ----------------------------------
        # Restore original image size
        # ----------------------------------

        distance_map = np.zeros(
            mask.shape,
            dtype=np.float32,
        )

        distance_map[
            y:y + height,
            x:x + width,
        ] = cropped_distance.astype(
            np.float32
        )

        # ----------------------------------
        # Safety validation
        # ----------------------------------

        if not np.isfinite(
            distance_map
        ).all():

            raise PostprocessingError(
                "Distance transform contains "
                "non-finite values."
            )

        if np.any(
            distance_map < 0
        ):

            raise PostprocessingError(
                "Distance transform contains "
                "negative values."
            )

        logger.info(
            "Distance transform computed successfully."
        )

        return distance_map

    except PostprocessingError:

        raise

    except Exception as e:

        logger.exception(
            "Failed to compute distance transform."
        )

        raise PostprocessingError(
            "Failed to compute distance transform."
        ) from e

def compute_contours(
    region: dict,
) -> list[dict]:
    """
    Extract and analyze contours of a foreground region.

    The region is cropped to its bounding box before contour
    extraction. Contour coordinates are then restored to the
    original image coordinate system.

    Contour information is later used for:

    - Boundary analysis
    - Shape irregularity detection
    - Protrusion analysis
    - Attached structure detection
    - False foreground analysis

    Args:
        region:
            Foreground region returned by
            compute_region_features().

    Returns:
        List of contour dictionaries containing:

        - contour
        - area
        - perimeter
        - bounding_box
        - convexity
        - point_count

    Raises:
        PostprocessingError:
            If contour extraction fails.
    """

    try:

        logger.info(
            f"Computing contours for "
            f"region {region['label']}..."
        )

        # ----------------------------------
        # Read region information
        # ----------------------------------

        mask = region["mask"]

        bbox = region["bbox"]

        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        # ----------------------------------
        # Crop region
        # ----------------------------------

        cropped_mask = mask[
            y:y + height,
            x:x + width,
        ]

        # ----------------------------------
        # Validate crop
        # ----------------------------------

        if cropped_mask.size == 0:

            raise PostprocessingError(
                "Foreground region is empty."
            )

        if not np.any(cropped_mask):

            raise PostprocessingError(
                "Foreground region contains no foreground pixels."
            )

        # ----------------------------------
        # Prepare binary mask
        # ----------------------------------

        binary_mask = (
            cropped_mask > 0
        ).astype(
            np.uint8
        ) * 255

        # ----------------------------------
        # Find contours
        # ----------------------------------

        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # ----------------------------------
        # Validate contours
        # ----------------------------------

        if not contours:

            raise PostprocessingError(
                "No contours detected in foreground region."
            )

        contour_results = []

        # ----------------------------------
        # Analyze contours
        # ----------------------------------

        for contour in contours:

            area = float(
                cv2.contourArea(
                    contour
                )
            )

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True,
                )
            )

            contour_x, contour_y, contour_width, contour_height = (
                cv2.boundingRect(
                    contour
                )
            )

            # Restore contour coordinates
            restored_contour = (
                contour.astype(
                    np.int32
                ).copy()
            )

            restored_contour[:, 0, 0] += x
            restored_contour[:, 0, 1] += y

            # ----------------------------------
            # Convex hull
            # ----------------------------------

            hull = cv2.convexHull(
                contour
            )

            hull_area = float(
                cv2.contourArea(
                    hull
                )
            )

            solidity = (
                area / hull_area
                if hull_area > 0
                else 0.0
            )

            contour_results.append(
                {
                    "contour": restored_contour,

                    "area": area,

                    "perimeter": perimeter,

                    "bounding_box": {
                        "x": int(
                            contour_x + x
                        ),
                        "y": int(
                            contour_y + y
                        ),
                        "width": int(
                            contour_width
                        ),
                        "height": int(
                            contour_height
                        ),
                    },

                    "solidity": float(
                        solidity
                    ),

                    "point_count": int(
                        len(contour)
                    ),
                }
            )

        # ----------------------------------
        # Sort largest contour first
        # ----------------------------------

        contour_results.sort(
            key=lambda item: item["area"],
            reverse=True,
        )

        logger.info(
            f"Detected {len(contour_results)} "
            f"contour(s) successfully."
        )

        return contour_results

    except Exception as e:

        logger.exception(
            "Failed to compute contours."
        )

        raise PostprocessingError(
            "Failed to compute contours."
        ) from e

def analyze_region_shape(
    regions: list[dict],
) -> list[dict]:
    """
    Perform comprehensive geometric analysis of foreground regions.

    This function orchestrates the low-level shape-analysis
    operations and combines their results into a standardized
    representation.

    The analysis includes:

    - Morphological skeleton
    - Convex hull
    - Distance transform
    - Contours
    - Skeleton statistics
    - Hull statistics
    - Thickness statistics
    - Contour statistics

    This function does NOT remove or modify foreground pixels.
    It only produces structural information that will later be
    consumed by attached-structure detection and foreground
    cleaning.

    Args:
        regions:
            Foreground regions enriched by
            compute_region_features().

    Returns:
        List of regions containing comprehensive shape analysis.

    Raises:
        PostprocessingError:
            If shape analysis fails.
    """

    try:

        logger.info(
            "Analyzing foreground region shapes..."
        )

        analyzed_regions = []

        for region in regions:

            logger.info(
                f"Analyzing shape for "
                f"region {region['label']}..."
            )

            # ----------------------------------
            # Skeleton
            # ----------------------------------

            skeleton = compute_skeleton(
                region
            )

            skeleton_pixels = int(
                np.count_nonzero(
                    skeleton
                )
            )

            # ----------------------------------
            # Convex hull
            # ----------------------------------

            convex_hull = compute_convex_hull(
                region
            )

            hull_area = int(
                np.count_nonzero(
                    convex_hull
                )
            )

            foreground_area = int(
                np.count_nonzero(
                    region["mask"]
                )
            )

            hull_area_ratio = (
                hull_area /
                foreground_area
                if foreground_area > 0
                else 0.0
            )

            hull_excess_pixels = max(
                0,
                hull_area -
                foreground_area,
            )

            # ----------------------------------
            # Distance transform
            # ----------------------------------

            distance_transform = (
                compute_distance_transform(
                    region
                )
            )

            foreground_mask = (
                region["mask"] > 0
            )

            foreground_distances = (
                distance_transform[
                    foreground_mask
                ]
            )

            if foreground_distances.size > 0:

                min_thickness = float(
                    foreground_distances.min()
                )

                mean_thickness = float(
                    foreground_distances.mean()
                )

                max_thickness = float(
                    foreground_distances.max()
                )

            else:

                min_thickness = 0.0
                mean_thickness = 0.0
                max_thickness = 0.0

            # ----------------------------------
            # Contours
            # ----------------------------------

            contours = compute_contours(
                region
            )

            contour_count = len(
                contours
            )

            if contour_count > 0:

                largest_contour = (
                    contours[0]
                )

                largest_contour_area = (
                    float(
                        largest_contour["area"]
                    )
                )

                largest_contour_perimeter = (
                    float(
                        largest_contour["perimeter"]
                    )
                )

                largest_contour_solidity = (
                    float(
                        largest_contour["solidity"]
                    )
                )

                largest_contour_points = (
                    int(
                        largest_contour[
                            "point_count"
                        ]
                    )
                )

            else:

                largest_contour_area = 0.0
                largest_contour_perimeter = 0.0
                largest_contour_solidity = 0.0
                largest_contour_points = 0

            # ----------------------------------
            # Shape analysis
            # ----------------------------------

            shape_features = {

                "skeleton_pixels": (
                    skeleton_pixels
                ),

                "hull_area": (
                    hull_area
                ),

                "hull_area_ratio": float(
                    hull_area_ratio
                ),

                "hull_excess_pixels": (
                    hull_excess_pixels
                ),

                "min_thickness": (
                    min_thickness
                ),

                "mean_thickness": (
                    mean_thickness
                ),

                "max_thickness": (
                    max_thickness
                ),

                "contour_count": (
                    contour_count
                ),

                "largest_contour_area": (
                    largest_contour_area
                ),

                "largest_contour_perimeter": (
                    largest_contour_perimeter
                ),

                "largest_contour_solidity": (
                    largest_contour_solidity
                ),

                "largest_contour_points": (
                    largest_contour_points
                ),
            }

            # ----------------------------------
            # Build analyzed region
            # ----------------------------------

            analyzed_region = {
                **region,

                "shape_analysis": {

                    "skeleton": (
                        skeleton
                    ),

                    "convex_hull": (
                        convex_hull
                    ),

                    "distance_transform": (
                        distance_transform
                    ),

                    "contours": (
                        contours
                    ),

                    "features": (
                        shape_features
                    ),
                },
            }

            analyzed_regions.append(
                analyzed_region
            )

            logger.info(
                f"Shape analysis completed for "
                f"region {region['label']}."
            )

        logger.info(
            "Foreground region shape analysis "
            "completed successfully."
        )

        return analyzed_regions

    except Exception as e:

        logger.exception(
            "Failed to analyze foreground region shapes."
        )

        raise PostprocessingError(
            "Failed to analyze foreground region shapes."
        ) from e



def detect_branch_points(
    skeleton: np.ndarray,
) -> dict:
    """
    Detect and analyze branch points in a binary foreground skeleton.

    The function performs two stages:

    1. Detect raw branch-point candidate pixels using
       8-connected skeleton neighborhood analysis.

    2. Cluster adjacent candidate pixels into meaningful
       branch-point nodes using scikit-image connected-component
       labeling and region properties.

    A branch-point candidate is a skeleton pixel connected
    to three or more neighboring skeleton pixels.

    The function does not modify the skeleton or foreground mask.

    Args:
        skeleton:
            Binary skeleton represented as a NumPy array.

    Returns:
        Dictionary containing:

        - candidate_pixels:
            Raw branch-point pixel coordinates.

        - candidate_mask:
            Binary mask containing raw branch-point candidates.

        - branch_clusters:
            Grouped branch-point clusters.

        - branch_count:
            Number of detected branch nodes.

    Raises:
        PostprocessingError:
            If branch-point detection fails.
    """

    try:

        logger.info(
            "Detecting skeleton branch points..."
        )

        # ----------------------------------
        # Validate input
        # ----------------------------------

        if not isinstance(
            skeleton,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Skeleton must be a NumPy array."
            )

        if skeleton.ndim != 2:
            raise PostprocessingError(
                "Skeleton must be a 2D array."
            )

        if skeleton.size == 0:
            raise PostprocessingError(
                "Skeleton cannot be empty."
            )

        # ----------------------------------
        # Convert skeleton to binary
        # ----------------------------------

        binary = (
            skeleton > 0
        ).astype(
            np.uint8
        )

        # ----------------------------------
        # Pad skeleton
        # ----------------------------------

        padded = np.pad(
            binary,
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        # ----------------------------------
        # Count 8-connected neighbors
        # ----------------------------------

        neighbor_count = np.zeros_like(
            binary,
            dtype=np.uint8,
        )

        for dy in (-1, 0, 1):

            for dx in (-1, 0, 1):

                if (
                    dy == 0
                    and dx == 0
                ):
                    continue

                neighbor_count += (
                    padded[
                        1 + dy:
                        1 + dy + binary.shape[0],
                        1 + dx:
                        1 + dx + binary.shape[1],
                    ]
                )

        # ----------------------------------
        # Detect raw branch-point pixels
        # ----------------------------------

        candidate_mask = (
            (binary == 1)
            &
            (neighbor_count >= 3)
        )

        candidate_mask = (
            candidate_mask.astype(
                np.uint8
            )
        )

        candidate_pixels = np.argwhere(
            candidate_mask > 0
        )

        raw_candidate_count = (
            len(candidate_pixels)
        )

        logger.info(
            f"Detected "
            f"{raw_candidate_count} "
            f"raw branch-point candidate(s)."
        )

        # ----------------------------------
        # No candidates
        # ----------------------------------

        if raw_candidate_count == 0:

            logger.info(
                "No branch-point clusters detected."
            )

            return {
                "candidate_pixels":
                    candidate_pixels,

                "candidate_mask":
                    candidate_mask,

                "branch_clusters":
                    [],

                "branch_count":
                    0,
            }

        # ----------------------------------
        # Label candidate clusters
        #
        # scikit-image
        # ----------------------------------

        labeled_candidates = label(
            candidate_mask,
            connectivity=2,
            background=0,
        )

        # ----------------------------------
        # Extract cluster properties
        #
        # scikit-image regionprops
        # ----------------------------------

        properties = regionprops(
            labeled_candidates
        )

        branch_clusters = []

        # ----------------------------------
        # Build branch cluster records
        # ----------------------------------

        for prop in properties:

            coords = (
                prop.coords
                .astype(
                    np.int32
                )
            )

            min_row, min_col, max_row, max_col = (
                prop.bbox
            )

            cluster = {

                "pixels": coords,

                "pixel_count": int(
                    prop.area
                ),

                "centroid": (
                    float(
                        prop.centroid[0]
                    ),
                    float(
                        prop.centroid[1]
                    ),
                ),

                "bbox": {

                    "x": int(
                        min_col
                    ),

                    "y": int(
                        min_row
                    ),

                    "width": int(
                        max_col - min_col
                    ),

                    "height": int(
                        max_row - min_row
                    ),
                },
            }

            branch_clusters.append(
                cluster
            )

        # ----------------------------------
        # Sort largest cluster first
        # ----------------------------------

        branch_clusters.sort(
            key=lambda cluster:
                cluster["pixel_count"],
            reverse=True,
        )

        # ----------------------------------
        # Final statistics
        # ----------------------------------

        branch_count = len(
            branch_clusters
        )

        logger.info(
            f"Detected "
            f"{branch_count} "
            f"branch-point cluster(s)."
        )

        return {

            "candidate_pixels":
                candidate_pixels,

            "candidate_mask":
                candidate_mask,

            "branch_clusters":
                branch_clusters,

            "branch_count":
                branch_count,
        }

    except PostprocessingError:

        raise

    except Exception as e:

        logger.exception(
            "Failed to detect skeleton branch points."
        )

        raise PostprocessingError(
            "Failed to detect skeleton branch points."
        ) from e

def detect_end_points(
    skeleton: np.ndarray,
) -> np.ndarray:
    """
    Detect endpoint pixels in a binary foreground skeleton.

    An endpoint is a skeleton pixel that has exactly one
    8-connected neighboring skeleton pixel.

    Endpoint information is later used by
    detect_attached_structures() for:

    - Protrusion analysis
    - Thin-structure analysis
    - Skeleton path analysis
    - Structural connectivity analysis

    This function only analyzes the skeleton and does not
    modify foreground pixels.

    Args:
        skeleton:
            Binary skeleton represented as a NumPy array.

    Returns:
        NumPy array containing endpoint coordinates in
        (y, x) format.

    Raises:
        PostprocessingError:
            If endpoint detection fails.
    """
    try:

        logger.info(
            "Detecting skeleton endpoints..."
        )

        # ----------------------------------
        # Validate input
        # ----------------------------------

        if not isinstance(
            skeleton,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Skeleton must be a NumPy array."
            )

        if skeleton.ndim != 2:
            raise PostprocessingError(
                "Skeleton must be a 2D array."
            )

        if skeleton.size == 0:
            raise PostprocessingError(
                "Skeleton cannot be empty."
            )

        # ----------------------------------
        # Convert to binary
        # ----------------------------------

        binary = (
            skeleton > 0
        ).astype(
            np.uint8
        )

        # ----------------------------------
        # Pad skeleton
        # ----------------------------------

        padded = np.pad(
            binary,
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        # ----------------------------------
        # Count 8-connected neighbors
        # ----------------------------------

        neighbor_count = np.zeros_like(
            binary,
            dtype=np.uint8,
        )

        for dy in (-1, 0, 1):

            for dx in (-1, 0, 1):

                if (
                    dy == 0
                    and dx == 0
                ):
                    continue

                neighbor_count += (
                    padded[
                        1 + dy:
                        1 + dy + binary.shape[0],
                        1 + dx:
                        1 + dx + binary.shape[1],
                    ]
                )

        # ----------------------------------
        # Endpoint condition
        # ----------------------------------

        endpoints = np.argwhere(
            (binary == 1)
            &
            (neighbor_count == 1)
        )

        logger.info(
            f"Detected {len(endpoints)} "
            f"skeleton endpoint(s)."
        )

        return endpoints

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to detect skeleton endpoints."
        )

        raise PostprocessingError(
            "Failed to detect skeleton endpoints."
        ) from e

def detect_protrusions(
    region: dict,
    thin_ratio: float = 0.35,
    core_ratio: float = 0.55,
    min_length_ratio: float = 0.03,
) -> dict:
    """
    Detect geometrically suspicious protrusions extending from
    the main body of a foreground region.

    Protrusion detection uses:

    - Skeleton topology
    - Skeleton endpoints
    - Branch points
    - Euclidean distance transform
    - Adaptive thickness thresholds
    - Relative skeleton-path length

    A protrusion candidate is a skeleton path that:

    1. Starts at a skeleton endpoint.
    2. Begins with relatively thin foreground structure.
    3. Extends toward a substantially thicker core region
       or terminates at a branch point.
    4. Has sufficient relative path length.

    This function performs analysis only. It does not remove
    or modify foreground pixels.

    Args:
        region:
            Region containing shape analysis.

        thin_ratio:
            Maximum endpoint thickness relative to the maximum
            skeleton thickness.

        core_ratio:
            Relative thickness threshold used to identify the
            main foreground core.

        min_length_ratio:
            Minimum protrusion path length relative to the
            foreground region diagonal.

    Returns:
        Dictionary containing:

        - protrusions
        - protrusion_count
        - candidate_mask

    Raises:
        PostprocessingError:
            If protrusion detection fails.
    """

    try:

        logger.info(
            f"Detecting protrusions for "
            f"region {region['label']}..."
        )

        # --------------------------------------------------
        # Validate parameters
        # --------------------------------------------------

        if not 0.0 < thin_ratio < 1.0:
            raise PostprocessingError(
                "thin_ratio must be between 0 and 1."
            )

        if not 0.0 < core_ratio <= 1.0:
            raise PostprocessingError(
                "core_ratio must be between 0 and 1."
            )

        if not 0.0 < min_length_ratio < 1.0:
            raise PostprocessingError(
                "min_length_ratio must be between 0 and 1."
            )

        if thin_ratio >= core_ratio:
            raise PostprocessingError(
                "thin_ratio must be smaller than core_ratio."
            )

        # --------------------------------------------------
        # Retrieve shape analysis
        # --------------------------------------------------

        if "shape_analysis" not in region:
            raise PostprocessingError(
                "Region does not contain shape analysis."
            )

        shape_analysis = region[
            "shape_analysis"
        ]

        skeleton = shape_analysis[
            "skeleton"
        ]

        distance_transform = shape_analysis[
            "distance_transform"
        ]

        # --------------------------------------------------
        # Validate arrays
        # --------------------------------------------------

        if not isinstance(
            skeleton,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Skeleton must be a NumPy array."
            )

        if not isinstance(
            distance_transform,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Distance transform must be a NumPy array."
            )

        if skeleton.shape != distance_transform.shape:
            raise PostprocessingError(
                "Skeleton and distance transform "
                "must have identical shapes."
            )

        if skeleton.shape != region["mask"].shape:
            raise PostprocessingError(
                "Skeleton and region mask must "
                "have identical shapes."
            )

        # --------------------------------------------------
        # Prepare binary skeleton
        # --------------------------------------------------

        skeleton_binary = (
            skeleton > 0
        ).astype(np.uint8)

        foreground = (
            region["mask"] > 0
        )

        # --------------------------------------------------
        # Empty skeleton
        # --------------------------------------------------

        if not np.any(skeleton_binary):

            logger.info(
                "No skeleton pixels available "
                "for protrusion analysis."
            )

            return {
                "protrusions": [],
                "protrusion_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        # --------------------------------------------------
        # Skeleton distance values
        # --------------------------------------------------

        skeleton_distances = (
            distance_transform[
                skeleton_binary > 0
            ]
        )

        if skeleton_distances.size == 0:

            return {
                "protrusions": [],
                "protrusion_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        max_thickness = float(
            skeleton_distances.max()
        )

        if max_thickness <= 0:

            return {
                "protrusions": [],
                "protrusion_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        # --------------------------------------------------
        # Adaptive thickness thresholds
        # --------------------------------------------------

        thin_threshold = (
            max_thickness * thin_ratio
        )

        core_threshold = (
            max_thickness * core_ratio
        )

        # --------------------------------------------------
        # Detect endpoints
        # --------------------------------------------------

        endpoints = detect_end_points(
            skeleton_binary
        )

        # --------------------------------------------------
        # Detect branch points
        # --------------------------------------------------

        branch_analysis = (
            detect_branch_points(
                skeleton_binary
            )
        )

        branch_mask = np.zeros_like(
            skeleton_binary,
            dtype=np.uint8,
        )

        for cluster in branch_analysis[
            "branch_clusters"
        ]:

            for y_coord, x_coord in cluster[
                "pixels"
            ]:

                branch_mask[
                    y_coord,
                    x_coord
                ] = 1

        # --------------------------------------------------
        # Foreground region diagonal
        #
        # IMPORTANT:
        # Use the actual foreground bounding box,
        # not the entire image dimensions.
        # --------------------------------------------------

        bbox = region["bbox"]

        region_diagonal = float(
            np.hypot(
                bbox["width"],
                bbox["height"],
            )
        )

        if region_diagonal <= 0:
            raise PostprocessingError(
                "Foreground region has invalid dimensions."
            )

        minimum_length = (
            region_diagonal *
            min_length_ratio
        )

        # --------------------------------------------------
        # Candidate mask
        # --------------------------------------------------

        candidate_mask = np.zeros_like(
            skeleton_binary,
            dtype=np.uint8,
        )

        protrusions = []

        image_height, image_width = (
            skeleton_binary.shape
        )

        # --------------------------------------------------
        # Analyze every endpoint
        # --------------------------------------------------

        for endpoint in endpoints:

            start_y = int(endpoint[0])
            start_x = int(endpoint[1])

            # ----------------------------------------------
            # Endpoint thickness
            # ----------------------------------------------

            endpoint_thickness = float(
                distance_transform[
                    start_y,
                    start_x
                ]
            )

            # ----------------------------------------------
            # Thin endpoint requirement
            # ----------------------------------------------

            if (
                endpoint_thickness
                > thin_threshold
            ):
                continue

            relative_thickness = (
                endpoint_thickness /
                max_thickness
            )

            # ----------------------------------------------
            # Trace skeleton path
            # ----------------------------------------------

            path = []

            visited = set()

            current = (
                start_y,
                start_x,
            )

            previous = None

            path_length = 0.0

            reached_core = False
            reached_branch = False

            while True:

                current_y, current_x = current

                if current in visited:
                    break

                visited.add(current)

                path.append(current)

                current_thickness = float(
                    distance_transform[
                        current_y,
                        current_x
                    ]
                )

                # ------------------------------------------
                # Core reached
                # ------------------------------------------

                if (
                    current_thickness
                    >= core_threshold
                ):

                    reached_core = True
                    break

                # ------------------------------------------
                # Branch reached
                # ------------------------------------------

                if (
                    branch_mask[
                        current_y,
                        current_x
                    ] > 0
                    and
                    current != (
                        start_y,
                        start_x,
                    )
                ):

                    reached_branch = True
                    break

                # ------------------------------------------
                # Find next skeleton pixel
                # ------------------------------------------

                neighbors = []

                for dy in (-1, 0, 1):

                    for dx in (-1, 0, 1):

                        if (
                            dy == 0
                            and dx == 0
                        ):
                            continue

                        ny = current_y + dy
                        nx = current_x + dx

                        if (
                            ny < 0
                            or ny >= image_height
                            or nx < 0
                            or nx >= image_width
                        ):
                            continue

                        if (
                            skeleton_binary[
                                ny,
                                nx
                            ] == 0
                        ):
                            continue

                        if (
                            previous is not None
                            and
                            (ny, nx) == previous
                        ):
                            continue

                        neighbors.append(
                            (ny, nx)
                        )

                # ------------------------------------------
                # No continuation
                # ------------------------------------------

                if not neighbors:
                    break

                # ------------------------------------------
                # Multiple continuations
                # ------------------------------------------

                if len(neighbors) > 1:

                    reached_branch = True
                    break

                next_pixel = neighbors[0]

                # ------------------------------------------
                # Euclidean path length
                # ------------------------------------------

                dy = (
                    next_pixel[0]
                    - current_y
                )

                dx = (
                    next_pixel[1]
                    - current_x
                )

                path_length += float(
                    np.hypot(
                        dx,
                        dy,
                    )
                )

                previous = current
                current = next_pixel

            # --------------------------------------------------
            # Candidate validation
            # --------------------------------------------------

            if path_length < minimum_length:
                continue

            if not (
                reached_core
                or reached_branch
            ):
                continue

            # --------------------------------------------------
            # Path thickness statistics
            # --------------------------------------------------

            path_thicknesses = np.asarray(
                [
                    distance_transform[
                        y_coord,
                        x_coord
                    ]
                    for y_coord, x_coord
                    in path
                ],
                dtype=np.float32,
            )

            mean_thickness = float(
                path_thicknesses.mean()
            )

            maximum_path_thickness = float(
                path_thicknesses.max()
            )

            relative_length = (
                path_length /
                region_diagonal
            )

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            thinness_score = max(
                0.0,
                1.0 -
                (
                    mean_thickness /
                    max_thickness
                ),
            )

            length_score = min(
                1.0,
                relative_length /
                max(
                    min_length_ratio,
                    1e-6,
                ),
            )

            confidence = (
                0.6 * thinness_score
                +
                0.4 * length_score
            )

            # --------------------------------------------------
            # Candidate mask
            # --------------------------------------------------

            for y_coord, x_coord in path:

                if foreground[
                    y_coord,
                    x_coord
                ]:

                    candidate_mask[
                        y_coord,
                        x_coord
                    ] = 1

            # --------------------------------------------------
            # Bounding box
            # --------------------------------------------------

            path_array = np.asarray(
                path,
                dtype=np.int32,
            )

            min_y = int(
                path_array[:, 0].min()
            )

            max_y = int(
                path_array[:, 0].max()
            )

            min_x = int(
                path_array[:, 1].min()
            )

            max_x = int(
                path_array[:, 1].max()
            )

            protrusions.append(
                {
                    "endpoint": (
                        start_y,
                        start_x,
                    ),

                    "path": path,

                    "path_length": float(
                        path_length
                    ),

                    "relative_length": float(
                        relative_length
                    ),

                    "endpoint_thickness": (
                        endpoint_thickness
                    ),

                    "relative_thickness": (
                        relative_thickness
                    ),

                    "mean_thickness": (
                        mean_thickness
                    ),

                    "max_thickness": (
                        maximum_path_thickness
                    ),

                    "thinness_score": float(
                        thinness_score
                    ),

                    "confidence": float(
                        confidence
                    ),

                    "reached_core": bool(
                        reached_core
                    ),

                    "reached_branch": bool(
                        reached_branch
                    ),

                    "bbox": {
                        "x": min_x,
                        "y": min_y,
                        "width": (
                            max_x - min_x + 1
                        ),
                        "height": (
                            max_y - min_y + 1
                        ),
                    },
                }
            )

        # --------------------------------------------------
        # Sort by confidence
        # --------------------------------------------------

        protrusions.sort(
            key=lambda item:
                item["confidence"],
            reverse=True,
        )

        logger.info(
            f"Detected "
            f"{len(protrusions)} "
            f"protrusion candidate(s)."
        )

        return {
            "protrusions": protrusions,

            "protrusion_count": (
                len(protrusions)
            ),

            "candidate_mask": (
                candidate_mask
            ),
        }

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to detect protrusions."
        )

        raise PostprocessingError(
            "Failed to detect protrusions."
        ) from e

def detect_thin_structures(
    region: dict,
    thin_ratio: float = 0.35,
    min_length_ratio: float = 0.01,
    max_thickness_ratio: float = 0.40,
) -> dict:
    """
    Detect thin foreground structures inside a region.

    Thin structures are identified using the foreground skeleton
    together with the Euclidean distance transform.

    The analysis uses:

    - NumPy for binary/skeleton processing and statistics.
    - SciPy distance-transform values for local thickness.
    - scikit-image connected-component labeling for grouping
      connected thin skeleton structures.

    A structure is considered a thin-structure candidate when
    its skeleton contains pixels whose local thickness is below
    the adaptive thickness threshold.

    The function performs analysis only. It does not modify
    or remove foreground pixels.

    Args:
        region:
            Foreground region containing shape analysis.

        thin_ratio:
            Relative thickness threshold based on the maximum
            skeleton thickness.

        min_length_ratio:
            Minimum candidate length relative to the foreground
            region diagonal.

        max_thickness_ratio:
            Maximum mean thickness relative to the maximum
            skeleton thickness for a candidate structure.

    Returns:
        Dictionary containing:

        - thin_structures:
            Detected thin-structure information.

        - thin_structure_count:
            Number of detected thin structures.

        - candidate_mask:
            Full-size binary mask containing thin-structure
            skeleton pixels.

    Raises:
        PostprocessingError:
            If thin-structure detection fails.
    """

    try:

        logger.info(
            f"Detecting thin structures for "
            f"region {region['label']}..."
        )

        # --------------------------------------------------
        # Validate parameters
        # --------------------------------------------------

        if not 0.0 < thin_ratio < 1.0:
            raise PostprocessingError(
                "thin_ratio must be between 0 and 1."
            )

        if not 0.0 < min_length_ratio < 1.0:
            raise PostprocessingError(
                "min_length_ratio must be between 0 and 1."
            )

        if not 0.0 < max_thickness_ratio <= 1.0:
            raise PostprocessingError(
                "max_thickness_ratio must be between 0 and 1."
            )

        # --------------------------------------------------
        # Retrieve shape analysis
        # --------------------------------------------------

        if "shape_analysis" not in region:
            raise PostprocessingError(
                "Region does not contain shape analysis."
            )

        shape_analysis = region[
            "shape_analysis"
        ]

        skeleton = shape_analysis[
            "skeleton"
        ]

        distance_transform = shape_analysis[
            "distance_transform"
        ]

        # --------------------------------------------------
        # Validate arrays
        # --------------------------------------------------

        if not isinstance(
            skeleton,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Skeleton must be a NumPy array."
            )

        if not isinstance(
            distance_transform,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Distance transform must be a NumPy array."
            )

        if skeleton.ndim != 2:
            raise PostprocessingError(
                "Skeleton must be a 2D array."
            )

        if distance_transform.ndim != 2:
            raise PostprocessingError(
                "Distance transform must be a 2D array."
            )

        if skeleton.shape != distance_transform.shape:
            raise PostprocessingError(
                "Skeleton and distance transform "
                "must have identical shapes."
            )

        if skeleton.shape != region["mask"].shape:
            raise PostprocessingError(
                "Skeleton and region mask must "
                "have identical shapes."
            )

        # --------------------------------------------------
        # Prepare binary skeleton
        # --------------------------------------------------

        skeleton_binary = (
            skeleton > 0
        ).astype(
            np.uint8
        )

        # --------------------------------------------------
        # Empty skeleton
        # --------------------------------------------------

        if not np.any(skeleton_binary):

            logger.info(
                "No skeleton pixels available "
                "for thin-structure analysis."
            )

            return {
                "thin_structures": [],
                "thin_structure_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        # --------------------------------------------------
        # Extract skeleton thickness values
        # --------------------------------------------------

        skeleton_distances = (
            distance_transform[
                skeleton_binary > 0
            ]
        )

        if skeleton_distances.size == 0:

            return {
                "thin_structures": [],
                "thin_structure_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        max_thickness = float(
            skeleton_distances.max()
        )

        if max_thickness <= 0:

            logger.info(
                "Skeleton has no valid thickness values."
            )

            return {
                "thin_structures": [],
                "thin_structure_count": 0,
                "candidate_mask": np.zeros_like(
                    skeleton_binary,
                    dtype=np.uint8,
                ),
            }

        # --------------------------------------------------
        # Adaptive thickness threshold
        # --------------------------------------------------

        thin_threshold = (
            max_thickness *
            thin_ratio
        )

        maximum_allowed_mean_thickness = (
            max_thickness *
            max_thickness_ratio
        )

        # --------------------------------------------------
        # Identify thin skeleton pixels
        # --------------------------------------------------

        thin_pixel_mask = (
            (skeleton_binary > 0)
            &
            (
                distance_transform
                <= thin_threshold
            )
        ).astype(
            np.uint8
        )

        thin_pixel_count = int(
            thin_pixel_mask.sum()
        )

        if thin_pixel_count == 0:

            logger.info(
                "No thin skeleton pixels detected."
            )

            return {
                "thin_structures": [],
                "thin_structure_count": 0,
                "candidate_mask": (
                    thin_pixel_mask
                ),
            }

        # --------------------------------------------------
        # Group connected thin skeleton pixels
        #
        # scikit-image is intentionally used here instead
        # of OpenCV connected components.
        # --------------------------------------------------

        labeled_thin, structure_count = (
            label(
                thin_pixel_mask,
                connectivity=2,
                return_num=True,
            )
        )

        # --------------------------------------------------
        # Region diagonal
        # --------------------------------------------------

        bbox = region["bbox"]

        region_diagonal = float(
            np.hypot(
                bbox["width"],
                bbox["height"],
            )
        )

        if region_diagonal <= 0:

            raise PostprocessingError(
                "Foreground region has invalid dimensions."
            )

        minimum_length = (
            region_diagonal *
            min_length_ratio
        )

        # --------------------------------------------------
        # Analyze each connected structure
        # --------------------------------------------------

        thin_structures = []

        image_height, image_width = (
            skeleton_binary.shape
        )

        for structure_label in range(
            1,
            structure_count + 1,
        ):

            structure_mask = (
                labeled_thin ==
                structure_label
            )

            coordinates = np.argwhere(
                structure_mask
            )

            if coordinates.size == 0:
                continue

            # ----------------------------------------------
            # Thickness statistics
            # ----------------------------------------------

            thickness_values = (
                distance_transform[
                    structure_mask
                ]
            )

            mean_thickness = float(
                thickness_values.mean()
            )

            minimum_thickness = float(
                thickness_values.min()
            )

            maximum_structure_thickness = float(
                thickness_values.max()
            )

            relative_thickness = (
                mean_thickness /
                max_thickness
            )

            # ----------------------------------------------
            # Approximate skeleton length
            #
            # Sum distances between neighboring pixels.
            # ----------------------------------------------

            coordinate_set = {
                (
                    int(y_coord),
                    int(x_coord),
                )
                for y_coord, x_coord
                in coordinates
            }

            path_length = 0.0

            for y_coord, x_coord in coordinates:

                y_coord = int(y_coord)
                x_coord = int(x_coord)

                # Count only forward directions to avoid
                # measuring every edge twice.
                for dy, dx in (
                    (0, 1),
                    (1, 0),
                    (1, 1),
                    (1, -1),
                ):

                    neighbor = (
                        y_coord + dy,
                        x_coord + dx,
                    )

                    if neighbor not in coordinate_set:
                        continue

                    if dy != 0 and dx != 0:
                        path_length += np.sqrt(2.0)
                    else:
                        path_length += 1.0

            relative_length = (
                path_length /
                region_diagonal
            )

            # ----------------------------------------------
            # Minimum length requirement
            # ----------------------------------------------

            if path_length < minimum_length:
                continue

            # ----------------------------------------------
            # Thickness requirement
            # ----------------------------------------------

            if (
                mean_thickness
                > maximum_allowed_mean_thickness
            ):
                continue

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            min_y = int(
                coordinates[:, 0].min()
            )

            max_y = int(
                coordinates[:, 0].max()
            )

            min_x = int(
                coordinates[:, 1].min()
            )

            max_x = int(
                coordinates[:, 1].max()
            )

            # ----------------------------------------------
            # Endpoint relationship
            # ----------------------------------------------

            endpoint_count = 0

            for y_coord, x_coord in coordinates:

                y_coord = int(y_coord)
                x_coord = int(x_coord)

                neighbors = 0

                for dy in (-1, 0, 1):

                    for dx in (-1, 0, 1):

                        if (
                            dy == 0
                            and dx == 0
                        ):
                            continue

                        ny = y_coord + dy
                        nx = x_coord + dx

                        if (
                            0 <= ny < image_height
                            and
                            0 <= nx < image_width
                            and
                            skeleton_binary[
                                ny,
                                nx
                            ] > 0
                        ):
                            neighbors += 1

                if neighbors == 1:
                    endpoint_count += 1

            # ----------------------------------------------
            # Thinness score
            # ----------------------------------------------

            thinness_score = max(
                0.0,
                1.0 -
                relative_thickness,
            )

            length_score = min(
                1.0,
                relative_length /
                max(
                    min_length_ratio,
                    1e-6,
                ),
            )

            confidence = (
                0.6 * thinness_score
                +
                0.4 * length_score
            )

            # ----------------------------------------------
            # Store structure
            # ----------------------------------------------

            thin_structures.append(
                {
                    "label": int(
                        structure_label
                    ),

                    "pixels": coordinates,

                    "pixel_count": int(
                        len(coordinates)
                    ),

                    "length": float(
                        path_length
                    ),

                    "relative_length": float(
                        relative_length
                    ),

                    "min_thickness": float(
                        minimum_thickness
                    ),

                    "mean_thickness": float(
                        mean_thickness
                    ),

                    "max_thickness": float(
                        maximum_structure_thickness
                    ),

                    "relative_thickness": float(
                        relative_thickness
                    ),

                    "endpoint_count": int(
                        endpoint_count
                    ),

                    "thinness_score": float(
                        thinness_score
                    ),

                    "confidence": float(
                        confidence
                    ),

                    "bbox": {
                        "x": min_x,
                        "y": min_y,
                        "width": (
                            max_x - min_x + 1
                        ),
                        "height": (
                            max_y - min_y + 1
                        ),
                    },
                }
            )

        # --------------------------------------------------
        # Sort by confidence
        # --------------------------------------------------

        thin_structures.sort(
            key=lambda item:
                item["confidence"],
            reverse=True,
        )

        # --------------------------------------------------
        # Candidate mask
        #
        # Keep only accepted structures.
        # --------------------------------------------------

        candidate_mask = np.zeros_like(
            skeleton_binary,
            dtype=np.uint8,
        )

        for structure in thin_structures:

            for y_coord, x_coord in (
                structure["pixels"]
            ):

                candidate_mask[
                    y_coord,
                    x_coord
                ] = 1

        logger.info(
            f"Detected "
            f"{len(thin_structures)} "
            f"thin structure(s)."
        )

        return {
            "thin_structures": (
                thin_structures
            ),

            "thin_structure_count": (
                len(thin_structures)
            ),

            "candidate_mask": (
                candidate_mask
            ),
        }

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to detect thin structures."
        )

        raise PostprocessingError(
            "Failed to detect thin structures."
        ) from e

def detect_attached_structures(
    region: dict,
    thin_weight: float = 0.25,
    protrusion_weight: float = 0.35,
    endpoint_weight: float = 0.15,
    branch_weight: float = 0.10,
    connectivity_weight: float = 0.15,
) -> dict:
    """
    Integrate structural evidence and detect foreground structures
    that are potentially attached to the main foreground region.

    This function combines the outputs of:

    - detect_branch_points()
    - detect_end_points()
    - detect_protrusions()
    - detect_thin_structures()

    The function does not remove, modify, or classify foreground
    pixels as false foreground.

    It produces structural evidence that is consumed later by
    classify_regions().

    Args:
        region:
            Foreground region containing shape analysis.

        thin_weight:
            Weight assigned to thin-structure evidence.

        protrusion_weight:
            Weight assigned to protrusion evidence.

        endpoint_weight:
            Weight assigned to endpoint evidence.

        branch_weight:
            Weight assigned to branch-point evidence.

        connectivity_weight:
            Weight assigned to structural connectivity evidence.

    Returns:
        Dictionary containing:

        - branch_points
        - end_points
        - protrusions
        - thin_structures
        - attached_structures
        - attached_structure_count
        - candidate_mask
        - attachment_features

    Raises:
        PostprocessingError:
            If attached-structure analysis fails.
    """

    try:

        logger.info(
            f"Detecting attached structures for "
            f"region {region['label']}..."
        )

        # --------------------------------------------------
        # Validate region
        # --------------------------------------------------

        if not isinstance(region, dict):
            raise PostprocessingError(
                "Region must be a dictionary."
            )

        if "mask" not in region:
            raise PostprocessingError(
                "Region does not contain a mask."
            )

        if "shape_analysis" not in region:
            raise PostprocessingError(
                "Region does not contain shape analysis."
            )

        # --------------------------------------------------
        # Validate weights
        # --------------------------------------------------

        weights = {
            "thin": thin_weight,
            "protrusion": protrusion_weight,
            "endpoint": endpoint_weight,
            "branch": branch_weight,
            "connectivity": connectivity_weight,
        }

        for name, weight in weights.items():

            if weight < 0:
                raise PostprocessingError(
                    f"{name}_weight cannot be negative."
                )

        weight_sum = sum(
            weights.values()
        )

        if weight_sum <= 0:
            raise PostprocessingError(
                "At least one structural weight "
                "must be greater than zero."
            )

        # Normalize weights so that the final score
        # always remains within [0, 1].
        normalized_weights = {
            name: weight / weight_sum
            for name, weight
            in weights.items()
        }

        # --------------------------------------------------
        # Retrieve shape analysis
        # --------------------------------------------------

        shape_analysis = region[
            "shape_analysis"
        ]

        skeleton = shape_analysis[
            "skeleton"
        ]

        # --------------------------------------------------
        # Validate skeleton
        # --------------------------------------------------

        if not isinstance(
            skeleton,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Skeleton must be a NumPy array."
            )

        if skeleton.ndim != 2:
            raise PostprocessingError(
                "Skeleton must be a 2D array."
            )

        skeleton_binary = (
            skeleton > 0
        ).astype(np.uint8)

        # --------------------------------------------------
        # 1. Branch-point analysis
        # --------------------------------------------------

        branch_analysis = (
            detect_branch_points(
                skeleton_binary
            )
        )

        # --------------------------------------------------
        # 2. Endpoint analysis
        # --------------------------------------------------

        endpoint_coordinates = (
            detect_end_points(
                skeleton_binary
            )
        )

        endpoint_count = int(
            len(endpoint_coordinates)
        )

        # --------------------------------------------------
        # 3. Protrusion analysis
        # --------------------------------------------------

        protrusion_analysis = (
            detect_protrusions(
                region
            )
        )

        # --------------------------------------------------
        # 4. Thin-structure analysis
        # --------------------------------------------------

        thin_analysis = (
            detect_thin_structures(
                region
            )
        )

        # --------------------------------------------------
        # Prepare candidate masks
        # --------------------------------------------------

        protrusion_mask = (
            protrusion_analysis[
                "candidate_mask"
            ] > 0
        )

        thin_mask = (
            thin_analysis[
                "candidate_mask"
            ] > 0
        )

        branch_mask = np.zeros_like(
            skeleton_binary,
            dtype=np.uint8,
        )

        for cluster in branch_analysis[
            "branch_clusters"
        ]:

            for y_coord, x_coord in (
                cluster["pixels"]
            ):

                branch_mask[
                    y_coord,
                    x_coord
                ] = 1

        # --------------------------------------------------
        # Combined structural candidate mask
        # --------------------------------------------------

        candidate_mask = (
            (
                protrusion_mask
                |
                thin_mask
            )
            .astype(np.uint8)
        )

        # --------------------------------------------------
        # Analyze thin structures
        # --------------------------------------------------

        attached_structures = []

        thin_structures = (
            thin_analysis[
                "thin_structures"
            ]
        )

        protrusions = (
            protrusion_analysis[
                "protrusions"
            ]
        )

        # --------------------------------------------------
        # Build protrusion lookup mask
        # --------------------------------------------------

        protrusion_labels = np.zeros_like(
            skeleton_binary,
            dtype=np.int32,
        )

        for index, protrusion in enumerate(
            protrusions,
            start=1,
        ):

            for y_coord, x_coord in (
                protrusion["path"]
            ):

                protrusion_labels[
                    y_coord,
                    x_coord
                ] = index

        # --------------------------------------------------
        # Analyze each thin structure
        # --------------------------------------------------

        for thin_structure in (
            thin_structures
        ):

            pixels = thin_structure[
                "pixels"
            ]

            if len(pixels) == 0:
                continue

            # ----------------------------------------------
            # Create structure mask
            # ----------------------------------------------

            structure_mask = np.zeros_like(
                skeleton_binary,
                dtype=np.uint8,
            )

            for y_coord, x_coord in pixels:

                structure_mask[
                    y_coord,
                    x_coord
                ] = 1

            # ----------------------------------------------
            # Protrusion overlap
            # ----------------------------------------------

            protrusion_overlap = (
                structure_mask.astype(bool)
                &
                protrusion_mask
            )

            has_protrusion_overlap = bool(
                np.any(
                    protrusion_overlap
                )
            )

            # ----------------------------------------------
            # Branch relationship
            # ----------------------------------------------

            branch_overlap = (
                structure_mask.astype(bool)
                &
                branch_mask.astype(bool)
            )

            has_branch_connection = bool(
                np.any(
                    branch_overlap
                )
            )

            # ----------------------------------------------
            # Endpoint evidence
            # ----------------------------------------------

            structure_endpoint_count = int(
                thin_structure[
                    "endpoint_count"
                ]
            )

            has_endpoint = (
                structure_endpoint_count > 0
            )

            # ----------------------------------------------
            # Thinness evidence
            # ----------------------------------------------

            thinness_score = float(
                thin_structure[
                    "thinness_score"
                ]
            )

            # ----------------------------------------------
            # Connectivity evidence
            #
            # A thin structure that overlaps a detected
            # protrusion or reaches a branch point has
            # stronger structural evidence.
            # ----------------------------------------------

            connectivity_score = 0.0

            if has_protrusion_overlap:
                connectivity_score += 0.5

            if has_branch_connection:
                connectivity_score += 0.5

            connectivity_score = min(
                1.0,
                connectivity_score,
            )

            # ----------------------------------------------
            # Protrusion evidence
            # ----------------------------------------------

            protrusion_score = 0.0

            if has_protrusion_overlap:

                overlapping_labels = (
                    protrusion_labels[
                        structure_mask > 0
                    ]
                )

                overlapping_labels = (
                    overlapping_labels[
                        overlapping_labels > 0
                    ]
                )

                if len(
                    overlapping_labels
                ) > 0:

                    best_label = int(
                        np.bincount(
                            overlapping_labels
                        ).argmax()
                    )

                    matched_protrusion = (
                        protrusions[
                            best_label - 1
                        ]
                    )

                    protrusion_score = float(
                        matched_protrusion[
                            "confidence"
                        ]
                    )

            # ----------------------------------------------
            # Endpoint evidence
            # ----------------------------------------------

            endpoint_score = (
                1.0
                if has_endpoint
                else 0.0
            )

            # ----------------------------------------------
            # Branch evidence
            # ----------------------------------------------

            branch_score = (
                1.0
                if has_branch_connection
                else 0.0
            )

            # ----------------------------------------------
            # Attachment confidence
            # ----------------------------------------------

            attachment_score = (
                normalized_weights["thin"]
                * thinness_score

                +

                normalized_weights[
                    "protrusion"
                ]
                * protrusion_score

                +

                normalized_weights[
                    "endpoint"
                ]
                * endpoint_score

                +

                normalized_weights[
                    "branch"
                ]
                * branch_score

                +

                normalized_weights[
                    "connectivity"
                ]
                * connectivity_score
            )

            attachment_score = float(
                np.clip(
                    attachment_score,
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Structural classification
            #
            # This is NOT semantic classification.
            # It only describes structural evidence.
            # ----------------------------------------------

            evidence = []

            if thinness_score > 0:
                evidence.append(
                    "thin_structure"
                )

            if has_protrusion_overlap:
                evidence.append(
                    "protrusion"
                )

            if has_endpoint:
                evidence.append(
                    "endpoint"
                )

            if has_branch_connection:
                evidence.append(
                    "branch_connection"
                )

            if connectivity_score > 0:
                evidence.append(
                    "connected"
                )

            # ----------------------------------------------
            # Store candidate
            # ----------------------------------------------

            attached_structures.append(
                {
                    "source": "thin_structure",

                    "pixels": pixels,

                    "pixel_count": int(
                        thin_structure[
                            "pixel_count"
                        ]
                    ),

                    "length": float(
                        thin_structure[
                            "length"
                        ]
                    ),

                    "mean_thickness": float(
                        thin_structure[
                            "mean_thickness"
                        ]
                    ),

                    "max_thickness": float(
                        thin_structure[
                            "max_thickness"
                        ]
                    ),

                    "thinness_score": float(
                        thinness_score
                    ),

                    "protrusion_score": float(
                        protrusion_score
                    ),

                    "endpoint_score": float(
                        endpoint_score
                    ),

                    "branch_score": float(
                        branch_score
                    ),

                    "connectivity_score": float(
                        connectivity_score
                    ),

                    "attachment_score": (
                        attachment_score
                    ),

                    "evidence": evidence,

                    "bbox": thin_structure[
                        "bbox"
                    ],
                }
            )

        # --------------------------------------------------
        # Add protrusions that were not represented by
        # thin structures.
        #
        # This prevents a legitimate protrusion from being
        # lost simply because it does not satisfy the thin
        # structure criteria.
        # --------------------------------------------------

        existing_pixel_sets = []

        for structure in attached_structures:

            existing_pixel_sets.append(
                {
                    tuple(pixel)
                    for pixel in structure[
                        "pixels"
                    ]
                }
            )

        for protrusion in protrusions:

            protrusion_pixels = {
                tuple(pixel)
                for pixel in protrusion[
                    "path"
                ]
            }

            already_represented = any(
                protrusion_pixels
                &
                existing_pixels
                for existing_pixels
                in existing_pixel_sets
            )

            if already_represented:
                continue

            protrusion_attachment_score = float(
                np.clip(
                    protrusion[
                        "confidence"
                    ],
                    0.0,
                    1.0,
                )
            )

            attached_structures.append(
                {
                    "source": "protrusion",

                    "pixels": np.asarray(
                        protrusion[
                            "path"
                        ],
                        dtype=np.int32,
                    ),

                    "pixel_count": len(
                        protrusion[
                            "path"
                        ]
                    ),

                    "length": float(
                        protrusion[
                            "path_length"
                        ]
                    ),

                    "mean_thickness": float(
                        protrusion[
                            "mean_thickness"
                        ]
                    ),

                    "max_thickness": float(
                        protrusion[
                            "max_thickness"
                        ]
                    ),

                    "thinness_score": float(
                        protrusion[
                            "thinness_score"
                        ]
                    ),

                    "protrusion_score": (
                        protrusion_attachment_score
                    ),

                    "endpoint_score": 1.0,

                    "branch_score": (
                        1.0
                        if protrusion[
                            "reached_branch"
                        ]
                        else 0.0
                    ),

                    "connectivity_score": (
                        1.0
                        if (
                            protrusion[
                                "reached_core"
                            ]
                            or
                            protrusion[
                                "reached_branch"
                            ]
                        )
                        else 0.0
                    ),

                    "attachment_score": (
                        protrusion_attachment_score
                    ),

                    "evidence": [
                        "protrusion",
                        "endpoint",
                        "connected",
                    ],

                    "bbox": protrusion[
                        "bbox"
                    ],
                }
            )

        # --------------------------------------------------
        # Sort candidates by attachment confidence
        # --------------------------------------------------

        attached_structures.sort(
            key=lambda structure:
                structure[
                    "attachment_score"
                ],
            reverse=True,
        )

        # --------------------------------------------------
        # Overall structural statistics
        # --------------------------------------------------

        attachment_scores = [
            structure[
                "attachment_score"
            ]
            for structure
            in attached_structures
        ]

        if attachment_scores:

            max_attachment_score = float(
                max(
                    attachment_scores
                )
            )

            mean_attachment_score = float(
                np.mean(
                    attachment_scores
                )
            )

        else:

            max_attachment_score = 0.0
            mean_attachment_score = 0.0

        attachment_features = {
            "branch_count": int(
                branch_analysis[
                    "branch_count"
                ]
            ),

            "endpoint_count": (
                endpoint_count
            ),

            "protrusion_count": int(
                protrusion_analysis[
                    "protrusion_count"
                ]
            ),

            "thin_structure_count": int(
                thin_analysis[
                    "thin_structure_count"
                ]
            ),

            "attached_structure_count": (
                len(
                    attached_structures
                )
            ),

            "max_attachment_score": (
                max_attachment_score
            ),

            "mean_attachment_score": (
                mean_attachment_score
            ),
        }

        logger.info(
            f"Attached structure analysis "
            f"completed for region "
            f"{region['label']}. "
            f"Detected "
            f"{len(attached_structures)} "
            f"candidate structure(s)."
        )

        return {
            "branch_points": branch_analysis,

            "end_points": endpoint_coordinates,

            "protrusions": protrusion_analysis,

            "thin_structures": thin_analysis,

            "attached_structures": (
                attached_structures
            ),

            "attached_structure_count": (
                len(
                    attached_structures
                )
            ),

            "candidate_mask": (
                candidate_mask
            ),

            "attachment_features": (
                attachment_features
            ),
        }

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to detect attached structures."
        )

        raise PostprocessingError(
            "Failed to detect attached structures."
        ) from e

def analyze_attached_structures(
    regions: list[dict],
) -> list[dict]:
    """
    Run attached-structure analysis for every foreground
    region and store the result inside the region.

    This connects:

        detect_attached_structures()
                    ↓
        classify_regions()

    No foreground pixels are modified.
    """

    try:

        logger.info(
            "Analyzing attached structures for all foreground regions..."
        )

        if not isinstance(
            regions,
            list,
        ):
            raise PostprocessingError(
                "Regions must be provided as a list."
            )

        analyzed_regions = []

        for region in regions:

            if not isinstance(
                region,
                dict,
            ):
                raise PostprocessingError(
                    "Each region must be a dictionary."
                )

            # -----------------------------------------
            # Run attached-structure analysis
            # -----------------------------------------

            attached_analysis = (
                detect_attached_structures(
                    region
                )
            )

            # -----------------------------------------
            # Copy region
            # -----------------------------------------

            enriched_region = region.copy()

            # -----------------------------------------
            # Store analysis result
            # -----------------------------------------

            enriched_region[
                "attached_structure_analysis"
            ] = attached_analysis

            analyzed_regions.append(
                enriched_region
            )

        logger.info(
            "Attached-structure analysis completed "
            "for all foreground regions."
        )

        return analyzed_regions

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to analyze attached structures."
        )

        raise PostprocessingError(
            "Failed to analyze attached structures."
        ) from e



def classify_regions(
    regions: list[dict],
    tiny_area_threshold: float = 0.001,
    main_subject_threshold: float = 0.55,
    false_foreground_threshold: float = 0.65,
) -> list[dict]:
    """
    Classify foreground regions using multiple structural features.

    The classifier assigns one of three classes:

    - MAIN_SUBJECT
    - VALID_FOREGROUND
    - FALSE_FOREGROUND

    Classification is based on explainable evidence collected by
    previous foreground-refinement stages.

    The function does not modify foreground pixels.

    Args:
        regions:
            Enriched foreground regions containing region features,
            shape analysis, and attached-structure analysis.

        tiny_area_threshold:
            Relative-area threshold used as strong evidence for
            very small foreground regions.

        main_subject_threshold:
            Minimum main-subject score required for MAIN_SUBJECT.

        false_foreground_threshold:
            Minimum false-foreground score required for
            FALSE_FOREGROUND.

    Returns:
        Classified regions.

    Raises:
        PostprocessingError:
            If region classification fails.
    """

    try:

        logger.info(
            "Classifying foreground regions..."
        )

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if not isinstance(
            regions,
            list,
        ):
            raise PostprocessingError(
                "Regions must be provided as a list."
            )

        if not regions:

            logger.info(
                "No foreground regions found."
            )

            return []

        # --------------------------------------------------
        # Validate thresholds
        # --------------------------------------------------

        if not (
            0.0
            <= tiny_area_threshold
            <= 1.0
        ):
            raise PostprocessingError(
                "tiny_area_threshold must be between 0 and 1."
            )

        if not (
            0.0
            <= main_subject_threshold
            <= 1.0
        ):
            raise PostprocessingError(
                "main_subject_threshold must be between 0 and 1."
            )

        if not (
            0.0
            <= false_foreground_threshold
            <= 1.0
        ):
            raise PostprocessingError(
                "false_foreground_threshold must be between 0 and 1."
            )

        # --------------------------------------------------
        # Total foreground area
        # --------------------------------------------------

        total_area = sum(
            float(
                region.get(
                    "area",
                    0,
                )
            )
            for region in regions
        )

        if total_area <= 0:

            raise PostprocessingError(
                "Foreground regions contain no valid area."
            )

        # --------------------------------------------------
        # Largest region
        #
        # Largest region is evidence for main subject,
        # NOT an automatic classification.
        # --------------------------------------------------

        largest_area = max(
            float(
                region.get(
                    "area",
                    0,
                )
            )
            for region in regions
        )

        # --------------------------------------------------
        # Maximum relative area
        # --------------------------------------------------

        max_relative_area = max(
            float(
                region.get(
                    "relative_area",
                    0.0,
                )
            )
            for region in regions
        )

        if max_relative_area <= 0:
            max_relative_area = 1.0

        classified_regions = []

        # ==================================================
        # Evaluate every region
        # ==================================================

        for region in regions:

            classified = region.copy()

            # --------------------------------------------------
            # Basic region measurements
            # --------------------------------------------------

            area = float(
                region.get(
                    "area",
                    0,
                )
            )

            relative_area = float(
                region.get(
                    "relative_area",
                    0.0,
                )
            )

            # --------------------------------------------------
            # Relative area score
            #
            # Largest regions receive stronger main-subject
            # evidence.
            # --------------------------------------------------

            area_score = (
                relative_area
                / max_relative_area
            )

            area_score = float(
                np.clip(
                    area_score,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Largest-region evidence
            # --------------------------------------------------

            largest_region_score = (
                1.0
                if area == largest_area
                else area_score
            )

            # --------------------------------------------------
            # Shape analysis
            # --------------------------------------------------
            shape_analysis = region.get(
                "shape_analysis",
                {},
            )

            shape_features = shape_analysis.get(
                "features",
                {},
            )

            # --------------------------------------------------
            # Convex hull evidence
            # --------------------------------------------------
            hull_area_ratio = float(
                shape_features.get(
                    "hull_area_ratio",
                    0.0,
                )
            )

            hull_area_ratio = float(
                np.clip(
                    hull_area_ratio,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Contour evidence
            # --------------------------------------------------

            contour_count = int(
                shape_features.get(
                    "contour_count",
                    0,
                )
            )

            largest_contour_area = float(
                shape_features.get(
                    "largest_contour_area",
                    0.0,
                )
            )

            logger.debug(
                f"Region {region.get('label', 'unknown')} "
                f"shape features: "
                f"contour_count={contour_count}, "
                f"largest_contour_area={largest_contour_area:.2f}"
            )

            # --------------------------------------------------
            # Compactness / contour-area evidence
            # --------------------------------------------------

            contour_area_ratio = 0.0

            if area > 0:

                contour_area_ratio = (
                    largest_contour_area
                    / area
                )

                contour_area_ratio = float(
                    np.clip(
                        contour_area_ratio,
                        0.0,
                        1.0,
                    )
                )

            # --------------------------------------------------
            # Attached-structure analysis
            # --------------------------------------------------

            attached_analysis = region.get(
                "attached_structure_analysis",
                region.get(
                    "attached_structures",
                    {},
                ),
            )

            if not isinstance(
                attached_analysis,
                dict,
            ):
                attached_analysis = {}

            attached_count = int(
                attached_analysis.get(
                    "attached_structure_count",
                    0,
                )
            )

            max_attachment_score = float(
                attached_analysis.get(
                    "attachment_features",
                    {},
                ).get(
                    "max_attachment_score",
                    0.0,
                )
            )

            max_attachment_score = float(
                np.clip(
                    max_attachment_score,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Structural complexity
            # --------------------------------------------------

            branch_count = int(
                attached_analysis.get(
                    "attachment_features",
                    {},
                ).get(
                    "branch_count",
                    0,
                )
            )

            endpoint_count = int(
                attached_analysis.get(
                    "attachment_features",
                    {},
                ).get(
                    "endpoint_count",
                    0,
                )
            )

            protrusion_count = int(
                attached_analysis.get(
                    "attachment_features",
                    {},
                ).get(
                    "protrusion_count",
                    0,
                )
            )

            thin_structure_count = int(
                attached_analysis.get(
                    "attachment_features",
                    {},
                ).get(
                    "thin_structure_count",
                    0,
                )
            )

            # --------------------------------------------------
            # Small-region evidence
            # --------------------------------------------------

            tiny_region_score = float(
                np.clip(
                    1.0
                    -
                    (
                        relative_area
                        /
                        max(
                            tiny_area_threshold,
                            1e-12,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Main-subject evidence
            #
            # Multiple independent features are combined.
            # --------------------------------------------------

            main_subject_score = (
                0.45
                * area_score

                +

                0.25
                * largest_region_score

                +

                0.15
                * contour_area_ratio

                +

                0.15
                * hull_area_ratio
            )

            # Attached structures are positive evidence that
            # a region may contain legitimate connected detail.
            if attached_count > 0:

                main_subject_score += (
                    0.10
                    * max_attachment_score
                )

            main_subject_score = float(
                np.clip(
                    main_subject_score,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # False-foreground evidence
            #
            # Small isolated regions receive stronger evidence.
            #
            # Structural complexity reduces confidence that a
            # region is simply an artifact.
            # --------------------------------------------------

            false_foreground_score = (
                0.65
                * tiny_region_score
            )

            # Very small regions are stronger candidates.
            if relative_area < (
                tiny_area_threshold
                * 0.25
            ):

                false_foreground_score += 0.20

            # Regions containing meaningful structure should
            # receive less false-foreground confidence.
            if (
                branch_count > 0
                or protrusion_count > 0
                or thin_structure_count > 0
                or attached_count > 0
            ):

                false_foreground_score -= 0.20

            false_foreground_score = float(
                np.clip(
                    false_foreground_score,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Valid foreground evidence
            #
            # Valid foreground is the middle state:
            # not sufficiently strong to be the main subject,
            # but not sufficiently suspicious to be false.
            # --------------------------------------------------

            valid_foreground_score = float(
                np.clip(
                    1.0
                    -
                    max(
                        main_subject_score,
                        false_foreground_score,
                    ),
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Determine class
            #
            # Main subject takes priority only when the evidence
            # reaches the configured threshold.
            # --------------------------------------------------

            if (
                main_subject_score
                >= main_subject_threshold
                and
                main_subject_score
                >= false_foreground_score
            ):

                classification = (
                    "MAIN_SUBJECT"
                )

            elif (
                false_foreground_score
                >= false_foreground_threshold
                and
                false_foreground_score
                >
                main_subject_score
            ):

                classification = (
                    "FALSE_FOREGROUND"
                )

            else:

                classification = (
                    "VALID_FOREGROUND"
                )

            # --------------------------------------------------
            # Classification confidence
            # --------------------------------------------------

            if classification == "MAIN_SUBJECT":

                confidence = (
                    main_subject_score
                )

            elif classification == "FALSE_FOREGROUND":

                confidence = (
                    false_foreground_score
                )

            else:

                confidence = (
                    valid_foreground_score
                )

            confidence = float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Explainable evidence
            # --------------------------------------------------

            evidence = []

            if area_score >= 0.75:
                evidence.append(
                    "large_region"
                )

            if largest_region_score >= 0.90:
                evidence.append(
                    "largest_region"
                )

            if relative_area < (
                tiny_area_threshold
            ):
                evidence.append(
                    "small_region"
                )

            if attached_count > 0:
                evidence.append(
                    "attached_structures"
                )

            if protrusion_count > 0:
                evidence.append(
                    "protrusions"
                )

            if thin_structure_count > 0:
                evidence.append(
                    "thin_structures"
                )

            if branch_count > 0:
                evidence.append(
                    "branch_structure"
                )

            if not evidence:

                evidence.append(
                    "neutral_region"
                )

            # --------------------------------------------------
            # Store classification metadata
            # --------------------------------------------------

            classified[
                "class"
            ] = classification

            classified[
                "classification_confidence"
            ] = confidence

            classified[
                "classification_scores"
            ] = {
                "main_subject": (
                    main_subject_score
                ),

                "valid_foreground": (
                    valid_foreground_score
                ),

                "false_foreground": (
                    false_foreground_score
                ),
            }

            classified[
                "classification_evidence"
            ] = evidence

            classified_regions.append(
                classified
            )

        # ==================================================
        # Ensure exactly one MAIN_SUBJECT when foreground
        # exists.
        #
        # This prevents several regions from independently
        # becoming MAIN_SUBJECT because of similar scores.
        # ==================================================

        main_subject_candidates = [
            (
                index,
                region[
                    "classification_scores"
                ][
                    "main_subject"
                ],
            )
            for index, region
            in enumerate(
                classified_regions
            )
        ]

        if main_subject_candidates:

            best_index, best_score = max(
                main_subject_candidates,
                key=lambda item: item[1],
            )

            # If no region reached the threshold, promote
            # the strongest region only when it has meaningful
            # main-subject evidence.
            if best_score >= 0.40:

                for index, region in enumerate(
                    classified_regions
                ):

                    if index == best_index:

                        region[
                            "class"
                        ] = "MAIN_SUBJECT"

                        region[
                            "classification_confidence"
                        ] = float(
                            region[
                                "classification_scores"
                            ][
                                "main_subject"
                            ]
                        )

                    elif (
                        region["class"]
                        == "MAIN_SUBJECT"
                    ):

                        region[
                            "class"
                        ] = "VALID_FOREGROUND"

                        region[
                            "classification_confidence"
                        ] = float(
                            region[
                                "classification_scores"
                            ][
                                "valid_foreground"
                            ]
                        )

        # --------------------------------------------------
        # Final statistics
        # --------------------------------------------------

        main_count = sum(
            region["class"]
            == "MAIN_SUBJECT"
            for region
            in classified_regions
        )

        valid_count = sum(
            region["class"]
            == "VALID_FOREGROUND"
            for region
            in classified_regions
        )

        false_count = sum(
            region["class"]
            == "FALSE_FOREGROUND"
            for region
            in classified_regions
        )

        logger.info(
            "Region classification completed successfully. "
            f"Main={main_count}, "
            f"Valid={valid_count}, "
            f"False={false_count}."
        )

        return classified_regions

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to classify regions."
        )

        raise PostprocessingError(
            "Failed to classify regions."
        ) from e



def remove_false_foreground(
    alpha: np.ndarray,
    classified_regions: list[dict],
) -> np.ndarray:
    """
    Remove foreground regions classified as FALSE_FOREGROUND.

    This stage performs editing only after the classification stage
    has decided which regions are false foreground.

    The function:
        1. Validates the alpha matte.
        2. Builds a removal mask from regions classified as
           FALSE_FOREGROUND.
        3. Uses connected-component validation to ensure that only
           identified regions are removed.
        4. Removes those pixels from the alpha matte.
        5. Preserves MAIN_SUBJECT and VALID_FOREGROUND regions.
        6. Preserves the original alpha values of all retained
           foreground pixels.

    Args:
        alpha:
            Input alpha matte as a 2D NumPy array.

        classified_regions:
            Regions returned by classify_regions().
            Each region must contain:
                - class
                - mask

    Returns:
        np.ndarray:
            Refined alpha matte with false foreground removed.

    Raises:
        PostprocessingError:
            If false-foreground removal fails.
    """

    try:

        logger.info(
            "Removing false foreground regions..."
        )

        # ----------------------------------
        # Validate alpha
        # ----------------------------------

        if not isinstance(alpha, np.ndarray):
            raise PostprocessingError(
                "Alpha matte must be a NumPy array."
            )

        if alpha.ndim != 2:
            raise PostprocessingError(
                "Alpha matte must be a 2D array."
            )

        if alpha.size == 0:
            raise PostprocessingError(
                "Alpha matte cannot be empty."
            )

        # ----------------------------------
        # Validate regions
        # ----------------------------------

        if classified_regions is None:
            raise PostprocessingError(
                "Classified regions cannot be None."
            )

        if not isinstance(
            classified_regions,
            list,
        ):
            raise PostprocessingError(
                "Classified regions must be a list."
            )

        # ----------------------------------
        # Copy alpha
        # ----------------------------------
        #
        # Never modify the caller's original
        # alpha matte.
        # ----------------------------------

        refined_alpha = alpha.copy()

        # ----------------------------------
        # Create removal mask
        # ----------------------------------

        removal_mask = np.zeros(
            alpha.shape,
            dtype=np.uint8,
        )

        false_region_count = 0

        # ----------------------------------
        # Collect FALSE_FOREGROUND regions
        # ----------------------------------

        for region in classified_regions:

            if not isinstance(region, dict):
                raise PostprocessingError(
                    "Each region must be a dictionary."
                )

            region_class = region.get(
                "class"
            )

            # Only FALSE_FOREGROUND regions
            # are eligible for removal.
            if region_class != "FALSE_FOREGROUND":
                continue

            region_mask = region.get(
                "mask"
            )

            if region_mask is None:
                raise PostprocessingError(
                    "FALSE_FOREGROUND region "
                    "is missing its mask."
                )

            if not isinstance(
                region_mask,
                np.ndarray,
            ):
                raise PostprocessingError(
                    "Region mask must be a NumPy array."
                )

            if region_mask.shape != alpha.shape:
                raise PostprocessingError(
                    "Region mask shape does not "
                    "match alpha matte shape."
                )

            false_region_count += 1

            # ----------------------------------
            # Add region to removal mask
            # ----------------------------------

            removal_mask[
                region_mask > 0
            ] = 255

        # ----------------------------------
        # No false foreground
        # ----------------------------------

        if false_region_count == 0:

            logger.info(
                "No false foreground regions "
                "found. Alpha matte unchanged."
            )

            return refined_alpha

        # ----------------------------------
        # Validate removal mask
        # ----------------------------------

        if not np.any(removal_mask):

            logger.info(
                "False foreground regions contain "
                "no removable pixels."
            )

            return refined_alpha

        # ----------------------------------
        # Connected-component validation
        # ----------------------------------
        #
        # Ensure the removal mask represents
        # independent foreground components.
        # ----------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                removal_mask,
                connectivity=8,
            )
        )

        validated_removal_mask = np.zeros(
            alpha.shape,
            dtype=np.uint8,
        )

        validated_components = 0

        for label in range(
            1,
            num_labels,
        ):

            component = (
                labels == label
            )

            if not np.any(component):
                continue

            validated_removal_mask[
                component
            ] = 255

            validated_components += 1

        # ----------------------------------
        # Remove false foreground
        # ----------------------------------
        #
        # Set only the identified false
        # foreground pixels to zero.
        # ----------------------------------

        refined_alpha[
            validated_removal_mask > 0
        ] = 0

        # ----------------------------------
        # Preserve alpha range
        # ----------------------------------

        if np.issubdtype(
            refined_alpha.dtype,
            np.integer,
        ):

            refined_alpha = np.clip(
                refined_alpha,
                0,
                255,
            )

        else:

            refined_alpha = np.clip(
                refined_alpha,
                0.0,
                1.0,
            )

        # ----------------------------------
        # Statistics
        # ----------------------------------

        removed_pixels = int(
            np.count_nonzero(
                validated_removal_mask
            )
        )

        logger.info(
            "False foreground removal completed "
            f"successfully. "
            f"Regions={false_region_count}, "
            f"Components={validated_components}, "
            f"Pixels Removed={removed_pixels}."
        )

        return refined_alpha

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to remove false foreground."
        )

        raise PostprocessingError(
            "Failed to remove false foreground."
        ) from e



# def repair_boundaries(
#     alpha: np.ndarray,
#     repair_radius: int = 2,
# ) -> np.ndarray:
#     """
#     Repair small foreground boundary gaps and discontinuities.

#     The repair is performed conservatively using:

#         1. OpenCV morphological closing
#         2. scikit-image morphological reconstruction
#            by dilation
#         3. Local neighborhood restriction

#     Morphological reconstruction is constrained by the
#     closed candidate mask, preventing disconnected regions
#     from being introduced.

#     Args:
#         alpha:
#             Foreground alpha matte as a 2D NumPy array.

#             Supported formats:
#                 - uint8 [0, 255]
#                 - floating point [0, 1]

#         repair_radius:
#             Radius controlling the maximum scale of local
#             boundary defects that may be repaired.

#             The resulting elliptical kernel size is:

#                 (2 * repair_radius + 1)

#             Must be >= 1.

#     Returns:
#         np.ndarray:
#             Repaired alpha matte with the same shape and
#             dtype as the input.

#     Raises:
#         PostprocessingError:
#             If boundary repair fails.
#     """

#     try:

#         logger.info(
#             "Repairing foreground boundaries..."
#         )

#         # ----------------------------------
#         # Validate alpha
#         # ----------------------------------

#         if not isinstance(
#             alpha,
#             np.ndarray,
#         ):
#             raise PostprocessingError(
#                 "Alpha matte must be a NumPy array."
#             )

#         if alpha.ndim != 2:
#             raise PostprocessingError(
#                 "Alpha matte must be a 2D array."
#             )

#         if alpha.size == 0:
#             raise PostprocessingError(
#                 "Alpha matte cannot be empty."
#             )

#         # ----------------------------------
#         # Validate repair radius
#         # ----------------------------------

#         if not isinstance(
#             repair_radius,
#             (int, np.integer),
#         ):
#             raise PostprocessingError(
#                 "Repair radius must be an integer."
#             )

#         if repair_radius < 1:
#             raise PostprocessingError(
#                 "Repair radius must be >= 1."
#             )

#         # ----------------------------------
#         # Preserve original dtype
#         # ----------------------------------

#         original_dtype = alpha.dtype

#         # ----------------------------------
#         # Convert alpha to binary foreground
#         # ----------------------------------

#         if np.issubdtype(
#             alpha.dtype,
#             np.floating,
#         ):

#             foreground = (
#                 alpha > 0.01
#             ).astype(
#                 np.uint8
#             )

#         else:

#             foreground = (
#                 alpha > 0
#             ).astype(
#                 np.uint8
#             )

#         # ----------------------------------
#         # Nothing to repair
#         # ----------------------------------

#         if not np.any(foreground):

#             logger.info(
#                 "No foreground pixels found. "
#                 "Boundary repair skipped."
#             )

#             return alpha.copy()

#         # ----------------------------------
#         # Build conservative kernel
#         # ----------------------------------

#         kernel_size = (
#             2 * repair_radius + 1
#         )

#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE,
#             (
#                 kernel_size,
#                 kernel_size,
#             ),
#         )

#         # ----------------------------------
#         # Morphological closing
#         # ----------------------------------
#         #
#         # Closing repairs local gaps and
#         # discontinuities:
#         #
#         # dilation → erosion
#         # ----------------------------------

#         closed = cv2.morphologyEx(
#             foreground,
#             cv2.MORPH_CLOSE,
#             kernel,
#             iterations=1,
#         )

#         # ----------------------------------
#         # Morphological reconstruction
#         # ----------------------------------
#         #
#         # IMPORTANT:
#         #
#         # seed = original foreground
#         # mask = closed candidate
#         #
#         # Therefore:
#         #
#         #       seed <= mask
#         #
#         # which is required for dilation
#         # reconstruction.
#         #
#         # Reconstruction prevents completely
#         # disconnected candidate regions from
#         # becoming foreground.
#         # ----------------------------------

#         reconstructed = reconstruction(
#             foreground.astype(
#                 np.uint8
#             ),
#             closed.astype(
#                 np.uint8
#             ),
#             method="dilation",
#             footprint=np.ones(
#                 (3, 3),
#                 dtype=np.uint8,
#             ),
#         )

#         reconstructed = (
#             reconstructed > 0
#         ).astype(
#             np.uint8
#         )

#         # ----------------------------------
#         # Identify newly repaired pixels
#         # ----------------------------------

#         repair_pixels = (
#             (reconstructed > 0)
#             &
#             (foreground == 0)
#         )

#         # ----------------------------------
#         # Restrict repairs to immediate
#         # neighborhood of original foreground
#         # ----------------------------------

#         local_neighborhood = cv2.dilate(
#             foreground,
#             kernel,
#             iterations=1,
#         )

#         repair_pixels &= (
#             local_neighborhood > 0
#         )

#         # ----------------------------------
#         # Apply repair
#         # ----------------------------------

#         repaired = foreground.copy()

#         repaired[
#             repair_pixels
#         ] = 1

#         # ----------------------------------
#         # Restore original alpha format
#         # ----------------------------------

#         if np.issubdtype(
#             original_dtype,
#             np.floating,
#         ):

#             repaired_alpha = (
#                 repaired.astype(
#                     original_dtype
#                 )
#             )

#         else:

#             repaired_alpha = (
#                 repaired * 255
#             ).astype(
#                 original_dtype
#             )

#         # ----------------------------------
#         # Statistics
#         # ----------------------------------

#         original_pixels = int(
#             np.count_nonzero(
#                 foreground
#             )
#         )

#         repaired_pixels = int(
#             np.count_nonzero(
#                 repaired
#             )
#         )

#         added_pixels = int(
#             np.count_nonzero(
#                 repair_pixels
#             )
#         )

#         logger.info(
#             "Foreground boundary repair "
#             "completed successfully. "
#             f"Original={original_pixels}, "
#             f"Repaired={repaired_pixels}, "
#             f"Added={added_pixels}, "
#             f"Radius={repair_radius}."
#         )

#         return repaired_alpha

#     except PostprocessingError:
#         raise

#     except Exception as e:

#         logger.exception(
#             "Failed to repair foreground boundaries."
#         )

#         raise PostprocessingError(
#             "Failed to repair foreground boundaries."
#         ) from e
   
# # till


def repair_boundaries(
    alpha: np.ndarray,
    repair_radius: int = 2,
) -> np.ndarray:
    """
    Conservatively repair small foreground boundary gaps while
    preserving the original soft alpha matte.

    Existing alpha values are never modified.

    Only newly detected repair pixels are filled using local
    alpha information from the original matte.

    Args:
        alpha:
            Foreground alpha matte as a 2D NumPy array.

            Supported formats:
                - uint8 [0, 255]
                - floating point [0, 1]

        repair_radius:
            Radius controlling the local boundary-repair scale.

    Returns:
        np.ndarray:
            Repaired alpha matte with the same shape and dtype
            as the input.
    """

    try:

        logger.info(
            "Repairing foreground boundaries..."
        )

        # ==================================================
        # 1. Validate alpha
        # ==================================================

        if not isinstance(
            alpha,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Alpha matte must be a NumPy array."
            )

        if alpha.ndim != 2:
            raise PostprocessingError(
                "Alpha matte must be a 2D array."
            )

        if alpha.size == 0:
            raise PostprocessingError(
                "Alpha matte cannot be empty."
            )

        # ==================================================
        # 2. Validate repair radius
        # ==================================================

        if not isinstance(
            repair_radius,
            (int, np.integer),
        ):
            raise PostprocessingError(
                "Repair radius must be an integer."
            )

        if repair_radius < 1:
            raise PostprocessingError(
                "Repair radius must be >= 1."
            )

        # ==================================================
        # 3. Preserve original dtype
        # ==================================================

        original_dtype = alpha.dtype

        # ==================================================
        # 4. Convert alpha to canonical float representation
        #
        # Internally everything is processed as [0, 1].
        # ==================================================

        if np.issubdtype(
            alpha.dtype,
            np.floating,
        ):

            alpha_float = np.asarray(
                alpha,
                dtype=np.float32,
            )

        elif np.issubdtype(
            alpha.dtype,
            np.integer,
        ):

            alpha_float = (
                np.asarray(
                    alpha,
                    dtype=np.float32,
                )
                / 255.0
            )

        else:

            raise PostprocessingError(
                "Alpha matte must use a floating-point "
                "or integer dtype."
            )

        alpha_float = np.clip(
            alpha_float,
            0.0,
            1.0,
        )

        # ==================================================
        # 5. Create binary foreground mask
        #
        # IMPORTANT:
        # This mask is ONLY used to detect where repair
        # may happen.
        #
        # It must NEVER replace the original alpha.
        # ==================================================

        foreground = (
            alpha_float > 0.01
        ).astype(
            np.uint8
        )

        # ==================================================
        # 6. Nothing to repair
        # ==================================================

        if not np.any(foreground):

            logger.info(
                "No foreground pixels found. "
                "Boundary repair skipped."
            )

            return alpha.copy()

        # ==================================================
        # 7. Build conservative morphology kernel
        # ==================================================

        kernel_size = (
            2 * repair_radius + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        # ==================================================
        # 8. Morphological closing
        # ==================================================

        closed = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        # ==================================================
        # 9. Morphological reconstruction
        #
        # Reconstruction is performed on the BINARY mask.
        #
        # This determines WHERE repair is allowed.
        #
        # It does not determine the alpha value.
        # ==================================================

        reconstructed = reconstruction(
            foreground.astype(
                np.uint8
            ),
            closed.astype(
                np.uint8
            ),
            method="dilation",
            footprint=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        reconstructed = (
            reconstructed > 0
        ).astype(
            np.uint8
        )

        # ==================================================
        # 10. Identify ONLY newly created pixels
        # ==================================================

        repair_pixels = (
            (reconstructed > 0)
            &
            (foreground == 0)
        )

        # ==================================================
        # 11. Restrict repairs to local neighborhood
        # ==================================================

        local_neighborhood = cv2.dilate(
            foreground,
            kernel,
            iterations=1,
        )

        repair_pixels &= (
            local_neighborhood > 0
        )

        # ==================================================
        # 12. Nothing new was detected
        # ==================================================

        if not np.any(repair_pixels):

            logger.info(
                "No boundary repair pixels detected. "
                "Original alpha preserved."
            )

            return alpha.copy()

        # ==================================================
        # 13. Estimate alpha for NEW repair pixels
        #
        # We use the original soft alpha.
        #
        # Existing alpha values remain untouched.
        # ==================================================

        # Smooth the original alpha only for estimating
        # newly created pixels.
        #
        # This does NOT replace the original alpha matte.
        smoothing_kernel_size = max(
            3,
            kernel_size,
        )

        if smoothing_kernel_size % 2 == 0:
            smoothing_kernel_size += 1

        local_alpha = cv2.GaussianBlur(
            alpha_float,
            (
                smoothing_kernel_size,
                smoothing_kernel_size,
            ),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_REPLICATE,
        )

        # ==================================================
        # 14. Build repaired alpha
        #
        # Start from the ORIGINAL alpha.
        # ==================================================

        repaired_float = alpha_float.copy()

        # Only new repair pixels receive estimated alpha.
        repaired_float[
            repair_pixels
        ] = local_alpha[
            repair_pixels
        ]

        # ==================================================
        # 15. Safety: repaired pixels must actually be
        # foreground-like
        #
        # Prevent tiny numerical values from creating
        # unintended foreground.
        # ==================================================

        repaired_float[
            repair_pixels
            &
            (repaired_float < 0.01)
        ] = 0.0

        # ==================================================
        # 16. Clip final alpha
        # ==================================================

        repaired_float = np.clip(
            repaired_float,
            0.0,
            1.0,
        )

        # ==================================================
        # 17. Restore original dtype
        # ==================================================

        if np.issubdtype(
            original_dtype,
            np.floating,
        ):

            repaired_alpha = (
                repaired_float.astype(
                    original_dtype
                )
            )

        elif np.issubdtype(
            original_dtype,
            np.integer,
        ):

            repaired_alpha = (
                repaired_float * 255.0
            ).round().astype(
                original_dtype
            )

        else:

            raise PostprocessingError(
                "Unsupported alpha dtype."
            )

        # ==================================================
        # 18. Statistics
        # ==================================================

        original_foreground_pixels = int(
            np.count_nonzero(
                foreground
            )
        )

        repaired_foreground = (
            repaired_float > 0.01
        )

        repaired_foreground_pixels = int(
            np.count_nonzero(
                repaired_foreground
            )
        )

        added_pixels = int(
            np.count_nonzero(
                repair_pixels
                &
                repaired_foreground
            )
        )

        # Count pixels whose alpha was preserved exactly
        preserved_pixels = int(
            np.count_nonzero(
                ~repair_pixels
            )
        )

        logger.info(
            "Foreground boundary repair completed "
            "successfully. "
            f"Original={original_foreground_pixels}, "
            f"Repaired={repaired_foreground_pixels}, "
            f"Added={added_pixels}, "
            f"Preserved={preserved_pixels}, "
            f"Radius={repair_radius}."
        )

        return repaired_alpha

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to repair foreground boundaries."
        )

        raise PostprocessingError(
            "Failed to repair foreground boundaries."
        ) from e


def refine_foreground(
    alpha: Image.Image,
) -> np.ndarray:
    """
    Run the complete foreground-refinement pipeline.

    This function orchestrates all foreground-refinement stages
    without duplicating the underlying image-processing logic.

    Pipeline:

        PIL Alpha Image
                ↓
        prepare_alpha()
                ↓
        analyze_foreground()
                ↓
        compute_region_features()
                ↓
        analyze_region_shape()
                ↓
        analyze_attached_structures()
                ↓
        classify_regions()
                ↓
        remove_false_foreground()
                ↓
        repair_boundaries()
                ↓
        Refined Alpha Matte

    The function performs:

    - Alpha normalization
    - Connected-component analysis
    - Region feature extraction
    - Skeleton analysis
    - Convex-hull analysis
    - Distance-transform analysis
    - Contour analysis
    - Attached-structure analysis
    - Foreground classification
    - False-foreground removal
    - Conservative boundary repair
    
    This function does not implement new detection algorithms.
    It only coordinates the already-tested foreground-refinement
    stages.

    Args:
        alpha:
            Alpha matte produced by the previous postprocessing
            stage as a PIL Image.

    Returns:
        np.ndarray:
            Refined alpha matte as float32 values in [0, 1].

            The returned array has the same spatial dimensions
            as the input alpha image.

    Raises:
        PostprocessingError:
            If any foreground-refinement stage fails.
    """

    try:

        logger.info(
            "Starting complete foreground refinement pipeline..."
        )

        # ==================================================
        # 1. Validate input
        # ==================================================

        if not isinstance(
            alpha,
            Image.Image,
        ):
            raise PostprocessingError(
                "Alpha matte must be a PIL Image."
            )

        if alpha.width <= 0 or alpha.height <= 0:
            raise PostprocessingError(
                "Alpha matte cannot have zero dimensions."
            )

        # Preserve original spatial dimensions
        original_height = alpha.height
        original_width = alpha.width

        # ==================================================
        # 2. Prepare alpha
        # ==================================================

        logger.info(
            "Foreground refinement stage 1/8: "
            "Preparing alpha matte..."
        )

        prepared_alpha = prepare_alpha(
            alpha
        )

        # --------------------------------------------------
        # Validate prepared alpha
        # --------------------------------------------------

        if not isinstance(
            prepared_alpha,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Prepared alpha must be a NumPy array."
            )

        if prepared_alpha.ndim != 2:
            raise PostprocessingError(
                "Prepared alpha must be a 2D array."
            )

        if prepared_alpha.shape != (
            original_height,
            original_width,
        ):
            raise PostprocessingError(
                "Prepared alpha dimensions do not "
                "match the input alpha."
            )

        prepared_alpha = np.asarray(
            prepared_alpha,
            dtype=np.float32,
        )

        prepared_alpha = np.clip(
            prepared_alpha,
            0.0,
            1.0,
        )

        # ==================================================
        # 3. Analyze foreground regions
        # ==================================================

        logger.info(
            "Foreground refinement stage 2/8: "
            "Analyzing foreground regions..."
        )

        regions = analyze_foreground(
            prepared_alpha
        )

        # --------------------------------------------------
        # Empty foreground
        # --------------------------------------------------

        if not regions:

            logger.info(
                "No foreground regions detected. "
                "Foreground refinement completed "
                "without modifications."
            )

            return prepared_alpha.copy()

        # ==================================================
        # 4. Compute region features
        # ==================================================

        logger.info(
            "Foreground refinement stage 3/8: "
            "Computing region features..."
        )

        regions = compute_region_features(
            regions,
            prepared_alpha.shape,
        )

        if not regions:

            logger.info(
                "No valid foreground regions remained "
                "after feature computation."
            )

            return prepared_alpha.copy()

        # ==================================================
        # 5. Analyze region shape
        # ==================================================

        logger.info(
            "Foreground refinement stage 4/8: "
            "Analyzing foreground region shapes..."
        )

        regions = analyze_region_shape(
            regions
        )

        if not regions:

            logger.info(
                "No foreground regions remained "
                "after shape analysis."
            )

            return prepared_alpha.copy()

        # ==================================================
        # 6. Analyze attached structures
        # ==================================================

        logger.info(
            "Foreground refinement stage 5/8: "
            "Analyzing attached structures..."
        )

        regions = analyze_attached_structures(
            regions
        )

        if not regions:

            logger.info(
                "No foreground regions remained "
                "after attached-structure analysis."
            )

            return prepared_alpha.copy()

        # ==================================================
        # 7. Classify foreground regions
        # ==================================================

        logger.info(
            "Foreground refinement stage 6/8: "
            "Classifying foreground regions..."
        )

        classified_regions = classify_regions(
            regions
        )

        if not classified_regions:

            logger.info(
                "No classified foreground regions found. "
                "Returning prepared alpha."
            )

            return prepared_alpha.copy()

        # ==================================================
        # Classification statistics
        # ==================================================

        main_subject_count = sum(
            region.get("class")
            == "MAIN_SUBJECT"
            for region
            in classified_regions
        )

        valid_foreground_count = sum(
            region.get("class")
            == "VALID_FOREGROUND"
            for region
            in classified_regions
        )

        false_foreground_count = sum(
            region.get("class")
            == "FALSE_FOREGROUND"
            for region
            in classified_regions
        )

        logger.info(
            "Foreground classification summary: "
            f"Main={main_subject_count}, "
            f"Valid={valid_foreground_count}, "
            f"False={false_foreground_count}."
        )

        # ==================================================
        # 8. Remove false foreground
        # ==================================================

        logger.info(
            "Foreground refinement stage 7/8: "
            "Removing false foreground..."
        )

        refined_alpha = remove_false_foreground(
            prepared_alpha,
            classified_regions,
        )

        # --------------------------------------------------
        # Validate removal output
        # --------------------------------------------------

        if not isinstance(
            refined_alpha,
            np.ndarray,
        ):
            raise PostprocessingError(
                "False-foreground removal must return "
                "a NumPy array."
            )

        if refined_alpha.shape != prepared_alpha.shape:
            raise PostprocessingError(
                "False-foreground removal changed "
                "the alpha dimensions."
            )

        # ==================================================
        # 9. Repair boundaries
        # ==================================================

        logger.info(
            "Foreground refinement stage 8/8: "
            "Repairing foreground boundaries..."
        )

        refined_alpha = repair_boundaries(
            refined_alpha
        )

        # ==================================================
        # Final output validation
        # ==================================================

        if not isinstance(
            refined_alpha,
            np.ndarray,
        ):
            raise PostprocessingError(
                "Final refined alpha must be a NumPy array."
            )

        if refined_alpha.ndim != 2:
            raise PostprocessingError(
                "Final refined alpha must be a 2D array."
            )

        if refined_alpha.shape != (
            original_height,
            original_width,
        ):
            raise PostprocessingError(
                "Final refined alpha dimensions do not "
                "match the input alpha."
            )

        # --------------------------------------------------
        # Convert to canonical representation
        #
        # Foreground refiner output:
        #
        #     dtype  = float32
        #     range  = [0, 1]
        # --------------------------------------------------

        refined_alpha = np.asarray(
            refined_alpha,
            dtype=np.float32,
        )

        refined_alpha = np.clip(
            refined_alpha,
            0.0,
            1.0,
        )
        # --------------------------------------------------
        # Final numerical validation
        # --------------------------------------------------

        if not np.isfinite(
            refined_alpha
        ).all():

            raise PostprocessingError(
                "Final refined alpha contains "
                "non-finite values."
            )

        # ==================================================
        # Final statistics
        # ==================================================

        original_foreground_pixels = int(
            np.count_nonzero(
                prepared_alpha > 0.01
            )
        )

        final_foreground_pixels = int(
            np.count_nonzero(
                refined_alpha > 0.01
            )
        )

        foreground_difference = (
            final_foreground_pixels
            -
            original_foreground_pixels
        )

        logger.info(
            "Foreground refinement completed successfully. "
            f"Regions={len(classified_regions)}, "
            f"Main={main_subject_count}, "
            f"Valid={valid_foreground_count}, "
            f"False={false_foreground_count}, "
            f"OriginalForeground="
            f"{original_foreground_pixels}, "
            f"FinalForeground="
            f"{final_foreground_pixels}, "
            f"Difference="
            f"{foreground_difference}."
        )

        return refined_alpha

    except PostprocessingError:
        raise

    except Exception as e:

        logger.exception(
            "Failed to complete foreground refinement pipeline."
        )

        raise PostprocessingError(
            "Failed to complete foreground refinement pipeline."
        ) from e

    
