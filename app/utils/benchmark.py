from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.core import logger


@dataclass
class BenchmarkResult:
    image_name: str

    input_width: int
    input_height: int

    output_width: int
    output_height: int

    device: str

    preprocessing_time: float
    inference_time: float
    postprocessing_time: float
    saving_time: float

    total_time: float

    images_per_second: float


def create_benchmark(
    image_path: str,
    input_image: Image.Image,
    output_image: Image.Image,
    device: str,
    preprocessing_time: float,
    inference_time: float,
    postprocessing_time: float,
    saving_time: float,
) -> BenchmarkResult:

    total_time = (
        preprocessing_time
        + inference_time
        + postprocessing_time
        + saving_time
    )

    images_per_second = (
        1 / total_time
        if total_time > 0
        else 0.0
    )

    return BenchmarkResult(
        image_name=Path(image_path).name,

        input_width=input_image.width,
        input_height=input_image.height,

        output_width=output_image.width,
        output_height=output_image.height,

        device=device.upper(),

        preprocessing_time=preprocessing_time,
        inference_time=inference_time,
        postprocessing_time=postprocessing_time,
        saving_time=saving_time,

        total_time=total_time,

        images_per_second=images_per_second,
    )


def log_benchmark(result: BenchmarkResult) -> None:

    logger.info("=" * 50)
    logger.info("Benchmark Report")
    logger.info("=" * 50)

    logger.info(f"Image            : {result.image_name}")

    logger.info(
        f"Input Size       : "
        f"{result.input_width} x {result.input_height}"
    )

    logger.info(
        f"Output Size      : "
        f"{result.output_width} x {result.output_height}"
    )

    logger.info(f"Device           : {result.device}")

    logger.info("-" * 50)

    logger.info(
        f"Preprocessing    : "
        f"{result.preprocessing_time:.2f} sec"
    )

    logger.info(
        f"Inference        : "
        f"{result.inference_time:.2f} sec"
    )

    logger.info(
        f"Postprocessing   : "
        f"{result.postprocessing_time:.2f} sec"
    )

    logger.info(
        f"Saving           : "
        f"{result.saving_time:.2f} sec"
    )

    logger.info("-" * 50)

    logger.info(
        f"Total Time       : "
        f"{result.total_time:.2f} sec"
    )

    logger.info(
        f"Images / Second  : "
        f"{result.images_per_second:.3f}"
    )

    logger.info("=" * 50)