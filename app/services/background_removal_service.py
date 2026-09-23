from pathlib import Path
from app.core import logger
from app.inference.model_loader import load_model
from app.inference.predictor import predict
from app.preprocessing.preprocessor import preprocess_image
from app.postprocessing.postprocessor import postprocess_prediction
from app.postprocessing.image_saver import save_image
from app.utils.file_utils import validate_image_path
from app.utils.device_info import log_device_info
from app.utils.timer import Timer
from app.utils.benchmark import (
    create_benchmark,
    log_benchmark,
)


class BackgroundRemovalService:

    def __init__(self):

        logger.info(
            "Initializing Background Removal Service..."
        )

        self.model, self.device = load_model()

        log_device_info()

        logger.info(
            "Background Removal Service initialized."
        )
        

    def remove_background(
        self,
        image_path: str,
        output_filename: str = "output.png",
    ) -> Path:

        logger.info(
            f"Processing image: {image_path}"
        )

        validated_path = validate_image_path(image_path)

        with Timer("Preprocessing") as preprocessing_timer:
            image, input_tensor = preprocess_image(
                str(validated_path),
                self.device,
            )

        with Timer("Inference") as inference_timer:
            preds = predict(
                self.model,
                input_tensor,
            )

        with Timer("Postprocessing") as postprocessing_timer:
            rgba_image = postprocess_prediction(
                preds,
                image,
            )

        with Timer("Saving Image") as saving_timer:
            saved_path = save_image(
                rgba_image,
                output_filename,
            )
        
        benchmark = create_benchmark(
            image_path=image_path,
            input_image=image,
            output_image=rgba_image,
            device=self.device,
            preprocessing_time=preprocessing_timer.elapsed_time,
            inference_time=inference_timer.elapsed_time,
            postprocessing_time=postprocessing_timer.elapsed_time,
            saving_time=saving_timer.elapsed_time,
        )

        log_benchmark(benchmark)


        return saved_path