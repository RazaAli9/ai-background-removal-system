import torch

from app.core import (
    logger,
    ImageProcessingError,
)


def predict(model, input_tensor):
    """
    Run inference using BiRefNet.

    Args:
        model: Loaded BiRefNet model
        input_tensor: Preprocessed image tensor

    Returns:
        Raw model prediction
    """

    logger.info("Running model inference...")

    try:

        with torch.no_grad():

            preds = model(input_tensor)

        logger.info("Inference completed successfully.")

        return preds

    except Exception as e:

        logger.exception("Inference failed.")

        raise ImageProcessingError(
            f"Model inference failed: {e}"
        )