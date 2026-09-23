import platform
import torch
from app.core import logger

def log_device_info() -> None:
    """
    Log information about the current execution environment.
    """

    logger.info("=" * 40)
    logger.info("Device Information")
    logger.info("=" * 40)

    logger.info(f"Python Version : {platform.python_version()}")
    logger.info(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():

        logger.info("Device         : GPU")
        logger.info(f"GPU Name       : {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version   : {torch.version.cuda}")

    else:

        logger.info("Device         : CPU")
        logger.info("CUDA Available : False")

    logger.info("=" * 40)