import os
from pathlib import Path

import gradio as gr
import spaces

# ---------------------------------------------------------
# Hugging Face / application configuration
# ---------------------------------------------------------

os.environ.setdefault("MODEL_PATH", "ZhengPeng7/BiRefNet")
os.environ.setdefault("DEVICE", "auto")
os.environ.setdefault("OUTPUT_DIR", "/tmp/bg_remover_output")
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "7860")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("IMAGE_SIZE", "1024")
os.environ.setdefault("NORMALIZE_MEAN", "[0.485, 0.456, 0.406]")
os.environ.setdefault("NORMALIZE_STD", "[0.229, 0.224, 0.225]")

Path(os.environ["OUTPUT_DIR"]).mkdir(
    parents=True,
    exist_ok=True,
)

from app.services.background_removal_service import (
    BackgroundRemovalService,
)


_service = None


def get_service():
    global _service

    if _service is None:
        _service = BackgroundRemovalService()

    return _service


@spaces.GPU
def remove_background(input_image):

    if input_image is None:
        raise gr.Error("Please upload an image.")

    input_path = Path(input_image)

    if not input_path.exists():
        raise gr.Error("Uploaded image could not be found.")

    service = get_service()

    output_path = service.remove_background(
        str(input_path),
        output_filename="result.png",
    )

    return str(output_path)


with gr.Blocks(
    title="AI Background Remover",
) as demo:

    gr.Markdown(
        """
        # AI Background Remover

        Upload an image and remove its background
        using **BiRefNet AI**.
        """
    )

    input_image = gr.Image(
        type="filepath",
        label="Upload Image",
    )

    remove_button = gr.Button(
        "Remove Background",
        variant="primary",
    )

    output_image = gr.Image(
        type="filepath",
        label="Result",
    )

    remove_button.click(
        fn=remove_background,
        inputs=input_image,
        outputs=output_image,
    )


if __name__ == "__main__":
    demo.launch(
    allowed_paths=[r"E:\tmp\bg_remover_output"]
)