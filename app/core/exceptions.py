class BackgroundRemoverError(Exception):
    """Base exception for the application."""
    pass

class ModelNotFoundError(BackgroundRemoverError):
    """Raised when model weights cannot be found."""
    pass


class InvalidImageError(BackgroundRemoverError):
    """Raised when the image cannot be opened or is corrupted."""
    pass


class UnsupportedImageFormatError(BackgroundRemoverError):
    """Raised when an unsupported image format is provided."""
    pass


class GPUUnavailableError(BackgroundRemoverError):
    """Raised when GPU is requested but unavailable."""
    pass


class ImageProcessingError(BackgroundRemoverError):
    """Raised when inference or image processing fails."""
    pass

class PreprocessingError(BackgroundRemoverError):
    """Raised when image preprocessing fails."""
    pass


class InferenceError(BackgroundRemoverError):
    """Raised when model inference fails."""
    pass


class PostprocessingError(BackgroundRemoverError):
    """Raised when image postprocessing fails."""
    pass