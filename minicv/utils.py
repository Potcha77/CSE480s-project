import numpy as np

def validate_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or RGB")

def normalize_color_value(color, channels):
    if channels == 1:
        return np.array(color, dtype=np.float32)
    return np.asarray(color, dtype=np.float32)
