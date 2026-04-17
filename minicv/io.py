import numpy as np
import matplotlib.pyplot as plt


def read_image(path):
    """
    Read image from file and return float32 numpy array
    """
    img = plt.imread(path)

    # If image is uint8 convert to float
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    return img


def save_image(path, image):
    """
    Save image to file
    """
    plt.imsave(path, image, cmap='gray')


def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale
    """
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:  # remove alpha channel
        image = image[:, :, :3]

    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )

    return gray.astype(np.float32)


def grayscale_to_rgb(image):
    """
    Convert grayscale image to RGB (FIXED)
    """
    if image.ndim != 2:
        raise ValueError("Input must be grayscale image")

    # stack along last axis -> (H, W, 3)
    rgb = np.dstack((image, image, image))

    return rgb.astype(np.float32)
