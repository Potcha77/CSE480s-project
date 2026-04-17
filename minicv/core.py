import numpy as np

def normalize_image(img):
    return (img - img.min())/(img.max()-img.min())

def clip_pixels(img, min_val=0, max_val=1):
    return np.clip(img, min_val, max_val)
