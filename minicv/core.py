import numpy as np

def normalize_image(img):
    return (img - img.min())/(img.max()-img.min())

def clip_pixels(img, min_val=0, max_val=1):
    return np.clip(img, min_val, max_val)
def pad_image(img, pad):
    return np.pad(img, pad, mode='edge')
def convolve2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh//2
    pad_w = kw//2
    
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)))
    output = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region*kernel)
    return output
