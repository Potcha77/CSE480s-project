import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    img = plt.imread(path)
    return img.astype(np.float32)

def save_image(path, image):
    plt.imsave(path, image, cmap='gray')

def rgb_to_grayscale(image):
    return 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

def grayscale_to_rgb(image):
    return np.stack([image,image,image], axis=-1)
