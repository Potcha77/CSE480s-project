def intensity_statistics(img):
    return {
        "mean": np.mean(img),
        "std": np.std(img),
        "min": np.min(img),
        "max": np.max(img)
    }
def histogram_descriptor(img, bins=16):
    hist,_ = np.histogram(img, bins=bins, range=(0,1))
    return hist
def gradient_statistics(img):
    grad = sobel_gradients(img)
    return {
        "mean": np.mean(grad),
        "std": np.std(grad)
    }
