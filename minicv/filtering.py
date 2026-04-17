def mean_filter(img, k=3):
    kernel = np.ones((k,k))/(k*k)
    return convolve2d(img, kernel)
def gaussian_kernel(size, sigma=1):
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return kernel/np.sum(kernel)
def gaussian_filter(img, size=5):
    return convolve2d(img, gaussian_kernel(size))
def median_filter(img, k=3):
    h,w = img.shape
    pad = k//2
    padded = np.pad(img, pad)
    out = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k, j:j+k]
            out[i,j] = np.median(region)
    return out
def sobel_gradients(img):
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    dx = convolve2d(img,gx)
    dy = convolve2d(img,gy)
    
    mag = np.sqrt(dx**2 + dy**2)
    return mag
