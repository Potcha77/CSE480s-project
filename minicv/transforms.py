def resize(img, new_h, new_w):
    h,w = img.shape
    out = np.zeros((new_h,new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            x = int(i*h/new_h)
            y = int(j*w/new_w)
            out[i,j] = img[x,y]
    return out
def rotate(img, angle):
    angle = np.deg2rad(angle)
    h,w = img.shape
    cx,cy = h//2, w//2
    
    out = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            x = int((i-cx)*np.cos(angle) - (j-cy)*np.sin(angle) + cx)
            y = int((i-cx)*np.sin(angle) + (j-cy)*np.cos(angle) + cy)
            
            if 0<=x<h and 0<=y<w:
                out[i,j] = img[x,y]
    return out
def translate(img, tx, ty):
    h,w = img.shape
    out = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            x = i - tx
            y = j - ty
            if 0<=x<h and 0<=y<w:
                out[i,j] = img[x,y]
    return out
