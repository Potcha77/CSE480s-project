def create_canvas(h,w):
    return np.zeros((h,w))
def draw_point(img, x, y, color=1):
    img[x,y] = color
def draw_line(img,x0,y0,x1,y1):
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx-dy
    
    while True:
        img[x0,y0]=1
        if x0==x1 and y0==y1:
            break
        e2 = 2*err
        if e2>-dy:
            err -= dy
            x0 += sx
        if e2<dx:
            err += dx
            y0 += sy
