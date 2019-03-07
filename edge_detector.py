import numpy as np
import math

# Implementation of forward operator
def forward_operator(img):
    img_h, img_w = img.shape
    ret = np.copy(img)
    for i in range(img_h):
        for j in range(img_w):
            if (i >= img_h - 1 or j >= img_w - 1):
                ret[i][j] = 0
            else:
                dx = float(img[i + 1][j]) - float(img[i][j])
                dy = float(img[i][j + 1]) - float(img[i][j])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx**2 + dy**2)))
    return ret

# Implementation of backward operator
def backward_operator(img):
    img_h, img_w = img.shape
    ret = np.copy(img)
    for i in range(img_h):
        for j in range(img_w):
            if (i <= 0 or j <= 0):
                ret[i][j] = 0
            else:
                dx = float(img[i][j]) - float(img[i - 1][j])
                dy = float(img[i][j]) - float(img[i][j - 1])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx**2 + dy**2)))
    return ret

# Implementation of central operator
def central_operator(img):
    img_h, img_w = img.shape
    ret = np.copy(img)
    for i in range(img_h):
        for j in range(img_w):
            if (i <= 0 or j <= 0 or i >= img_h - 1 or j >= img_w - 1):
                ret[i][j] = 0
            else:
                dx = float(img[i + 1][j]) - float(img[i - 1][j])
                dy = float(img[i][j + 1]) - float(img[i][j - 1])
                ret[i][j] = np.uint8(np.round(math.sqrt(dx**2 + dy**2) / 2))
    return ret

# Generalize the process of appling the mask on the pictures
def apply_filter(img, f):
    img_h, img_w = img.shape
    ret = np.array(img, dtype='float')
    img = np.array(img, dtype='float')
    f_h, f_w = f.shape
    assert f_h % 2 == 1, 'assume filter size is odd'
    f_size = np.int((f_h - 1) / 2)

    for i in range(img_h):
        for j in range(img_w):
            if (i - f_size < 0 or j - f_size < 0 
            	or i + f_size >= img_h or j + f_size >= img_w):
                ret[i][j] = 0
                continue
            v = 0
            for di in range(-f_size, f_size + 1):
                for dj in range(-f_size, f_size + 1):
                    ci = i + di
                    cj = j + dj
                    fi = di + f_size
                    fj = dj + f_size
                    v = v + f[fi, fj] * img[ci, cj]
            ret[i][j] = v

    return ret

# Implementation of sobel operator
def sobel_operator(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float') / 8
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') / 8

    dx = apply_filter(img, sobel_x)
    dy = apply_filter(img, sobel_y)

    ret = np.uint8(np.round(np.sqrt(dx**2 + dy**2)))

    return ret

# Implementation of prewitt operator
def prewitt_operator(img):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='float') / 6
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='float') / 6

    dx = apply_filter(img, prewitt_x)
    dy = apply_filter(img, prewitt_y)

    ret = np.uint8(np.round(np.sqrt(dx**2 + dy**2)))

    return ret


# Implementation of Gaussian blur
def gaussian_filter(img):
    gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float') / 16
    ret = apply_filter(img, gaussian)
    return np.uint8(np.round(ret))


# Implementation of Canny operator
def canny_operator(img):
    smooth_ret = gaussian_filter(img)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float') / 8
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') / 8

    dx = apply_filter(smooth_ret, sobel_x)
    dy = apply_filter(smooth_ret, sobel_y)

    ret_img = np.uint8(np.round(np.sqrt(dx**2 + dy**2)))
    theta = np.uint8(np.round((np.arctan2(dx, dy) + 180) / 360 * 255))
    return (ret_img, theta)


# Implementation of Laplacian operator
def laplacian_operator(img):    
    smooth_ret = gaussian_filter(img)
    
    laplacian_f = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='float')
    ret = apply_filter(smooth_ret, laplacian_f)
    
    return np.uint8(np.absolute(ret))

# Implementation of LoG filter generator
def build_LoG_filter(shape=(5,5), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    f = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    f = f / f.sum()
    f = f * (x*x + y*y - 2*sigma**2) / sigma**4

    return f - f.mean()

# Implementation of Laplacian of Gaussian (LoG)
def LoG(img, sigma):
	LoG_filter = build_LoG_filter(shape=(5, 5), sigma=sigma)
	ret = apply_filter(img, LoG_filter)
	return np.uint8(np.round(np.absolute(ret)))

# Implementation of DoG filter generator
def build_DoG_filter(shape=(5, 5), sigma1=0.5, sigma2=1.):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    f1 = np.exp(-(x*x + y*y) / (2.*sigma1*sigma1))
    f1 = f1 / f1.sum()
    f2 = np.exp(-(x*x + y*y) / (2.*sigma2*sigma2))
    f2 = f2 / f2.sum()
    f = f1 - f2
    return f - f.mean()
    
# Implementation of Difference of Gaussian (DoG)
def DoG(img, sigma1, sigma2):
    DoG_filter = build_DoG_filter(shape=(5, 5), sigma1=sigma1, sigma2=sigma2)
    ret = apply_filter(img, DoG_filter)
    return np.uint8(np.round(np.absolute(ret)))

# Implementation of binarize operation
def binarize_operation(img, threshold):
	new_img = img > threshold
	return new_img

# Implementation of dilation operation
def dilation_operation(img):
	element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='int')
	ret = apply_filter(img, element) >= 1
	return ret
# Implementation of erosion operation
def erosion_operation(img):
	element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='int')
	ret = apply_filter(img, element) >= 9
	return ret

# Implementation of open operation
def open_operation(img):
	ret = erosion_operation(img)
	ret = dilation_operation(ret)
	return ret

# Implementation of close operation
def close_operation(img):
	ret = dilation_operation(img)
	ret = erosion_operation(ret)
	return ret



