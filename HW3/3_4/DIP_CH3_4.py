import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.signal import convolve2d

# generate gradient kernel for adaptive Sobel filter
def generate_gradient_kernel(filter_size):
    r = filter_size // 2
    kx = np.zeros((filter_size, filter_size), dtype=np.float64)
    ky = np.zeros((filter_size, filter_size), dtype=np.float64)

    for dh in range(-r, r+1):
        for dw in range(-r, r+1):
            if dh == 0 and dw == 0:
                px = py = 0.0
            else:
                denom = (dw*dw + dh*dh)
                px = dw / denom
                py = dh / denom
            kx[dh + r, dw + r] = px
            ky[dh + r, dw + r] = py
            
    return kx, ky

# 3*3 Sobel filter with enhancement option (robust_norm)
def sobel_filtering(input_image, filter_size=3, robust_norm=True):
    img = np.asarray(input_image, dtype=np.float64)
    kx, ky = generate_gradient_kernel(filter_size)
    
    edge_x = convolve2d(img, kx, mode='same', boundary='fill')
    edge_y = convolve2d(img, ky, mode='same', boundary='fill')
    
    h, w = input_image.shape
    output_image = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            output_image[i, j] = np.sqrt( edge_x[i, j]**2 + edge_y[i, j]**2 )
    
    print(max(output_image[:,:]), min(output_image[:,:]))   
    # avoid pixel value distortion by extremes        
    if robust_norm:
        p1, p99 = np.percentile(output_image, (1, 99))
        scale = max(p99 - p1, 1e-9)
        output_image = np.clip((output_image - p1) / scale, 0, 1) * 255.0
    else:
        output_image = (output_image / (output_image.max() + 1e-12)) * 255.0
        
    return output_image.astype(np.uint8)

#def Laplacian_filtering(input_image):

# read image
image_path = "Bodybone.bmp"
with rasterio.open(image_path) as raw_img:
    #print(src.count)     # see how many bands are there
    image = raw_img.read(1)   # grayscale image => only one band 
        
sobel_image = sobel_filtering(image, robust_norm=False)    
enhanced_sobel = sobel_filtering(image, robust_norm=True)

plt.figure(figsize=(15,4))
# plt.subplot(1,2,1)
# plt.imshow(image, cmap='gray')
# plt.title('Original')
# plt.axis('off')

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(sobel_image, cmap='gray')
plt.title('Sobel filter')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(enhanced_sobel, cmap='gray')
plt.title('Enhanced Sobel ')
plt.axis('off')

plt.show()
