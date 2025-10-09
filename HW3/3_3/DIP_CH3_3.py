import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np
import math
from scipy.signal import convolve2d

# n*n Gaussian filter
def Gaussian_Filter(kernel_size, sigma):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    x, y = np.meshgrid(ax, ax)
    
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    return gaussian_kernel

# read image
#image_path = "checkerboard1024-shaded.tif"
image_path = "N1.bmp"

with rasterio.open(image_path) as raw_img:
    #print(src.count)     # see how many bands are there
    image = raw_img.read(1)   # grayscale image => only one band 

# Gaussian filter parameters
kernel_size = 111
sigma = 20

# Gaussian filter image processing
gaussian_filter = Gaussian_Filter(kernel_size=kernel_size, sigma=sigma)

shade_image = convolve2d(image, gaussian_filter, mode='same', boundary='symm').astype(np.float32)    
    
eps = 1e-6 # avoid dividing by 0
processed_image = image / (shade_image + eps)               

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("shade (Gaussian LP)")
plt.imshow(shade_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.title("shade corrected")
plt.imshow(processed_image, cmap='gray'); plt.axis('off')
plt.show()

