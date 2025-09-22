import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np

# read image
image_path = "aerial_view.tif"
with rasterio.open(image_path) as raw_img:
    #print(src.count)     # see how many bands are there
    image = raw_img.read(1)   # grayscale image => only one band 


### Question 1 : demonstrate the raw image and corresponding histogram
plt.figure(figsize=(10, 5))

# left plot : raw image
plt.subplot(1, 2, 1)
plt.title("Aerial View Image")
plt.imshow(image, cmap="gray")

# right plot : histogram of raw image
plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(image.flatten(), bins=256, color="gray")
plt.xlabel("Pixel Intensity")
plt.ylabel("Occurrences")

plt.tight_layout()
#plt.show()

### Qusetion 2 : demonstrate the histogram after "equalization"
px_min, px_max = np.min(image), np.max(image) # min and max value in the raw_image

# step 1 : normalize to [0, 255]
image_norm = ((image - px_min) / (px_max - px_min) * 255).astype(np.uint8)

# step 2 : infomation of normalize image
hist, bins = np.histogram(image_norm.flatten(), bins=256, range=[0, 255])

# step 3: Cumulative Distribution Function (CDF)
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1] * 255

# step 4 : apply equalization
iamge_equ = np.interp(image_norm.flatten(), bins[:-1], cdf_normalized)
iamge_equ = iamge_equ.reshape(image.shape).astype(np.uint8)

# Step 5: Visualization
plt.figure(figsize=(12,6))

# left : equalized image
plt.subplot(1, 2, 1)
plt.title("Equalized Image")
plt.imshow(iamge_equ, cmap="gray")
plt.axis("off")

# right plot : histogram of equalized image
plt.subplot(1, 2, 2)
plt.title("Histogram after equalization")
plt.hist(iamge_equ.flatten(), bins=256, color="gray")
plt.xlabel("Pixel Intensity")
plt.ylabel("Occurrences")

plt.tight_layout()
plt.show()

### Question 3 : demonstrate the histogram after "specification" pz(zq) = c.zq^0.4