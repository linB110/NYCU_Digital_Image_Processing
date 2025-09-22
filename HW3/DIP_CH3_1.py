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
image_equ = np.interp(image_norm.flatten(), bins[:-1], cdf_normalized)
image_equ = image_equ.reshape(image.shape).astype(np.uint8)

# Step 5: Visualization
plt.figure(figsize=(12,6))

# left : equalized image
plt.subplot(1, 2, 1)
plt.title("Equalized Image")
plt.imshow(image_equ, cmap="gray")
plt.axis("off")

# right plot : histogram of equalized image
plt.subplot(1, 2, 2)
plt.title("Histogram after equalization")
plt.hist(image_equ.flatten(), bins=256, color="gray")
plt.xlabel("Pixel Intensity")
plt.ylabel("Occurrences")

plt.tight_layout()
plt.show()

### Question 3 : demonstrate the histogram after "specification" pz(zq) = c*zq^0.4
# step 1 : normalization

# step 2 : input histogram + CDF, where hist and bins came from image_norm
cdf_in = hist.cumsum() / hist.sum()  # normalize to [0,1], 
s_k = np.rint(cdf_in * 255).astype(np.uint8) 

# step 3: target distribution pz(z) = c * z^0.4
zq = np.linspace(0, 255, 256)
pz = (zq + 1e-6)**0.4
pz /= pz.sum()  
cdf_target = np.cumsum(pz)
qz_G = np.rint(cdf_target * 255).astype(np.uint8) 

# step 4: mapping to closet pixel value
mapping = np.searchsorted(qz_G, s_k, side="left")
mapping = np.clip(mapping, 0, 255).astype(np.uint8)

# step 5 : apply specification
image_spec = mapping[image_norm]

# step 6 : visualization
plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.title("Image after specification")
plt.imshow(image_spec, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Histogram after specification")
plt.hist(image_spec.flatten(), bins=256, color="gray")
plt.xlabel("Pixel Intensity")
plt.ylabel("Occurrences")

plt.tight_layout()
plt.show()
