import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

image_path = "hidden object.jpg"   
img = Image.open(image_path).convert("L")            
image = np.array(img, dtype=np.uint8)

H, W = image.shape

hist_raw, _ = np.histogram(image, bins=256, range=(0, 256))
pdf_raw = hist_raw / (H * W)

cdf = np.cumsum(pdf_raw)
lut = np.floor(255 * cdf + 0.5).astype(np.uint8)   
image_eq = lut[image]                               

hist_eq, _ = np.histogram(image_eq, bins=256, range=(0, 256))
pdf_eq = hist_eq / (H * W)

counter = 0
for i in range(256) :
    if i <= 100:
        counter += hist_eq[i]
print("histogram after equalization in low intensity", counter)

levels = np.arange(256, dtype=np.float64)
mean_global = (levels * pdf_eq).sum() 

std_global  = np.sqrt(((levels - mean_global) ** 2 * pdf_eq).sum())

# local enhancement
f = image_eq.astype(np.float32)
win = 3
# local mean
m_local = uniform_filter(f, size=win, mode="reflect")
# local mean of squares
m2_local = uniform_filter(f * f, size=win, mode="reflect")
# local variance -> std
var_local = np.maximum(m2_local - m_local**2, 0.0)
std_local = np.sqrt(var_local)

# parameters
k0, k1 = 0.0, 0.25
k2, k3 = 0.0, 1.0
C = 3.0

# condition： k0*mean <= m_local <= k1*mean  and  k2*std_global <= std_local <= k3*std_global
mask = (
    (m_local >= k0 * mean_global) & (m_local <= k1 * mean_global) &
    (std_local >= k2 * std_global) & (std_local <= k3 * std_global)
)

out = f.copy()
out[mask] = C * out[mask]
out = np.clip(out, 0, 255).astype(np.uint8)

hist_local, _ = np.histogram(out, bins=256, range=(0, 256))
pdf_out = hist_local / (H * W)

counter = 0
for i in range(256) :
    if i <= 100:
        counter += hist_local[i]
print("histogram after local enhancement in low intensity", counter)

# Original
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Histogram")
plt.hist(image.flatten(), bins=256)
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Equalized
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Equalized image")
plt.imshow(image_eq, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Histogram")
plt.hist(image_eq.flatten(), bins=256)
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Local Enhanced
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Local Enhanced")
plt.imshow(out, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Histogram")
plt.hist(out.flatten(), bins=256)
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()