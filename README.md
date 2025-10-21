### DIP Homework 3-1: Histogram Processing

**Description:**
This assignment involves histogram-based image processing on the image `aerial_view.tif`. The tasks include visualizing histograms, performing histogram equalization, and histogram matching.

**Tasks:**

1. Plot the original image and its histogram.
2. Apply **Histogram Equalization**, plot the equalized image and its histogram.
3. Apply **Histogram Matching** using the transformation function ( p_z(z_q) = c \cdot z_q^{0.4} ) (compute the constant ( c ) beforehand).
4. Provide comments comparing the original, equalized, and matched images.

**Files:**

* `aerial_view.tif`
* Python code implementing histogram operations.

---

### DIP Homework 3-2: Hidden Image Enhancement

**Description:**
This assignment focuses on image enhancement techniques for revealing hidden images within `hidden_object.jpg`.

**Tasks:**

1. Use a **Histogram Statistics method** to extract the hidden image.
2. Use a **Local Enhancement method** for the same purpose.
3. Describe parameters and methods used in both cases in detail.
4. Compare and comment on the two enhancement approaches.

**Files:**

* `hidden_object.jpg`
* Python code for histogram statistics and local enhancement methods.

---

### DIP Homework 3-3: Lowpass Gaussian Filtering

**Description:**
This assignment applies self-designed **Lowpass Gaussian Filters** to remove shading noise patterns from images.

**Tasks:**

1. Design a custom lowpass Gaussian kernel to remove shaded noise from `checkerboard1024-shaded.tif` (as in Fig. 3.42(b)-(c)).
2. Repeat the same process for the image `N1.bmp`.
3. Describe both filter kernels and compare their performance.

**Files:**

* `checkerboard1024-shaded.tif`
* `N1.bmp`
* Python code for Gaussian filter design and application.

---

### DIP Homework 3-4: Highboost Filtering

**Description:**
This assignment focuses on **Highboost Filtering** using **Sobel** and **Laplacian** filters for image enhancement.

**Tasks:**

1. Design a Highboost filter based on Sobel and Laplacian methods (pp.183–195). Apply it to `bodybone.bmp` as shown in Fig. 3.49(e).
2. Repeat the same procedure on `fish.jpg`.
3. Compare and comment on the two filter results.

**Files:**

* `bodybone.bmp`
* `fish.jpg`
* Python code for Highboost filtering using Sobel and Laplacian operators.

---

### DIP Homework 4-2: Fast Fourier Transform

**Description:**
Remove high‑frequency periodic interference / moiré by transforming the image to the frequency domain, identifying interference peaks, applying notch (band‑reject) filtering, and reconstructing via IFFT.

**Tasks**

1. Display the original image and the log amplitude spectrum (use fftshift and percentile contrast stretching for visibility).
2. Design a band‑reject (notch) mask around the interference peaks (manual selection or programmatic detection).
3. Multiply the complex spectrum by the mask, then perform IFFT to obtain the restored image.
4. Comment on the visual improvement (stripe removal vs. detail loss) and list the parameters you used.

**Files**
* `astronaut-interference.tif`
* `car-moire-pattern.tif`
python code for denoise unwanted noise in frequency domain

---


