import numpy as np
import matplotlib.pyplot as plt

def image_fft_band_reject(
    img,
    w_centers=(480, 530),  
    h_centers=(390, 442),
    expand=4,             
    method="two_1d",       # "two_1d" or "fft2"
    show=True
):
   
    if img.ndim == 3:
        img_gray = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
    else:
        img_gray = img.astype(np.uint8)

    H, W = img_gray.shape

    # ====== FFT ======
    if method == "two_1d":
        F = np.zeros((H, W), dtype=np.complex128)
        for h in range(H):
            F[h, :] = np.fft.fft(img_gray[h, :])
        for w in range(W):
            F[:, w] = np.fft.fft(F[:, w])
        F = np.fft.fftshift(F)
        F = F / (H * W)  
    elif method == "fft2":
        F = np.fft.fftshift(np.fft.fft2(img_gray))
    else:
        raise ValueError('method must be "two_1d" or "fft2"')

    amp = np.abs(F)

    # build band-reject filter
    mask = np.ones((H, W), dtype=np.float64)
    wc0 = [w - 1 for w in w_centers]  # 0-based
    hc0 = [h - 1 for h in h_centers]
    rng = range(-expand, expand + 1)

    for wc in wc0:
        for hc in hc0:
            for dh in rng:
                for dw in rng:
                    r = hc + dh
                    c = wc + dw
                    if 0 <= r < H and 0 <= c < W:
                        mask[r, c] = 0.0

    # visualization
    if show:
        plt.figure(); plt.imshow(img_gray, cmap='gray'); plt.title('original image'); plt.axis('off')

        log_amp = np.log1p(amp)
        lo, hi = np.percentile(log_amp, [5, 99.7])
        plt.figure(); plt.imshow(np.clip(log_amp, lo, hi), cmap='gray'); plt.title('log |FFT|'); plt.axis('off')

        plt.figure(); plt.imshow(mask, cmap='gray'); plt.title('band_reject_filter'); plt.axis('off')

        masked_log = np.log1p(amp * mask)
        lo2, hi2 = np.percentile(masked_log, [5, 99.7])
        plt.figure(); plt.imshow(np.clip(masked_log, lo2, hi2), cmap='gray'); plt.title('masked log |FFT|'); plt.axis('off')

    Ff = F * mask

    if method == "two_1d":
        tmp = np.zeros_like(Ff, dtype=np.complex128)
        for h in range(H):
            tmp[h, :] = np.fft.ifft(Ff[h, :])
        for w in range(W):
            tmp[:, w] = np.fft.ifft(tmp[:, w])
        out = np.abs(tmp) * (H * W)  
    else:  # "fft2"
        out = np.abs(np.fft.ifft2(np.fft.ifftshift(Ff)))

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    if show:
        plt.figure(); plt.imshow(out_u8, cmap='gray'); plt.title(f'output image ({method})'); plt.axis('off')
        plt.show()

    return out_u8, mask, amp


if __name__ == "__main__":
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    #img = iio.imread("astronaut-interference.tif")
    img = iio.imread("car-moire-pattern.tif")

    
    # astronaut parameter
    # out_u8, mask, amp = image_fft_band_reject(
    #     img,
        
    #     w_centers=(480, 530),
    #     h_centers=(390, 442),
    #     expand=4,
    #     method="two_1d",   # 或 "fft2"
    #     show=True
    # )
    
    # car parameter
    out_u8_b, mask_b, amp_b = image_fft_band_reject(
    img,
    w_centers=(58, 114),
    h_centers=(86, 45, 166, 207),
    expand=10,
    method="fft2",
    show=True
)
