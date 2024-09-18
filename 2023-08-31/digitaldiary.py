import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Compute the CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # Mask all pixels with value=0 and normalize the CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Use the CDF to map the original image
    image_equalized = cdf[image]
    
    return image_equalized, hist, cdf_normalized

# Load image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
image_equalized, hist_original, cdf_original = histogram_equalization(image)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_original)
plt.title('Original Histogram')

plt.subplot(2, 2, 3)
plt.imshow(image_equalized, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(cdf_original)
plt.title('Equalized Histogram')

plt.show()
