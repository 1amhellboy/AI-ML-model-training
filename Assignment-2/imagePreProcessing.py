import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
img = cv2.imread("sample.jpg")
if img is None:
    raise FileNotFoundError("Image file not found. Make sure 'sample.jpg' is in the same folder.")

print("Original shape:", img.shape)

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show original image
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis('off')
plt.show()

# Resize image to 128x128
img_resized = cv2.resize(img_rgb, (128, 128))
print("Resized shape:", img_resized.shape)

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# Convert to binary using thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis('off')
plt.show()

# Normalize pixel values to [0,1]
normalized = gray / 255.0
print("Normalized range: min =", normalized.min(), ", max =", normalized.max())

# Flipping and rotating the image
flipped = cv2.flip(img_resized, 1)  # horizontal flip
rotated = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

plt.subplot(1, 2, 1)
plt.imshow(flipped)
plt.title("Flipped")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated)
plt.title("Rotated")
plt.axis('off')
plt.show()

# Apply Gaussian and Median Blur
gaussian_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
median_blur = cv2.medianBlur(img_resized, 5)

plt.subplot(1, 2, 1)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(median_blur)
plt.title("Median Blur")
plt.axis('off')
plt.show()

# Flatten the resized image
flattened = img_resized.flatten()
print("Flattened shape:", flattened.shape)
