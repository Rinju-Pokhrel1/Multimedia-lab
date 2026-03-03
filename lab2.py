import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Input image file from user
image_path = input("Enter the path of the image: ")
# Read the image in grayscale
gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
# Read the image in RGB
rgb_image = cv.imread(image_path, cv.IMREAD_COLOR)
# Check if images are loaded successfully
if gray_image is None or rgb_image is None:
   print("Error: Image not found or unable to read")
else:
# Display the inputted image
 input_image = cv.imread(image_path)
 cv.imshow('Input Image', input_image)
 cv.imshow('Grayscale Image', gray_image)
 cv.imshow('RGB Image', rgb_image)
 cv.waitKey(0)
 cv.destroyAllWindows()
# Compute and display histogram of grayscale image
plt.figure()
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='black',
alpha=0.75)
plt.show()
# Compute and display histograms for RGB channels (Optimized)
colors = ('b', 'g', 'r') # BGR channel order
plt.figure()
plt.title("RGB Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
for i, color in enumerate(colors):
# Calculate histogram for each channel
  hist = cv.calcHist([rgb_image], [i], None, [256], [0, 256])
  plt.plot(hist, color=color, label=f'{color.upper()} Channel')
  plt.legend()
  plt.show()