import cv2 as cv
import numpy as np
# Input image file from user
image_path = input("Enter the path of the image: ")
# Read the image in grayscale
gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
# Read the image in RGB
rgb_image = cv.imread(image_path, cv.IMREAD_COLOR)
# Convert BGR to RGB
# cv.IMREAD_COLOR reads the image in BGR(Blue-Green-Red) format(default in OpenCV).
#The output (rgb_image) is a 3D array (height × width × 3 channels)
rgb_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2RGB)
# Check if the images are loaded successfully
if gray_image is None or rgb_image is None:
 print("Error: Image not found or unable to read")
else:
# Store pixel values in a 2D matrix for grayscale
 gray_pixel_matrix = np.array(gray_image)
# Store pixel values in a 3D matrix for RGB
rgb_pixel_matrix = np.array(rgb_image)
# Display the pixel matrices
print("Grayscale Pixel Matrix:")
print(gray_pixel_matrix)
print("Size of Grayscale Matrix:", gray_pixel_matrix.shape)
print("\nRGB Pixel Matrix:")
print(rgb_pixel_matrix)
print("Size of RGB Matrix:", rgb_pixel_matrix.shape)
# Show the images
input_image = cv.imread(image_path) # Read original image
cv.imshow('Input Image', input_image)
cv.imshow('Grayscale Image', gray_image)
cv.imshow('RGB Image', cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()