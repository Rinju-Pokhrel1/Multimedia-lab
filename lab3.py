import numpy as np # For array and numerical operations
import cv2 # For image loading and processing
import matplotlib.pyplot as plt # For image display
import os # For checking file paths

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read image in grayscale mode
    return image # Return grayscale image

# Predefined 2x2 cluster dot patterns for 5 intensity levels
dot_patterns = {
    0: np.array([[0, 0], [0, 0]], dtype=np.uint8), # White (no dots)
    1: np.array([[255, 0], [0, 0]], dtype=np.uint8), # Light gray
    2: np.array([[255, 255], [0, 0]], dtype=np.uint8), # Medium-light
    3: np.array([[255, 255], [255, 0]], dtype=np.uint8), # Medium-dark
    4: np.array([[255, 255], [255, 255]], dtype=np.uint8), # Black (all dots)
}

def cluster_dot_halftoning(grayscale_image, cluster_size=2):
    height, width = grayscale_image.shape # Get image dimensions
    halftoned_image = np.zeros_like(grayscale_image) # Initialize empty halftoned image

    for i in range(0, height, cluster_size): # Loop over rows in steps
        for j in range(0, width, cluster_size): # Loop over columns in steps
            cluster = grayscale_image[i:i+cluster_size, j:j+cluster_size] # Extract cluster
            avg_intensity = np.mean(cluster) # Compute average brightness
            dot_level = int(avg_intensity / 255 * 4) # Map brightness to dot level (0–4)
            pattern = dot_patterns[dot_level] # Get matching pattern

            halftoned_image[i:i+cluster.shape[0], j:j+cluster.shape[1]] = \
                pattern[:cluster.shape[0], :cluster.shape[1]] # Apply pattern

    return halftoned_image # Return the halftoned image


def display_images(original, halftoned):
    plt.figure(figsize=(10, 5)) # Set figure size

    plt.subplot(1, 2, 1) # First subplot (original)
    plt.imshow(original, cmap='gray') # Show original grayscale image
    plt.title("Original Grayscale Image") # Title
    plt.axis('off') # Hide axes

    plt.subplot(1, 2, 2) # Second subplot (halftoned)
    plt.imshow(halftoned, cmap='gray') # Show halftoned image
    plt.title("Halftoned Image (2x2 Cluster)") # Title
    plt.axis('off') # Hide axes

    plt.tight_layout() # Adjust spacing
    plt.show() # Display both images


def main():
    image_path = input("Enter the path to the image: ") # Prompt user for image path

    if not os.path.isfile(image_path): # Check if path is valid
        print("Invalid file path. Please try again.") # Show error message
        return # Exit program

    grayscale_image = convert_to_grayscale(image_path) # Convert to grayscale
    halftoned_image = cluster_dot_halftoning(grayscale_image) # Apply halftoning
    display_images(grayscale_image, halftoned_image) # Show both images


if __name__ == "__main__":
    main()