import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Complete JPEG Compression & Decompression using OpenCV ---

def jpeg_opencv_compress_decompress(img_path, quality=50):
    """
    Reads an image, compresses it to JPEG format in-memory using OpenCV,
    then decompresses back to an image array.

    Args:
        img_path (str): Path to the input image.
        quality (int): JPEG quality (1-100, higher => better quality, larger size).

    Returns:
        original (ndarray): Original BGR image array.
        decompressed (ndarray): Decompressed BGR image array after JPEG compression.
        quality (int): Quality used for compression.
    """
    original = cv2.imread(img_path)
    if original is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    # JPEG compression parameters
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Compress to memory buffer
    result, encimg = cv2.imencode('.jpg', original, encode_params)
    if not result:
        raise RuntimeError("JPEG encoding failed")

    # Decompress back to BGR image
    decompressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if decompressed is None:
        raise RuntimeError("JPEG decoding failed")

    return original, decompressed, quality


def plot_comparison(original, compressed, quality):
    """
    Display side-by-side comparison of original vs compressed images.
    """
    # Convert BGR to RGB for matplotlib
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    comp_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(comp_rgb)
    plt.title(f'JPEG Compressed (Quality={quality})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Prompt user to enter image path
    img_path = input("Enter image file path: ")
    if not img_path:
        print("No path provided. Exiting.")
        exit()

    # Fixed quality for demo
    quality = 50

    # Compress & decompress
    original, compressed, used_quality = jpeg_opencv_compress_decompress(img_path, quality)

    # Plot side-by-side comparison
    plot_comparison(original, compressed, used_quality)

    # Show the compressed image in an OpenCV window
    cv2.imshow('Compressed Image', compressed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()