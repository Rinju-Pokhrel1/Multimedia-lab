import cv2

# Abstract image handler class
class ImageHandler:
    def __init__(self, path):
        self.image = cv2.imread(path)

    def show(self):
        if self.image is not None:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Unable to load image.")


# Get image path from user
path = input("Enter the path to the image: ")

# Use the class
img = ImageHandler(path)
img.show()