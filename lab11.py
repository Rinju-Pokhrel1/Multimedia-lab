import cv2
import time

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press 'c' to capture image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the live video
    cv2.imshow("Multimedia Interface", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Capture and save the current frame
        filename = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved frame as {filename}")
    elif key == ord('q'):
        # Quit the loop
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()