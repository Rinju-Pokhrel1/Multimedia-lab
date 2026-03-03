import cv2
import matplotlib.pyplot as plt
# Ask user for video path .mp4
video_path = input("Enter the path to the video file in mp4 : ")
# Load video
cap = cv2.VideoCapture(video_path)
# Check if video is opened
if not cap.isOpened():
    print("Error: Cannot open video.")
exit()
# Display video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("--- Video Properties ---")
print(f"Resolution : {width} x {height}")
print(f"Frame Rate : {fps} fps")
print(f"Frame Count : {frame_count}")
print("------------------------")
# Show the video properties using matplotlib
labels = ['Width', 'Height', 'FPS', 'Total Frames']
values = [width, height, fps, frame_count]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['skyblue', 'orange', 'green', 'red'])
plt.title('Video Properties')
plt.ylabel('Values')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Release the video capture object
cap.release()