import cv2
import numpy as np
import requests
import tempfile
import os

# Step 1: Download the video from the URL
video_url = "https://avtshare01.rz.tu-ilmenau.de/avt-vqdb-uhd1/test_1/segments/bigbuck_bunny_8bit_15000kbps_1080p_60.0fps_h264.mp4"
response = requests.get(video_url, stream=True)

# Step 2: Save it temporarily to a file
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
    for chunk in response.iter_content(chunk_size=1024*1024):
        if chunk:
            tmp_file.write(chunk)
    temp_video_path = tmp_file.name

# Step 3: Load video using OpenCV
cap = cv2.VideoCapture(temp_video_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

processed_frames = []

# Step 4: Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to 128x128
    resized = cv2.resize(gray, (128, 128))

    # Append to list
    processed_frames.append(resized)

# Cleanup
cap.release()
os.remove(temp_video_path)  # Delete temp file

print(f"Total frames processed: {len(processed_frames)}")
