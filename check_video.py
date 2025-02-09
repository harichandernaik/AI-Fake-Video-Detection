import os
import cv2

def check_video_path(video_path):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print("Error: Video file not found.")
        return False  # Return False if the file doesn't exist
    
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return False  # Return False if the video can't be opened
    
    print("Video opened successfully.")
    return True  # Return True if everything is fine
