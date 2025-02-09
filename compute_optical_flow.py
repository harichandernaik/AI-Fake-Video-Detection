import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_optical_flow(video_path, save_heatmap=False):
    """Computes dense optical flow and optionally saves a motion heatmap."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return None

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame.")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_magnitudes = []
    heatmap_frames = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        motion_magnitudes.append(np.mean(magnitude))
        heatmap_frames.append(magnitude)

        prev_gray = gray
        frame_count += 1

    cap.release()

    if len(motion_magnitudes) == 0:
        print("❌ Error: No motion detected in video.")
        return None

    # 🔥 Save Heatmap of Motion if requested
    if save_heatmap:
        os.makedirs("heatmaps", exist_ok=True)
        print(f"✅ Saving motion heatmaps for {min(10, frame_count)} frames...")

        for i, heatmap in enumerate(heatmap_frames[:10]):  # Save first 10 frames only
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f"Optical Flow Magnitude - Frame {i}")
            plt.savefig(f"heatmaps/frame_{i}.png")
            plt.close()

        print("✅ Motion heatmaps saved in 'heatmaps/' directory.")

    return motion_magnitudes
