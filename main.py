import cv2
import numpy as np
import os

# =========================================================
# Utility Functions
# =========================================================

def extract_frames(path):
    """Extracts all frames from the video in sequential order."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames, path, fps=30):
    """Saves the provided frames as a video file."""
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def compute_background(frames):
    """Computes the median background image."""
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def centroid_mask(frame, bg):
    """Generates a binary mask highlighting the moving object."""
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    return mask


def compute_centroids(frames, bg):
    """Computes horizontal centroid positions for each frame."""
    centroids = []
    for frame in frames:
        mask = centroid_mask(frame, bg)
        M = cv2.moments(mask)
        if M["m00"] > 0:
            centroids.append(M["m10"] / M["m00"])
        else:
            centroids.append(np.nan)

    centroids = np.array(centroids)
    idx = np.arange(len(centroids))
    good = ~np.isnan(centroids)
    if np.any(~good):
        centroids[np.isnan(centroids)] = np.interp(idx[np.isnan(centroids)], idx[good], centroids[good])
    return centroids


def zoom_frame(frame, factor=1.5):
    """Zooms into the center region by a given factor."""
    h, w = frame.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return zoomed


# =========================================================
# Main Reconstruction Logic
# =========================================================

if __name__ == "__main__":
    video_path = "videos/jumbled_video.mp4"
    output_path = "results/reconstructed.mp4"
    fps = 30

    # Define breakpoints (in frames)
    bp1 = 90   # 3 seconds
    bp2 = 180  # 6 seconds

    print("Step 1: Extracting frames from input video...")
    frames = extract_frames(video_path)
    print(f"Total frames extracted: {len(frames)}")

    # Step 2: Compute the initial full ordering
    print("Step 2: Performing full centroid-based ordering...")
    bg_full = compute_background(frames)
    cx_full = compute_centroids(frames, bg_full)
    order_full = np.argsort(cx_full)[::-1]
    ordered_frames = [frames[i] for i in order_full]

    # Step 3: Split into three logical segments
    print("Step 3: Splitting into segments at 3s and 6s marks...")
    seg1 = ordered_frames[:bp1]         # 0–3s (stable)
    seg2 = ordered_frames[bp1:bp2]      # 3–6s (medium zoom)
    seg3 = ordered_frames[bp2:]         # 6–10s (strong zoom)

    # Step 4: Reorder mid and tail segments after zoom
    print("Step 4: Refining mid (3–6s) with moderate zoom...")
    zoomed_seg2 = [zoom_frame(f, factor=1.4) for f in seg2]
    bg2 = compute_background(zoomed_seg2)
    cx2 = compute_centroids(zoomed_seg2, bg2)
    order2 = np.argsort(cx2)[::-1]
    refined_seg2 = [seg2[i] for i in order2]

    print("Step 5: Refining tail (6–10s) with stronger zoom...")
    zoomed_seg3 = [zoom_frame(f, factor=1.8) for f in seg3]
    bg3 = compute_background(zoomed_seg3)
    cx3 = compute_centroids(zoomed_seg3, bg3)
    order3 = np.argsort(cx3)[::-1]
    refined_seg3 = [seg3[i] for i in order3]

    # Step 6: Merge all refined parts
    print("Step 6: Combining all refined segments...")
    final_frames = seg1 + refined_seg2 + refined_seg3

    # Step 7: Save final output
    print("Step 7: Saving the final reconstructed video...")
    save_video(final_frames, output_path, fps=fps)
    print(f"Reconstruction completed successfully. Video saved at: {output_path}")
