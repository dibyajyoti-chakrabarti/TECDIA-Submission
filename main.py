import cv2
import numpy as np
import os

# =========================================================
# Utility Functions
# =========================================================

def extract_frames(path):
    """
    Step 1: Extract frames from the input video.
    Reads all frames in sequential order and returns them as a list.
    """
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
    """
    Step 2: Save frames as a video.
    Reconstructs a video from a list of frames at the given frame rate.
    """
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def compute_background(frames):
    """
    Step 3: Compute a static background.
    Uses the median of all frames to estimate a clean background.
    """
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def centroid_mask(frame, bg):
    """
    Step 4: Create a binary mask isolating the moving object.
    Compares each frame with the background and thresholds the difference.
    """
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    return mask


def compute_centroids(frames, bg):
    """
    Step 5: Compute horizontal centroids for all frames.
    Determines the object's approximate x-position in each frame.
    """
    centroids = []
    for frame in frames:
        mask = centroid_mask(frame, bg)
        M = cv2.moments(mask)
        if M["m00"] > 0:
            centroids.append(M["m10"] / M["m00"])
        else:
            centroids.append(np.nan)

    # Handle missing centroid values using interpolation
    centroids = np.array(centroids)
    idx = np.arange(len(centroids))
    good = ~np.isnan(centroids)
    if np.any(~good):
        centroids[np.isnan(centroids)] = np.interp(idx[np.isnan(centroids)], idx[good], centroids[good])
    return centroids


def zoom_frame(frame, factor=1.8):
    """
    Step 6: Zoom into the center of the frame.
    This improves visibility of small objects by enlarging the region of interest.
    """
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
    # Step 1: Specify input and output details
    video_path = "videos/jumbled_video.mp4"
    output_path = "results/reconstructed.mp4"
    fps = 30
    split_point = 180  # First 6 seconds (at 30 fps) are considered correct

    print("Step 1: Extracting frames from input video...")
    frames = extract_frames(video_path)
    print(f"Total frames extracted: {len(frames)}")

    # Step 2: Compute centroid-based full reconstruction
    print("Step 2: Performing full centroid-based ordering...")
    bg_full = compute_background(frames)
    cx_full = compute_centroids(frames, bg_full)
    order_full = np.argsort(cx_full)[::-1]  # Reverse for correct direction
    ordered_frames = [frames[i] for i in order_full]

    # Step 3: Split into stable (head) and unstable (tail) parts
    print("Step 3: Splitting ordered frames into head (first 180) and tail (remaining frames)...")
    head = ordered_frames[:split_point]
    tail = ordered_frames[split_point:]

    # Step 4: Refine the tail by re-ordering after zooming in
    print("Step 4: Refining tail region using zoomed-in centroid computation...")
    zoomed_tail = [zoom_frame(f, factor=1.8) for f in tail]
    bg_tail = compute_background(zoomed_tail)
    cx_tail = compute_centroids(zoomed_tail, bg_tail)
    order_tail = np.argsort(cx_tail)[::-1]
    refined_tail = [tail[i] for i in order_tail]

    # Step 5: Merge head and refined tail
    print("Step 5: Combining the stable head and refined tail...")
    final_frames = head + refined_tail

    # Step 6: Save the final reconstructed video
    print("Step 6: Saving the reconstructed video...")
    save_video(final_frames, output_path, fps=fps)
    print(f"Reconstruction completed successfully. Video saved at: {output_path}")