import cv2
import numpy as np

def extract_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in frames: writer.write(f)
    writer.release()

def compute_background(frames):
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

def centroid_mask(frame, bg):
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    return mask

def compute_centroids(frames, bg):
    centroids = []
    for f in frames:
        m = centroid_mask(f, bg)
        M = cv2.moments(m)
        if M["m00"] > 0:
            centroids.append(M["m10"]/M["m00"])
        else:
            centroids.append(np.nan)
    centroids = np.array(centroids)
    # fill missing values
    idx = np.arange(len(centroids))
    good = ~np.isnan(centroids)
    centroids[np.isnan(centroids)] = np.interp(idx[np.isnan(centroids)], idx[good], centroids[good])
    return centroids

# --- MAIN ---
frames = extract_frames("videos/jumbled_video.mp4")
bg = compute_background(frames)
cx = compute_centroids(frames, bg)

# sort frames by centroid → reconstruct forward motion
#order = np.argsort(cx) -> Use this if the direction is wrong
order = np.argsort(cx)[::-1]
ordered_frames = [frames[i] for i in order]

save_video(ordered_frames, "reconstructed.mp4", fps=30)
print("Done → reconstructed.mp4 saved ✅")