# Jumbled Frames Reconstruction

## Overview

This project performs **video frame reordering** to reconstruct the original temporal sequence from a jumbled video.  
It assumes the input video contains a **single moving object** (for example, a walking person) against a **static background**.

The reconstruction is based on analyzing the **horizontal movement** of the object across frames using **centroid tracking**.

## Note

- The entire project was created and run on a Linux OS and as such, some commands might vary according to the host OS.
- If you are using Windows, please verify beforehand if the commands are Windows compatible.
- You can either use ChatGPT or any other AI tools for doing this.
- Please run the following commands in <ins>**Terminal**</ins> and not VS code's in-built Terminal.
- Instances of VS Code crashing are common.

## Cloning the repository
### 1. Create a project folder
```
mkdir project
cd project
```
### 2. Initialize git and clone
```
git init
git clone https://github.com/dibyajyoti-chakrabarti/reconstruct_frames.git
cd reconstruct_frames/
```
## Setup
### 1. Create a virtualenv
```
python -m venv venv
source venv/bin/activate
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Create a Directory for Videos
```
mkdir videos
```
*And store you video in this directory*

### 4. Create a Directory for Resultant Videos
```
mkdir results
```

*main.py will generate the resultant video here*

### 5. Fix File name in main.py

In the line `video_path = "videos/jumbled_video.mp4")` on `line number 101`

Change `jumbled_video.mp4` with the name of the video to be reconstructed.

### 6. Create the Reconstructed Video
```
python3 main.py
```
### 7. View the Reconstructed Video

You can either navigate to the `results` directory and manually play the video or use the following command:
```
cvlc results/reconstructed.mp4  
```
## Algorithm Explanation
### 1. Approach Overview

The algorithm reconstructs the temporal order of frames in a jumbled 10-second video by analyzing the motion of the primary moving object (e.g., a person walking).
The assumption is that the background remains static while the object moves in a consistent direction (e.g., right to left).

The process is divided into two main phases:

1. `Global Centroid Ordering`: Orders all frames based on the horizontal centroid of the moving object.

2. `Refined Tail Correction`: Reprocesses the final portion of the video (where the object becomes small and centroid detection becomes unstable) by zooming in and recalculating centroids.

### 2. Techniques Used
**<ins>Centroid Tracking</ins>**

For each frame:
- A background image is computed using the median of all frames. 
- The difference between the frame and background isolates the moving object. 
- The binary mask of this object is used to compute its centroid (x-position).
- Sorting the frames by centroid position yields the approximate temporal sequence.

Mathematically:

$$
cx_i = \frac{M_{10}}{M_{00}}
$$



**where 𝑀<sub>10</sub>
and 𝑀<sub>00</sub>** are image moments.

This method effectively reconstructs the movement when the object is large and clearly visible.

**<ins>Tail Refinement via Zoom</ins>**

As the object becomes smaller (e.g., further from the camera), centroid accuracy drops due to:

- Low pixel coverage
- Increased background noise

To counter this:

- The last 120 frames (approx. 4 seconds) of the already reconstructed sequence are zoomed by a factor of 1.8×.
- The centroid computation is reapplied to this zoomed region.
- The new order is merged back into the first 180 stable frames.
- This localized reprocessing helps stabilize small-object motion and reduce jitter.

### 3. Why This Method Was Chosen

- **Simplicity and Interpretability:** Centroid tracking is easy to understand and tune without requiring complex ML training.
- **Efficiency:** The median background and centroid extraction are 𝑂(𝑛) operations over frames, making it practical for 10-second (≈300-frame) videos.
- **Consistency:** Since the same background is used across all frames, centroid positions provide consistent spatial reference.
- **Fallback-Resilient:** Even if some frames fail to produce centroids, missing values are interpolated linearly, ensuring full reconstruction.
### 4. Key Design Considerations
   
- **Accuracy:** Combined centroid-based and zoom-based analysis maintains temporal consistency even for small distant objects.
- **Time Complexity:** O(n) per stage (centroid + sorting), fast and scalable for short clips.
- **Parallelism:** Not required for this dataset size, avoids multi-threading overhead.
- **Stability:** Median background removes noise and illumination changes.
- **Directional Flexibility:** Simple reversal flag handles right-to-left or left-to-right motion.

### 5. Possible Enhancements
- Integrating optical flow for more precise motion estimation.
- Using object tracking (e.g., Kalman filters or SORT) for complex scenes.
- Adding temporal smoothing between consecutive centroids to reduce local jitter.

## Execution Time Log
- System: `Ubuntu 22.04 (x86_64)`
- CPU: `Intel i5 12th Gen`
- RAM: `16 GB`
- Video Duration: `10 seconds (300 frames at 30 FPS)`

| Stage | Description | Time (seconds) |
|--------|-------------|----------------|
| Frame Extraction | Reading 300 frames | 2.1 s |
| Background Computation | Median background calculation | 1.8 s |
| Full Centroid Ordering | Primary reconstruction | 3.4 s |
| Tail Refinement | Zoom + centroid recomputation | 2.6 s |
| Saving Output Video | Writing reconstructed frames | 1.2 s |
| Total Execution Time |  | ≈ 11.1 s |

**<ins>Note:</ins>** Execution time varies depending on resolution, system performance, and Python environment.

## Output
After running the script successfully, the final reconstructed video will be available at ```results/reconstructed.mp4```

The reconstructed video shows:
- Smooth motion for the first 6 seconds.
- Reduced jitter in the final 4 seconds after refinement.

## Author
**Dibyajyoti Chakrabarti, VIT Vellore (October 2025)**