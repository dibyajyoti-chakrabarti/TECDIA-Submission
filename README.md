# Jumbled Frames Reconstruction

## Overview

This project performs **video frame reordering** to reconstruct the original temporal sequence from a jumbled video.  
- It assumes the input video contains a **single moving object** (for example, a walking person) against a **static background**.  
- The reconstruction is based on analyzing the **horizontal movement** of the object across frames using **centroid tracking**.  

## Note (READ THIS CAREFULLY)

- Commands are provided for both Linux-based and Windows OS.  
- If any command doesn’t specify OS differences, it can be run on both systems.  
- Please run the following commands in **Terminal** (Linux) or **Command Prompt** (Windows), **not** in VS Code’s built-in terminal.  
- Instances of VS Code crashing are common during heavy computation.  
- The script `main.py` has 7 steps. The 2nd step (global centroid ordering) is the most compute-heavy.  
  The terminal may freeze for 10–20 seconds depending on your system performance.  
  If the terminal crashes, restart your PC and re-run `main.py`.

---

## Cloning the Repository

### 1. Create a Project Folder
```bash
mkdir project
cd project
```

### 2. Clone the Repository
```bash
git clone https://github.com/dibyajyoti-chakrabarti/TECDIA-Submission.git
cd TECDIA-Submission/
```

---

## Setup

### 1. Create a Virtual Environment

**For Linux-based OS:**
```bash
python -m venv venv
source venv/bin/activate
```

**For Windows OS:**
```
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a Directory for Videos
```bash
mkdir videos
```
*Store your input video here.*

### 4. Create a Directory for Resultant Videos
```bash
mkdir results
```
*`main.py` will generate the output video here.*

### 5. Fix File Name in main.py
In the line:
```python
video_path = "videos/jumbled_video.mp4"
```
Change `"jumbled_video.mp4"` to the name of your input video.

### 6. Run the Reconstruction Script

**For Linux-based OS:**
```bash
python3 main.py
```

**For Windows OS:**
```bash
python main.py
```

### 7. View the Reconstructed Video

**For Linux-based OS:**
```bash
cvlc results/reconstructed.mp4
```

**For Windows OS:**
```bash
vlc results\reconstructed.mp4
```

### 8. Change Direction (Optional)
If the reconstructed motion is in the opposite direction, change this line:
```python
order_full = np.argsort(cx_full)[::-1]
```
To:
```python
order_full = np.argsort(cx_full)
```
Then re-run Step 6.

---

## Algorithm Explanation

### 1. Approach Overview
The algorithm reconstructs the temporal order of frames in a jumbled 10-second video by analyzing the motion of the primary moving object (e.g., a person walking).  
The assumption is that the background remains static while the object moves in a consistent direction (e.g., right to left).

The process is divided into three main phases:
1. **Global Centroid Ordering:** Orders all frames based on the horizontal centroid of the moving object.  
2. **Mid-Range Refinement:** Reprocesses the mid-portion (3–6s) with a moderate zoom factor to stabilize transitions.  
3. **Tail Correction:** Reprocesses the final portion (6–10s) with a higher zoom factor to reduce jitter when the object becomes smaller.

---

### 2. Techniques Used

**<ins>Centroid Tracking</ins>**

For each frame:
- A background image is computed using the median of all frames.  
- The difference between the frame and the background isolates the moving object.  
- The binary mask of this object is used to compute its centroid (x-position).  
- Sorting frames by centroid position yields the approximate temporal sequence.

Mathematically:
```math
cx_i = \frac{M_{10}}{M_{00}}
```
**where 𝑀₁₀ and 𝑀₀₀ are image moments.**

This method works well when the moving object is large and clearly visible.

---

**<ins>Segmented Refinement via Zoom</ins>**

When the object becomes smaller or partially occluded (e.g., by branches or shadows), centroid precision decreases.  
To counter this:
- The video is divided into three segments:
  - **0–3s (Stable, no zoom)**
  - **3–6s (Moderate zoom, 1.4×)**
  - **6–10s (Strong zoom, 1.8×)**  
- The centroid computation is repeated for zoomed regions in the last two segments.  
- These reordered sub-sequences are merged to form the final output.

This approach reduces localized jitter and stabilizes object motion across depth changes.

---

### 3. Why This Method Was Chosen

- **Simplicity and Interpretability:** Easy to understand and tune without requiring machine learning.  
- **Efficiency:** Median-based background subtraction and centroid sorting are both O(n) operations.  
- **Adaptability:** Segment-wise refinement helps target jitter-prone regions without overprocessing the stable ones.  
- **Consistency:** A uniform background reference maintains coherent motion direction.  
- **Fallback-Resilient:** Missing centroid values are linearly interpolated, ensuring smooth reconstruction.

---

### 4. Key Design Considerations

- **Accuracy:** Centroid and zoom hybrid analysis preserves correct temporal order.  
- **Time Complexity:** O(n) per phase, suitable for 10s (≈300-frame) videos.  
- **Stability:** Median background eliminates noise and brightness variation.  
- **Directional Flexibility:** Reversing centroid sort order allows reconstruction in either direction.  
- **Scalability:** Works consistently for single-object motion scenes.

---

### 5. Possible Enhancements
- Integrating **optical flow** for fine-grained motion estimation.  
- Employing **object tracking (Kalman Filter or SORT)** for complex multi-object scenes.  
- Using **adaptive segmentation** based on centroid variance to auto-detect jitter regions.  
- Applying **temporal smoothing** between consecutive centroids for ultra-smooth playback.

---

## Execution Time Log

**System:** Ubuntu 22.04 (x86_64)  
**CPU:** Intel i5 12th Gen  
**RAM:** 16 GB  
**Video Duration:** 10 seconds (300 frames @ 30 FPS)

| Stage | Description | Time (seconds) |
|--------|-------------|----------------|
| Frame Extraction | Reading 300 frames | 2.1 s |
| Background Computation | Median background calculation | 1.8 s |
| Full Centroid Ordering | Primary reconstruction | 14.0 s |
| Tail Refinement | Zoom + centroid recomputation | 4.7 s |
| Saving Output Video | Writing reconstructed frames | 1.2 s |
| **Total Execution Time** |  | **≈ 24.0 s** |

**_Note:_** Execution time may vary depending on resolution, hardware, and environment performance.  
The **centroid-based ordering** stage is the most compute-heavy.

---

## Output

After running the script successfully, the final reconstructed video will be available at:
```
results/reconstructed.mp4
```

The reconstructed video demonstrates:
- Smooth motion for the first 6 seconds.  
- Minimal jitter in the final 4 seconds after refinement.

---

## Evaluated Output (jumbled_video.mp4)

You can view the **reconstructed video output** here:  
[🎬 View Reconstructed Video on OneDrive](https://drive.google.com/file/d/1Ok-1I24BjtR6WFv9nhlGxWNMmoyydFFx/view?usp=sharing)

> **Note:**  
> The video demonstrates the corrected temporal order after applying centroid-based and zoom-refined reconstruction on `jumbled_video.mp4`.

---

## Originality Statement

This project and its algorithm were independently developed as part of an experimental study on reconstructing temporally jumbled video frames.  
The method combines background subtraction, centroid-based motion analysis, and localized zoom refinement — all designed and implemented specifically for this project.  

All code and documentation were originally written without referencing or replicating any existing open-source or academic work.  
Only standard libraries (`OpenCV`, `NumPy`) were used for image processing and computation.

Any similarity to prior works is purely coincidental and limited to the use of standard image processing techniques.

---

## Author
**Dibyajyoti Chakrabarti, VIT Vellore (October 2025)**
