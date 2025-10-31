# Jumbled Frames Reconstruction

## Overview

This project performs **video frame reordering** to reconstruct the original temporal sequence from a jumbled video. 
- It assumes the input video contains a **single moving object** (for example, a walking person) against a **static background**.
- The reconstruction is based on analyzing the **horizontal movement** of the object across frames using **centroid tracking**.

## Note (READ THIS CAREFULLY)

- Commands are provided for both Linux-based and Winows OS. If any command doesn't have that distinction, the command can be run on both- Windows or Linux.
- Please run the following commands in `Terminal` for Linux and `Command Prompt` for Windows, and not VS code's in-built Terminal.
- Instances of VS Code crashing are common.
- The script- `main.py` has 6 steps. The 2nd step involves computing complete centroid-based ordering. This step is the most compute-heavy step and the terminal might freeze for a few seconds (10-20s depending on OS performance). In case the terminal crashes, restart the PC and try running main.py again.

## Cloning the repository
### 1. Create a project folder
```
mkdir project
cd project
```
### 2. Clone the repository

```
git clone https://github.com/dibyajyoti-chakrabarti/TECDIA-Submission.git
cd TECDIA-Submission/
```
## Setup
### 1. Create a virtualenv

For Linux-based OS
```
python -m venv venv
source venv/bin/activate
```
For Windows OS
```
cd TECDIA-Submission/python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Create a Directory for Videos
```
mkdir videos
```
*And store your video in this directory*

### 4. Create a Directory for Resultant Videos
```
mkdir results
```

*main.py will generate the resultant video here*

### 5. Fix File name in main.py

In the line `video_path = "videos/jumbled_video.mp4"` on `line number 111`

Change `jumbled_video.mp4` with the name of the video to be reconstructed.

### 6. Create the Reconstructed Video
For Linux-based OS
```
python3 main.py
```
For Windows OS
```
python main.py
```
### 7. View the Reconstructed Video

You can either navigate to the `results` directory and manually play the video or use the following command:

For Linux-based OS
```
cvlc results/reconstructed.mp4  
```
For Windows OS
```
vlc results\reconstructed.mp4
```
### 8. Change Direction (Optional)
If the observed direction of the moving object is the opposite of what it's suposed to be, change line 127 from:
```
order_full = np.argsort(cx_full)[::-1]
```
To the following code:
```
order_full = np.argsort(cx_full)
```
This changes the frame order and effectively changes the order in which the video is reconstructed. Now, visit `Step 6` again.
## Algorithm Explanation
### 1. Approach Overview

The algorithm reconstructs the temporal order of frames in a jumbled 10-second video by analyzing the motion of the primary moving object (e.g., a person walking).  
The assumption is that the background remains static while the object moves in a consistent direction (e.g., right to left).

The process is divided into two main phases:

1. `Global Centroid Ordering (Parallelized)`: Orders all frames based on the horizontal centroid of the moving object, now computed concurrently across multiple threads.

2. `Refined Tail Correction`: Reprocesses the final portion of the video (where the object becomes small and centroid detection becomes unstable) by zooming in and recalculating centroids in parallel.

---

### 2. Techniques Used
**<ins>Centroid Tracking (Multithreaded)</ins>**

For each frame:
- A background image is computed using the median of all frames.  
- The difference between the frame and background isolates the moving object.  
- The binary mask of this object is used to compute its centroid (x-position).  
- Multiple centroid computations are now distributed across threads for faster processing.  
- Sorting the frames by centroid position yields the approximate temporal sequence.

Mathematically:

$$
cx_i = \frac{M_{10}}{M_{00}}
$$

**where 𝑀<sub>10</sub> and 𝑀<sub>00</sub>** are image moments.

This method effectively reconstructs the movement when the object is large and clearly visible while leveraging parallel threads for efficiency.

---

**<ins>Tail Refinement via Zoom (Parallelized)</ins>**

As the object becomes smaller (e.g., further from the camera), centroid accuracy drops due to:

- Low pixel coverage  
- Increased background noise  

To counter this:

- The last 120 frames (approx. 4 seconds) of the already reconstructed sequence are zoomed by a factor of 1.8×.  
- Both zooming and centroid computation are executed in parallel across threads.  
- The new order is merged back into the first 180 stable frames.  
- This localized reprocessing helps stabilize small-object motion and reduce jitter with reduced processing time.

---

### 3. Why This Method Was Chosen

- **Simplicity and Interpretability:** Centroid tracking remains straightforward to understand and tune while being efficiently parallelized.  
- **Efficiency:** Thread-based centroid extraction reduces the total runtime while maintaining an overall O(n) complexity.  
- **Consistency:** The same background reference ensures stable centroid-based ordering even when computations run concurrently.  
- **Scalability:** The threaded approach scales efficiently with CPU cores, improving processing speed for longer or higher-resolution videos.  
- **Fallback-Resilient:** Missing centroid values are still interpolated linearly, ensuring reliable reconstruction.

---

### 4. Key Design Considerations

- **Accuracy:** Combined centroid-based and zoom-based analysis maintains temporal consistency even for small distant objects.  
- **Time Complexity:** O(n) per stage (centroid + sorting), with parallel execution reducing wall-clock time significantly.  
- **Parallelism:** Implemented using Python’s `ThreadPoolExecutor` for per-frame centroid and zoom operations, achieving ~1.5×–2× speedup.  
- **Stability:** Median background removes noise and illumination changes, ensuring robust motion detection.  
- **Directional Flexibility:** Simple reversal flag handles right-to-left or left-to-right motion.

---

### 5. Possible Enhancements

- Extending parallelism using **multiprocessing** for CPU-bound workloads on large videos.  
- Integrating optical flow for more precise motion estimation.  
- Using object tracking (e.g., Kalman filters or SORT) for complex multi-object scenes.  
- Adding temporal smoothing between consecutive centroids to further reduce local jitter.


## Execution Time Log
- System: `Ubuntu 22.04 (x86_64)`
- CPU: `Intel i5 12th Gen`
- RAM: `16 GB`
- Video Duration: `10 seconds (300 frames at 30 FPS)`

| Stage | Description | Time (seconds) |
|--------|-------------|----------------|
| Frame Extraction | Reading 300 frames | 2.1 s |
| Background Computation | Median background calculation | 1.8 s |
| Full Centroid Ordering (Parallel) | Primary reconstruction using 8 threads | 12.6 s |
| Tail Refinement (Parallel) | Zoom + centroid recomputation | 4.7 s |
| Saving Output Video | Writing reconstructed frames | 1.2 s |
| **Total Execution Time** |  | **≈ 22.4 s** |

**<ins>Note:</ins>** Execution time varies depending on resolution, system performance, and Python environment. The centroid-based reordering (Step 3) is the most computationally intensive stage due to pixel-level moment calculations on each frame.

## Output
After running the script successfully, the final reconstructed video will be available at ```results/reconstructed.mp4```

The reconstructed video shows:
- Smooth motion for the first 6 seconds.
- Minimal jitter in the final 4 seconds after refinement.

## Evaluated Output (jumbled_video.mp4)

You can view the **reconstructed video output** here:  
[🎬 View Reconstructed Video on OneDrive](https://drive.google.com/file/d/1j-hY36c38bN8NGFFVm7iDTiZXx5PH7RS/view?usp=sharing)

> **Note:**  
> The video demonstrates the corrected temporal order after applying centroid-based and zoom-refined reconstruction on `jumbled_video.mp4` provided for evaluation.


## Originality Statement
This project and its accompanying algorithm were independently developed as part of an experimental approach to reconstructing temporally jumbled video frames.
The implementation is based on a custom heuristic pipeline combining background subtraction, centroid-based motion analysis, and localized refinement through region zooming.

All code, methodology, and documentation have been authored originally for this work, without replication of any existing public or academic source.
Standard open-source libraries such as `OpenCV` and `NumPy` were used solely for computational and image-processing support.

Any resemblance to previously published methods or open repositories is purely coincidental and limited to the use of commonly available image-processing functions.

## Author
**Dibyajyoti Chakrabarti, VIT Vellore (October 2025)**