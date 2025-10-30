# Jumbled Frames Reconstruction

## Overview
This program attempts to reconstruct the original ordering of frames from a shuffled video (10s, 30 fps). It first tries a very fast centroid-based method using a background median. If that fails, it falls back to a deep-feature similarity + spectral ordering method.

## Setup
1. Create a virtualenv (optional):
```
python -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Store video:
```
mkdir videos
```
***And store you video in this directory***
4. Test and tune:
```
python main.py --input videos/video_name.mp4 --output reconstructed.mp4 --fps 30
```