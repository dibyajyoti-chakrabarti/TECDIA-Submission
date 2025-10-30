# Jumbled Frames Reconstruction

## Overview

This project performs **video frame reordering** to reconstruct the original temporal sequence from a jumbled video.  
It assumes the input video contains a **single moving object** (for example, a walking person) against a **static background**.

The reconstruction is based on analyzing the **horizontal movement** of the object across frames using **centroid tracking**.


## Setup
1. Create a virtualenv:
```
python -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a Directory for Videos:
```
mkdir videos
```
**And store you video in this directory**

4. Create a Directory for Resultant Videos:
```
mkdir results
```

**main.py will generate the resultant video here**

5. Create the Reconstructed Video:
```
python3 main.py
```