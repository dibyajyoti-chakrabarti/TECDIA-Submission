# Jumbled Frames Reconstruction

## Overview

This project performs **video frame reordering** to reconstruct the original temporal sequence from a jumbled video.  
It assumes the input video contains a **single moving object** (for example, a walking person) against a **static background**.

The reconstruction is based on analyzing the **horizontal movement** of the object across frames using **centroid tracking**.

## Note

The entire project was created and run on a Linux OS and as such, some commands might vary according to the host OS. If you are using Windows, please verify beforehand if the commands are Windows compatible. You can either use ChatGPT or any other AI tools for doing this.

## Cloning the repository
### 1. Create a project folder
```
mkdir project
cd project
```
### 2. Get git in the project directory
```
git init
```
### 3. Clone the directory
```
git clone https://github.com/dibyajyoti-chakrabarti/reconstruct_frames.git
```
### 4. Move to project directory 
```
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

In the line `frames = extract_frames("videos/jumbled_video.mp4")`

Change `jumbled_video.mp4` with the name of the video to be reconstructed.

### 6. Create the Reconstructed Video
```
python3 main.py
```
### 7. View the Reconstructed Video

You can either navigate to the `results` directory and manually play the video or use the following command:
```
cvlc videos/jumbled_video.mp4  
```