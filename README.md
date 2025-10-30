# Jumbled Frames Reconstruction

## Overview
This program attempts to reconstruct the original ordering of frames from a shuffled video (10s, 30 fps). It first tries a very fast centroid-based method using a background median. If that fails, it falls back to a deep-feature similarity + spectral ordering method.

## Setup
1. Create a virtualenv (optional):