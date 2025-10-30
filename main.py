# main.py
import argparse
import numpy as np
import cv2
import time
from video_io import extract_frames, save_video
from utils import (compute_background_median, compute_centroids, monotonicity_score,
                   make_order_by_similarity)
import os

def try_centroid_ordering(frames, fps=30, visualize=False):
    bg = compute_background_median(frames, subsample=1)
    centroids, masks = compute_centroids(frames, bg)
    # choose main axis: compute variance to pick horizontal or vertical
    xs = centroids[:,0]
    ys = centroids[:,1]
    # if centroid NaNs occur, fill by previous/next interpolation
    nan_mask = np.isnan(xs)
    if nan_mask.any():
        # linear interpolation
        idx = np.arange(len(xs))
        good = ~nan_mask
        if good.sum() >= 2:
            xs[nan_mask] = np.interp(idx[nan_mask], idx[good], xs[good])
        else:
            xs[nan_mask] = np.nanmedian(xs) if not np.all(np.isnan(xs)) else 0
    nan_mask_y = np.isnan(ys)
    if nan_mask_y.any():
        idx = np.arange(len(ys))
        good = ~nan_mask_y
        if good.sum() >= 2:
            ys[nan_mask_y] = np.interp(idx[nan_mask_y], idx[good], ys[good])
        else:
            ys[nan_mask_y] = np.nanmedian(ys) if not np.all(np.isnan(ys)) else 0
    var_x = np.nanvar(xs)
    var_y = np.nanvar(ys)
    if var_x >= var_y:
        proj = xs
    else:
        proj = ys
    # attempt ordering by proj (either increasing or decreasing)
    order_inc = np.argsort(proj)
    order_dec = order_inc[::-1]
    # compute monotonicity scores for both directions by checking centroid trend
    # if real time sequence is increasing or decreasing should be consistent, check adjacency similarity
    def adjacency_consistency(ordr):
        # fraction of adjacent centroid differences that have consistent sign
        vals = proj[ordr]
        diffs = np.diff(vals)
        return np.mean(np.abs(diffs))  # larger is better (more movement)
    score_inc = adjacency_consistency(order_inc)
    score_dec = adjacency_consistency(order_dec)
    # choose the one with larger spread (movement)
    order = order_inc if score_inc >= score_dec else order_dec
    # compute a heuristic monotonicity score (how consistent direction is)
    mono_score = monotonicity_score(np.sort(proj) if order is order_inc else np.sort(proj)[::-1])
    return order, bg, masks, mono_score

def main(args):
    start_time = time.time()
    frames = extract_frames(args.input)
    N = len(frames)
    print(f"Loaded {N} frames.")

    # STEP 1: centroid-based ordering attempt
    order_centroid, bg, masks, mono_score = try_centroid_ordering(frames)
    print(f"Centroid monotonicity heuristic score: {mono_score:.3f}")

    accept_centroid = False
    # heuristic: if monotonicity score > 0.6 we accept centroid method
    if mono_score >= 0.2:
        accept_centroid = True
        final_order = order_centroid
        method = 'centroid'
        print("Accepting centroid-based ordering.")
    else:
        print("Centroid ordering not reliable, using similarity fallback.")
        # fallback: deep feature similarity & spectral ordering
        device = 'cuda' if (args.use_cuda and __import__('torch').cuda.is_available()) else 'cpu'
        print("Using device:", device)
        final_order, sim = make_order_by_similarity(frames, device=device)
        method = 'similarity'

    # build output frames in final_order
    ordered_frames = [frames[i] for i in final_order]
    
    save_video(ordered_frames, args.output, fps=args.fps)
    np.save("ordering.npy", final_order)
    elapsed = time.time() - start_time
    with open("timing.log", "a") as f:
        f.write(f"{args.input}, method={method}, time={elapsed:.2f}s, frames={N}\n")
    print(f"Saved reconstructed video to {args.output} using method {method}. Time: {elapsed:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jumbled_video.mp4")
    parser.add_argument("--output", type=str, default="reconstructed.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA for feature extraction if available")
    args = parser.parse_args()
    main(args)