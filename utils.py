# utils.py
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
import torch
import torchvision.models as models
import torchvision.transforms as T
from scipy.sparse import csgraph
from scipy.linalg import eigh
from tqdm import tqdm

def compute_background_median(frames, subsample=1):
    # frames: list of BGR numpy arrays
    # convert to RGB or keep BGR — median is per-channel
    # subsample helps memory: use every k-th frame for median if desired
    sampled = frames[::subsample]
    stack = np.stack(sampled, axis=0).astype(np.uint8)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median

def foreground_mask(frame, background, blur_kernel=5, min_area=50):
    # returns binary mask of moving regions
    # convert to gray, subtract background, threshold
    fg = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    # Otsu threshold
    try:
        th = threshold_otsu(gray)
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    except Exception:
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    # morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # remove small blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean

def compute_centroids(frames, background):
    centroids = []
    masks = []
    for f in frames:
        m = foreground_mask(f, background)
        masks.append(m)
        M = cv2.moments(m)
        if M["m00"] != 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
            centroids.append((cx, cy))
        else:
            centroids.append((np.nan, np.nan))
    centroids = np.array(centroids)  # (N,2)
    return centroids, masks

def monotonicity_score(values, window=3):
    # measure how monotonic the sequence is: fraction of adjacent pairs matching trend
    diffs = np.diff(values)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return 0.0
    # compute sign consistency
    signs = np.sign(diffs)
    # count nonzero sign
    nz = signs != 0
    if nz.sum() == 0:
        return 1.0
    score = (np.sum(signs[nz] == signs[nz][0]) / len(signs[nz]))  # fraction matching initial direction
    # also return mean absolute monotonicity
    return float(score)

# -----------------------
# Deep feature extraction (fallback)
# -----------------------
def build_resnet_feature_extractor(device='cpu'):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # remove final fc
    model.eval()
    model.to(device)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def extract_deep_features(frames, model, transform, device='cpu', batch=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            batch_frames = frames[i:i+batch]
            tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            x = torch.stack(tensors).to(device)
            out = model(x).cpu().numpy()
            feats.append(out)
    feats = np.vstack(feats)
    return feats

def spectral_seriation_from_similarity(sim):
    # sim: NxN similarity (higher = more similar)
    # Convert to dissimilarity and run spectral ordering (Fiedler vector)
    # Build Laplacian
    # Using symmetric sim; if not symmetric, symmetrize
    sim = (sim + sim.T) / 2.0
    # Convert to affinity (already similarity) but ensure positivity
    sim = np.maximum(sim, 0)
    deg = np.diag(sim.sum(axis=1))
    L = deg - sim
    # compute eigenvectors of L; second smallest eigenvector = Fiedler vector
    # use eigh for symmetric
    try:
        vals, vecs = eigh(L, eigvals=(1,1))
        fiedler = vecs[:,0]
    except Exception:
        # fallback to eigen decomposition of normalized laplacian from csgraph
        Lnorm = csgraph.laplacian(sim, normed=True)
        vals, vecs = eigh(Lnorm, eigvals=(1,1))
        fiedler = vecs[:,0]
    order = np.argsort(fiedler)
    return order

def greedy_path_from_dissimilarity(diss, start=None):
    # Simple greedy nearest neighbor on dissimilarity (lower = similar)
    N = diss.shape[0]
    visited = np.zeros(N, dtype=bool)
    if start is None:
        start = 0
    path = [start]
    visited[start] = True
    for _ in range(N-1):
        last = path[-1]
        candidates = np.where(~visited)[0]
        next_idx = candidates[np.argmin(diss[last, candidates])]
        path.append(next_idx)
        visited[next_idx] = True
    return np.array(path)

def make_order_by_similarity(frames, device='cpu'):
    # extract features
    model, transform = build_resnet_feature_extractor(device=device)
    feats = extract_deep_features(frames, model, transform, device=device, batch=32)
    feats_norm = normalize(feats, axis=1)
    sim = cosine_similarity(feats_norm)
    # spectral seriation
    order = spectral_seriation_from_similarity(sim)
    # refine by greedy path using dissimilarity
    diss = 1 - sim
    start = np.argmin(np.sum(diss, axis=1))  # most central
    path = greedy_path_from_dissimilarity(diss, start=start)
    # combine: choose whichever path yields higher internal similarity
    def path_score(p):
        return np.sum([sim[p[i], p[i+1]] for i in range(len(p)-1)])
    if path_score(path) > path_score(order):
        final = path
    else:
        final = order
    return final, sim