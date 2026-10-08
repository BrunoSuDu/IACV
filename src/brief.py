"""教学版 BRIEF：固定采样点，比较灰度，打包成二进制描述子。

参考 Calonder et al. (2012), Section 3.2 的 uniform / Gaussian 思路。
这是简化实现，不声称逐位复现 OpenCV BRIEF；不做尺度或方向归一化。
"""
import cv2
import numpy as np


def make_brief_pattern(strategy, seed=0, bits=256, patch_size=31):
    if bits <= 0 or bits % 8 or patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("bits must be a positive multiple of 8; patch_size must be odd and >= 3")
    rng = np.random.default_rng(seed)
    radius = patch_size // 2
    if strategy == "uniform":
        pattern = rng.integers(-radius, radius + 1, size=(bits, 2, 2))
    elif strategy == "gaussian":
        pattern = rng.normal(0, patch_size / 5, size=(bits, 2, 2))
        pattern = np.clip(np.rint(pattern), -radius, radius).astype(int)
    else:
        raise ValueError(f"Unknown BRIEF sampling strategy: {strategy}")
    return pattern


def compute_brief(image_gray, keypoints, pattern, patch_size=31):
    # 留出平滑核的边界，两种采样策略使用完全相同的有效区域。
    margin = patch_size // 2 + 3
    height, width = image_gray.shape
    kept, centers = [], []
    for keypoint in keypoints:
        x, y = np.rint(keypoint.pt).astype(int)
        if margin <= x < width - margin and margin <= y < height - margin:
            kept.append(keypoint)
            centers.append((x, y))
    if not kept:
        return [], np.empty((0, len(pattern) // 8), dtype=np.uint8)
    smooth = cv2.GaussianBlur(image_gray, (7, 7), np.sqrt(2))
    descriptors = []
    # 每个关键点做 256 次灰度比较；< 为 1，否则为 0。
    for x, y in centers:
        first = smooth[y + pattern[:, 0, 1], x + pattern[:, 0, 0]]
        second = smooth[y + pattern[:, 1, 1], x + pattern[:, 1, 0]]
        descriptors.append(np.packbits(first < second))
    return kept, np.asarray(descriptors, dtype=np.uint8)
