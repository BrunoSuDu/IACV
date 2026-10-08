"""Common post-extraction mask policy with synchronized feature/native indexing."""
import hashlib

import cv2
import numpy as np

from features import ImageFeatures
from transformations import erode_valid_mask


MASK_POLICY = {"minimum_margin_px": 48, "size_multiplier": 3.0,
               "rule": "distance to invalid pixel > max(48, 3*OpenCV keypoint size)",
               "stage": "post extraction, after feature cap; no replenishment",
               "limitation": "finite support heuristic; padding may affect scale space, CNN and top-k"}


def support_keep(keypoints, valid_mask):
    # Pad with invalid pixels so outer canvas edges also count as boundaries.
    padded = np.pad(valid_mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    keep = []
    for kp in keypoints:
        x, y = np.rint(kp.pt).astype(int)
        radius = max(MASK_POLICY["minimum_margin_px"], MASK_POLICY["size_multiplier"] * kp.size)
        keep.append(0 <= y < distance.shape[0] and 0 <= x < distance.shape[1]
                    and distance[y, x] > radius)
    return np.asarray(keep, bool)


def filter_features(features, valid_mask):
    detected_keep = support_keep(features.detected_keypoints, valid_mask)
    keep = support_keep(features.keypoints, valid_mask)
    indices = np.flatnonzero(keep)
    native = {}
    feature_fields = {"keypoints", "descriptors", "keypoint_scores", "scales", "oris", "native_indices"}
    for key, value in features.native.items():
        if key in feature_fields:
            if value.ndim < 2 or value.shape[1] != len(keep):
                raise ValueError(f"Misaligned native field {key}: {value.shape}")
            # Torch tensors and NumPy arrays both support index selection here.
            if hasattr(value, "index_select"):
                import torch
                value = value.index_select(1, torch.as_tensor(indices, device=value.device))
            else:
                value = value[:, indices]
        native[key] = value
    return ImageFeatures([kp for kp, k in zip(features.detected_keypoints, detected_keep) if k],
                         [kp for kp, k in zip(features.keypoints, keep) if k],
                         features.descriptors[keep], dict(features.timings), native,
                         features.detected_before_limit)


def evaluation_mask(valid_mask):
    return erode_valid_mask(valid_mask, MASK_POLICY["minimum_margin_px"])


def feature_fingerprint(features):
    """Content digest verifies shared extraction without claiming independent timing."""
    digest = hashlib.sha256()
    for points in (features.detected_keypoints, features.keypoints):
        values = np.array([[*kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id]
                           for kp in points], np.float64).reshape(-1, 7)
        digest.update(values.tobytes())
    digest.update(str(features.descriptors.shape).encode())
    digest.update(str(features.descriptors.dtype).encode())
    digest.update(features.descriptors.tobytes())
    for name in ("keypoints", "descriptors", "keypoint_scores", "image_size", "scales", "oris", "native_indices"):
        if name in features.native:
            value = features.native[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            digest.update(name.encode())
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()
