"""方法配置和各模块共同使用的简单数据结构。"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    detector: str
    descriptor: str
    norm: int
    matcher: str = "bf_ratio"


CONFIGS = {
    "sift": FeatureConfig("sift", "sift", "sift", cv2.NORM_L2),
    "orb": FeatureConfig("orb", "orb", "orb", cv2.NORM_HAMMING),
    "kaze": FeatureConfig("kaze", "kaze", "kaze", cv2.NORM_L2),
    "fast_brief": FeatureConfig("fast_brief", "fast", "brief", cv2.NORM_HAMMING),
    "fast_brisk": FeatureConfig("fast_brisk", "fast", "brisk", cv2.NORM_HAMMING),
    "fast_freak": FeatureConfig("fast_freak", "fast", "freak", cv2.NORM_HAMMING),
    "superpoint": FeatureConfig("superpoint", "superpoint", "superpoint", cv2.NORM_L2),
    "superpoint_lightglue": FeatureConfig(
        "superpoint_lightglue", "superpoint", "superpoint", cv2.NORM_L2, "lightglue"
    ),
}
CLASSICAL_METHODS = ["sift", "orb", "kaze", "fast_brief", "fast_brisk", "fast_freak"]


@dataclass
class ImageFeatures:
    # compute() 可能删除边缘上的关键点；保留检测结果，便于单独评价 detector。
    detected_keypoints: list
    keypoints: list
    descriptors: np.ndarray
    timings: dict
    native: dict = field(default_factory=dict)
    detected_before_limit: int | None = None


def descriptor_statistics(features, image_index):
    descriptors = features.descriptors
    suffix = f"image{image_index}"
    return {
        f"detected_before_limit_{suffix}": (
            features.detected_before_limit if features.detected_before_limit is not None
            else len(features.detected_keypoints)
        ),
        f"detected_keypoints_{suffix}": len(features.detected_keypoints),
        f"features_{suffix}_total": len(features.keypoints),
        f"descriptor_memory_bytes_{suffix}": descriptors.nbytes,
        "descriptor_dim": descriptors.shape[1],
        "descriptor_dtype": str(descriptors.dtype),
        "bytes_per_descriptor": descriptors.shape[1] * descriptors.dtype.itemsize,
    }
