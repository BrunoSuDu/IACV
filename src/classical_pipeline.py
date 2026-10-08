"""普通 OpenCV 特征流程：检测 → 描述 → 匹配。"""
import cv2
import numpy as np

from descriptors import create_descriptor
from detectors import create_detector
from features import ImageFeatures
from matching import match_descriptors
from timing import measure


def extract_classical_features(image_gray, config, warmup=2, repetitions=5, max_keypoints=2048):
    # ORB 自带默认 500 点上限；调整到共同预算。其余方法先检测再筛选。
    params = {"nfeatures": max_keypoints} if config.detector == "orb" and max_keypoints else None
    detector = create_detector(config.detector, params)
    descriptor = create_descriptor(config.descriptor)

    def detect():
        keypoints = list(detector.detect(image_gray, None))
        count_before_limit = len(keypoints)
        if max_keypoints:
            keypoints = sorted(keypoints, key=lambda point: point.response, reverse=True)[:max_keypoints]
        return keypoints, count_before_limit

    (detected, count_before_limit), detection_ms, detection_std = measure(detect, warmup, repetitions)
    (keypoints, values), description_ms, description_std = measure(
        lambda: descriptor.compute(image_gray, detected), warmup, repetitions
    )
    if values is None:
        dtype = np.uint8 if descriptor.descriptorType() == cv2.CV_8U else np.float32
        values = np.empty((0, descriptor.descriptorSize()), dtype=dtype)
    timings = {
        "detection_ms": detection_ms,
        "detection_std_ms": detection_std,
        "description_ms": description_ms,
        "description_std_ms": description_std,
        "joint_extraction_ms": np.nan,
        "joint_extraction_std_ms": np.nan,
    }
    return ImageFeatures(list(detected), list(keypoints or []), values, timings,
                         detected_before_limit=count_before_limit)


def match_classical_features(features1, features2, config, ratio_threshold=0.8,
                             warmup=2, repetitions=5, matching_strategy="ratio"):
    def operation():
        matches, _ = match_descriptors(
            features1.descriptors, features2.descriptors, config.norm, ratio_threshold,
            matching_strategy=matching_strategy,
        )
        return matches
    matches, mean_ms, std_ms = measure(operation, warmup, repetitions)
    return matches, {"matching_ms": mean_ms, "matching_std_ms": std_ms}


def run_classical_pair(image1_gray, image2_gray, config, ratio_threshold=0.8,
                       warmup=2, repetitions=5, max_keypoints=2048, matching_strategy="ratio"):
    features1 = extract_classical_features(image1_gray, config, warmup, repetitions, max_keypoints)
    features2 = extract_classical_features(image2_gray, config, warmup, repetitions, max_keypoints)
    matches, timings = match_classical_features(
        features1, features2, config, ratio_threshold, warmup, repetitions, matching_strategy
    )
    return features1, features2, matches, timings
