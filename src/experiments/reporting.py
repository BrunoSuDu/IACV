import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy


def save_metadata(options, pairs, extra=None):
    from experiments.protocol import PROTOCOL_VERSION, code_identity
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(options).items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "opencv_threads": cv2.getNumThreads(), "pair_count": len(pairs),
        "image_resize": None,
        "classical_parameters": "OpenCV defaults except ORB nfeatures=cap when cap>0; top response cap",
        "feature_cap": options.max_keypoints,
        "ratio_test": "best < ratio * second_best; greedy one-to-one by distance",
        "evaluation_features": "descriptor-retained query keypoints projected inside target",
        "detector_repeatability": "separately evaluated before descriptor border filtering",
        "undefined_ratios": "NaN; aggregation reports valid sample counts",
        "timing_std": "population std of repetitions; summary std is between pairs (sample std)",
        "ransac": {"max_iterations": options.ransac_max_iters, "confidence": options.ransac_confidence, "seed": options.seed},
        "code_sha256": code_identity(),
        "timing_sharing": "cached reference and shared feature-source extraction; repeated row timings are not independent measurements",
        "classical_defaults": {
            "sift": {"contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6, "nOctaveLayers": 3},
            "orb": {"nfeatures": options.max_keypoints, "scaleFactor": 1.2, "nlevels": 8, "WTA_K": 2,
                    "edgeThreshold": 31, "patchSize": 31, "fastThreshold": 20, "scoreType": "HARRIS_SCORE"},
            "kaze": {"threshold": 0.001, "nOctaves": 4, "nOctaveLayers": 4, "extended": False,
                     "upright": False, "diffusivity": "DIFF_PM_G2"},
            "fast": {"threshold": 10, "nonmaxSuppression": True, "type": "TYPE_9_16"},
            "brief": {"bytes": 32, "use_orientation": False},
            "brisk": {"thresh": 30, "octaves": 3, "patternScale": 1.0},
            "freak": {"orientationNormalized": True, "scaleNormalized": True, "patternScale": 22.0, "nOctaves": 4}},
    }
    metadata.update(extra or {})
    (options.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_summary(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "part1_raw_results.csv", index=False)
    numeric = list(frame.select_dtypes(include=["number", "bool"]).columns)
    keys = ["dataset", "method", "matching_strategy", "category"]
    if "transformation_type" in frame:
        keys = ["dataset", "method", "matching_strategy", "transformation_type", "transformation_strength"]
    grouped = frame.groupby(keys, sort=False)
    numeric = [name for name in numeric if name not in keys]
    summary = grouped[numeric].agg(["mean", "median", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.copy()
    summary.insert(0, "pair_count", grouped.size())
    summary.reset_index().to_csv(output / "part1_summary.csv", index=False)
    costs = [name for name in summary.columns
             if "_ms" in name or "descriptor" in name or "keypoints" in name]
    summary[costs].reset_index().to_csv(output / "part1_computational_cost.csv", index=False)
