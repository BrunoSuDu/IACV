import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy


def save_metadata(options, pairs):
    metadata = {
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
        "ransac": {"max_iterations": 5000, "confidence": 0.995, "seed": options.seed},
    }
    (options.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_summary(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "part1_raw_results.csv", index=False)
    numeric = list(frame.select_dtypes(include=["number", "bool"]).columns)
    grouped = frame.groupby(["dataset", "method", "category"], sort=False)
    summary = grouped[numeric].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.copy()
    summary.insert(0, "pair_count", grouped.size())
    summary.reset_index().to_csv(output / "part1_summary.csv", index=False)
    costs = [name for name in summary.columns
             if "_ms" in name or "descriptor" in name or "keypoints" in name]
    summary[costs].reset_index().to_csv(output / "part1_computational_cost.csv", index=False)
