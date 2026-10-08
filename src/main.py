"""命令行入口；算法实现见 classical_pipeline.py 等模块。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib

# 批量实验只保存图片，不依赖 Tk 等桌面 GUI 组件。
matplotlib.use("Agg")

from datasets.graf import iter_graf_pairs
from datasets.hpatches import iter_hpatches_pairs
from experiments.runner import run_experiment
from features import CLASSICAL_METHODS, CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Part 1: feature matching evaluation")
    parser.add_argument("--dataset", choices=["graf", "hpatches"], default="graf")
    parser.add_argument("--data-root", type=Path, help="Dataset directory; defaults to data/<dataset>")
    parser.add_argument("--methods", nargs="+", choices=list(CONFIGS), default=CLASSICAL_METHODS)
    parser.add_argument("--sequences", nargs="+", help="Optional HPatches sequence names")
    parser.add_argument("--limit-pairs", type=int, help="Small smoke test only")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--correctness-threshold", type=float, default=3.0)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048,
                        help="Common feature cap; 0 keeps OpenCV defaults (classical only)")
    parser.add_argument("--figures", action="store_true", help="One pair per category and method")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.warmup < 0 or options.repetitions < 1 or options.threads < 1:
        parser.error("warmup >= 0, repetitions >= 1 and threads >= 1 required")
    if not 0 < options.ratio < 1 or min(options.correctness_threshold, options.ransac_threshold) <= 0:
        parser.error("ratio must be in (0, 1); geometric thresholds must be positive")
    if options.limit_pairs is not None and options.limit_pairs < 1:
        parser.error("limit-pairs must be positive")
    if options.max_keypoints < 0:
        parser.error("max-keypoints must be >= 0")
    if options.max_keypoints == 0 and any("superpoint" in m for m in options.methods):
        parser.error("Use a positive max-keypoints value for learned methods")
    options.data_root = options.data_root or PROJECT_ROOT / "data" / options.dataset
    options.output = options.output or PROJECT_ROOT / "results" / options.dataset
    cv2.setNumThreads(options.threads)
    cv2.setRNGSeed(options.seed)
    np.random.seed(options.seed)
    if options.dataset == "graf":
        pairs = list(iter_graf_pairs(options.data_root))
    else:
        pairs = list(iter_hpatches_pairs(options.data_root, options.sequences))
    if options.limit_pairs:
        pairs = pairs[:options.limit_pairs]
    run_experiment(pairs, options)


if __name__ == "__main__":
    main()
