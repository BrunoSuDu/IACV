"""FAST 参数与教学版 BRIEF 的小规模补充实验；不混入全量主表。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from brief import compute_brief, make_brief_pattern
from datasets.graf import iter_graf_pairs
from datasets.hpatches import iter_hpatches_pairs
from detectors import create_detector
from evaluation import (count_ground_truth_correspondences, evaluate_feature_matching,
                        get_covisible_feature_mask, load_homography, project_keypoints)
from image_io import load_image
from matching import match_descriptors
from timing import measure
from experiments.protocol import add_protocol_arguments, input_manifest, reserve_outputs, validate_options
from experiments.reporting import save_metadata


ROOT = Path(__file__).resolve().parents[2]


def detector_study(pairs, warmup, repetitions):
    rows = []
    settings = [("sift", {})]
    for threshold in (10, 20, 40):
        for nonmax in (True, False):
            settings.append(("fast", {"threshold": threshold, "nonmaxSuppression": nonmax}))
    for method, params in settings:
        detector = create_detector(method, params)
        for pair in pairs:
            _, image1 = load_image(pair.image1)
            _, image2 = load_image(pair.image2)
            first, ms1, std1 = measure(lambda: detector.detect(image1, None), warmup, repetitions)
            second, ms2, std2 = measure(lambda: detector.detect(image2, None), warmup, repetitions)
            _, projected = project_keypoints(first, load_homography(pair.homography))
            visible = get_covisible_feature_mask(projected, image2.shape)
            count = count_ground_truth_correspondences(projected, second, visible)
            rows.append({"method": method, "threshold": params.get("threshold", np.nan),
                         "nonmax": params.get("nonmaxSuppression", np.nan),
                         "dataset": pair.dataset, "pair": pair.name, "category": pair.category,
                         "keypoints1": len(first), "keypoints2": len(second),
                         "n_features": int(visible.sum()), "n_correspondences": count,
                         "repeatability": count / visible.sum() if visible.any() else np.nan,
                         "detection_ms_image1": ms1, "detection_ms_image2": ms2,
                         "detection_std_ms_image1": std1, "detection_std_ms_image2": std2})
    return pd.DataFrame(rows)


def brief_study(pairs, warmup, repetitions, seed=0):
    rows = []
    detector = create_detector("fast")
    for strategy in ("uniform", "gaussian"):
        pattern = make_brief_pattern(strategy, seed=seed)
        for pair in pairs:
            _, image1 = load_image(pair.image1)
            _, image2 = load_image(pair.image2)
            detected1 = detector.detect(image1, None)
            detected2 = detector.detect(image2, None)
            (first, desc1), ms1, std1 = measure(
                lambda: compute_brief(image1, detected1, pattern), warmup, repetitions
            )
            (second, desc2), ms2, std2 = measure(
                lambda: compute_brief(image2, detected2, pattern), warmup, repetitions
            )
            (matches, _), match_ms, match_std = measure(
                lambda: match_descriptors(desc1, desc2, cv2.NORM_HAMMING), warmup, repetitions
            )
            metrics = evaluate_feature_matching(first, second, matches,
                                               load_homography(pair.homography), image2.shape)[0]
            rows.append({"strategy": strategy, "dataset": pair.dataset, "pair": pair.name,
                         "category": pair.category, "description_ms_image1": ms1,
                         "description_ms_image2": ms2, "description_std_ms_image1": std1,
                         "description_std_ms_image2": std2, "matching_ms": match_ms,
                         "matching_std_ms": match_std, **metrics})
    return pd.DataFrame(rows)


def run_supplementary(args):
    pairs = list(iter_graf_pairs(args.graf_root))
    pairs += list(iter_hpatches_pairs(args.hpatches_root, ["i_ajuntament", "v_adam"]))
    output = args.output
    reserve_outputs([output])
    metadata = {"pairs": [p.name for p in pairs], "brief_seed": args.seed, "brief_bits": 256,
                "input_manifest": input_manifest([path for pair in pairs for path in (pair.image1, pair.image2, pair.homography)]),
                "brief_patch_size": 31, "brief_blur": "7x7, sigma=sqrt(2)",
                "brief_gaussian": "sigma=31/5; rounded and clamped to patch bounds",
                "ratio": 0.8, "correctness_threshold_px": 3,
                "scope": "12 selected pairs; not a full HPatches benchmark"}
    save_metadata(args, pairs, metadata)
    detector_study(pairs, args.warmup, args.repetitions).to_csv(output / "fast_parameters.csv", index=False)
    brief_study(pairs, args.warmup, args.repetitions, args.seed).to_csv(output / "brief_sampling.csv", index=False)
    (output / "complete.json").write_text(json.dumps({"fast_records": 7*len(pairs), "brief_records": 2*len(pairs)}), encoding="utf-8")
    print(f"Supplementary results saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument("--graf-root", type=Path, default=ROOT / "data/graf")
    parser.add_argument("--hpatches-root", type=Path, default=ROOT / "data/hpatches")
    parser.add_argument("--output", type=Path, default=ROOT / "results/supplementary")
    args = parser.parse_args()
    validate_options(args)
    # This fixed supplementary protocol intentionally has no threshold sweeps.
    if args.ratio != 0.8 or args.correctness_threshold != 3.0:
        parser.error("Supplementary protocol fixes ratio=0.8 and correctness=3px")
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    run_supplementary(args)


if __name__ == "__main__":
    main()
