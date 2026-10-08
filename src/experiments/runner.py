"""每对图像依次完成 matching、GT evaluation 和 RANSAC。"""
import csv

import numpy as np

from classical_pipeline import extract_classical_features, match_classical_features
from evaluation import (count_ground_truth_correspondences, evaluate_feature_matching,
                        get_covisible_feature_mask, load_homography, project_keypoints)
from experiments.reporting import save_metadata, save_summary
from features import CONFIGS, descriptor_statistics
from homography import estimate_homography, homography_accuracy
from image_io import load_image
from visualization import save_keypoint_visualization, save_match_visualization


def evaluate_pair(pair, features1, features2, matches, matching_times, shape1, shape2, options):
    ground_truth = load_homography(pair.homography)
    evaluation, evaluated, correct, incorrect, _ = evaluate_feature_matching(
        features1.keypoints, features2.keypoints, matches, ground_truth, shape2,
        options.correctness_threshold,
    )
    # 检测阶段的 repeatability 另算；匹配指标继续使用描述子保留的点。
    _, projected = project_keypoints(features1.detected_keypoints, ground_truth)
    visible = get_covisible_feature_mask(projected, shape2)
    detector_correspondences = count_ground_truth_correspondences(
        projected, features2.detected_keypoints, visible, options.correctness_threshold
    )
    estimated, inliers, homography_metrics = estimate_homography(
        features1.keypoints, features2.keypoints, matches, options.ransac_threshold, options.seed
    )
    row = {
        "dataset": pair.dataset, "sequence": pair.sequence, "category": pair.category,
        "pair": pair.name, "reference_index": 1, "target_index": pair.target_index,
        "image1": str(pair.image1), "image2": str(pair.image2),
        "raw_putative_matches": len(matches),
        "detector_n_features": int(visible.sum()),
        "detector_n_correspondences": detector_correspondences,
        "detector_repeatability": detector_correspondences / visible.sum() if visible.any() else np.nan,
        "ransac_threshold_px": options.ransac_threshold,
        "homography_error_px": homography_accuracy(estimated, ground_truth, shape1),
        **evaluation, **homography_metrics, **matching_times,
        **descriptor_statistics(features1, 1), **descriptor_statistics(features2, 2),
    }
    for index, features in enumerate((features1, features2), start=1):
        for name, value in features.timings.items():
            row[f"{name}_image{index}"] = value
    selections = {"putative": evaluated, "correct": correct,
                  "incorrect": incorrect, "ransac_inliers": inliers}
    return row, selections


def save_figures(image1, image2, features1, features2, selections, folder, title):
    for index, (image, features) in enumerate(((image1, features1), (image2, features2)), 1):
        save_keypoint_visualization(image, features.keypoints,
                                    f"{title}: image {index} descriptor-retained keypoints",
                                    folder / f"keypoints{index}.png")
    for label, matches in selections.items():
        save_match_visualization(image1, features1.keypoints, image2, features2.keypoints,
                                 matches, f"{title}: {label} ({len(matches)} total; up to 100 shown)",
                                 folder / f"{label}.png")


def run_experiment(pairs, options):
    options.output.mkdir(parents=True, exist_ok=True)
    save_metadata(options, pairs)
    rows = []
    # 一边跑一边保存，意外中断时已经完成的 pair 仍然可检查。
    with (options.output / "progress.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for method in options.methods:
            config = CONFIGS[method]
            learned = None
            if config.detector == "superpoint":
                from learned_features import LearnedPipeline
                learned = LearnedPipeline(config, options.device, options.max_keypoints,
                                          options.output / "models", options.seed)
            reference_path = None
            figures_saved = set()
            for pair_index, pair in enumerate(pairs, 1):
                # 每个序列的 image 1 只提取/计时一次；缓存不跨方法。
                if pair.image1 != reference_path:
                    image1, gray1 = load_image(pair.image1)
                    if learned:
                        features1 = learned.extract(gray1, options.warmup, options.repetitions)
                    else:
                        features1 = extract_classical_features(gray1, config, options.warmup,
                                                                options.repetitions, options.max_keypoints)
                    reference_path = pair.image1
                image2, gray2 = load_image(pair.image2)
                if learned:
                    features2 = learned.extract(gray2, options.warmup, options.repetitions)
                    matches, times = learned.match(features1, features2, options.ratio,
                                                   options.warmup, options.repetitions)
                else:
                    features2 = extract_classical_features(gray2, config, options.warmup,
                                                           options.repetitions, options.max_keypoints)
                    matches, times = match_classical_features(
                        features1, features2, config, options.ratio, options.warmup, options.repetitions
                    )
                row, selections = evaluate_pair(pair, features1, features2, matches, times,
                                                gray1.shape, gray2.shape, options)
                row.update({"method": method, "detector": config.detector,
                            "descriptor": config.descriptor, "matcher": config.matcher,
                            "ratio_threshold": options.ratio if config.matcher == "bf_ratio" else np.nan})
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                if options.figures and pair.category not in figures_saved:
                    save_figures(image1, image2, features1, features2, selections,
                                 options.output / "figures" / method / pair.name, f"{method}: {pair.name}")
                    figures_saved.add(pair.category)
                print(f"{method} [{pair_index}/{len(pairs)}] {pair.name}: "
                      f"correct={row['n_correct']}/{row['n_putative']}, "
                      f"H error={row['homography_error_px']:.2f}px", flush=True)
            save_summary(rows, options.output)
    return rows
