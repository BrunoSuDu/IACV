"""Shared feature extraction, independently timed matching and GT-only evaluation."""
import csv
import json
import uuid
from contextlib import ExitStack
from copy import copy

import numpy as np
import cv2

from classical_pipeline import extract_classical_features, match_classical_features
from evaluation import (count_ground_truth_correspondences, covisible_pair_masks,
                        evaluate_feature_matching, load_homography)
from experiments.protocol import input_manifest, reserve_outputs, result_folder, validate_pair_inventory
from experiments.reporting import save_metadata, save_summary
from feature_masks import feature_fingerprint
from features import LEARNED_MATCHERS, CONFIGS, CONTROL_CONFIG, descriptor_statistics, strategy_name
from homography import estimate_homography, homography_accuracy
from image_io import load_image
from visualization import save_keypoint_visualization, save_match_visualization


def evaluate_pair(pair, features1, features2, matches, matching_times, shape1, shape2, options,
                  source_valid_mask=None, target_valid_mask=None):
    ground_truth = pair.homography if isinstance(pair.homography, np.ndarray) else load_homography(pair.homography)
    evaluation, evaluated, correct, incorrect, _ = evaluate_feature_matching(
        features1.keypoints, features2.keypoints, matches, ground_truth, shape2,
        options.correctness_threshold, source_valid_mask, target_valid_mask)
    projected, visible, target_visible = covisible_pair_masks(
        features1.detected_keypoints, features2.detected_keypoints, ground_truth, shape2,
        source_valid_mask, target_valid_mask)
    detector_correspondences = count_ground_truth_correspondences(
        projected, [kp for kp, keep in zip(features2.detected_keypoints, target_visible) if keep],
        visible, options.correctness_threshold)
    # Estimator sees raw descriptor matches, no GT H or GT co-visibility filters.
    estimated, inliers, homography_metrics = estimate_homography(
        features1.keypoints, features2.keypoints, matches, options.ransac_threshold, options.seed,
        options.ransac_max_iters, options.ransac_confidence)
    row = {
        "dataset": pair.dataset, "sequence": pair.sequence, "category": pair.category,
        "pair": pair.name, "reference_index": 1, "target_index": pair.target_index,
        "image1": str(pair.image1), "image2": str(pair.image2),
        "height_image1": shape1[0], "width_image1": shape1[1],
        "height_image2": shape2[0], "width_image2": shape2[1],
        "raw_putative_matches": len(matches),
        "detector_n_features": int(visible.sum()), "detector_n_correspondences": detector_correspondences,
        "detector_repeatability": detector_correspondences / visible.sum() if visible.any() else np.nan,
        "ransac_threshold_px": options.ransac_threshold,
        "homography_error_px": homography_accuracy(estimated, ground_truth, shape1),
        "H_gt": json.dumps(ground_truth.tolist()),
        **evaluation, **homography_metrics, **matching_times,
        **descriptor_statistics(features1, 1), **descriptor_statistics(features2, 2),
    }
    row["feature_repeatability"] = row["repeatability"]
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


def feature_groups(methods, include_control=False):
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Require unique nonempty methods")
    groups = {}
    for method in methods:
        config = CONFIGS[method]
        # Compatible SIFT and baseline SIFT intentionally have different source keys.
        source = "superpoint" if config.detector == "superpoint" else config.name
        if config.detector == "sift_compatible":
            source = "sift_compatible"
        groups.setdefault(source, []).append(config)
    if include_control:
        if "sift_compatible" not in groups:
            raise ValueError("SIFT-compatible BF control requires sift_lightglue")
        groups["sift_compatible"].append(CONTROL_CONFIG)
    return groups


def config_strategies(config, options):
    if config.matcher in LEARNED_MATCHERS:
        return [config.matcher]
    if config.name == CONTROL_CONFIG.name:
        return ["ratio", "crosscheck"]
    return list(getattr(options, "bf_strategies", None) or [getattr(options, "matching_strategy", None) or "ratio"])


def record_configuration(row, config, strategy, options):
    row.update({"method": config.name, "detector": config.detector, "descriptor": config.descriptor,
                "matcher": strategy if strategy in LEARNED_MATCHERS else "bf",
                "matching_strategy": strategy_name(config, strategy),
                "distance_metric": "learned_confidence" if strategy in LEARNED_MATCHERS else (
                    "Hamming" if config.norm == cv2.NORM_HAMMING else "L2"),
                "ratio_threshold": options.ratio if strategy == "ratio" else np.nan,
                "feature_configuration": config.detector + "+" + config.descriptor,
                "is_primary": config.name != CONTROL_CONFIG.name,
                "extraction_device": options.device if config.detector == "superpoint" else "cpu",
                "matching_device": options.device if strategy in LEARNED_MATCHERS else "cpu"})


def extract_image(gray, config, pipeline, options):
    if pipeline is not None:
        return pipeline.extract(gray, options.warmup, options.repetitions)
    return extract_classical_features(gray, config, options.warmup, options.repetitions, options.max_keypoints)


def match_features(first, second, config, strategy, pipeline, options):
    if strategy in LEARNED_MATCHERS:
        return pipeline.match(first, second, options.ratio, options.warmup,
                              options.repetitions, matching_strategy=strategy)
    return match_classical_features(first, second, config, options.ratio, options.warmup,
                                    options.repetitions, matching_strategy=strategy)


def make_pipeline(config, options):
    if config.detector in ("superpoint", "sift_compatible"):
        from learned_features import LearnedPipeline
        return LearnedPipeline(config, options.device, options.max_keypoints, seed=options.seed)
    return None


def planned_outputs(pairs, options):
    groups = feature_groups(options.methods, getattr(options, "sift_control", False))
    routes = {}
    for configs in groups.values():
        for config in configs:
            for strategy in config_strategies(config, options):
                strategy_name(config, strategy)
                output = result_folder(options.results_root, pairs[0].dataset, config.name, strategy) if getattr(options, "routed", False) else options.output
                routes[(config.name, strategy)] = output
    return groups, routes


def run_experiment(pairs, options):
    validate_pair_inventory(pairs, pairs[0].dataset if pairs else "", getattr(options, "formal", False))
    groups, routes = planned_outputs(pairs, options)
    reserve_outputs(routes.values())
    run_id = uuid.uuid4().hex
    inputs = input_manifest([path for pair in pairs for path in (pair.image1, pair.image2, pair.homography)])
    rows_by_output = {folder: [] for folder in routes.values()}
    for folder in rows_by_output:
        local = copy(options)
        local.output = folder
        combinations = [(method, strategy) for (method, strategy), target in routes.items() if target == folder]
        save_metadata(local, pairs, {"run_id": run_id, "input_manifest": inputs,
                                     "combinations": combinations, "formal": getattr(options, "formal", False),
                                     "expected_records": len(pairs) * len(combinations)})
    with ExitStack() as stack:
        streams = {folder: stack.enter_context((folder / "progress.csv").open("w", newline="", encoding="utf-8"))
                   for folder in rows_by_output}
        writers = {}
        for source, configs in groups.items():
            extraction_config = configs[0]
            pipeline = make_pipeline(extraction_config, options)
            if pipeline is not None:
                for matcher in sorted({config.matcher for config in configs} & set(LEARNED_MATCHERS)):
                    pipeline.ensure_matcher(matcher)
                for config in configs:
                    for strategy in config_strategies(config, options):
                        pipeline.save_manifest(routes[(config.name, strategy)] / "models", config.name)
            reference_path = None
            figures_saved = set()
            for pair_index, pair in enumerate(pairs, 1):
                cached = pair.image1 == reference_path
                if not cached:
                    image1, gray1 = load_image(pair.image1)
                    first = extract_image(gray1, extraction_config, pipeline, options)
                    first_hash = feature_fingerprint(first)
                    reference_path = pair.image1
                image2, gray2 = load_image(pair.image2)
                second = extract_image(gray2, extraction_config, pipeline, options)
                second_hash = feature_fingerprint(second)
                for config in configs:
                    for strategy in config_strategies(config, options):
                        matches, times = match_features(first, second, config, strategy, pipeline, options)
                        row, selections = evaluate_pair(pair, first, second, matches, times,
                                                        gray1.shape, gray2.shape, options)
                        record_configuration(row, config, strategy, options)
                        row.update({"run_id": run_id, "feature_source": source,
                                    "feature_sha256_image1": first_hash, "feature_sha256_image2": second_hash,
                                    "extraction_id_image1": f"{run_id}:{source}:{pair.image1}",
                                    "extraction_id_image2": f"{run_id}:{source}:{pair.image2}",
                                    "reference_extraction_cached": cached,
                                    "extraction_shared_configurations": len(configs),
                                    "extraction_shared_strategies": len(config_strategies(config, options))})
                        folder = routes[(config.name, strategy)]
                        rows_by_output[folder].append(row)
                        if folder not in writers:
                            writers[folder] = csv.DictWriter(streams[folder], fieldnames=list(row))
                            writers[folder].writeheader()
                        writers[folder].writerow(row)
                        streams[folder].flush()
                        figure_key = (config.name, strategy, pair.name if pair.dataset == "graf" else pair.category)
                        if options.figures and figure_key not in figures_saved:
                            save_figures(image1, image2, first, second, selections,
                                         folder / "figures" / config.name / strategy / pair.name,
                                         f"{config.name}/{strategy}: {pair.name}")
                            figures_saved.add(figure_key)
                        print(f"{config.name}/{strategy} [{pair_index}/{len(pairs)}] {pair.name}: "
                              f"correct={row['n_correct']}/{row['n_putative']}; H={row['homography_error_px']:.2f}px", flush=True)
            del pipeline
    for folder, rows in rows_by_output.items():
        combinations = {(m, s) for (m, s), output in routes.items() if output == folder}
        expected = {(m, s if s in LEARNED_MATCHERS else f"bf_{s}", pair.name)
                    for m, s in combinations for pair in pairs}
        actual = {(r["method"], r["matching_strategy"], r["pair"]) for r in rows}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError(f"Incomplete or duplicate results: {folder}")
        save_summary(rows, folder)
        (folder / "complete.json").write_text(json.dumps({"records": len(rows), "run_id": run_id}), encoding="utf-8")
    return [row for rows in rows_by_output.values() for row in rows]
