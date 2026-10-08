"""Validate new CSV inventories and report measured results only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import validate_metrics
from experiments.protocol import FORMAL, check_output, primary_combinations, reserve_outputs, result_folder
from features import BF_METHODS, CONFIGS, CONTROL_CONFIG, MAIN_METHODS, LEARNED_MATCHERS
from transformations import CURVE_VALUES, is_identity, transformation_grid


PAIRED_METRICS = ["n_correct", "precision", "recall", "pmr", "matching_score", "ransac_inliers",
                  "ransac_inlier_ratio", "ransac_reprojection_error_px", "homography_error_px", "homography_success", "matching_ms"]
QUALITY = ["detector_repeatability", "feature_repeatability", "pmr", "precision", "recall",
           "matching_score", "n_correct", "n_putative", "n_features", "ransac_inliers",
           "ransac_inlier_ratio", "ransac_reprojection_error_px", "homography_error_px",
           "homography_success", "matching_ms", "detection_ms_image1", "detection_ms_image2",
           "description_ms_image1", "description_ms_image2", "joint_extraction_ms_image1",
           "joint_extraction_ms_image2", "bytes_per_descriptor", "descriptor_memory_bytes_image1",
           "descriptor_memory_bytes_image2", "detected_keypoints_image1", "detected_keypoints_image2"]


def required_folders(root, scope="all"):
    paths = []
    if scope in ("all", "hpatches"):
        for method, strategy in primary_combinations():
            stage = strategy.removeprefix("bf_")
            paths.append(result_folder(root, "hpatches", method, stage))
    if scope == "all":
        paths.extend(Path(root) / stage / "graf" for stage in ("ratio", "crosscheck", "lightglue", "superglue"))
        paths.append(Path(root) / "supplementary")
    if scope in ("all", "synthetic"):
        paths.append(Path(root) / "synthetic/raw")
    return sorted(set(paths))


def check_metadata(folder):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in FORMAL.items():
        if metadata["options"].get(key) != expected:
            raise ValueError(f"Nonformal {key} in {folder}")
    if metadata.get("image_resize") is not None:
        raise ValueError("Formal images must stay full resolution")
    return metadata


def read_completed(folder):
    metadata = check_metadata(folder)
    frame = pd.read_csv(folder / "part1_raw_results.csv")
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if len(frame) != complete["records"] or len(frame) != metadata["expected_records"]:
        raise ValueError(f"Mismatched completion count in {folder}")
    if frame.run_id.nunique() != 1 or frame.run_id.iloc[0] != metadata["run_id"] or complete["run_id"] != metadata["run_id"]:
        raise ValueError(f"Inconsistent run identity in {folder}")
    if "combinations" in metadata:
        expected = {(method, strategy if strategy in LEARNED_MATCHERS else "bf_" + strategy)
                    for method, strategy in metadata["combinations"]}
        if set(zip(frame.method, frame.matching_strategy)) != expected:
            raise ValueError("CSV configurations differ from stage metadata")
    check_model_manifests(folder, frame)
    return frame, metadata


def check_model_manifests(folder, frame):
    for method in frame.method.unique():
        config = CONTROL_CONFIG if method == CONTROL_CONFIG.name else CONFIGS[method]
        if config.detector not in ("superpoint", "sift_compatible"):
            continue
        path = folder / "models" / f"{method}.json"
        if not path.is_file():
            raise ValueError(f"Missing model/extractor manifest: {path}")
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("commit") != "eb42fee2d71449efb0aa5c10549752b5d75384d8" or model.get("resize") is not None:
            raise ValueError("Incorrect pretrained implementation or image resizing")
        if model["extractor"]["max_num_keypoints"] != 2048:
            raise ValueError("Incorrect learned/compatible feature budget")
        source = "sift" if config.detector == "sift_compatible" else "superpoint"
        if model.get("extractor_source") != source:
            raise ValueError("Wrong pretrained feature source")
        required = (["superpoint"] if source == "superpoint" else [])
        if config.matcher == "lightglue":
            required.append(source + "_lightglue")
            if model["matcher"]["input_dim"] != (128 if source == "sift" else 256):
                raise ValueError("Wrong pretrained descriptor dimension")
            if source == "sift" and not model["matcher"]["add_scale_ori"]:
                raise ValueError("SIFT LightGlue must use scales/orientations")
        if config.matcher == "superglue":
            from superglue_adapter import COMMIT, FILES, SETTINGS, SUPERPOINT_SHA256
            required.append("superpoint_superglue")
            provenance = model.get("matcher_provenance", {})
            if provenance.get("commit") != COMMIT or provenance.get("weights") != "outdoor":
                raise ValueError("Incorrect official SuperGlue implementation or weights")
            if any(model["matcher"].get(key) != value for key, value in SETTINGS.items()):
                raise ValueError("Incorrect SuperGlue configuration")
            for relative, blob in FILES.items():
                if provenance.get("files", {}).get(relative, {}).get("git_blob_sha1") != blob:
                    raise ValueError("SuperGlue provenance differs from pinned official files")
            if model["checkpoint_sha256"].get("superpoint") != SUPERPOINT_SHA256:
                raise ValueError("Shared SuperPoint checkpoint differs from official SuperGlue checkpoint")
            if model["checkpoint_sha256"].get("superpoint_superglue") != provenance["files"]["models/weights/superglue_outdoor.pth"]["sha256"]:
                raise ValueError("Inconsistent SuperGlue checkpoint identity")
        for kind in required:
            digest = model["checkpoint_sha256"].get(kind, "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Missing checkpoint SHA-256")


def validate_configuration_rows(frame):
    for column in ("homography_success", "is_primary"):
        if frame[column].isna().any() or not pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"Expected explicit booleans in {column}")
    if frame.duplicated(["dataset", "method", "matching_strategy", "pair"]).any():
        raise ValueError("Duplicate method/strategy/pair records")
    for (method, strategy), group in frame.groupby(["method", "matching_strategy"]):
        config = CONTROL_CONFIG if method == CONTROL_CONFIG.name else CONFIGS[method]
        expected_matcher = strategy if strategy in LEARNED_MATCHERS else "bf"
        if strategy not in (*LEARNED_MATCHERS, "bf_ratio", "bf_crosscheck"):
            raise ValueError("Unrecognized matching strategy")
        if config.matcher != expected_matcher:
            raise ValueError("Unsupported method/strategy result")
        for key, expected in (("detector", config.detector), ("descriptor", config.descriptor),
                              ("matcher", expected_matcher), ("is_primary", method != CONTROL_CONFIG.name)):
            if not group[key].eq(expected).all():
                raise ValueError(f"Wrong {key} for {method}")
        distance = "learned_confidence" if strategy in LEARNED_MATCHERS else ("Hamming" if config.norm == cv2.NORM_HAMMING else "L2")
        if not group.distance_metric.eq(distance).all():
            raise ValueError("Incorrect descriptor distance")
        if strategy == "bf_ratio":
            if not group.ratio_threshold.eq(0.8).all():
                raise ValueError("Formal ratio must be 0.8")
        elif not group.ratio_threshold.isna().all():
            raise ValueError("Ratio threshold must be undefined for crosscheck/learned matchers")
    for row in frame.to_dict("records"):
        validate_metrics(row)
        if row["correctness_threshold_px"] != 3 or row["ransac_threshold_px"] != 3:
            raise ValueError("Incorrect formal geometric threshold")
        for key in ("features_image1_total", "features_image2_total", "ransac_inliers", "raw_putative_matches"):
            if not np.isfinite(row[key]) or row[key] < 0 or row[key] != int(row[key]):
                raise ValueError(f"Invalid count {key}")
        if not np.isfinite(row["matching_ms"]) or row["matching_ms"] < 0:
            raise ValueError("Invalid matching runtime")
        if not row["homography_success"] and pd.notna(row["homography_error_px"]):
            raise ValueError("Failed homography must have NaN error")
        if max(row["features_image1_total"], row["features_image2_total"]) > 2048:
            raise ValueError("Exceeded formal feature budget")
        if row["ransac_inliers"] > row["raw_putative_matches"]:
            raise ValueError("RANSAC inliers exceed raw putative matches")


def validate_hpatches(frame):
    expected = set(primary_combinations())
    if len(frame) != 9860 or set(zip(frame.method, frame.matching_strategy)) != expected:
        raise ValueError("Expected exactly 9860 primary HPatches records / 17 method-strategy combinations")
    if set(frame.dataset) != {"hpatches"} or not frame.is_primary.all():
        raise ValueError("Controls or another dataset mixed into HPatches primary results")
    inventory = None
    for (method, strategy), group in frame.groupby(["method", "matching_strategy"]):
        if (len(group) != 580 or group.sequence.nunique() != 116
                or group.category.value_counts().to_dict() != {"illumination": 285, "viewpoint": 295}):
            raise ValueError(f"Incomplete HPatches categories for {method}/{strategy}")
        if not group.sequence.str.startswith(("i_", "v_")).all():
            raise ValueError("Invalid HPatches sequence prefix")
        expected_categories = np.where(group.sequence.str.startswith("i_"), "illumination", "viewpoint")
        if not np.array_equal(group.category.to_numpy(), expected_categories):
            raise ValueError("HPatches category classification mismatch")
        for sequence, targets in group.groupby("sequence"):
            if len(targets) != 5 or set(targets.target_index) != {2, 3, 4, 5, 6}:
                raise ValueError(f"Incomplete sequence {sequence}")
        current = set(zip(group.sequence, group.target_index, group.pair, group.image1, group.image2))
        if inventory is not None and current != inventory:
            raise ValueError("HPatches methods use different pair inventories")
        inventory = current
    validate_configuration_rows(frame)
    paired_bf(frame)  # Raises on mismatched feature contents.


def validate_synthetic(frame):
    if len(frame) != 2240 or set(frame.method) != set(MAIN_METHODS) or set(frame.dataset) != {"synthetic"}:
        raise ValueError("Expected 2240 synthetic records / ten primary methods")
    settings = {(s.kind, s.parameter) for s in transformation_grid()}
    sources = set(frame.source_image)
    if (len(sources) != 8 or sum(Path(p).parent.name.startswith("i_") for p in sources) != 4
            or sum(Path(p).parent.name.startswith("v_") for p in sources) != 4):
        raise ValueError("Synthetic requires 4 illumination and 4 viewpoint sources")
    for (method, source), group in frame.groupby(["method", "source_image"]):
        if len(group) != 28 or set(zip(group.transformation_type, group.transformation_strength)) != settings:
            raise ValueError(f"Missing or duplicate transformation setting: {method}/{source}")
        if group.transformation_type.eq("identity").sum() != 1:
            raise ValueError("Identity must be measured once per source and method")
    if not frame.overlap_ratio.between(0, 1).all() or not frame.support_overlap_ratio.between(0, 1).all():
        raise ValueError("Invalid synthetic overlap")
    validate_configuration_rows(frame)
    expected_strategies = {m: CONFIGS[m].matcher if CONFIGS[m].matcher in LEARNED_MATCHERS else "bf_ratio"
                           for m in MAIN_METHODS}
    if any(strategy != expected_strategies[method] for method, strategy in zip(frame.method, frame.matching_strategy)):
        raise ValueError("Synthetic uses ratio for BF and the registered learned matcher")
    for _, group in frame.groupby("synthetic_pair_id"):
        if len(group) != len(MAIN_METHODS) or set(group.method) != set(MAIN_METHODS):
            raise ValueError("Synthetic pair missing a primary method")
        if group.H_gt.nunique() != 1 or group.overlap_ratio.nunique() != 1:
            raise ValueError("Methods used inconsistent synthetic GT/canvas")
        H = np.asarray(json.loads(group.H_gt.iloc[0]), dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
            raise ValueError("Invalid synthetic GT H")


def paired_bf(frame):
    keys = ["dataset", "method", "sequence", "pair", "category"]
    ratio = frame[frame.matching_strategy == "bf_ratio"]
    cross = frame[frame.matching_strategy == "bf_crosscheck"]
    paired = ratio.merge(cross, on=keys, suffixes=("_ratio", "_crosscheck"), validate="one_to_one")
    if len(paired) != len(ratio) or len(paired) != len(cross):
        raise ValueError("Ratio/crosscheck pairs are missing")
    for index in (1, 2):
        column = f"feature_sha256_image{index}"
        if not paired[column + "_ratio"].eq(paired[column + "_crosscheck"]).all():
            raise ValueError("BF strategies did not use identical keypoints/descriptors; use shared --only bf")
    paired["shared_extraction_measurement"] = (
        paired.extraction_id_image1_ratio.eq(paired.extraction_id_image1_crosscheck)
        & paired.extraction_id_image2_ratio.eq(paired.extraction_id_image2_crosscheck))
    for metric in PAIRED_METRICS:
        paired[metric + "_delta_crosscheck_minus_ratio"] = paired[metric + "_crosscheck"].astype(float) - paired[metric + "_ratio"].astype(float)
    return paired


def aggregate(frame, keys, metrics=QUALITY):
    table = frame.groupby(keys)[metrics].agg(["mean", "std", "count"])
    table.columns = [f"{name}_{stat}" for name, stat in table.columns]
    return table.reset_index()


def homography_table(frame, keys):
    rows = []
    for labels, group in frame.groupby(keys):
        if not isinstance(labels, tuple):
            labels = (labels,)
        errors = group.homography_error_px
        row = dict(zip(keys, labels))
        row.update({"pair_count": len(group), "mean_error_px": errors.mean(), "median_error_px": errors.median(),
                    "valid_count": int(errors.count()), "failure_count": int((~group.homography_success).sum()),
                    "invalid_error_count": int(errors.isna().sum()), "success_rate": group.homography_success.mean(),
                    "accuracy_3px": errors.le(3).mean(), "accuracy_5px": errors.le(5).mean(),
                    "mean_inliers": group.ransac_inliers.mean(),
                    "mean_reprojection_px": group.ransac_reprojection_error_px.mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def curve_data(frame):
    """Reference one identity measurement in each curve; never add raw observations."""
    identity = frame[frame.transformation_type == "identity"]
    pieces = []
    for kind, values in CURVE_VALUES.items():
        part = frame[frame.transformation_type == kind].copy()
        part["curve"] = kind
        part["curve_parameter"] = part.transformation_strength
        baseline = identity.copy()
        baseline["curve"] = kind
        baseline["curve_parameter"] = 1.0 if kind == "scale" else 0.0
        # Vertical translation also references the same zero baseline.
        pieces.extend([part, baseline])
    return pd.concat(pieces, ignore_index=True)


def plot_synthetic(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = curve_data(frame)
    metrics = ["detector_repeatability", "feature_repeatability", "matching_score", "precision", "recall", "homography_success", "homography_error_px"]
    summary = aggregate(curves, ["curve", "curve_parameter", "method"], metrics + ["overlap_ratio", "support_overlap_ratio"])
    summary.to_csv(output / "synthetic_curves.csv", index=False)
    folder = output / "synthetic_figures"
    folder.mkdir()
    for kind, values in summary.groupby("curve"):
        for metric in metrics:
            figure, axis = plt.subplots(figsize=(10, 6))
            for method, group in values.groupby("method", sort=False):
                group = group.sort_values("curve_parameter")
                axis.errorbar(group.curve_parameter, group[metric + "_mean"],
                              yerr=group[metric + "_std"].fillna(0), marker="o", label=method, capsize=2)
            axis.set_xlabel(kind + (" (degrees)" if kind == "rotation" else " (factor)" if kind == "scale" else " (relative fraction)"))
            axis.set_ylabel(metric + " (mean ± sample std)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(folder / f"{kind}_{metric}.png", dpi=150)
            plt.close(figure)
    translation = curves[curves.curve.isin(["translation_x", "translation_y"])]
    for axis_name in ("translation_x", "translation_y"):
        figure, axis = plt.subplots(figsize=(10, 6))
        for method, group in translation[translation.curve == axis_name].groupby("method"):
            table = group.groupby("curve_parameter")[["overlap_ratio", "matching_score"]].mean().sort_values("overlap_ratio")
            axis.plot(table.overlap_ratio, table.matching_score, "o-", label=method)
        axis.set_xlabel("Geometric source overlap ratio")
        axis.set_ylabel("Matching score (mean)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(folder / f"{axis_name}_overlap.png", dpi=150)
        plt.close(figure)


def learned_comparison(frame, control=None):
    comparisons = [("sift", "sift_lightglue", "extractor and normalization differ"),
                   ("superpoint", "superpoint_lightglue", "same SuperPoint features"),
                   ("superpoint", "superpoint_superglue", "same SuperPoint features"),
                   ("superpoint_lightglue", "superpoint_superglue", "same SuperPoint features"),
                   ("sift_lightglue", "superpoint_lightglue", "different extractor, descriptor and pretrained matcher")]
    if control is not None:
        comparisons.append((CONTROL_CONFIG.name, "sift_lightglue", "same official-compatible RootSIFT features"))
        frame = pd.concat([frame, control], ignore_index=True)
    rows = []
    for first, second, interpretation in comparisons:
        left = frame[(frame.method == first) & (frame.matching_strategy != "bf_crosscheck")]
        right = frame[(frame.method == second) & frame.matching_strategy.isin(LEARNED_MATCHERS)]
        merged = left.merge(right, on=["dataset", "sequence", "pair", "category"], suffixes=("_first", "_second"), validate="one_to_one")
        if not len(merged) or len(merged) != len(left) or len(merged) != len(right):
            raise ValueError("Missing learned comparison pairs")
        shared = (merged.feature_sha256_image1_first.eq(merged.feature_sha256_image1_second)
                  & merged.feature_sha256_image2_first.eq(merged.feature_sha256_image2_second))
        isolated_matcher = (first.startswith("superpoint") and second.startswith("superpoint")) or first == CONTROL_CONFIG.name
        if isolated_matcher and not shared.all():
            raise ValueError(f"Feature contents differ for matcher comparison {first}/{second}")
        same_measurement = (merged.extraction_id_image1_first.eq(merged.extraction_id_image1_second)
                            & merged.extraction_id_image2_first.eq(merged.extraction_id_image2_second))
        group_keys = ["dataset", "category"]
        if "transformation_strength_first" in merged:
            group_keys.append("transformation_strength_first")
        for labels, group in merged.groupby(group_keys):
            row = dict(zip(group_keys, labels))
            row.update({"first": first, "second": second, "pair_count": len(group),
                        "interpretation": interpretation, "identical_feature_contents": bool(shared.loc[group.index].all()),
                        "shared_extraction_measurement": bool(same_measurement.loc[group.index].all())})
            for metric in PAIRED_METRICS:
                a, b = group[metric + "_first"].astype(float), group[metric + "_second"].astype(float)
                delta = b - a
                row.update({metric + "_first_mean": a.mean(), metric + "_second_mean": b.mean(),
                            metric + "_delta_mean": delta.mean(), metric + "_delta_std": delta.std(),
                            metric + "_paired_valid_count": int(delta.count())})
            for suffix in ("first", "second"):
                errors = group["homography_error_px_" + suffix]
                row.update({f"homography_error_px_{suffix}_median": errors.median(),
                            f"homography_error_px_{suffix}_valid_count": int(errors.count()),
                            f"homography_failure_count_{suffix}": int((~group["homography_success_" + suffix]).sum()),
                            f"homography_accuracy_3px_{suffix}": errors.le(3).mean(),
                            f"homography_accuracy_5px_{suffix}": errors.le(5).mean()})
            rows.append(row)
    return pd.DataFrame(rows)


def build_report(results_root, output=None, scope="all", validate_only=False):
    root = Path(results_root)
    folders = required_folders(root, scope)
    missing = []
    for folder in folders:
        names = ["metadata.json", "complete.json"]
        names += ["fast_parameters.csv", "brief_sampling.csv"] if folder.name == "supplementary" else ["part1_raw_results.csv"]
        missing.extend(str(folder / name) for name in names if not (folder / name).is_file())
    if missing:
        print("PENDING — missing completed experiment outputs; no formal report generated:\n" + "\n".join(missing))
        return False
    datasets, metadata = {}, []
    supplementary = None
    for folder in folders:
        if folder.name == "supplementary":
            meta = check_metadata(folder)
            metadata.append(meta)
            fast = pd.read_csv(folder / "fast_parameters.csv", dtype={"nonmax": "string"})
            invalid_nonmax = fast.nonmax.notna() & ~fast.nonmax.isin(["True", "False"])
            if invalid_nonmax.any():
                raise ValueError("Invalid FAST nonmax boolean")
            fast["nonmax"] = fast.nonmax.map({"True": True, "False": False})
            brief = pd.read_csv(folder / "brief_sampling.csv")
            complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
            if len(fast) != 84 or len(brief) != 24 or complete != {"fast_records": 84, "brief_records": 24}:
                raise ValueError("Incomplete FAST/BRIEF supplementary results")
            if fast.duplicated(["method", "threshold", "nonmax", "dataset", "pair"]).any() or brief.duplicated(["strategy", "dataset", "pair"]).any():
                raise ValueError("Duplicate supplementary records")
            settings = set(zip(fast[fast.method == "fast"].threshold, fast[fast.method == "fast"].nonmax))
            if settings != {(t, n) for t in (10, 20, 40) for n in (True, False)} or set(brief.strategy) != {"uniform", "gaussian"}:
                raise ValueError("Incorrect supplementary parameter settings")
            if fast.pair.nunique() != 12 or brief.pair.nunique() != 12:
                raise ValueError("Supplementary requires the same twelve selected pairs")
            expected_pairs = set(zip(brief.dataset, brief.pair))
            if set(zip(fast.dataset, fast.pair)) != expected_pairs:
                raise ValueError("FAST and BRIEF supplementary pair inventories differ")
            for _, group in fast.groupby(["method", "threshold", "nonmax"], dropna=False):
                if len(group) != 12 or set(zip(group.dataset, group.pair)) != expected_pairs:
                    raise ValueError("Incomplete supplementary detector setting")
            for _, group in brief.groupby("strategy"):
                if len(group) != 12 or set(zip(group.dataset, group.pair)) != expected_pairs:
                    raise ValueError("Incomplete supplementary BRIEF setting")
            supplementary = (fast, brief)
            continue
        frame, meta = read_completed(folder)
        metadata.append(meta)
        if frame.dataset.nunique() != 1:
            raise ValueError("Mixed datasets in a raw CSV")
        datasets.setdefault(frame.dataset.iloc[0], []).append(frame)
    datasets = {name: pd.concat(parts, ignore_index=True) for name, parts in datasets.items()}
    if len({meta["code_sha256"] for meta in metadata}) != 1:
        raise ValueError("Experiment code/config identities differ")
    for key in ("python", "opencv", "numpy", "scipy", "platform", "opencv_threads"):
        if len({str(meta[key]) for meta in metadata}) != 1:
            raise ValueError(f"Inconsistent execution environment: {key}")
    primary_devices = {meta["options"]["device"] for meta in metadata if "expected_records" in meta}
    if len(primary_devices) != 1:
        raise ValueError("Primary stages must use the same requested device")
    # Compare real image/GT hashes, not just pair labels, across dataset stages.
    for dataset in ("hpatches", "graf"):
        manifests = [json.dumps(meta["input_manifest"], sort_keys=True) for meta in metadata
                     if meta["options"].get("dataset") == dataset]
        if len(set(manifests)) > 1:
            raise ValueError(f"{dataset} input content differs across stages")
    if "hpatches" in datasets:
        validate_hpatches(datasets["hpatches"])
    if "synthetic" in datasets:
        validate_synthetic(datasets["synthetic"])
        synthetic_meta = next(meta for meta in metadata if meta.get("dataset") == "synthetic")
        source_paths = {item["path"] for item in synthetic_meta["input_manifest"]}
        if set(datasets["synthetic"].source_image) != source_paths:
            raise ValueError("Synthetic CSV source paths differ from saved source manifest")
        hpatches_manifests = [meta["input_manifest"] for meta in metadata if meta["options"].get("dataset") == "hpatches"]
        if hpatches_manifests:
            reference_hashes = {item["path"]: item["sha256"] for item in hpatches_manifests[0]}
            for item in synthetic_meta["input_manifest"]:
                if reference_hashes.get(item["path"]) != item["sha256"]:
                    raise ValueError("Synthetic references differ from the HPatches source images")
    if "graf" in datasets:
        graf = datasets["graf"]
        if len(graf) != 34 or set(zip(graf.method, graf.matching_strategy)) != set(primary_combinations()):
            raise ValueError("GRAF requires 17 method/strategy configurations x 2 pairs")
        for _, group in graf.groupby(["method", "matching_strategy"]):
            if len(group) != 2 or set(group.target_index) != {2, 4}:
                raise ValueError("Missing GRAF pair")
        validate_configuration_rows(graf)
        paired_bf(graf)
    controls = {}
    for dataset, count in (("hpatches", 580), ("graf", 2)):
        suffix = "hpatches_sift_control" if dataset == "hpatches" else "graf_control"
        control_folder = root / "lightglue" / suffix
        if dataset not in datasets or not control_folder.exists():
            continue
        for name in ("metadata.json", "complete.json", "part1_raw_results.csv"):
            if not (control_folder / name).is_file():
                raise ValueError("SIFT-compatible control started but is incomplete")
        control_frame, meta = read_completed(control_folder)
        if (len(control_frame) != count or set(control_frame.method) != {CONTROL_CONFIG.name}
                or set(control_frame.matching_strategy) != {"bf_ratio"} or control_frame.is_primary.any()
                or set(control_frame.dataset) != {dataset} or meta["code_sha256"] != metadata[0]["code_sha256"]):
            raise ValueError("Invalid auxiliary SIFT-compatible BF control")
        source_meta = next(item for item in metadata if item["options"].get("dataset") == dataset)
        if meta["input_manifest"] != source_meta["input_manifest"] or meta["options"]["device"] != source_meta["options"]["device"]:
            raise ValueError("Auxiliary control uses different inputs/device")
        validate_configuration_rows(control_frame)
        controls[dataset] = control_frame
    control = controls.get("hpatches")
    # Validate learned comparisons for every dataset even in validation-only mode.
    learned = {name: learned_comparison(frame, controls.get(name)) for name, frame in datasets.items()}
    if validate_only:
        print("Validated completed CSV inventories: " + ", ".join(f"{key}={len(frame)}" for key, frame in datasets.items()))
        return True
    output = Path(output) if output is not None else root / "comparison"
    reserve_outputs([output])
    notes = ["# Part 1 measured results", "Tables below are generated from completed CSV files; no predicted ranking."]
    if "hpatches" in datasets:
        frame = datasets["hpatches"]
        keys = ["method", "matching_strategy", "category"]
        aggregate(frame, keys).to_csv(output / "hpatches_summary.csv", index=False)
        htable = homography_table(frame, keys)
        htable.to_csv(output / "hpatches_homography.csv", index=False)
        paired = paired_bf(frame)
        paired.to_csv(output / "ratio_crosscheck_paired.csv", index=False)
        deltas = [metric + "_delta_crosscheck_minus_ratio" for metric in PAIRED_METRICS]
        aggregate(paired, ["method", "category"], deltas).to_csv(output / "ratio_crosscheck_summary.csv", index=False)
        learned["hpatches"].to_csv(output / "learned_comparisons.csv", index=False)
        if control is not None:
            aggregate(control, keys).to_csv(output / "sift_compatible_bf_control.csv", index=False)
        runtime = frame.copy()
        classic_time = sum(runtime[name] for name in ["detection_ms_image1", "detection_ms_image2", "description_ms_image1", "description_ms_image2"])
        runtime["extraction_pair_ms"] = classic_time.fillna(runtime.joint_extraction_ms_image1 + runtime.joint_extraction_ms_image2)
        runtime["pair_stage_total_ms"] = runtime.extraction_pair_ms + runtime.matching_ms
        aggregate(runtime, keys, ["extraction_pair_ms", "matching_ms", "pair_stage_total_ms", "bytes_per_descriptor",
                                  "descriptor_memory_bytes_image1", "descriptor_memory_bytes_image2"]).to_csv(output / "runtime_memory.csv", index=False)
        notes.extend([f"Primary HPatches records: {len(frame)}. Illumination/viewpoint are reported separately.",
                      f"BF comparisons have identical feature hashes. Shared extraction measurements: {int(paired.shared_extraction_measurement.sum())}/{len(paired)} paired rows.",
                      "Separately invoked ratio/crosscheck stages can have identical contents but independent extraction measurements; shared --only bf is preferred.",
                      "Baseline SIFT versus SIFT+LightGlue changes both extraction parameters and RootSIFT normalization. Only the compatible BF control isolates the matcher.",
                      "SuperPoint BF, LightGlue and official outdoor SuperGlue share native features, scores and image dimensions; paired hashes and extraction identities are checked.",
                      "GPU and CPU timings exclude I/O, transfers, model initialization, GT evaluation, masks and RANSAC; repeated reference timings are shared measurements."])
        notes.append(markdown_table(htable))
    if "synthetic" in datasets:
        frame = datasets["synthetic"]
        keys = ["method", "matching_strategy", "transformation_type", "transformation_strength"]
        aggregate(frame, keys).to_csv(output / "synthetic_summary.csv", index=False)
        homography_table(frame, keys).to_csv(output / "synthetic_homography.csv", index=False)
        learned["synthetic"].to_csv(output / "synthetic_learned_comparisons.csv", index=False)
        plot_synthetic(frame, output)
        notes.extend([f"Synthetic records: {len(frame)}; one identity per method/source, referenced by curves.",
                      "Synthetic and HPatches are separate benchmarks. Error bars are sample standard deviations across source images; CSV includes valid counts.",
                      "Overlap reduction and support filtering reduce the available content. The post-filter policy cannot guarantee removal of every padding influence; inspect overlap and retained counts alongside invariance curves."])
    if "graf" in datasets:
        if "graf" in controls:
            aggregate(controls["graf"], ["method", "matching_strategy"]).to_csv(output / "graf_sift_compatible_bf_control.csv", index=False)
        learned["graf"].to_csv(output / "graf_learned_comparisons.csv", index=False)
        homography_table(datasets["graf"], ["method", "matching_strategy"]).to_csv(output / "graf_homography.csv", index=False)
        aggregate(datasets["graf"], ["method", "matching_strategy"]).to_csv(output / "graf_summary.csv", index=False)
        notes.append("GRAF has two pairs only and is supplementary qualitative/geometric evidence.")
    if supplementary is not None:
        fast, brief = supplementary
        fast.groupby(["method", "threshold", "nonmax"], dropna=False)[["keypoints1", "keypoints2", "repeatability", "detection_ms_image1", "detection_ms_image2"]].agg(["mean", "std", "count"]).to_csv(output / "fast_summary.csv")
        brief.groupby("strategy")[["pmr", "precision", "recall", "matching_score", "description_ms_image1", "description_ms_image2", "matching_ms"]].agg(["mean", "std", "count"]).to_csv(output / "brief_summary.csv")
        notes.append("FAST/BRIEF supplementary studies cover twelve selected pairs, not full HPatches.")
    notes.append("Homography mean, median, valid error count, failure count and success rate are separate. NaN estimates count as failures in threshold accuracy; RANSAC inliers do not imply GT-correct correspondences.")
    (output / "PART1_RESULTS.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    (output / "validation.json").write_text(json.dumps({"counts": {key: len(frame) for key, frame in datasets.items()},
                                                        "code_sha256": metadata[0]["code_sha256"],
                                                        "control_count": len(control) if control is not None else 0}, indent=2), encoding="utf-8")
    print(f"Measured report saved to {output}")
    return True


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scope", choices=["all", "hpatches", "synthetic"], default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    success = build_report(args.results_root, args.output, args.scope, args.validate_only)
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
