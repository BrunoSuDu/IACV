"""Manual synthetic experiment runner; never executed during code preparation."""
import argparse
import csv
import json
import sys
import uuid
from copy import copy
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.synthetic import DEFAULT_MANIFEST, SyntheticPair, select_sources
from experiments.protocol import (add_protocol_arguments, input_manifest, reserve_outputs,
                                  validate_options)
from experiments.reporting import save_metadata, save_summary
from experiments.runner import (evaluate_pair, extract_image, feature_groups, make_pipeline,
                                match_features, record_configuration, save_figures)
from feature_masks import MASK_POLICY, evaluation_mask, feature_fingerprint, filter_features
from features import LEARNED_MATCHERS, MAIN_METHODS
from image_io import load_image
from transformations import erode_valid_mask, homography_for, transformation_grid, warp_with_mask


def selected_plan(options):
    sources, manifest = select_sources(options.data_root, options.source_manifest)
    settings = transformation_grid()
    if options.limit_sources is not None:
        if options.limit_sources < 1:
            raise ValueError("limit-sources must be positive")
        sources = sources[:options.limit_sources]
    if options.settings:
        requested = set(options.settings)
        if len(requested) != len(options.settings) or requested - {s.name for s in settings}:
            raise ValueError("Unknown or duplicate transformation IDs")
        settings = [setting for setting in settings if setting.name in requested]
    return sources, settings, manifest


def run_synthetic(options):
    sources, settings, manifest = selected_plan(options)
    groups = feature_groups(options.methods)
    reserve_outputs([options.output])
    raw = options.output / "raw"
    summary = options.output / "summary"
    raw.mkdir()
    summary.mkdir()
    run_id = uuid.uuid4().hex
    local = copy(options)
    local.output = raw
    save_metadata(local, [], {"dataset": "synthetic", "run_id": run_id,
                              "source_manifest": manifest, "input_manifest": input_manifest(sources),
                              "settings": [{"kind": s.kind, "parameter": s.parameter} for s in settings],
                              "source_count": len(sources), "setting_count": len(settings),
                              "expected_records": len(sources) * len(settings) * len(options.methods),
                              "mask_policy": MASK_POLICY, "formal": getattr(options, "formal", False),
                              "pair_count": len(sources) * len(settings)})
    rows, source_dimensions, pair_manifest = [], {}, {}
    with (raw / "progress.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for source_group, configs in groups.items():
            pipeline = make_pipeline(configs[0], options)
            if pipeline is not None:
                for matcher in sorted({config.matcher for config in configs} & set(LEARNED_MATCHERS)):
                    pipeline.ensure_matcher(matcher)
                for config in configs:
                    pipeline.save_manifest(raw / "models", config.name)
            for path in sources:
                original, gray = load_image(path)
                source_dimensions[str(path)] = {"height": gray.shape[0], "width": gray.shape[1]}
                source_valid = erode_valid_mask(np.ones(gray.shape, bool), 1)
                source_eval = evaluation_mask(source_valid)
                reference = extract_image(gray, configs[0], pipeline, options)
                source_before_mask = len(reference.keypoints)
                source_detected_before_mask = len(reference.detected_keypoints)
                first = filter_features(reference, source_valid)
                first_hash = feature_fingerprint(first)
                for setting_index, setting in enumerate(settings):
                    H = homography_for(setting, gray.shape)
                    transformed, valid, geometry = warp_with_mask(original, H)
                    target_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
                    target_eval = evaluation_mask(valid)
                    if setting.kind == "identity":
                        second = first
                        target_before_mask = source_before_mask
                        target_detected_before_mask = source_detected_before_mask
                    else:
                        extracted = extract_image(target_gray, configs[0], pipeline, options)
                        target_before_mask = len(extracted.keypoints)
                        target_detected_before_mask = len(extracted.detected_keypoints)
                        second = filter_features(extracted, valid)
                    second_hash = feature_fingerprint(second)
                    pair = SyntheticPair(path, setting, H, setting_index)
                    overlap = cv2.warpPerspective(target_eval.astype(np.uint8), np.linalg.inv(H),
                                                  (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                    geometry["support_overlap_ratio"] = float((overlap & source_eval).mean())
                    pair_manifest[pair.name] = {"source": str(path), "shape": list(gray.shape),
                                                "transformation_type": setting.kind,
                                                "transformation_parameter": setting.parameter,
                                                "H_gt": H.tolist(), **geometry}
                    for config in configs:
                        strategy = config.matcher if config.matcher in LEARNED_MATCHERS else "ratio"
                        matches, times = match_features(first, second, config, strategy, pipeline, options)
                        row, selections = evaluate_pair(pair, first, second, matches, times, gray.shape,
                                                        target_gray.shape, options, source_eval, target_eval)
                        record_configuration(row, config, strategy, options)
                        row.update({"run_id": run_id, "source_image": str(path), "synthetic_pair_id": pair.name,
                                    "transformation_type": setting.kind, "transformation_strength": setting.parameter,
                                    "feature_source": source_group, "mask_policy": json.dumps(MASK_POLICY),
                                    "features_before_mask_image1": source_before_mask,
                                    "features_before_mask_image2": target_before_mask,
                                    "detected_keypoints_before_mask_image1": source_detected_before_mask,
                                    "detected_keypoints_before_mask_image2": target_detected_before_mask,
                                    "feature_sha256_image1": first_hash, "feature_sha256_image2": second_hash,
                                    "extraction_id_image1": f"{run_id}:{source_group}:{path}",
                                    "extraction_id_image2": f"{run_id}:{source_group}:{path if setting.kind == 'identity' else pair.name}",
                                    "reference_extraction_cached": setting_index > 0,
                                    "extraction_shared_configurations": len(configs), **geometry})
                        rows.append(row)
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        stream.flush()
                        # Deterministic qualitative selection: first source per group,
                        # first setting per kind. Quantitative evaluation uses all matches.
                        first_of_kind = setting == next(s for s in settings if s.kind == setting.kind)
                        if options.figures and path == sources[0] and first_of_kind:
                            folder = options.output / "figures" / config.name / pair.name
                            save_figures(original, transformed, first, second, selections, folder,
                                         f"{config.name}: {pair.name}; overlap={geometry['overlap_ratio']:.3f}")
                            from visualization import save_synthetic_context
                            save_synthetic_context(original, transformed, valid, folder / "context.png", pair.name)
                        print(f"synthetic {config.name} {pair.name}: {row['n_correct']}/{row['n_putative']}", flush=True)
            del pipeline
    expected = {(m, p.parent.name + "__" + s.name) for m in options.methods for p in sources for s in settings}
    actual = {(r["method"], r["synthetic_pair_id"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Incomplete or duplicate synthetic records")
    save_summary(rows, raw)
    import pandas as pd
    frame = pd.DataFrame(rows)
    keys = ["method", "matching_strategy", "transformation_type", "transformation_strength"]
    numeric = [c for c in frame.select_dtypes(include=["number", "bool"]).columns if c not in keys]
    table = frame.groupby(keys)[numeric].agg(["mean", "std", "count"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table.reset_index().to_csv(summary / "robustness.csv", index=False)
    (raw / "sources.json").write_text(json.dumps(source_dimensions, indent=2), encoding="utf-8")
    (raw / "pairs.json").write_text(json.dumps(pair_manifest, indent=2), encoding="utf-8")
    (raw / "complete.json").write_text(json.dumps({"records": len(rows), "run_id": run_id}), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_protocol_arguments(parser)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/hpatches")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--methods", nargs="+", choices=MAIN_METHODS, default=MAIN_METHODS)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--settings", nargs="+", help="Exact IDs, e.g. identity_0 rotation_30 scale_0.75")
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic")
    options = parser.parse_args()
    validate_options(options)
    if len(set(options.methods)) != len(options.methods):
        parser.error("Duplicate methods")
    options.formal = False
    cv2.setNumThreads(options.threads)
    cv2.setRNGSeed(options.seed)
    np.random.seed(options.seed)
    run_synthetic(options)


if __name__ == "__main__":
    main()
