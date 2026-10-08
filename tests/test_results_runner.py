"""Output safety and orchestration mocks; no feature extraction or models. NOT RUN."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import ImagePair
from experiments.build_report import (PAIRED_METRICS, build_report, learned_comparison,
                                      paired_bf, validate_hpatches, validate_synthetic)
from experiments.protocol import FORMAL, ROOT, check_output, primary_combinations, result_folder
from experiments.runner import feature_groups, run_experiment
from features import CONFIGS, MAIN_METHODS, ImageFeatures


class ResultsRunnerTests(unittest.TestCase):
    def test_registry_and_primary_scale(self):
        self.assertEqual(len(MAIN_METHODS), 10)
        combinations = primary_combinations()
        self.assertEqual(len(combinations), 17)
        self.assertEqual(len(combinations)*580 + 8*28*len(MAIN_METHODS), 12100)
        self.assertNotIn("sift_compatible_bf", CONFIGS)
        self.assertEqual(len(feature_groups(["superpoint", "superpoint_lightglue", "superpoint_superglue"])), 1)
        self.assertEqual(len(feature_groups(["sift", "sift_lightglue"])), 2)

    def test_output_isolation_and_no_overwrite(self):
        self.assertEqual(result_folder(Path("results"), "hpatches", "superpoint_superglue", "superglue"),
                         Path("results/superglue/hpatches_superpoint"))
        self.assertNotEqual(result_folder(Path("results"), "hpatches", "sift", "ratio"),
                            result_folder(Path("results"), "hpatches", "sift", "crosscheck"))
        for protected in (ROOT, ROOT / "data/generated", ROOT / "models", ROOT / "src/results"):
            with self.assertRaises(ValueError):
                check_output(protected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            check_output(path)
            (path / "sentinel.txt").write_text("preserve")
            with self.assertRaises(FileExistsError):
                check_output(path)
            self.assertEqual((path / "sentinel.txt").read_text(), "preserve")

    def test_pending_report_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            self.assertFalse(build_report(Path(directory)))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_completeness_rejects_missing_results(self):
        with self.assertRaises(ValueError):
            validate_hpatches(pd.DataFrame())
        with self.assertRaises(ValueError):
            validate_synthetic(pd.DataFrame())

    def test_shared_pass_extracts_each_image_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / name for name in ("1.ppm", "2.ppm", "H")]
            for path in inputs:
                path.write_text("mock input; never decoded")
            pair = ImagePair("graf", "graf", "viewpoint", 2, *inputs)
            options = SimpleNamespace(**FORMAL, methods=["sift"], bf_strategies=["ratio", "crosscheck"],
                                      output=root / "output", device="cpu", figures=False, formal=False,
                                      routed=False, sift_control=False)
            features = ImageFeatures([cv2.KeyPoint(10, 10, 1)], [cv2.KeyPoint(10, 10, 1)],
                                     np.zeros((1, 128), np.float32), {})
            observed = []
            def match(first, second, config, strategy, pipeline, options):
                observed.append((first, second, strategy))
                return [], {"matching_ms": 1 if strategy == "ratio" else 2}
            with patch("experiments.runner.load_image", return_value=(np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), np.uint8))), patch("experiments.runner.extract_image", return_value=features) as extract, patch("experiments.runner.match_features", side_effect=match), patch("experiments.runner.evaluate_pair", side_effect=lambda *args: ({"pair": args[0].name, "dataset": args[0].dataset, "sequence": args[0].sequence, "category": args[0].category, "n_correct": 0, "n_putative": 0, "homography_error_px": float("nan"), **args[4]}, {})), patch("experiments.runner.save_summary"), patch("builtins.print"):
                rows = run_experiment([pair], options)
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(len(rows), 2)
            self.assertIs(observed[0][0], observed[1][0])
            self.assertIs(observed[0][1], observed[1][1])
            self.assertEqual([r["matching_ms"] for r in rows], [1, 2])
            self.assertEqual(rows[0]["extraction_id_image1"], rows[1]["extraction_id_image1"])
            metadata = json.loads((options.output / "metadata.json").read_text())
            self.assertEqual(metadata["expected_records"], 2)
            self.assertEqual(metadata["options"]["max_keypoints"], 2048)
            self.assertEqual(metadata["ransac"], {"max_iterations": 5000, "confidence": .995, "seed": 0})
            self.assertEqual(len(metadata["input_manifest"]), 3)
            self.assertEqual(len(metadata["code_sha256"]), 64)
            self.assertEqual(json.loads((options.output / "complete.json").read_text())["run_id"], rows[0]["run_id"])

    def test_missing_dataset_raises_before_extraction(self):
        from datasets.graf import iter_graf_pairs
        from datasets.hpatches import iter_hpatches_pairs
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                list(iter_graf_pairs(directory))
            with self.assertRaises(ValueError):
                list(iter_hpatches_pairs(directory))

    def test_three_matchers_share_one_superpoint_extraction_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("1.ppm", "2.ppm", "H")]
            for path in paths:
                path.write_text("mock bytes; never decoded")
            pair = ImagePair("graf", "graf", "viewpoint", 2, *paths)
            methods = ["superpoint", "superpoint_lightglue", "superpoint_superglue"]
            options = SimpleNamespace(**FORMAL, methods=methods, bf_strategies=["ratio"],
                                      output=root / "output", device="cpu", figures=False,
                                      formal=False, routed=False, sift_control=False)
            features = ImageFeatures([], [], np.empty((0, 256), np.float32), {})
            pipeline, observations = MagicMock(), []
            def match(first, second, config, strategy, provided, options):
                self.assertIs(provided, pipeline)
                observations.append((first, second, strategy))
                return [], {"matching_ms": 1.0}
            row = {"pair": pair.name, "dataset": "graf", "sequence": "graf", "category": "viewpoint",
                   "n_correct": 0, "n_putative": 0, "homography_error_px": float("nan")}
            with patch("experiments.runner.make_pipeline", return_value=pipeline) as make, patch("experiments.runner.load_image", return_value=(None, np.zeros((2, 2), np.uint8))), patch("experiments.runner.extract_image", return_value=features) as extract, patch("experiments.runner.match_features", side_effect=match), patch("experiments.runner.evaluate_pair", side_effect=lambda *args: (dict(row), {})), patch("experiments.runner.save_summary"), patch("builtins.print"):
                rows = run_experiment([pair], options)
            self.assertEqual(make.call_count, 1)
            self.assertEqual(extract.call_count, 2)
            self.assertEqual({call.args[0] for call in pipeline.ensure_matcher.call_args_list}, {"lightglue", "superglue"})
            self.assertEqual([item[2] for item in observations], ["ratio", "lightglue", "superglue"])
            self.assertTrue(all(a is features and b is features for a, b, _ in observations))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({r["extraction_id_image1"] for r in rows}), 1)

    def test_paired_report_requires_identical_feature_contents(self):
        common = {"dataset": "hpatches", "method": "sift", "sequence": "i_test", "pair": "p", "category": "illumination",
                  "feature_sha256_image1": "a", "feature_sha256_image2": "b", "extraction_id_image1": "one", "extraction_id_image2": "two"}
        from experiments.build_report import PAIRED_METRICS
        common.update({metric: 0.0 for metric in PAIRED_METRICS})
        records = [{**common, "matching_strategy": strategy} for strategy in ("bf_ratio", "bf_crosscheck")]
        self.assertTrue(paired_bf(pd.DataFrame(records)).shared_extraction_measurement.all())
        records[1]["feature_sha256_image2"] = "different"
        with self.assertRaises(ValueError):
            paired_bf(pd.DataFrame(records))

    def test_three_superpoint_matchers_are_paired_and_hash_checked(self):
        common = {"dataset": "graf", "sequence": "graf", "pair": "pair", "category": "viewpoint",
                  "feature_sha256_image1": "a", "feature_sha256_image2": "b",
                  "extraction_id_image1": "one", "extraction_id_image2": "two"}
        common.update({metric: 1.0 for metric in PAIRED_METRICS})
        common["homography_success"] = True
        methods = [("sift", "bf_ratio"), ("sift_lightglue", "lightglue"), ("superpoint", "bf_ratio"),
                   ("superpoint_lightglue", "lightglue"), ("superpoint_superglue", "superglue")]
        records = [{**common, "method": method, "matching_strategy": strategy} for method, strategy in methods]
        table = learned_comparison(pd.DataFrame(records))
        comparisons = set(zip(table["first"], table["second"]))
        self.assertTrue({("superpoint", "superpoint_lightglue"), ("superpoint", "superpoint_superglue"),
                         ("superpoint_lightglue", "superpoint_superglue")} <= comparisons)
        self.assertTrue(table.shared_extraction_measurement.all())
        self.assertEqual(table.homography_success_delta_mean.tolist(), [0.] * len(table))
        records[-1]["feature_sha256_image2"] = "changed"
        with self.assertRaises(ValueError):
            learned_comparison(pd.DataFrame(records))
