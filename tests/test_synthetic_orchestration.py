"""Synthetic orchestration with every image/model/evaluation operation mocked."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.protocol import FORMAL
from experiments.run_synthetic import run_synthetic
from features import ImageFeatures
from transformations import Transformation


class SyntheticOrchestrationTests(unittest.TestCase):
    def test_shared_features_identity_and_two_auxiliary_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "i_fixture/1.ppm"
            source.parent.mkdir()
            source.write_text("fixture bytes, never decoded")
            options = SimpleNamespace(**FORMAL, output=root / "out", device="cpu", figures=False,
                                      formal=False, sift_control=True, bf_strategies=["ratio", "crosscheck"],
                                      methods=["superpoint", "superpoint_lightglue", "superpoint_superglue", "sift_lightglue"])
            settings = [Transformation("identity", 0), Transformation("brightness", 30)]
            gray, color, mask = np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool)
            features = [ImageFeatures([], [], np.zeros((0, 256), np.float32), {}) for _ in range(4)]
            observations = []
            def match(first, second, config, strategy, pipeline, args):
                observations.append((config.name, strategy, first, second))
                return [], {"matching_ms": float(len(observations))}
            def evaluate(pair, first, second, matches, times, *args):
                return {"dataset": pair.dataset, "category": pair.category, "sequence": pair.sequence,
                        "pair": pair.name, "n_correct": 0, "n_putative": 0, "H_gt": json.dumps(pair.homography.tolist()),
                        **times}, {}
            geometry = {"overlap_ratio": 1., "target_valid_ratio": 1., "canvas": "fixed source width/height; origin (0,0); no canvas translation",
                        "interpolation": "INTER_LINEAR", "border_mode": "BORDER_REFLECT_101", "mask_interpolation": "fixture"}
            photo = {"photometric_clipped_fraction": 0., "photometric_saturated_fraction": 0., "photometric_source_saturated_fraction": 1.}
            from contextlib import ExitStack
            with ExitStack() as stack:
                mocks = {
                    "selected_plan": ([source], settings, {"sources": ["i_fixture/1.ppm"]}),
                    "make_pipeline": MagicMock(), "load_image": (color, gray),
                    "erode_valid_mask": mask, "evaluation_mask": mask,
                    "homography_for": np.eye(3), "warp_with_mask": (color, mask, geometry),
                    "apply_photometric": (gray, photo), "feature_fingerprint": "shared",
                }
                for name, result in mocks.items():
                    stack.enter_context(patch("experiments.run_synthetic." + name, return_value=result))
                extract = stack.enter_context(patch("experiments.run_synthetic.extract_image", side_effect=features))
                stack.enter_context(patch("experiments.run_synthetic.filter_features", side_effect=lambda value, mask: value))
                stack.enter_context(patch("experiments.run_synthetic.match_features", side_effect=match))
                stack.enter_context(patch("experiments.run_synthetic.evaluate_pair", side_effect=evaluate))
                stack.enter_context(patch("experiments.run_synthetic.cv2.cvtColor", return_value=color))
                stack.enter_context(patch("experiments.run_synthetic.cv2.warpPerspective", return_value=mask))
                stack.enter_context(patch("builtins.print"))
                rows = run_synthetic(options)
            self.assertEqual(extract.call_count, 4)  # two feature sources x (reference + brightness)
            self.assertEqual(len(rows), 14)
            self.assertEqual(sum(r["is_primary"] for r in rows), 10)
            self.assertEqual(sum(not r["is_primary"] for r in rows), 4)
            sp = observations[:8]
            self.assertTrue(all(first is second for _, _, first, second in sp[:4]))
            self.assertTrue(all(first is features[0] and second is features[1] for _, _, first, second in sp[4:]))
            self.assertEqual({strategy for method, strategy, _, _ in observations if method == "sift_compatible_bf"}, {"ratio", "crosscheck"})
            self.assertEqual(len({r["matching_ms"] for r in rows}), 14)
            metadata = json.loads((options.output / "raw/metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["expected_primary_records"], 10)
            self.assertEqual(metadata["expected_auxiliary_records"], 4)
            self.assertEqual(metadata["setting_count"], 2)
            self.assertEqual(json.loads((options.output / "raw/complete.json").read_text())["records"], 14)
