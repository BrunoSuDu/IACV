"""Tensor/API tests and opt-in real inference tests. NOT RUN during authoring."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("lightglue") is not None
if AVAILABLE:
    import torch
    from features import CONFIGS
    from feature_masks import filter_features
    from learned_features import (CHECKPOINTS, MODEL_ROOT, PINNED_COMMIT, LearnedPipeline,
                                  prediction_matches, require_checkpoint, validate_native,
                                  validate_checkpoint_parameters, verify_pinned_installation)


@unittest.skipUnless(AVAILABLE, "Requires pinned LightGlue and torch; no installation/download in tests")
class LearnedContractTests(unittest.TestCase):
    def native(self, source="sift"):
        data = {"keypoints": torch.tensor([[[100., 100.], [150., 100.]]]),
                "descriptors": torch.zeros((1, 2, 128 if source == "sift" else 256)),
                "image_size": torch.tensor([[256., 256.]]),
                "native_indices": torch.tensor([[8, 9]]), "keypoint_scores": torch.tensor([[.8, .9]])}
        if source == "sift":
            data.update({"scales": torch.tensor([[4., 8.]]), "oris": torch.tensor([[0., np.pi/2]])})
        return data

    def test_sift_batched_input_and_device_validation(self):
        native = self.native()
        validate_native(native, "sift", "cpu")
        self.assertEqual(native["scales"].shape, (1, 2))
        self.assertAlmostEqual(native["oris"][0, 1].item(), np.pi/2, places=6)
        native["descriptors"] = torch.zeros((1, 1, 128))
        with self.assertRaises(ValueError):
            validate_native(native, "sift", "cpu")

    def test_superpoint_shape_and_dim(self):
        native = self.native("superpoint")
        validate_native(native, "superpoint", "cpu")
        native["descriptors"] = torch.zeros((1, 2, 128))
        with self.assertRaises(ValueError):
            validate_native(native, "superpoint", "cpu")

    def test_lightglue_output_indices_and_alignment(self):
        prediction = {"matches": [torch.tensor([[0, 1], [1, 0]])], "scores": [torch.tensor([.9, .8])]}
        matches = prediction_matches(prediction, 2, 2)
        self.assertEqual({(m.queryIdx, m.trainIdx) for m in matches}, {(0, 1), (1, 0)})
        prediction["matches"] = [torch.tensor([[0, 2]])]
        prediction["scores"] = [torch.tensor([.9])]
        with self.assertRaises(ValueError):
            prediction_matches(prediction, 2, 2)

    def test_duplicate_lightglue_output_is_error(self):
        prediction = {"matches": [torch.tensor([[0, 0], [1, 0]])], "scores": [torch.tensor([.9, .8])]}
        with self.assertRaises(ValueError):
            prediction_matches(prediction, 2, 2)

    def test_pinned_sift_pretrained_contract(self):
        from lightglue import LightGlue, SIFT
        from lightglue.sift import sift_to_rootsift
        self.assertEqual(LightGlue.features["sift"]["input_dim"], 128)
        self.assertTrue(LightGlue.features["sift"]["add_scale_ori"])
        self.assertTrue(SIFT.default_conf["rootsift"])
        self.assertEqual(SIFT.default_conf["detection_threshold"], .0066667)
        self.assertEqual(SIFT.default_conf["num_octaves"], 4)
        descriptors = torch.tensor([[[1., 4., 0.]]])
        expected = torch.sqrt(torch.clamp(descriptors / descriptors.sum(-1, keepdim=True), min=1e-6))
        expected = expected / expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(sift_to_rootsift(descriptors.clone()), expected)
        verify_pinned_installation()

    def test_missing_checkpoint_no_download(self):
        with tempfile.TemporaryDirectory() as directory, patch("learned_features.MODEL_ROOT", Path(directory)), patch("torch.hub.load_state_dict_from_url") as loader:
            with self.assertRaises(FileNotFoundError):
                require_checkpoint("sift_lightglue")
            loader.assert_not_called()

    def test_checkpoint_parameter_coverage_rejects_missing_or_wrong_shape(self):
        model = torch.nn.Linear(2, 3)
        with patch("learned_features.torch.load", return_value={}):
            with self.assertRaises(RuntimeError):
                validate_checkpoint_parameters(model, Path("mock.pth"))
        with patch("learned_features.torch.load", return_value={"weight": torch.zeros(3, 1), "bias": torch.zeros(3)}):
            with self.assertRaises(RuntimeError):
                validate_checkpoint_parameters(model, Path("mock.pth"))
        with patch("learned_features.torch.load", return_value=model.state_dict()):
            validate_checkpoint_parameters(model, Path("mock.pth"))

    def test_torch_native_filter_preserves_feature_fields_and_indices(self):
        import cv2
        from features import ImageFeatures
        native = self.native()
        native["keypoints"][0, 1] = torch.tensor([5., 100.])
        points = [cv2.KeyPoint(100, 100, 4), cv2.KeyPoint(5, 100, 8)]
        features = ImageFeatures(points, points, native["descriptors"][0].numpy(), {}, native)
        result = filter_features(features, np.ones((256, 256), bool))
        self.assertEqual(len(result.keypoints), 1)
        self.assertEqual(result.native["descriptors"].shape, (1, 1, 128))
        self.assertEqual(result.native["scales"][0, 0].item(), 4)
        self.assertEqual(result.native["native_indices"][0, 0].item(), 8)

    def test_lightglue_crosscheck_rejected_without_loading_model(self):
        pipeline = object.__new__(LearnedPipeline)
        pipeline.config = CONFIGS["sift_lightglue"]
        with self.assertRaises(ValueError):
            pipeline.match(None, None, matching_strategy="crosscheck")


@unittest.skipUnless(AVAILABLE and os.environ.get("IACV_RUN_MODEL_TESTS") == "1",
                     "NOT RUN: opt in with IACV_RUN_MODEL_TESTS=1; requires local checkpoints")
class RealLearnedIntegrationTests(unittest.TestCase):
    def test_superpoint_bf_and_lightglue_cpu_and_optional_cuda(self):
        self.check_models(["superpoint", "superpoint_lightglue"])
        self.exercise("superpoint")
        self.exercise("superpoint_lightglue")

    def test_superpoint_superglue_cpu_and_optional_cuda(self):
        self.check_models(["superpoint"])
        from superglue_adapter import require_official_files
        require_official_files()
        self.exercise("superpoint_superglue")

    def test_sift_lightglue_cpu_and_optional_cuda(self):
        self.check_models(["sift_lightglue"])
        self.exercise("sift_lightglue")

    def check_models(self, kinds):
        for kind in kinds:
            if not (MODEL_ROOT / "checkpoints" / CHECKPOINTS[kind][0]).is_file():
                self.skipTest(f"Missing local checkpoint {kind}; tests never download")

    def exercise(self, method):
        from classical_pipeline import match_classical_features
        from features import CONTROL_CONFIG
        image = np.random.default_rng(0).integers(0, 256, (256, 256), np.uint8)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                pipeline = LearnedPipeline(CONFIGS[method], device, 128, Path(directory))
                first = pipeline.extract(image, warmup=0, repetitions=1)
                source = "sift" if method == "sift_lightglue" else "superpoint"
                validate_native(first.native, source, pipeline.device)
                self.assertEqual(len(first.keypoints), len(first.descriptors))
                bf_config = CONTROL_CONFIG if source == "sift" else CONFIGS["superpoint"]
                bf, _ = match_classical_features(first, first, bf_config, warmup=0, repetitions=1)
                lg, _ = pipeline.match(first, first, warmup=0, repetitions=1, matching_strategy="superglue" if method == "superpoint_superglue" else "lightglue")
                self.assertGreater(len(first.keypoints), 0)
                self.assertGreater(len(bf), 0)
                self.assertGreater(len(lg), 0)
                import json
                metadata = json.loads((Path(directory) / f"{method}.json").read_text())
                self.assertEqual(metadata["commit"], PINNED_COMMIT)
                self.assertTrue(metadata["checkpoint_sha256"])
