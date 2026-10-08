"""Small tensor/provenance tests; no extraction, matcher forward or model loading.

Status: NOT RUN — requires manual execution.
"""
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_masks import feature_fingerprint
from features import CONFIGS, ImageFeatures, strategy_name
from superglue_adapter import matcher_inputs, prediction_matches, require_official_files

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch


class SuperGlueProvenanceTests(unittest.TestCase):
    def test_missing_and_altered_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                require_official_files(Path(folder))
            path = Path(folder) / "models/superglue.py"
            path.parent.mkdir()
            path.write_text("raise AssertionError('must never be imported')")
            with self.assertRaises(ValueError):
                require_official_files(Path(folder))

    def test_git_blob_identity_and_sha256(self):
        payload = b"test fixture, not a model"
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder, patch("superglue_adapter.FILES", {"fixture": blob}):
            (Path(folder) / "fixture").write_bytes(payload)
            record = require_official_files(Path(folder))["fixture"]
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["git_blob_sha1"], blob)

    def test_registry_rejects_bf_flags(self):
        self.assertEqual(strategy_name(CONFIGS["superpoint_superglue"], None), "superglue")
        for strategy in ("ratio", "crosscheck", "lightglue"):
            with self.assertRaises(ValueError):
                strategy_name(CONFIGS["superpoint_superglue"], strategy)


@unittest.skipUnless(AVAILABLE, "Requires torch; tests never install dependencies")
class SuperGlueContractTests(unittest.TestCase):
    def native(self, n=3):
        return {"keypoints": torch.arange(2*n, dtype=torch.float32).reshape(1, n, 2),
                "descriptors": torch.arange(n*256, dtype=torch.float32).reshape(1, n, 256),
                "keypoint_scores": torch.full((1, n), .8),
                "image_size": torch.tensor([[320., 240.]])}

    def prediction(self):
        return {"matches0": torch.tensor([[1, -1, 0]]), "matches1": torch.tensor([[2, 0]]),
                "matching_scores0": torch.tensor([[.9, .1, .8]]),
                "matching_scores1": torch.tensor([[.8, .9]])}

    def test_inputs_preserve_pixels_scores_and_descriptor_values(self):
        first, second = self.native(), self.native(2)
        original = {key: value.clone() for key, value in first.items()}
        inputs = matcher_inputs(first, second)
        self.assertIs(inputs["keypoints0"], first["keypoints"])
        self.assertIs(inputs["scores0"], first["keypoint_scores"])
        self.assertEqual(tuple(inputs["descriptors0"].shape), (1, 256, 3))
        self.assertEqual(tuple(inputs["image0"].shape), (1, 1, 240, 320))
        torch.testing.assert_close(inputs["descriptors0"].transpose(1, 2), first["descriptors"])
        for key in first:
            torch.testing.assert_close(first[key], original[key])

    def test_inputs_reject_missing_scores_wrong_shapes_and_invalid_dimensions(self):
        native = self.native()
        del native["keypoint_scores"]
        with self.assertRaises(KeyError):
            matcher_inputs(native, self.native())
        for key, value in (("descriptors", torch.zeros(1, 3, 128)),
                           ("keypoint_scores", torch.tensor([[.1, float("nan"), .3]])),
                           ("image_size", torch.tensor([[320., 0.]]))):
            native = self.native()
            native[key] = value
            with self.assertRaises(ValueError):
                matcher_inputs(native, self.native())

    def test_output_unmatched_indices_scores_and_mutuality(self):
        matches = prediction_matches(self.prediction(), 3, 2)
        self.assertEqual([(m.queryIdx, m.trainIdx) for m in matches], [(0, 1), (2, 0)])
        self.assertAlmostEqual(matches[0].distance, .1, places=6)
        for value in (torch.tensor([[2, -1, 0]]), torch.tensor([[1, -2, 0]]),
                      torch.tensor([[1, 1, 0]]), torch.tensor([[1., -1., 0.]])):
            prediction = self.prediction()
            prediction["matches0"] = value
            with self.assertRaises(ValueError):
                prediction_matches(prediction, 3, 2)

    def test_empty_output_and_inputs(self):
        inputs = matcher_inputs(self.native(0), self.native(2))
        self.assertEqual(tuple(inputs["descriptors0"].shape), (1, 256, 0))
        prediction = {"matches0": torch.empty((1, 0), dtype=torch.long),
                      "matches1": torch.full((1, 2), -1, dtype=torch.long),
                      "matching_scores0": torch.empty((1, 0)), "matching_scores1": torch.zeros(1, 2)}
        self.assertEqual(prediction_matches(prediction, 0, 2), [])

    def test_fingerprint_includes_scores_and_image_sizes(self):
        native = self.native()
        features = ImageFeatures([], [], np.zeros((3, 256), np.float32), {}, native)
        initial = feature_fingerprint(features)
        native["keypoint_scores"][0, 0] = .7
        self.assertNotEqual(initial, feature_fingerprint(features))
        before = feature_fingerprint(features)
        native["image_size"][0, 0] = 640
        self.assertNotEqual(before, feature_fingerprint(features))
