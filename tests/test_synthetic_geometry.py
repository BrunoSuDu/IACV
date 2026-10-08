"""Known-coordinate geometry and mask tests. NOT RUN during authoring."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets.synthetic import DEFAULT_MANIFEST, expected_synthetic_records, select_sources
from evaluation import evaluate_feature_matching
from feature_masks import filter_features, support_keep
from features import ImageFeatures
from homography import transform_points
from transformations import (Transformation, corners, homography_for, perspective_destination,
                             transformation_grid, warp_with_mask)


class SyntheticGeometryTests(unittest.TestCase):
    shape = (201, 301)

    def H(self, kind, value):
        return homography_for(Transformation(kind, value), self.shape)

    def test_identity_grid_counts_and_dtype(self):
        np.testing.assert_array_equal(self.H("identity", 0), np.eye(3))
        grid = transformation_grid()
        self.assertEqual(len(grid), 28)
        self.assertEqual(sum(s.kind == "identity" for s in grid), 1)
        self.assertEqual(expected_synthetic_records(), 2240)
        for setting in grid:
            H = homography_for(setting, self.shape)
            self.assertEqual(H.dtype, np.float64)
            self.assertTrue(np.isfinite(H).all())
            np.testing.assert_allclose(H @ np.linalg.inv(H), np.eye(3), atol=1e-10)

    def test_rotation_centre_and_known_point(self):
        projected = transform_points([[150, 100], [160, 100]], self.H("rotation", 90))
        np.testing.assert_allclose(projected, [[150, 100], [150, 90]], atol=1e-10)

    def test_scale_translation_shear(self):
        np.testing.assert_allclose(transform_points([[160, 110]], self.H("scale", 2)), [[170, 120]])
        np.testing.assert_allclose(transform_points([[0, 0]], self.H("translation_x", .1)), [[30.1, 0]])
        np.testing.assert_allclose(transform_points([[0, 0]], self.H("translation_y", -.1)), [[0, -20.1]])
        np.testing.assert_allclose(transform_points([[150, 120]], self.H("shear", .25)), [[155, 120]])

    def test_perspective_corner_rule(self):
        for strength in (-.12, -.06, 0, .06, .12):
            H = self.H("perspective", strength)
            np.testing.assert_allclose(transform_points(corners(self.shape), H),
                                       perspective_destination(self.shape, strength), atol=1e-8)
        with self.assertRaises(ValueError):
            self.H("perspective", .5)

    def test_canvas_mask_and_overlap(self):
        image = np.ones(self.shape, np.uint8) * 100
        H = self.H("translation_x", .1)
        warped, valid, metadata = warp_with_mask(image, H)
        self.assertEqual(warped.shape, image.shape)
        self.assertFalse(valid[:, :30].any())
        self.assertTrue(valid[100, 100])
        self.assertAlmostEqual(metadata["overlap_ratio"], .9, delta=.02)
        _, _, identity = warp_with_mask(image, np.eye(3))
        self.assertEqual(identity["overlap_ratio"], 1)

    def test_support_filter_aligns_descriptors_and_all_native_indices(self):
        points = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(5, 100, 1), cv2.KeyPoint(160, 100, 1)]
        descriptors = np.arange(12).reshape(3, 4)
        native = {"keypoints": np.array([[p.pt for p in points]]), "descriptors": descriptors[None],
                  "scales": np.array([[1, 2, 3]]), "oris": np.array([[.1, .2, .3]]),
                  "native_indices": np.array([[10, 11, 12]]), "image_size": np.array([[301, 201]])}
        features = ImageFeatures(points, points, descriptors, {}, native)
        result = filter_features(features, np.ones(self.shape, bool))
        self.assertEqual(len(result.keypoints), 2)
        np.testing.assert_array_equal(result.descriptors, descriptors[[0, 2]])
        np.testing.assert_array_equal(result.native["native_indices"], [[10, 12]])
        np.testing.assert_array_equal(result.native["scales"], [[1, 3]])
        np.testing.assert_array_equal(result.native["oris"], [[.1, .3]])
        np.testing.assert_array_equal(result.native["image_size"], [[301, 201]])

    def test_invalid_hole_and_large_support_are_rejected(self):
        valid = np.ones(self.shape, bool)
        valid[100, 120] = False
        points = [cv2.KeyPoint(100, 100, 1), cv2.KeyPoint(220, 100, 30)]
        self.assertFalse(support_keep(points, valid).any())

    def test_gt_correctness_and_covisible_masks(self):
        first = [cv2.KeyPoint(60, 100, 1), cv2.KeyPoint(280, 100, 1)]
        second = [cv2.KeyPoint(90.1, 100, 1)]
        source = np.ones(self.shape, bool)
        target = np.ones(self.shape, bool)
        target[:, :31] = False
        metrics = evaluate_feature_matching(first, second, [cv2.DMatch(0, 0, 0.)],
                                             self.H("translation_x", .1), self.shape,
                                             source_valid_mask=source, target_valid_mask=target)[0]
        self.assertEqual(metrics["n_features"], 1)
        self.assertEqual(metrics["n_correct"], 1)
        self.assertEqual(metrics["precision"], 1)
        target[100, 90] = False
        self.assertEqual(evaluate_feature_matching(first, second, [], self.H("translation_x", .1),
                                                  self.shape, source_valid_mask=source,
                                                  target_valid_mask=target)[0]["n_features"], 0)

    def test_frozen_source_manifest_and_missing_failure(self):
        spec = json.loads(DEFAULT_MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                select_sources(root)
            for relative in spec["sources"]:
                path = root / relative
                path.parent.mkdir()
                path.touch()
            first, _ = select_sources(root)
            second, _ = select_sources(root)
            self.assertEqual(first, second)
            self.assertEqual([p.relative_to(root).as_posix() for p in first], spec["sources"])
