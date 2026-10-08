"""Small array tests; no dataset or learned model. NOT RUN during authoring."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import BF_METHODS, CONFIGS, ImageFeatures, strategy_name
from classical_pipeline import match_classical_features
from matching import enforce_one_to_one, match_descriptors


class MatchingStrategyTests(unittest.TestCase):
    def test_ratio_preserves_original_algorithm(self):
        first = np.array([[0., 0.], [5., 0.], [5.1, 0.]], np.float32)
        second = np.array([[0., 0.], [0.5, 0.], [5., 0.]], np.float32)
        candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(first, second, k=2)
        expected = enforce_one_to_one([a for a, b in candidates if a.distance < 0.8*b.distance])
        actual, _ = match_descriptors(first, second)
        self.assertEqual([(m.queryIdx, m.trainIdx) for m in actual],
                         [(m.queryIdx, m.trainIdx) for m in expected])

    def test_knn_two_and_strict_ratio(self):
        descriptors = np.zeros((2, 3), np.float32)
        with patch("matching.cv2.BFMatcher") as factory:
            matcher = factory.return_value
            matcher.knnMatch.return_value = [[cv2.DMatch(0, 0, 8.), cv2.DMatch(0, 1, 10.)]]
            self.assertEqual(match_descriptors(descriptors, descriptors)[0], [])
            self.assertEqual(matcher.knnMatch.call_args.kwargs["k"], 2)
            factory.assert_called_once_with(normType=cv2.NORM_L2, crossCheck=False)

    def test_crosscheck_mutual_nearest_and_one_to_one(self):
        first = np.array([[0.], [0.2], [10.]], np.float32)
        second = np.array([[0.05], [10.]], np.float32)
        matches, _ = match_descriptors(first, second, matching_strategy="crosscheck")
        self.assertEqual({(m.queryIdx, m.trainIdx) for m in matches}, {(0, 0), (2, 1)})
        self.assertEqual(len({m.queryIdx for m in matches}), len(matches))
        self.assertEqual(len({m.trainIdx for m in matches}), len(matches))

    def test_crosscheck_never_calls_knn_or_ratio(self):
        descriptors = np.zeros((1, 3), np.float32)
        with patch("matching.cv2.BFMatcher") as factory:
            factory.return_value.match.return_value = [cv2.DMatch(0, 0, 0.)]
            self.assertEqual(len(match_descriptors(descriptors, descriptors, ratio_threshold=None,
                                                  matching_strategy="crosscheck")[0]), 1)
            factory.assert_called_once_with(normType=cv2.NORM_L2, crossCheck=True)
            factory.return_value.knnMatch.assert_not_called()

    def test_empty_and_one_neighbour(self):
        empty = np.empty((0, 32), np.uint8)
        one = np.zeros((1, 32), np.uint8)
        for strategy in ("ratio", "crosscheck"):
            for first, second in ((None, one), (one, None), (empty, one), (one, empty)):
                self.assertEqual(match_descriptors(first, second, cv2.NORM_HAMMING,
                                                  matching_strategy=strategy)[0], [])
        self.assertEqual(match_descriptors(one, one, cv2.NORM_HAMMING)[0], [])
        self.assertEqual(len(match_descriptors(one, one, cv2.NORM_HAMMING,
                                              matching_strategy="crosscheck")[0]), 1)

    def test_norm_registry_and_unsupported_combinations(self):
        for method in BF_METHODS:
            expected = cv2.NORM_L2 if method in ("sift", "kaze", "superpoint") else cv2.NORM_HAMMING
            self.assertEqual(CONFIGS[method].norm, expected)
        for method in ("sift_lightglue", "superpoint_lightglue", "superpoint_superglue"):
            for strategy in ("ratio", "crosscheck"):
                with self.assertRaises(ValueError):
                    strategy_name(CONFIGS[method], strategy)
        with self.assertRaises(ValueError):
            match_descriptors(None, None, matching_strategy="lightglue")

    def test_two_strategies_receive_identical_arrays_without_extraction(self):
        descriptors = np.array([[0.], [5.]], np.float32)
        features = ImageFeatures([], [], descriptors, {})
        with patch("classical_pipeline.match_descriptors", return_value=([], 0)) as match:
            for strategy in ("ratio", "crosscheck"):
                match_classical_features(features, features, CONFIGS["sift"], warmup=0,
                                         repetitions=1, matching_strategy=strategy)
            self.assertEqual(match.call_count, 2)
            for call in match.call_args_list:
                self.assertIs(call.args[0], descriptors)
                self.assertIs(call.args[1], descriptors)
