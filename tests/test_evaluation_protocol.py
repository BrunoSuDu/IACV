"""Known answers, optional masks and estimator separation; NOT RUN."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import evaluate_feature_matching, validate_metrics
from homography import estimate_homography, homography_accuracy


def points(coordinates):
    return [cv2.KeyPoint(float(x), float(y), 1) for x, y in coordinates]


class EvaluationProtocolTests(unittest.TestCase):
    def test_validation_rejects_plausible_but_wrong_ratios(self):
        record = {"n_features": 4, "n_putative": 2, "n_correct": 1, "n_correspondences": 2,
                  "n_incorrect": 1, "pmr": .5, "precision": .5, "recall": .5,
                  "matching_score": .25, "repeatability": .5}
        validate_metrics(record)
        for key, wrong in (("recall", .75), ("n_incorrect", 2), ("n_features", 4.5),
                           ("repeatability", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_metrics({**record, key: wrong})

    def test_formulas_and_inclusive_threshold(self):
        first = points([(10, 10), (20, 20), (30, 30)])
        second = points([(13, 10), (20, 20), (80, 80)])
        matches = [cv2.DMatch(0, 0, 0.), cv2.DMatch(1, 2, 0.)]
        metrics = evaluate_feature_matching(first, second, matches, np.eye(3), (100, 100))[0]
        self.assertEqual([metrics[k] for k in ("n_features", "n_putative", "n_correct", "n_correspondences")], [3, 2, 1, 2])
        self.assertAlmostEqual(metrics["pmr"], 2/3)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], .5)
        self.assertAlmostEqual(metrics["matching_score"], 1/3)

    def test_empty_undefined_and_invalid_indices(self):
        metrics = evaluate_feature_matching([], [], [], np.eye(3), (100, 100))[0]
        for key in ("pmr", "precision", "recall", "matching_score"):
            self.assertTrue(np.isnan(metrics[key]))
        with self.assertRaises(ValueError):
            evaluate_feature_matching(points([(10, 10)]), points([(10, 10)]),
                                      [cv2.DMatch(0, 1, 0.)], np.eye(3), (100, 100))

    def test_estimator_direction_and_no_gt_argument(self):
        first = points([(10, 10), (80, 10), (80, 80), (10, 80)])
        second = points([(15, 20), (85, 20), (85, 90), (15, 90)])
        expected = np.array([[1., 0, 5], [0, 1, 10], [0, 0, 1]])
        with patch("homography.cv2.findHomography", return_value=(expected, np.ones((4, 1), np.uint8))) as estimator:
            H, inliers, metrics = estimate_homography(first, second, [cv2.DMatch(i, i, 0.) for i in range(4)])
            np.testing.assert_array_equal(estimator.call_args.args[0], [kp.pt for kp in first])
            np.testing.assert_array_equal(estimator.call_args.args[1], [kp.pt for kp in second])
            self.assertEqual(estimator.call_args.kwargs, {"maxIters": 5000, "confidence": .995})
            self.assertEqual(homography_accuracy(H, expected, (100, 100)), 0)
            self.assertTrue(metrics["homography_success"])

    def test_estimator_failure_keeps_nan_error(self):
        first = points([(10, 10)]*4)
        matches = [cv2.DMatch(i, i, 0.) for i in range(4)]
        for result in ((None, None), (np.zeros((3, 3)), np.ones((4, 1), np.uint8))):
            with patch("homography.cv2.findHomography", return_value=result):
                H, _, metrics = estimate_homography(first, first, matches)
                self.assertFalse(metrics["homography_success"])
                self.assertTrue(np.isnan(homography_accuracy(H, np.eye(3), (100, 100))))
