"""用已知答案验证评估公式、空输入、投影和 RANSAC。"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import (count_ground_truth_correspondences, evaluate_feature_matching,
                        get_covisible_feature_mask, project_keypoints)
from homography import estimate_homography, homography_accuracy
from matching import enforce_one_to_one, match_descriptors
from image_io import load_image


def keypoints(points):
    return [cv2.KeyPoint(float(x), float(y), 5) for x, y in points]


class GeometryTests(unittest.TestCase):
    def test_known_translation_and_counts(self):
        first = keypoints([(5, 5), (15, 5), (25, 5), (90, 90)])
        second = keypoints([(15, 5), (25, 5), (35, 5), (80, 80)])
        gt = np.array([[1., 0, 10], [0, 1, 0], [0, 0, 1]])
        matches = [cv2.DMatch(0, 0, 1.), cv2.DMatch(1, 3, 1.), cv2.DMatch(3, 1, 1.)]
        metrics, evaluated, correct, incorrect, _ = evaluate_feature_matching(
            first, second, matches, gt, (100, 100)
        )
        self.assertEqual([metrics[k] for k in ("n_features", "n_putative", "n_correct",
                                               "n_correspondences")], [3, 2, 1, 3])
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(len(correct), 1)
        self.assertEqual(len(incorrect), 1)
        self.assertAlmostEqual(metrics["matching_score"], 1 / 3)
        self.assertEqual(metrics["repeatability"], 1)

    def test_maximum_matching_not_greedy(self):
        # 第一行能连两列，第二行只能连第一列；最大匹配数应为 2。
        projected = np.array([[1., 0], [0., 0]])
        self.assertEqual(count_ground_truth_correspondences(
            projected, keypoints([(0, 0), (2, 0)]), np.ones(2, dtype=bool), 1.1
        ), 2)

    def test_sparse_graph_matches_original_dense_definition(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            projected = rng.uniform(0, 20, (30, 2))
            second = keypoints(rng.uniform(0, 20, (35, 2)))
            dense = csr_matrix(cdist(projected, [point.pt for point in second]) <= 3)
            expected = np.count_nonzero(maximum_bipartite_matching(dense, perm_type="column") >= 0)
            actual = count_ground_truth_correspondences(projected, second, np.ones(30, bool))
            self.assertEqual(actual, expected)

    def test_projection_at_infinity_is_not_visible(self):
        _, projected = project_keypoints(keypoints([(0, 5)]),
                                          np.array([[1., 0, 0], [0, 1, 0], [1, 0, 0]]))
        self.assertFalse(get_covisible_feature_mask(projected, (100, 100))[0])

    def test_empty_inputs_and_insufficient_neighbours(self):
        result = evaluate_feature_matching([], [], [], np.eye(3), (100, 100))[0]
        self.assertEqual(result["n_features"], 0)
        self.assertTrue(np.isnan(result["precision"]))
        descriptors = np.zeros((1, 32), np.uint8)
        self.assertEqual(match_descriptors(descriptors, descriptors, cv2.NORM_HAMMING)[0], [])
        self.assertIsNone(estimate_homography([], [], [])[0])

    def test_one_to_one_rejects_duplicate_target(self):
        matches = [cv2.DMatch(0, 0, 2.), cv2.DMatch(1, 0, 1.)]
        self.assertEqual(enforce_one_to_one(matches)[0].queryIdx, 1)
        with self.assertRaises(ValueError):
            evaluate_feature_matching(keypoints([(5, 5), (8, 8)]), keypoints([(5, 5)]),
                                      matches, np.eye(3), (100, 100))

    def test_ransac_with_outliers(self):
        rng = np.random.default_rng(5)
        first = rng.uniform(0, 200, (40, 2))
        second = first + [10, 5]
        second[-8:] = rng.uniform(0, 200, (8, 2))
        estimated, inliers, metrics = estimate_homography(
            keypoints(first), keypoints(second), [cv2.DMatch(i, i, 0.) for i in range(40)]
        )
        expected = np.array([[1., 0, 10], [0, 1, 5], [0, 0, 1]])
        self.assertEqual(len(inliers), 32)
        self.assertTrue(metrics["homography_success"])
        self.assertLess(homography_accuracy(estimated, expected, (200, 200)), 0.001)

    def test_unicode_image_path(self):
        path = Path(__file__).resolve().parents[1] / "data" / "graf" / "img1.ppm"
        color, gray = load_image(path)
        self.assertEqual(color.shape[:2], gray.shape)


if __name__ == "__main__":
    unittest.main()
