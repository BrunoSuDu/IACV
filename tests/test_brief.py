import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brief import compute_brief, make_brief_pattern


class BriefTests(unittest.TestCase):
    def test_pattern_is_fixed_and_strategies_differ(self):
        uniform = make_brief_pattern("uniform")
        np.testing.assert_array_equal(uniform, make_brief_pattern("uniform"))
        gaussian = make_brief_pattern("gaussian")
        self.assertFalse(np.array_equal(uniform, gaussian))
        self.assertTrue((np.abs(gaussian) <= 15).all())

    def test_intensity_comparison_and_border_filtering(self):
        # 水平递增的图像：左像素 < 右像素，所以每位应为 1。
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        pattern = np.tile(np.array([[[-1, 0], [1, 0]]]), (256, 1, 1))
        points = [cv2.KeyPoint(32, 32, 5), cv2.KeyPoint(1, 1, 5)]
        kept, descriptors = compute_brief(image, points, pattern)
        self.assertEqual(len(kept), 1)
        self.assertEqual(descriptors.shape, (1, 32))
        self.assertTrue((descriptors == 255).all())
        empty = compute_brief(image, [], pattern)[1]
        self.assertEqual(empty.shape, (0, 32))


if __name__ == "__main__":
    unittest.main()
