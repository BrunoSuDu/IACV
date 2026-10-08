"""估计 H 不接触 GT；只有 accuracy 函数使用 GT。"""
import cv2
import numpy as np


def transform_points(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    projected = np.full((len(points), 2), np.nan)
    valid = np.abs(homogeneous[:, 2]) > 1e-12
    projected[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return projected


def estimate_homography(keypoints1, keypoints2, matches, threshold_px=3.0, seed=0):
    """用全部 raw putative matches 做 RANSAC，不用 GT 筛选。"""
    metrics = {
        "homography_success": False,
        "ransac_inliers": 0,
        "ransac_inlier_ratio": 0.0 if matches else np.nan,
        "ransac_reprojection_error_px": np.nan,
    }
    if len(matches) < 4:
        return None, [], metrics
    points1 = np.float64([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float64([keypoints2[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(seed)
    estimated, mask = cv2.findHomography(
        points1, points2, cv2.RANSAC, threshold_px, maxIters=5000, confidence=0.995
    )
    if (estimated is None or mask is None or not np.isfinite(estimated).all()
            or np.linalg.matrix_rank(estimated) < 3):
        return None, [], metrics
    inlier_mask = mask.ravel().astype(bool)
    residuals = np.linalg.norm(transform_points(points1, estimated) - points2, axis=1)
    if inlier_mask.sum() < 4 or not np.isfinite(residuals[inlier_mask]).all():
        return None, [], metrics
    inlier_matches = [match for match, keep in zip(matches, inlier_mask) if keep]
    metrics.update({
        "homography_success": True,
        "ransac_inliers": len(inlier_matches),
        "ransac_inlier_ratio": len(inlier_matches) / len(matches),
        "ransac_reprojection_error_px": float(np.mean(residuals[inlier_mask])),
    })
    return estimated, inlier_matches, metrics


def homography_accuracy(estimated, ground_truth, image1_shape):
    """Image 1 四个角点的 mean corner transfer error，单位为像素。"""
    if estimated is None:
        return np.nan
    height, width = image1_shape[:2]
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    errors = np.linalg.norm(
        transform_points(corners, estimated) - transform_points(corners, ground_truth), axis=1
    )
    return float(errors.mean()) if np.isfinite(errors).all() else np.nan
