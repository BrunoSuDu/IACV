from pathlib import Path

import cv2
import numpy as np

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def load_homography(path):
    """
    Load a 3x3 ground-truth homography matrix.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the homography text file.

    Returns
    -------
    H : np.ndarray
        3x3 ground-truth homography.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Homography file not found: {path}"
        )

    H = np.loadtxt(path, dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 homography matrix, "
            f"but got {H.shape} from {path}"
        )

    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
        raise ValueError(f"Invalid homography: {path}")
    return H


def project_keypoints(keypoints, H):
    """
    Project OpenCV keypoints using a homography.

    Returns
    -------
    points : np.ndarray
        Original keypoint coordinates, shape (N, 2).

    projected_points : np.ndarray
        Coordinates after homography projection,
        shape (N, 2).
    """
    if len(keypoints) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    points = np.float32(
        [kp.pt for kp in keypoints]
    )

    # 显式处理投影到无穷远的点，避免被 OpenCV 当成 (0, 0)。
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ H.T
    projected_points = np.full((len(points), 2), np.nan)
    finite = np.abs(homogeneous[:, 2]) > 1e-12
    projected_points[finite] = homogeneous[finite, :2] / homogeneous[finite, 2:3]

    return points, projected_points


def get_covisible_feature_mask(
    projected_points,
    image2_shape,
    valid_region_mask=None,
):
    """
    Determine which projected Image-1 features remain
    inside Image 2.

    Parameters
    ----------
    projected_points : np.ndarray
        Image-1 points projected into Image 2.

    image2_shape : tuple
        Shape of Image 2.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask identifying co-visible features.
    """
    height, width = image2_shape[:2]

    x = projected_points[:, 0]
    y = projected_points[:, 1]

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    inside = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    valid = finite & inside
    if valid_region_mask is not None:
        if valid_region_mask.shape != (height, width):
            raise ValueError("Valid-region mask and image canvas must have the same shape")
        indices = np.flatnonzero(valid)
        coordinates = np.floor(projected_points[indices]).astype(int)
        valid[indices] &= valid_region_mask[coordinates[:, 1], coordinates[:, 0]].astype(bool)
    return valid


def covisible_pair_masks(keypoints1, keypoints2, H, image2_shape,
                         source_valid_mask=None, target_valid_mask=None):
    points1, projected = project_keypoints(keypoints1, H)
    query_visible = get_covisible_feature_mask(projected, image2_shape, target_valid_mask)
    target_visible = np.ones(len(keypoints2), dtype=bool)
    if source_valid_mask is not None:
        query_visible &= get_covisible_feature_mask(points1, source_valid_mask.shape, source_valid_mask)
        points2, inverse = project_keypoints(keypoints2, np.linalg.inv(H))
        target_visible &= get_covisible_feature_mask(inverse, source_valid_mask.shape, source_valid_mask)
        target_visible &= get_covisible_feature_mask(points2, image2_shape, target_valid_mask)
    elif target_valid_mask is not None:
        points2 = np.asarray([kp.pt for kp in keypoints2], np.float64).reshape(-1, 2)
        target_visible &= get_covisible_feature_mask(points2, image2_shape, target_valid_mask)
    return projected, query_visible, target_visible


def count_ground_truth_correspondences(
    projected_points1,
    keypoints2,
    valid_feature_mask,
    correctness_threshold=3.0,
):
    """
    Count the maximum number of one-to-one geometrically
    valid correspondences between detected keypoints.

    Descriptor information is NOT used here.

    A geometrically valid correspondence exists when the
    distance between the GT-projected Image-1 point and an
    Image-2 keypoint is <= correctness_threshold.

    Returns
    -------
    n_correspondences : int
        Maximum number of available one-to-one GT
        correspondences.
    """
    valid_projected_points = (
        projected_points1[valid_feature_mask]
    )

    if (
        len(valid_projected_points) == 0
        or len(keypoints2) == 0
    ):
        return 0

    points2 = np.float64(
        [kp.pt for kp in keypoints2]
    )

    # 只查找半径内的邻居，避免 FAST 点很多时创建巨大的 N×M 距离矩阵。
    # 阈值和最大一对一匹配定义与原来的 cdist 实现完全相同。
    neighbours = cKDTree(points2).query_ball_point(
        valid_projected_points, r=correctness_threshold
    )
    rows, columns = [], []
    for row, candidates in enumerate(neighbours):
        rows.extend([row] * len(candidates))
        columns.extend(candidates)
    sparse_graph = csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)),
        shape=(len(valid_projected_points), len(points2)),
    )

    gt_matching = maximum_bipartite_matching(
        sparse_graph,
        perm_type="column",
    )

    n_correspondences = int(
        np.sum(gt_matching != -1)
    )

    return n_correspondences


def evaluate_feature_matching(
    keypoints1,
    keypoints2,
    putative_matches,
    H_gt,
    image2_shape,
    correctness_threshold=3.0,
    source_valid_mask=None,
    target_valid_mask=None,
):
    """
    Evaluate descriptor matching using ground-truth geometry.

    Metrics
    -------
    PMR =
        N_putative / N_features

    Precision =
        N_correct / N_putative

    Matching Score =
        N_correct / N_features

    Recall =
        N_correct / N_correspondences

    Protocol
    --------
    N_features:
        Number of Image-1 detected keypoints whose
        GT projection falls inside Image 2.

    N_putative:
        One-to-one descriptor matches whose Image-1
        feature belongs to the evaluation region.

    N_correct:
        Putative matches whose GT geometric error
        is <= correctness_threshold.

    N_correspondences:
        Maximum number of one-to-one detected-feature
        correspondences allowed by GT geometry alone.

    Returns
    -------
    evaluation : dict

    evaluated_putative_matches : list

    correct_matches : list

    incorrect_matches : list

    errors : np.ndarray
        GT geometric error of each evaluated putative match.
    """

    if correctness_threshold <= 0:
        raise ValueError("correctness_threshold must be positive")
    if (len({m.queryIdx for m in putative_matches}) != len(putative_matches)
            or len({m.trainIdx for m in putative_matches}) != len(putative_matches)):
        raise ValueError("Evaluation requires one-to-one putative matches")
    for match in putative_matches:
        if not (0 <= match.queryIdx < len(keypoints1) and 0 <= match.trainIdx < len(keypoints2)):
            raise ValueError("Match index outside aligned feature arrays")
    projected_points1, valid_feature_mask, target_visible = covisible_pair_masks(
        keypoints1, keypoints2, H_gt, image2_shape, source_valid_mask, target_valid_mask
    )

    points2 = np.float64(
        [kp.pt for kp in keypoints2]
    )

    # --------------------------------------------------
    # 1. Determine co-visible Image-1 features
    # --------------------------------------------------

    valid_feature_indices = set(
        np.flatnonzero(
            valid_feature_mask
        ).tolist()
    )

    n_features = len(
        valid_feature_indices
    )

    # --------------------------------------------------
    # 2. Restrict putative matches to evaluation region
    # --------------------------------------------------

    evaluated_putative_matches = [
        match
        for match in putative_matches
        if match.queryIdx in valid_feature_indices and target_visible[match.trainIdx]
    ]

    n_putative = len(
        evaluated_putative_matches
    )

    # --------------------------------------------------
    # 3. Compute GT geometric error
    # --------------------------------------------------

    errors = []

    correct_matches = []
    incorrect_matches = []

    for match in evaluated_putative_matches:

        gt_position = projected_points1[
            match.queryIdx
        ]

        matched_position = points2[
            match.trainIdx
        ]

        error = float(
            np.linalg.norm(
                gt_position - matched_position
            )
        )

        errors.append(error)

        if error <= correctness_threshold:
            correct_matches.append(match)
        else:
            incorrect_matches.append(match)

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    n_correct = len(correct_matches)

    # --------------------------------------------------
    # 4. Number of available GT correspondences
    # --------------------------------------------------

    n_correspondences = (
        count_ground_truth_correspondences(
            projected_points1,
            [kp for kp, visible in zip(keypoints2, target_visible) if visible],
            valid_feature_mask,
            correctness_threshold=
                correctness_threshold,
        )
    )

    # --------------------------------------------------
    # 5. Required Assignment metrics
    # --------------------------------------------------

    pmr = (
        n_putative / n_features
        if n_features > 0
        else np.nan
    )

    precision = (
        n_correct / n_putative
        if n_putative > 0
        else np.nan
    )

    matching_score = (
        n_correct / n_features
        if n_features > 0
        else np.nan
    )

    recall = (
        n_correct / n_correspondences
        if n_correspondences > 0
        else np.nan
    )

    # GT error among correct correspondences only
    correct_errors = errors[
        errors <= correctness_threshold
    ]

    mean_correct_gt_error = (
        float(np.mean(correct_errors))
        if len(correct_errors) > 0
        else np.nan
    )

    median_correct_gt_error = (
        float(np.median(correct_errors))
        if len(correct_errors) > 0
        else np.nan
    )

    evaluation = {
        "n_features": n_features,
        "n_putative": n_putative,
        "n_correct": n_correct,
        "n_incorrect": (
            n_putative - n_correct
        ),
        "n_correspondences": n_correspondences,

        "pmr": pmr,
        "precision": precision,
        "matching_score": matching_score,
        "recall": recall,
        "repeatability": n_correspondences / n_features if n_features else np.nan,

        "correctness_threshold_px":
            correctness_threshold,

        "mean_correct_gt_error_px":
            mean_correct_gt_error,

        "median_correct_gt_error_px":
            median_correct_gt_error,
    }

    validate_metrics(evaluation)
    return (
        evaluation,
        evaluated_putative_matches,
        correct_matches,
        incorrect_matches,
        errors,
    )


def validate_metrics(result):
    """任何计数/分母不一致都应停止实验，不能悄悄写入结果表。"""
    for key in ("n_features", "n_putative", "n_correct", "n_correspondences", "n_incorrect"):
        value = result[key]
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError(f"Invalid integer count {key}: {value}")
    if result["n_incorrect"] != result["n_putative"] - result["n_correct"]:
        raise ValueError("Incorrect-match count is inconsistent")
    if not (0 <= result["n_correct"] <= result["n_putative"] <= result["n_features"]):
        raise ValueError(f"Invalid matching counts: {result}")
    if not (result["n_correct"] <= result["n_correspondences"] <= result["n_features"]):
        raise ValueError(f"Invalid GT correspondence count: {result}")
    for name in ("pmr", "precision", "matching_score", "recall", "repeatability"):
        value = result[name]
        if not np.isnan(value) and not 0 <= value <= 1:
            raise ValueError(f"Invalid {name}: {value}")
    fractions = {"pmr": ("n_putative", "n_features"), "precision": ("n_correct", "n_putative"),
                 "matching_score": ("n_correct", "n_features"), "recall": ("n_correct", "n_correspondences"),
                 "repeatability": ("n_correspondences", "n_features")}
    for name, (numerator, denominator) in fractions.items():
        expected = result[numerator] / result[denominator] if result[denominator] else np.nan
        if not np.isclose(result[name], expected, equal_nan=True):
            raise ValueError(f"Inconsistent {name} numerator/denominator")
    if np.isfinite(result["precision"]) and np.isfinite(result["pmr"]):
        if not np.isclose(result["matching_score"], result["pmr"] * result["precision"]):
            raise ValueError("Matching score must equal PMR * precision")
