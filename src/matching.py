import time
import cv2


def enforce_one_to_one(matches):
    """
    Enforce one-to-one correspondence between keypoints.

    Matches with smaller descriptor distance are preferred.
    """
    matches = sorted(
        matches,
        key=lambda match: match.distance,
    )

    used_query = set()
    used_train = set()
    unique_matches = []

    for match in matches:
        if (
            match.queryIdx not in used_query
            and match.trainIdx not in used_train
        ):
            unique_matches.append(match)
            used_query.add(match.queryIdx)
            used_train.add(match.trainIdx)

    return unique_matches


def match_descriptors(
    descriptors1,
    descriptors2,
    norm=cv2.NORM_L2,
    ratio_threshold=0.8,
):
    """
    Match descriptors using KNN matching with the requested distance
    followed by Lowe's ratio test.

    Returns
    -------
    matches : list[cv2.DMatch]
        Putative one-to-one correspondences.
    matching_time : float
        Matching runtime in seconds.
    """
    if not 0 < ratio_threshold < 1:
        raise ValueError("ratio_threshold must be between 0 and 1")
    if (descriptors1 is None or descriptors2 is None
            or len(descriptors1) == 0 or len(descriptors2) < 2):
        return [], 0.0

    matcher = cv2.BFMatcher(
        normType=norm,
        crossCheck=False,
    )

    start = time.perf_counter()

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2,
    )

    ratio_matches = []

    for neighbours in knn_matches:
        if len(neighbours) < 2:
            continue

        m, n = neighbours

        if m.distance < ratio_threshold * n.distance:
            ratio_matches.append(m)

    matches = enforce_one_to_one(ratio_matches)

    matching_time = time.perf_counter() - start

    return matches, matching_time


def match_sift_descriptors(descriptors1, descriptors2, ratio_threshold=0.8):
    """保留原有 SIFT 接口。"""
    return match_descriptors(descriptors1, descriptors2, cv2.NORM_L2, ratio_threshold)
