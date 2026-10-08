import time
import cv2


def create_detector(method, params=None):
    """只负责创建 detector，与数据集无关。"""
    factories = {
        "sift": cv2.SIFT_create,
        "orb": cv2.ORB_create,
        "fast": cv2.FastFeatureDetector_create,
        "kaze": cv2.KAZE_create,
    }
    if method not in factories:
        raise ValueError(f"Unknown detector: {method}")
    return factories[method](**(params or {}))


def detect_keypoints(image_gray, method, params=None):
    detector = create_detector(method, params)
    start = time.perf_counter()
    keypoints = detector.detect(image_gray, None)
    return keypoints, time.perf_counter() - start


def detect_sift_keypoints(image_gray, nfeatures=0):
    """
    Detect SIFT keypoints in a grayscale image.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale input image.
    nfeatures : int
        Maximum number of features retained by SIFT.
        0 uses OpenCV's default behaviour.

    Returns
    -------
    keypoints : tuple
        Detected OpenCV KeyPoint objects.
    detection_time : float
        Detection runtime in seconds.
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    start = time.perf_counter()
    keypoints = sift.detect(image_gray, None)
    detection_time = time.perf_counter() - start

    return keypoints, detection_time
