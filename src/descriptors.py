import time
import cv2


def create_descriptor(method):
    factories = {
        "sift": cv2.SIFT_create,
        "orb": cv2.ORB_create,
        "kaze": cv2.KAZE_create,
        "brisk": cv2.BRISK_create,
        "brief": cv2.xfeatures2d.BriefDescriptorExtractor_create,
        "freak": cv2.xfeatures2d.FREAK_create,
    }
    if method not in factories:
        raise ValueError(f"Unknown descriptor: {method}")
    return factories[method]()


def compute_descriptors(image_gray, keypoints, method):
    descriptor = create_descriptor(method)
    start = time.perf_counter()
    keypoints, values = descriptor.compute(image_gray, keypoints)
    return keypoints, values, time.perf_counter() - start


def compute_sift_descriptors(image_gray, keypoints):
    """
    Compute SIFT descriptors for previously detected keypoints.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale input image.
    keypoints : sequence of cv2.KeyPoint
        Detected keypoints.

    Returns
    -------
    keypoints : tuple
        Keypoints for which descriptors were successfully computed.
    descriptors : np.ndarray or None
        SIFT descriptors with shape (N, 128).
    description_time : float
        Descriptor computation runtime in seconds.
    """
    sift = cv2.SIFT_create()

    start = time.perf_counter()
    keypoints, descriptors = sift.compute(
        image_gray,
        keypoints,
    )
    description_time = time.perf_counter() - start

    return keypoints, descriptors, description_time
