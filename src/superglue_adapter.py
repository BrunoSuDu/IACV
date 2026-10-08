"""Strict adapter to manually installed, unmodified official SuperGlue.

Importing this module does not load a model, download files or run inference.
"""
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np

COMMIT = "ddcf11f42e7e0732a0c4607648f9448ea8d73590"
REPOSITORY = "https://github.com/magicleap/SuperGluePretrainedNetwork"
RAW_ROOT = f"https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/{COMMIT}/"
INSTALL_ROOT = Path(__file__).resolve().parents[1] / "models/superglue_official"
FILES = {
    "models/superglue.py": "5a89b0348075bcb918eab123bc988c7102137a3d",
    "models/weights/superglue_outdoor.pth": "5ca3392bd5349790d7bc1c246961fc041b88c1e8",
    "LICENSE": "afc1ed1db5d14d18e546de546762dd45fc78ef1d",
}
SUPERPOINT_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
SETTINGS = {"weights": "outdoor", "descriptor_dim": 256,
            "sinkhorn_iterations": 100, "match_threshold": 0.2}


def require_shared_superpoint(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SUPERPOINT_SHA256:
        raise ValueError("SuperGlue requires the official shared SuperPoint checkpoint")


def require_official_files(root=INSTALL_ROOT):
    """Verify Git blob identities without importing code or deserializing weights."""
    records = {}
    for relative, expected in FILES.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; manually download {RAW_ROOT}{relative}")
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                blob.update(block)
                sha256.update(block)
        if blob.hexdigest() != expected:
            raise ValueError(f"Official SuperGlue file differs from pinned Git blob: {path}")
        records[relative] = {"git_blob_sha1": expected, "sha256": sha256.hexdigest(),
                             "url": RAW_ROOT + relative}
    return records


def load_matcher(device):
    provenance = require_official_files()
    spec = importlib.util.spec_from_file_location("iacv_official_superglue", INSTALL_ROOT / "models/superglue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matcher = module.SuperGlue(dict(SETTINGS)).eval().to(device)
    return matcher, {"repository": REPOSITORY, "commit": COMMIT, "files": provenance,
                     "weights": "outdoor", "training_dataset": "MegaDepth",
                     "selection": "a priori outdoor setting; not tuned on HPatches",
                     "shared_features": "pinned LightGlue SuperPoint, same checkpoint; no re-extraction"}


def matcher_inputs(first, second):
    """Reuse scores/pixel coordinates; only transpose descriptor storage.

    Official forward uses image.shape exclusively. Expanded scalar tensors supply
    [1,1,H,W] without allocating or copying image pixels. Normalization remains
    inside official SuperGlue: (xy - [W,H]/2)/(0.7*max(W,H)).
    """
    result = {}
    for index, native in enumerate((first, second)):
        points, descriptors, scores = (native[key] for key in ("keypoints", "descriptors", "keypoint_scores"))
        n = points.shape[1]
        if (tuple(points.shape) != (1, n, 2) or tuple(descriptors.shape) != (1, n, 256)
                or tuple(scores.shape) != (1, n) or tuple(native["image_size"].shape) != (1, 2)):
            raise ValueError("Invalid SuperGlue input shapes")
        values = [points, descriptors, scores, native["image_size"]]
        if any(value.device != points.device or not value.is_floating_point()
               or not bool(value.isfinite().all()) for value in values):
            raise ValueError("SuperGlue requires finite floating tensors on one device")
        if any(value.dtype != points.dtype for value in (descriptors, scores)):
            raise ValueError("SuperGlue feature dtypes differ")
        if bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("Invalid SuperPoint scores")
        dimensions = native["image_size"][0].detach().cpu().tolist()
        if any(value < 1 or value != int(value) for value in dimensions):
            raise ValueError("Invalid image size")
        width, height = map(int, dimensions)
        result[f"keypoints{index}"] = points
        result[f"descriptors{index}"] = descriptors.transpose(1, 2).contiguous()
        result[f"scores{index}"] = scores
        result[f"image{index}"] = points.new_zeros((1, 1, 1, 1)).expand(1, 1, height, width)
    return result


def prediction_matches(prediction, n_first, n_second):
    arrays = {}
    for side, count, other in ((0, n_first, n_second), (1, n_second, n_first)):
        for stem in ("matches", "matching_scores"):
            value = prediction[f"{stem}{side}"].detach().cpu().numpy()
            if value.shape != (1, count):
                raise ValueError("Misaligned SuperGlue output")
            arrays[f"{stem}{side}"] = value[0]
        indices, scores = arrays[f"matches{side}"], arrays[f"matching_scores{side}"]
        if (not np.issubdtype(indices.dtype, np.integer) or np.any(indices < -1)
                or np.any(indices >= other) or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Invalid SuperGlue indices/scores")
    for side in (0, 1):
        indices = arrays[f"matches{side}"]
        valid = np.flatnonzero(indices >= 0)
        if not np.array_equal(arrays[f"matches{1-side}"][indices[valid]], valid):
            raise ValueError("SuperGlue matches must be mutual and one-to-one")
    return [cv2.DMatch(int(q), int(t), float(1-arrays["matching_scores0"][q]))
            for q, t in enumerate(arrays["matches0"]) if t >= 0]
