"""Shared pinned feature extraction with BF, LightGlue and official SuperGlue."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import cv2
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from lightglue import LightGlue, SIFT, SuperPoint
from lightglue.sift import filter_dog_point, run_opencv_sift

from classical_pipeline import match_classical_features
from features import ImageFeatures, LEARNED_MATCHERS
from matching import enforce_one_to_one
from timing import measure

PINNED_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
CHECKPOINTS = {
    "superpoint": ("superpoint_v1.pth", "superpoint_v1.pth"),
    "superpoint_lightglue": ("superpoint_lightglue_v0-1_arxiv.pth", "superpoint_lightglue.pth"),
    "sift_lightglue": ("sift_lightglue_v0-1_arxiv.pth", "sift_lightglue.pth"),
}
RELEASE_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/"


def verify_pinned_installation():
    distribution = importlib.metadata.distribution("lightglue")
    record = distribution.read_text("direct_url.json")
    if not record or json.loads(record).get("vcs_info", {}).get("commit_id") != PINNED_COMMIT:
        raise RuntimeError("LightGlue must have the exact pinned commit in direct_url.json")


def require_checkpoint(kind):
    filename, remote = CHECKPOINTS[kind]
    path = MODEL_ROOT / "checkpoints" / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing local checkpoint {path}. Download manually: {RELEASE_URL}{remote}")
    return path


def validate_checkpoint_parameters(model, path, lightglue=False):
    """Pinned upstream loads with strict=False; require coverage of every parameter.

    Buffer differences are allowed; missing/wrong-shaped trained parameters are not.
    This is model setup, outside measured extraction/matching.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if lightglue:
        for i in range(model.conf.n_layers):
            state = {key.replace(f"self_attn.{i}", f"transformers.{i}.self_attn")
                     .replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn"): value
                     for key, value in state.items()}
    for name, parameter in model.named_parameters():
        if name not in state or tuple(state[name].shape) != tuple(parameter.shape):
            raise RuntimeError(f"Incompatible pretrained checkpoint parameter {name}: {path}")


class EmptySafeSIFT(SIFT):
    """Official OpenCV SIFT path with well-shaped empty outputs.

    Uses pinned run_opencv_sift and filter_dog_point; inherited forward applies
    official sift_to_rootsift (L1, clamp eps=1e-6, sqrt, L2). Parameters unchanged.
    """
    def extract_single_image(self, image):
        points, scores, scales, angles, descriptors = run_opencv_sift(
            self.sift, (image.cpu().numpy().squeeze(0) * 255.0).astype(np.uint8))
        if descriptors is None or len(points) == 0:
            return {"keypoints": torch.empty((0, 2)), "keypoint_scores": torch.empty(0),
                    "scales": torch.empty(0), "oris": torch.empty(0),
                    "descriptors": torch.empty((0, 128))}
        pred = {"keypoints": points, "keypoint_scores": scores, "scales": scales,
                "oris": angles, "descriptors": descriptors}
        if self.conf.nms_radius is not None:
            keep = filter_dog_point(points, scales, angles, image.shape[-2:],
                                    self.conf.nms_radius, scores=scores)
            pred = {key: value[keep] for key, value in pred.items()}
        pred = {key: torch.from_numpy(value) for key, value in pred.items()}
        cap = self.conf.max_num_keypoints
        if cap is not None and len(pred["keypoints"]) > cap:
            indices = torch.topk(pred["keypoint_scores"], cap).indices
            pred = {key: value[indices] for key, value in pred.items()}
        return pred


def validate_native(native, feature_source, device):
    dim = 128 if feature_source == "sift" else 256
    n = native["keypoints"].shape[1]
    expected = {"keypoints": (1, n, 2), "descriptors": (1, n, dim), "image_size": (1, 2)}
    if feature_source == "sift":
        expected.update({"scales": (1, n), "oris": (1, n)})
    if feature_source == "superpoint":
        expected["keypoint_scores"] = (1, n)
    for key, shape in expected.items():
        if tuple(native[key].shape) != shape or native[key].device != torch.device(device):
            raise ValueError(f"Invalid {key}: expected {shape} on {device}")
        if not torch.isfinite(native[key]).all():
            raise ValueError(f"Nonfinite native features: {key}")


def prediction_matches(prediction, n_first, n_second):
    pairs = prediction["matches"][0].detach().cpu().numpy().reshape(-1, 2)
    scores = prediction["scores"][0].detach().cpu().numpy().reshape(-1)
    if (len(pairs) != len(scores) or not np.isfinite(scores).all()
            or not np.issubdtype(pairs.dtype, np.integer) or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Misaligned LightGlue matches/scores")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= n_first)
                       or np.any(pairs[:, 1] >= n_second)):
        raise ValueError("LightGlue index outside retained features")
    matches = [cv2.DMatch(int(q), int(t), float(1 - score)) for (q, t), score in zip(pairs, scores)]
    unique = enforce_one_to_one(matches)
    if len(unique) != len(matches):
        raise ValueError("LightGlue output is not one-to-one")
    return unique


class LearnedPipeline:
    def __init__(self, config, device, max_keypoints, manifest_dir=None, seed=0):
        verify_pinned_installation()
        if max_keypoints <= 0:
            raise ValueError("Compatible/learned extraction requires a positive feature cap")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu")
        self.config = config
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        self.feature_source = "sift" if config.detector == "sift_compatible" else "superpoint"
        torch.manual_seed(seed)
        torch.set_num_threads(cv2.getNumThreads())
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.hub.set_dir(str(MODEL_ROOT))
        self.checkpoints = {}
        if self.feature_source == "superpoint":
            self.checkpoints["superpoint"] = require_checkpoint("superpoint")
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            validate_checkpoint_parameters(self.extractor, self.checkpoints["superpoint"])
        else:
            self.extractor = EmptySafeSIFT(backend="opencv", rootsift=True,
                                           max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = None
        self.superglue_matcher = None
        self.superglue_provenance = None
        if config.matcher in LEARNED_MATCHERS:
            self.ensure_matcher(config.matcher)
        if manifest_dir is not None:
            self.save_manifest(manifest_dir, config.name)

    def ensure_matcher(self, strategy="lightglue"):
        if strategy == "superglue":
            if self.feature_source != "superpoint":
                raise ValueError("SuperGlue requires SuperPoint features")
            if self.superglue_matcher is None:
                from superglue_adapter import load_matcher, require_shared_superpoint
                require_shared_superpoint(self.checkpoints["superpoint"])
                self.superglue_matcher, self.superglue_provenance = load_matcher(self.device)
            return
        if strategy != "lightglue":
            raise ValueError("Unsupported learned matcher")
        if self.matcher is None:
            kind = f"{self.feature_source}_lightglue"
            self.checkpoints[kind] = require_checkpoint(kind)
            self.matcher = LightGlue(features=self.feature_source).eval().to(self.device)
            validate_checkpoint_parameters(self.matcher, self.checkpoints[kind], lightglue=True)

    def save_manifest(self, directory, method):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "implementation": "https://github.com/cvg/LightGlue", "commit": PINNED_COMMIT,
            "extractor_source": self.feature_source, "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "extractor": vars(self.extractor.conf),
            "matcher": vars(self.matcher.conf) if self.matcher is not None and "lightglue" in method else "CPU BF",
            "resize": None,
            "checkpoint_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for key, path in self.checkpoints.items()},
            "checkpoint_sources": {key: RELEASE_URL + CHECKPOINTS[key][1] for key in self.checkpoints},
            "timing": "synchronized extraction/matching; excludes model load, transfers, post-filtering",
            "sift_adapter": "official OpenCV-compatible SIFT; empty-safe; scales=kp.size, oris=radians",
            "normalization": "official RootSIFT" if self.feature_source == "sift" else "SuperPoint L2",
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "checkpoint_parameter_coverage_checked": True,
        }
        if method == "superpoint_superglue":
            metadata["matcher"] = dict(self.superglue_matcher.config)
            metadata["matcher_provenance"] = self.superglue_provenance
            metadata["checkpoint_sha256"]["superpoint_superglue"] = self.superglue_provenance["files"]["models/weights/superglue_outdoor.pth"]["sha256"]
            metadata["checkpoint_sources"]["superpoint_superglue"] = self.superglue_provenance["files"]["models/weights/superglue_outdoor.pth"]["url"]
            metadata["input_contract"] = "shared pixel keypoints and scores; descriptors transposed BND to BDN; official internal coordinate normalization"
        (directory / f"{method}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def extract(self, image_gray, warmup=2, repetitions=5):
        # Compatible SIFT is executed on CPU; move native results after timing.
        extract_device = self.device if self.feature_source == "superpoint" else torch.device("cpu")
        tensor = torch.from_numpy(image_gray.astype(np.float32) / 255)[None].to(extract_device)
        sync = self.synchronize if self.feature_source == "superpoint" else None
        native, mean_ms, std_ms = measure(lambda: self.extractor.extract(tensor, resize=None),
                                          warmup, repetitions, sync)
        native = {key: value.to(self.device) for key, value in native.items()}
        validate_native(native, self.feature_source, self.device)
        points = native["keypoints"][0].cpu().numpy()
        descriptors = native["descriptors"][0].cpu().numpy()
        if self.feature_source == "sift":
            scales = native["scales"][0].cpu().numpy()
            angles = np.rad2deg(native["oris"][0].cpu().numpy())
            keypoints = [cv2.KeyPoint(float(x), float(y), float(s), float(a))
                         for (x, y), s, a in zip(points, scales, angles)]
        else:
            keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points]
        native["native_indices"] = torch.arange(len(points), device=self.device)[None]
        timings = {"detection_ms": np.nan, "detection_std_ms": np.nan,
                   "description_ms": np.nan, "description_std_ms": np.nan,
                   "joint_extraction_ms": mean_ms, "joint_extraction_std_ms": std_ms}
        return ImageFeatures(keypoints, keypoints, descriptors, timings, native,
                             detected_before_limit=np.nan)

    @torch.inference_mode()
    def match(self, first, second, ratio_threshold=0.8, warmup=2, repetitions=5,
              matching_strategy=None):
        strategy = matching_strategy or (self.config.matcher if self.config.matcher in LEARNED_MATCHERS else "ratio")
        if strategy in ("ratio", "crosscheck"):
            if self.config.matcher in LEARNED_MATCHERS:
                raise ValueError("BF strategy is unsupported for a learned matcher configuration")
            return match_classical_features(first, second, self.config, ratio_threshold,
                                            warmup, repetitions, strategy)
        if strategy not in LEARNED_MATCHERS:
            raise ValueError("Unsupported matching strategy")
        self.ensure_matcher(strategy)
        validate_native(first.native, self.feature_source, self.device)
        validate_native(second.native, self.feature_source, self.device)
        if not first.keypoints or not second.keypoints:
            return [], {"matching_ms": 0.0, "matching_std_ms": 0.0}
        if strategy == "superglue":
            from superglue_adapter import matcher_inputs, prediction_matches as convert
            inputs = matcher_inputs(first.native, second.native)
            matcher = self.superglue_matcher
        else:
            inputs = {"image0": first.native, "image1": second.native}
            matcher, convert = self.matcher, prediction_matches
        prediction, mean_ms, std_ms = measure(lambda: matcher(inputs), warmup, repetitions, self.synchronize)
        matches = convert(prediction, len(first.keypoints), len(second.keypoints))
        return matches, {"matching_ms": mean_ms, "matching_std_ms": std_ms}
