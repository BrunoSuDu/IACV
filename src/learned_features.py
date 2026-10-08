"""将官方 SuperPoint / LightGlue 输出转换为现有评估接口。"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from classical_pipeline import match_classical_features
from features import ImageFeatures
from matching import enforce_one_to_one
from timing import measure


class LearnedPipeline:
    """模型只加载一次；每张图和每对匹配都复用它们。"""

    def __init__(self, config, device, max_keypoints, manifest_dir, seed=0):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(seed)
        torch.set_num_threads(cv2.getNumThreads())
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 所有实验共用项目内的下载缓存，不写入用户全局模型目录。
        model_dir = Path(__file__).resolve().parents[1] / "models"
        torch.hub.set_dir(str(model_dir))
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = None
        if config.matcher == "lightglue":
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = {}
        for path in (model_dir / "checkpoints").glob("*.pth"):
            checkpoints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "implementation": "https://github.com/cvg/LightGlue",
            "commit": "eb42fee2d71449efb0aa5c10549752b5d75384d8",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "extractor": vars(self.extractor.conf),
            "matcher": vars(self.matcher.conf) if self.matcher else "CPU BF L2 ratio test",
            "resize": None, "checkpoint_sha256": checkpoints,
            "timing": "synchronized forward inference; excludes model load and CPU/GPU transfers",
        }
        (manifest_dir / f"{config.name}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def extract(self, image_gray, warmup=2, repetitions=10):
        tensor = torch.from_numpy(image_gray.astype(np.float32) / 255)[None].to(self.device)
        native, mean_ms, std_ms = measure(
            lambda: self.extractor.extract(tensor, resize=None),
            warmup, repetitions, self.synchronize,
        )
        points = native["keypoints"][0].cpu().numpy()
        descriptors = native["descriptors"][0].cpu().numpy()
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points]
        # SuperPoint 共享卷积网络，不能把联合耗时伪装成独立的 detection/description。
        timings = {
            "detection_ms": np.nan, "detection_std_ms": np.nan,
            "description_ms": np.nan, "description_std_ms": np.nan,
            "joint_extraction_ms": mean_ms, "joint_extraction_std_ms": std_ms,
        }
        # 上游模型内部已做 top-k，未截断点数没有暴露；明确记为缺失值。
        return ImageFeatures(keypoints, keypoints, descriptors, timings, native,
                             detected_before_limit=np.nan)

    @torch.inference_mode()
    def match(self, first, second, ratio_threshold=0.8, warmup=2, repetitions=10):
        if self.matcher is None:
            return match_classical_features(first, second, self.config, ratio_threshold,
                                            warmup, repetitions)
        if not first.keypoints or not second.keypoints:
            return [], {"matching_ms": 0.0, "matching_std_ms": 0.0}
        prediction, mean_ms, std_ms = measure(
            lambda: self.matcher({"image0": first.native, "image1": second.native}),
            warmup, repetitions, self.synchronize,
        )
        pairs = prediction["matches"][0].cpu().numpy()
        scores = prediction["scores"][0].cpu().numpy()
        # 1-confidence 越小越好，仅用于统一排序；它不是描述子距离。
        matches = [cv2.DMatch(int(query), int(train), float(1 - score))
                   for (query, train), score in zip(pairs, scores)]
        return enforce_one_to_one(matches), {"matching_ms": mean_ms, "matching_std_ms": std_ms}
